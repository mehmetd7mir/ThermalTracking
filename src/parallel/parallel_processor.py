"""
ThermalTracking - Multi-threaded Video Processing Module
=========================================================
Paralel video işleme için multi-threading altyapısı.

Özellikler:
- Producer-Consumer pattern
- Thread-safe frame queue
- Paralel inference
- Real-time video display

Kullanım:
    python parallel_processor.py --source video.mp4 --workers 4
"""

import argparse
import threading
import queue
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class FrameData:
    """Frame verisi container'ı"""
    frame: np.ndarray
    frame_id: int
    timestamp: float


@dataclass
class ResultData:
    """İşlenmiş frame verisi"""
    frame: np.ndarray
    frame_id: int
    detections: list
    processing_time: float


class ThreadSafeCounter:
    """Thread-safe sayaç"""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class VideoReader(threading.Thread):
    """
    Video okuma thread'i (Producer).
    Frame'leri queue'ya ekler.
    """
    
    def __init__(
        self,
        source: str,
        frame_queue: queue.Queue,
        max_queue_size: int = 30,
        skip_frames: int = 0
    ):
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.skip_frames = skip_frames
        
        self.running = True
        self.frame_count = 0
        self.fps = 30.0
        
    def run(self):
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"❌ Video açılamadı: {self.source}")
            return
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video Reader başlatıldı: {total_frames} frames @ {self.fps}fps")
        
        frame_id = 0
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_id += 1
            
            # Frame skip
            if self.skip_frames > 0 and frame_id % (self.skip_frames + 1) != 0:
                continue
            
            self.frame_count += 1
            
            # Queue dolu ise bekle
            while self.frame_queue.qsize() >= self.max_queue_size and self.running:
                time.sleep(0.01)
            
            if not self.running:
                break
            
            # Frame'i queue'ya ekle
            frame_data = FrameData(
                frame=frame,
                frame_id=frame_id,
                timestamp=time.time()
            )
            self.frame_queue.put(frame_data)
        
        cap.release()
        
        # Sentinel değer (işlem bitti sinyali)
        self.frame_queue.put(None)
        print(f"✅ Video Reader tamamlandı: {self.frame_count} frames")
    
    def stop(self):
        self.running = False


class InferenceWorker(threading.Thread):
    """
    Inference worker thread'i (Consumer).
    Frame'leri işler ve sonuçları output queue'ya ekler.
    """
    
    def __init__(
        self,
        worker_id: int,
        model: YOLO,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        conf_threshold: float = 0.25
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.conf_threshold = conf_threshold
        
        self.running = True
        self.processed_count = 0
        self.total_time = 0.0
    
    def run(self):
        print(f"🔧 Worker-{self.worker_id} başlatıldı")
        
        while self.running:
            try:
                # Frame al (timeout ile)
                frame_data = self.input_queue.get(timeout=1.0)
                
                if frame_data is None:
                    # Sentinel - diğer worker'lara da ilet
                    self.input_queue.put(None)
                    break
                
                start_time = time.time()
                
                # Inference yap
                results = self.model.predict(
                    frame_data.frame,
                    conf=self.conf_threshold,
                    verbose=False
                )
                
                # Annotated frame
                annotated = results[0].plot()
                
                # Detections
                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        detections.append({
                            'class': int(box.cls),
                            'conf': float(box.conf),
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        })
                
                processing_time = time.time() - start_time
                self.total_time += processing_time
                self.processed_count += 1
                
                # Sonucu output queue'ya ekle
                result_data = ResultData(
                    frame=annotated,
                    frame_id=frame_data.frame_id,
                    detections=detections,
                    processing_time=processing_time
                )
                self.output_queue.put(result_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Worker-{self.worker_id} hata: {e}")
        
        avg_time = self.total_time / max(1, self.processed_count)
        print(f"✅ Worker-{self.worker_id} tamamlandı: {self.processed_count} frames, avg {avg_time*1000:.1f}ms")
    
    def stop(self):
        self.running = False


class VideoWriter(threading.Thread):
    """
    Video yazma thread'i.
    İşlenmiş frame'leri sıralı olarak yazar.
    """
    
    def __init__(
        self,
        output_path: str,
        result_queue: queue.Queue,
        fps: float = 30.0,
        frame_size: tuple = (640, 480)
    ):
        super().__init__(daemon=True)
        self.output_path = output_path
        self.result_queue = result_queue
        self.fps = fps
        self.frame_size = frame_size
        
        self.running = True
        self.written_count = 0
        
        # Frame buffer (sıralı yazım için)
        self.frame_buffer: Dict[int, ResultData] = {}
        self.next_frame_id = 1
    
    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = None
        
        print(f"💾 Video Writer başlatıldı: {self.output_path}")
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=1.0)
                
                if result is None:
                    break
                
                # Writer'ı ilk frame'de oluştur
                if writer is None:
                    h, w = result.frame.shape[:2]
                    writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
                
                # Buffer'a ekle
                self.frame_buffer[result.frame_id] = result
                
                # Sıralı yazım
                while self.next_frame_id in self.frame_buffer:
                    frame_result = self.frame_buffer.pop(self.next_frame_id)
                    writer.write(frame_result.frame)
                    self.written_count += 1
                    self.next_frame_id += 1
                
            except queue.Empty:
                continue
        
        # Kalan frame'leri yaz
        for frame_id in sorted(self.frame_buffer.keys()):
            writer.write(self.frame_buffer[frame_id].frame)
            self.written_count += 1
        
        if writer:
            writer.release()
        
        print(f"✅ Video Writer tamamlandı: {self.written_count} frames yazıldı")
    
    def stop(self):
        self.running = False


class ParallelVideoProcessor:
    """
    Paralel video işleme orchestrator'ı.
    
    Mimari:
    VideoReader -> [Input Queue] -> [InferenceWorkers] -> [Output Queue] -> VideoWriter
    """
    
    def __init__(
        self,
        weights: str = "best.pt",
        num_workers: int = 2,
        conf_threshold: float = 0.25,
        queue_size: int = 30
    ):
        self.weights = weights
        self.num_workers = num_workers
        self.conf_threshold = conf_threshold
        self.queue_size = queue_size
        
        # Model (her worker için ayrı kopya gerekebilir)
        self.model = YOLO(weights)
        
        # Queues
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        
        # Threads
        self.reader = None
        self.workers = []
        self.writer = None
        
        # Stats
        self.start_time = 0
        self.end_time = 0
    
    def process(
        self,
        source: str,
        output_path: Optional[str] = None,
        show: bool = False
    ) -> Dict[str, Any]:
        """
        Video'yu paralel olarak işle.
        
        Args:
            source: Kaynak video
            output_path: Çıktı video yolu
            show: Canlı göster
        
        Returns:
            İstatistikler
        """
        print(f"\n🚀 Paralel İşleme Başlıyor")
        print(f"   Workers: {self.num_workers}")
        print(f"   Queue Size: {self.queue_size}")
        print("-" * 40)
        
        self.start_time = time.time()
        
        # Video Reader başlat
        self.reader = VideoReader(source, self.input_queue)
        self.reader.start()
        
        # Inference Workers başlat
        self.workers = []
        for i in range(self.num_workers):
            worker = InferenceWorker(
                worker_id=i,
                model=self.model,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                conf_threshold=self.conf_threshold
            )
            worker.start()
            self.workers.append(worker)
        
        # Video Writer başlat (eğer kayıt isteniyorsa)
        if output_path:
            self.writer = VideoWriter(
                output_path=output_path,
                result_queue=self.output_queue,
                fps=self.reader.fps
            )
            self.writer.start()
        
        # Tüm thread'lerin bitmesini bekle
        self.reader.join()
        
        for worker in self.workers:
            worker.join()
        
        # Writer'a sonlanma sinyali gönder
        if self.writer:
            self.output_queue.put(None)
            self.writer.join()
        
        self.end_time = time.time()
        
        # İstatistikler
        total_time = self.end_time - self.start_time
        total_frames = self.reader.frame_count
        avg_fps = total_frames / total_time if total_time > 0 else 0
        
        stats = {
            'total_frames': total_frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'num_workers': self.num_workers
        }
        
        print("\n" + "=" * 40)
        print("📊 İşlem Tamamlandı!")
        print(f"   Toplam frame: {total_frames}")
        print(f"   Toplam süre: {total_time:.2f}s")
        print(f"   Ortalama FPS: {avg_fps:.1f}")
        print(f"   Speedup: ~{self.num_workers}x (teorik)")
        print("=" * 40)
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="ThermalTracking - Paralel Video İşleme"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Kaynak video"
    )
    parser.add_argument(
        "--weights", type=str, default="best.pt",
        help="Model ağırlıkları"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Worker sayısı"
    )
    parser.add_argument(
        "--output", type=str, default="output_parallel.mp4",
        help="Çıktı video"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold"
    )
    
    args = parser.parse_args()
    
    processor = ParallelVideoProcessor(
        weights=args.weights,
        num_workers=args.workers,
        conf_threshold=args.conf
    )
    
    processor.process(
        source=args.source,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
