"""
ThermalTracking - BoTSORT Multi-Object Tracking Module
=======================================================
Thermal görüntülerde nesne takibi ve trajectory analizi.

BoTSORT Avantajları:
- Camera Motion Compensation (CMC)
- Re-identification (ReID) desteği
- ByteTrack'ten daha yüksek MOTA/IDF1

Kullanım:
    python tracker.py --source video.mp4 --weights best.pt
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class ThermalTracker:
    """
    BoTSORT tabanlı multi-object tracker.
    Thermal görüntülerde hava araçlarını takip eder.
    """
    
    def __init__(
        self,
        weights: str = "best.pt",
        tracker_config: str = "botsort.yaml",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ):
        """
        Args:
            weights: YOLOv8 model ağırlıkları
            tracker_config: Tracker konfigürasyonu (botsort.yaml veya bytetrack.yaml)
            conf_threshold: Minimum confidence
            iou_threshold: NMS IoU threshold
        """
        self.model = YOLO(weights)
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Track history: {track_id: [(x, y, frame_num), ...]}
        self.track_history: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
        
        # Class names
        self.class_names = {0: 'bird', 1: 'drone', 2: 'helicopter', 3: 'plane'}
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'unique_tracks': set(),
            'class_counts': defaultdict(int)
        }
    
    def get_center(self, box: np.ndarray) -> Tuple[float, float]:
        """Bounding box'ın merkez koordinatlarını hesapla."""
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update_track_history(
        self,
        track_id: int,
        center: Tuple[float, float],
        frame_num: int,
        max_history: int = 50
    ):
        """Track history'yi güncelle."""
        self.track_history[track_id].append((*center, frame_num))
        
        # Eski kayıtları temizle
        if len(self.track_history[track_id]) > max_history:
            self.track_history[track_id].pop(0)
    
    def calculate_velocity(
        self,
        track_id: int,
        fps: float = 30.0,
        scale_factor: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Track için hız hesapla.
        
        Args:
            track_id: Takip ID'si
            fps: Video frame rate
            scale_factor: Pixel -> metre dönüşüm faktörü
        
        Returns:
            (velocity_px_sec, velocity_m_sec, direction_degrees)
        """
        history = self.track_history.get(track_id, [])
        
        if len(history) < 2:
            return (0.0, 0.0, 0.0)
        
        # Son iki nokta
        p1 = history[-2]
        p2 = history[-1]
        
        # Displacement
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Frame farkı
        frame_diff = p2[2] - p1[2]
        if frame_diff == 0:
            frame_diff = 1
        
        # Pixel cinsinden hız
        distance_px = math.sqrt(dx**2 + dy**2)
        velocity_px_sec = (distance_px / frame_diff) * fps
        
        # Metre cinsinden hız
        velocity_m_sec = velocity_px_sec * scale_factor
        
        # Hareket yönü (derece)
        direction = math.degrees(math.atan2(dy, dx))
        
        return (velocity_px_sec, velocity_m_sec, direction)
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        results,
        show_trajectory: bool = True,
        show_velocity: bool = True,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        Frame üzerine tracking bilgilerini çiz.
        
        Args:
            frame: Video frame
            results: YOLO tracking results
            show_trajectory: Trajectory çizgisi göster
            show_velocity: Hız bilgisi göster
            fps: Video FPS
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return annotated
        
        boxes = results[0].boxes
        
        for box in boxes:
            # Tracking ID
            if box.id is None:
                continue
            
            track_id = int(box.id)
            
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Class ve confidence
            cls_id = int(box.cls)
            conf = float(box.conf)
            cls_name = self.class_names.get(cls_id, 'unknown')
            
            # Merkez
            center = self.get_center(box.xyxy[0].cpu().numpy())
            
            # Renk (class'a göre)
            colors = {
                0: (0, 255, 0),    # bird - yeşil
                1: (0, 0, 255),    # drone - kırmızı
                2: (255, 0, 0),    # helicopter - mavi
                3: (0, 255, 255)   # plane - sarı
            }
            color = colors.get(cls_id, (255, 255, 255))
            
            # Bounding box çiz
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"ID:{track_id} {cls_name} {conf:.2f}"
            
            # Hız hesapla
            if show_velocity:
                vel_px, vel_m, direction = self.calculate_velocity(track_id, fps)
                if vel_px > 0:
                    label += f" | {vel_px:.1f}px/s"
            
            # Label çiz
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-20), (x1+w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Trajectory çiz
            if show_trajectory:
                history = self.track_history.get(track_id, [])
                if len(history) > 1:
                    points = [(int(p[0]), int(p[1])) for p in history]
                    for i in range(1, len(points)):
                        # Gradient renk (eski = soluk, yeni = parlak)
                        alpha = i / len(points)
                        line_color = tuple(int(c * alpha) for c in color)
                        cv2.line(annotated, points[i-1], points[i], line_color, 2)
            
            # Merkez nokta
            cv2.circle(annotated, (int(center[0]), int(center[1])), 4, color, -1)
        
        return annotated
    
    def process_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        show: bool = False,
        save: bool = True
    ) -> Dict:
        """
        Video üzerinde tracking yap.
        
        Args:
            source: Video dosyası veya kamera (0)
            output_path: Çıktı video yolu
            show: Canlı göster
            save: Kaydet
        
        Returns:
            İstatistikler
        """
        # Video aç
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {source}")
        
        # Video özellikleri
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Video writer
        writer = None
        if save and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                self.stats['total_frames'] = frame_num
                
                # Tracking yap
                results = self.model.track(
                    frame,
                    tracker=self.tracker_config,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    persist=True,
                    verbose=False
                )
                
                # Track history güncelle
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.id is not None:
                            track_id = int(box.id)
                            center = self.get_center(box.xyxy[0].cpu().numpy())
                            self.update_track_history(track_id, center, frame_num)
                            
                            # İstatistikler
                            self.stats['total_detections'] += 1
                            self.stats['unique_tracks'].add(track_id)
                            cls_id = int(box.cls)
                            self.stats['class_counts'][self.class_names.get(cls_id, 'unknown')] += 1
                
                # Frame annotate
                annotated = self.draw_tracks(results, results, fps=fps)
                
                # Göster
                if show:
                    cv2.imshow('ThermalTracker', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Kaydet
                if writer:
                    writer.write(annotated)
                
                # Progress
                if frame_num % 100 == 0:
                    print(f"  İşlenen: {frame_num}/{total_frames} ({100*frame_num/total_frames:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Final istatistikler
        self.stats['unique_tracks'] = len(self.stats['unique_tracks'])
        print(f"\n✅ Tamamlandı!")
        print(f"   Toplam frame: {self.stats['total_frames']}")
        print(f"   Toplam tespit: {self.stats['total_detections']}")
        print(f"   Benzersiz track: {self.stats['unique_tracks']}")
        print(f"   Sınıf dağılımı: {dict(self.stats['class_counts'])}")
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="ThermalTracking - BoTSORT Multi-Object Tracking"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Video dosyası veya kamera (0)"
    )
    parser.add_argument(
        "--weights", type=str, default="best.pt",
        help="YOLOv8 model ağırlıkları"
    )
    parser.add_argument(
        "--tracker", type=str, default="botsort.yaml",
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="Tracker konfigürasyonu"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="output_tracked.mp4",
        help="Çıktı video yolu"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Canlı göster"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Kaydetme"
    )
    
    args = parser.parse_args()
    
    # Tracker oluştur
    tracker = ThermalTracker(
        weights=args.weights,
        tracker_config=args.tracker,
        conf_threshold=args.conf
    )
    
    # Video işle
    tracker.process_video(
        source=args.source,
        output_path=args.output if not args.no_save else None,
        show=args.show,
        save=not args.no_save
    )


if __name__ == "__main__":
    main()
