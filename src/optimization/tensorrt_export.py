"""
TensorRT Export Module
-----------------------
Export YOLO model to TensorRT for faster inference.

TensorRT is NVIDIA's optimizer that can make model
run 3-5x faster on NVIDIA GPUs.

Supported precisions:
    - FP32: full precision, most accurate
    - FP16: half precision, 2x faster, minimal accuracy loss
    - INT8: 8-bit integer, fastest but needs calibration

Requirements:
    - NVIDIA GPU with CUDA
    - TensorRT installed
    - OR just use ultralytics export (handles everything)

Author: Mehmet Demir
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import time


class TensorRTExporter:
    """
    Export YOLO model to TensorRT engine.
    
    The engine file is optimized for specific GPU.
    If you change GPU, you need re-export.
    
    Example:
        exporter = TensorRTExporter("best.pt")
        engine_path = exporter.export(precision="fp16")
        
        # then use engine for inference
        model = YOLO(engine_path)
    """
    
    def __init__(self, model_path: str):
        """
        Initialize exporter.
        
        Args:
            model_path: path to YOLO .pt file
        """
        self.model_path = model_path
        
        # check if model file exist
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # try import ultralytics
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        self.model = None
        self.engine_path = None
    
    def check_tensorrt(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt
            print(f"TensorRT version: {tensorrt.__version__}")
            return True
        except ImportError:
            print("TensorRT not found. Install it for best performance.")
            return False
    
    def export(
        self,
        output_path: Optional[str] = None,
        precision: str = "fp16",
        imgsz: int = 640,
        batch_size: int = 1,
        workspace: int = 4  # GB
    ) -> str:
        """
        Export model to TensorRT.
        
        Args:
            output_path: where to save engine (auto if None)
            precision: 'fp32', 'fp16', or 'int8'
            imgsz: input image size
            batch_size: batch size for engine
            workspace: GPU memory for building (GB)
        
        Returns:
            Path to exported engine file
        """
        print(f"Starting TensorRT export...")
        print(f"  Model: {self.model_path}")
        print(f"  Precision: {precision}")
        print(f"  Image size: {imgsz}")
        
        # load model
        self.model = self.YOLO(self.model_path)
        
        # determine export settings
        half = precision in ["fp16", "fp16"]
        int8 = precision == "int8"
        
        start_time = time.time()
        
        try:
            # ultralytics handles the export
            # it will create .engine file
            export_path = self.model.export(
                format="engine",
                imgsz=imgsz,
                half=half,
                int8=int8,
                batch=batch_size,
                workspace=workspace,
                verbose=True
            )
            
            export_time = time.time() - start_time
            
            print(f"\nExport complete!")
            print(f"  Time: {export_time:.1f} seconds")
            print(f"  Output: {export_path}")
            
            self.engine_path = export_path
            return export_path
            
        except Exception as e:
            print(f"Export failed: {e}")
            raise
    
    def benchmark(
        self,
        engine_path: Optional[str] = None,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark the exported engine.
        
        Compare with original PyTorch model.
        """
        import numpy as np
        
        engine_path = engine_path or self.engine_path
        
        if engine_path is None:
            raise ValueError("No engine path. Run export() first.")
        
        # load both models
        pytorch_model = self.YOLO(self.model_path)
        trt_model = self.YOLO(engine_path)
        
        # create dummy input
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # warmup
        print("Warming up...")
        for _ in range(warmup):
            pytorch_model.predict(dummy_input, verbose=False)
            trt_model.predict(dummy_input, verbose=False)
        
        # benchmark pytorch
        print("Benchmarking PyTorch...")
        pytorch_times = []
        for _ in range(num_iterations):
            start = time.time()
            pytorch_model.predict(dummy_input, verbose=False)
            pytorch_times.append(time.time() - start)
        
        # benchmark tensorrt
        print("Benchmarking TensorRT...")
        trt_times = []
        for _ in range(num_iterations):
            start = time.time()
            trt_model.predict(dummy_input, verbose=False)
            trt_times.append(time.time() - start)
        
        # calculate stats
        pytorch_avg = sum(pytorch_times) / len(pytorch_times) * 1000
        trt_avg = sum(trt_times) / len(trt_times) * 1000
        speedup = pytorch_avg / trt_avg
        
        results = {
            "pytorch_ms": pytorch_avg,
            "tensorrt_ms": trt_avg,
            "speedup": speedup,
            "pytorch_fps": 1000 / pytorch_avg,
            "tensorrt_fps": 1000 / trt_avg
        }
        
        print(f"\nBenchmark Results:")
        print(f"  PyTorch:  {pytorch_avg:.2f}ms ({results['pytorch_fps']:.1f} FPS)")
        print(f"  TensorRT: {trt_avg:.2f}ms ({results['tensorrt_fps']:.1f} FPS)")
        print(f"  Speedup:  {speedup:.2f}x")
        
        return results


class TensorRTInference:
    """
    Helper class for running inference with TensorRT engine.
    
    This is basically thin wrapper around YOLO with engine.
    """
    
    def __init__(self, engine_path: str):
        """Load TensorRT engine."""
        from ultralytics import YOLO
        
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        self.model = YOLO(engine_path)
        self.engine_path = engine_path
    
    def predict(self, image, conf: float = 0.25, **kwargs):
        """Run inference on image."""
        return self.model.predict(image, conf=conf, **kwargs)
    
    def track(self, source, tracker: str = "botsort.yaml", **kwargs):
        """Run tracking on video."""
        return self.model.track(source, tracker=tracker, **kwargs)


# test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tensorrt_export.py <model.pt>")
        print("Example: python tensorrt_export.py best.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    exporter = TensorRTExporter(model_path)
    
    # check tensorrt
    if exporter.check_tensorrt():
        # export
        engine_path = exporter.export(precision="fp16")
        
        # benchmark
        exporter.benchmark(engine_path)
    else:
        print("Install TensorRT to export model")
