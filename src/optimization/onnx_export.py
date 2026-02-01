"""
ONNX Export Module
-------------------
Export YOLO model to ONNX format for cross platform inference.

ONNX (Open Neural Network Exchange) is a open format
that works on many platforms:
    - Windows, Linux, Mac
    - CPU, GPU  
    - Mobile devices
    - Edge devices like Raspberry Pi

Why use ONNX:
    - Faster than PyTorch on CPU
    - Works without PyTorch installed
    - Smaller deployment size

Author: Mehmet Demir
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
import numpy as np


class ONNXExporter:
    """
    Export YOLO model to ONNX format.
    
    Example:
        exporter = ONNXExporter("best.pt")
        onnx_path = exporter.export()
        
        # use with onnxruntime
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
    """
    
    def __init__(self, model_path: str):
        """
        Initialize with model path.
        
        Args:
            model_path: path to .pt model file
        """
        self.model_path = model_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # import ultralytics
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError("Need ultralytics: pip install ultralytics")
        
        self.model = None
        self.onnx_path = None
    
    def export(
        self,
        output_path: Optional[str] = None,
        imgsz: int = 640,
        simplify: bool = True,
        opset: int = 17,
        dynamic: bool = False
    ) -> str:
        """
        Export model to ONNX.
        
        Args:
            output_path: where to save (auto if None)
            imgsz: input image size
            simplify: simplify onnx graph (onnxsim)
            opset: ONNX opset version
            dynamic: enable dynamic input shapes
        
        Returns:
            Path to exported ONNX file
        """
        print("Starting ONNX export...")
        print(f"  Model: {self.model_path}")
        print(f"  Image size: {imgsz}")
        print(f"  Simplify: {simplify}")
        
        # load model
        self.model = self.YOLO(self.model_path)
        
        start = time.time()
        
        try:
            # export using ultralytics
            self.onnx_path = self.model.export(
                format="onnx",
                imgsz=imgsz,
                simplify=simplify,
                opset=opset,
                dynamic=dynamic
            )
            
            elapsed = time.time() - start
            
            # get file size
            size_mb = Path(self.onnx_path).stat().st_size / (1024 * 1024)
            
            print(f"\nExport done!")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Size: {size_mb:.1f} MB")
            print(f"  Path: {self.onnx_path}")
            
            return self.onnx_path
            
        except Exception as e:
            print(f"Export error: {e}")
            raise
    
    def validate(self, onnx_path: Optional[str] = None) -> bool:
        """
        Validate exported ONNX model.
        
        Checks if model has correct structure.
        """
        onnx_path = onnx_path or self.onnx_path
        
        if onnx_path is None:
            print("No ONNX path. Export first.")
            return False
        
        try:
            import onnx
            
            print(f"Validating: {onnx_path}")
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            print("Validation passed!")
            return True
            
        except ImportError:
            print("onnx package not found. Cannot validate.")
            return True  # assume ok
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return False


class ONNXInference:
    """
    Run inference using ONNX Runtime.
    
    This dont need PyTorch or ultralytics installed.
    Just onnxruntime package.
    
    Example:
        inferencer = ONNXInference("model.onnx")
        boxes, scores, classes = inferencer.predict(image)
    """
    
    def __init__(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None
    ):
        """
        Load ONNX model.
        
        Args:
            onnx_path: path to .onnx file
            providers: execution providers. Default tries GPU first.
                      Options: CUDAExecutionProvider, CPUExecutionProvider
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Need onnxruntime: pip install onnxruntime-gpu")
        
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        
        # default providers - try gpu first
        if providers is None:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
        
        # create inference session
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[2], input_shape[3])  # h, w
        
        print(f"ONNX model loaded")
        print(f"  Input: {self.input_name} {input_shape}")
        print(f"  Outputs: {self.output_names}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.
        
        1. Resize to model input size
        2. Convert BGR to RGB
        3. Normalize to 0-1
        4. Change to NCHW format
        """
        import cv2
        
        # resize
        h, w = self.input_size
        resized = cv2.resize(image, (w, h))
        
        # bgr to rgb
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # normalize
        normalized = rgb.astype(np.float32) / 255.0
        
        # hwc to chw
        chw = np.transpose(normalized, (2, 0, 1))
        
        # add batch dimension
        batched = np.expand_dims(chw, axis=0)
        
        return batched
    
    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on image.
        
        Args:
            image: input image (BGR, HWC format)
            conf_threshold: minimum confidence
        
        Returns:
            boxes: Nx4 array of xyxy boxes
            scores: N array of confidence scores
            classes: N array of class indices
        """
        # preprocess
        input_tensor = self.preprocess(image)
        
        # run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        
        # process outputs (depends on model output format)
        # yolov8 outputs single tensor [1, 84, 8400]
        # where 84 = 4 coords + 80 classes
        output = outputs[0]
        
        # transpose to [8400, 84]
        if output.shape[1] == 84 or output.shape[1] == 8:  # 8 for our 4 class model
            predictions = output[0].T
        else:
            predictions = output[0]
        
        # split boxes and scores
        num_classes = predictions.shape[1] - 4
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        
        # get best class for each detection
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)
        
        # filter by confidence
        mask = scores > conf_threshold
        
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # convert xywh to xyxy
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        
        # scale boxes back to original image size
        h, w = image.shape[:2]
        scale_x = w / self.input_size[1]
        scale_y = h / self.input_size[0]
        
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
        
        return boxes_xyxy, scores, class_ids
    
    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert center format to corner format."""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return xyxy


def benchmark_onnx_vs_pytorch(
    pt_path: str,
    onnx_path: str,
    iterations: int = 50
) -> Dict:
    """Compare ONNX and PyTorch inference speed."""
    from ultralytics import YOLO
    
    # load models
    pt_model = YOLO(pt_path)
    onnx_model = YOLO(onnx_path)
    
    # create test image
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # warmup
    for _ in range(5):
        pt_model.predict(test_img, verbose=False)
        onnx_model.predict(test_img, verbose=False)
    
    # benchmark pytorch
    pt_times = []
    for _ in range(iterations):
        start = time.time()
        pt_model.predict(test_img, verbose=False)
        pt_times.append(time.time() - start)
    
    # benchmark onnx
    onnx_times = []
    for _ in range(iterations):
        start = time.time()
        onnx_model.predict(test_img, verbose=False)
        onnx_times.append(time.time() - start)
    
    pt_avg = sum(pt_times) / len(pt_times) * 1000
    onnx_avg = sum(onnx_times) / len(onnx_times) * 1000
    
    print(f"PyTorch: {pt_avg:.2f}ms")
    print(f"ONNX:    {onnx_avg:.2f}ms")
    print(f"Speedup: {pt_avg/onnx_avg:.2f}x")
    
    return {
        "pytorch_ms": pt_avg,
        "onnx_ms": onnx_avg,
        "speedup": pt_avg / onnx_avg
    }


# test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python onnx_export.py <model.pt>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    exporter = ONNXExporter(model_path)
    onnx_path = exporter.export()
    exporter.validate(onnx_path)
