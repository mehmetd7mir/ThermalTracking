"""
C++ Extensions Wrapper Module
------------------------------
Python wrapper for C++ performance extensions.

This module provide Python interface to fast C++ implementations.
If C++ extensions not built, it falls back to pure Python.

Extensions included:
    - nms_cpp: Fast Non-Maximum Suppression
    - iou_cpp: Vectorized IoU calculation
    - kalman_cpp: High performance Kalman filter

Build with:
    python setup.py build_ext --inplace

Author: Mehmet Demir
"""

import numpy as np
from typing import List, Tuple, Optional

# try to import compiled extensions
CPP_AVAILABLE = False

try:
    from . import nms_cpp
    from . import iou_cpp
    CPP_AVAILABLE = True
    print("C++ extensions loaded successfully")
except ImportError:
    print("C++ extensions not built. Using Python fallback.")
    print("To build: cd cpp_extensions && python setup.py build_ext --inplace")


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> np.ndarray:
    """
    Non-Maximum Suppression.
    
    Remove overlapping boxes based on IoU threshold.
    
    Args:
        boxes: Nx4 array of [x1, y1, x2, y2]
        scores: N array of confidence scores
        iou_threshold: boxes with IoU > threshold get removed
    
    Returns:
        indices of kept boxes
    """
    if CPP_AVAILABLE:
        # use C++ implementation
        return nms_cpp.nms(boxes, scores, iou_threshold)
    else:
        # python fallback
        return _nms_python(boxes, scores, iou_threshold)


def _nms_python(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float
) -> np.ndarray:
    """Pure Python NMS implementation."""
    
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    # get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # calculate area
    areas = (x2 - x1) * (y2 - y1)
    
    # sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0:
        # pick highest score
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # calculate IoU with rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-6)
        
        # keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return np.array(keep, dtype=np.int64)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    if CPP_AVAILABLE:
        return iou_cpp.calculate_iou(box1, box2)
    else:
        return _iou_python(box1, box2)


def _iou_python(box1: np.ndarray, box2: np.ndarray) -> float:
    """Python IoU implementation."""
    # intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # union
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def batch_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU matrix between two sets of boxes.
    
    Args:
        boxes1: Nx4 array
        boxes2: Mx4 array
    
    Returns:
        NxM IoU matrix
    """
    if CPP_AVAILABLE:
        return iou_cpp.batch_iou(boxes1, boxes2)
    else:
        return _batch_iou_python(boxes1, boxes2)


def _batch_iou_python(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Python batch IoU."""
    N = len(boxes1)
    M = len(boxes2)
    
    iou_matrix = np.zeros((N, M), dtype=np.float32)
    
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = _iou_python(boxes1[i], boxes2[j])
    
    return iou_matrix


class KalmanFilterCpp:
    """
    Kalman Filter for object tracking.
    
    Uses C++ implementation if available for speed.
    State: [x, y, vx, vy] - position and velocity
    """
    
    def __init__(self, initial_state: Optional[np.ndarray] = None):
        """
        Initialize filter.
        
        Args:
            initial_state: [x, y, vx, vy] or None
        """
        if initial_state is not None:
            self.state = initial_state.astype(np.float32)
        else:
            self.state = np.zeros(4, dtype=np.float32)
        
        # state transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # measurement matrix (we only measure x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # covariance matrix
        self.P = np.eye(4, dtype=np.float32) * 100
        
        # process noise
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        
        # measurement noise
        self.R = np.eye(2, dtype=np.float32) * 1.0
    
    def predict(self) -> np.ndarray:
        """
        Predict next state.
        
        Returns:
            Predicted [x, y, vx, vy]
        """
        # state prediction
        self.state = self.F @ self.state
        
        # covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[:2]  # return only position
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with measurement.
        
        Args:
            measurement: [x, y] measured position
        
        Returns:
            Updated [x, y, vx, vy]
        """
        # measurement residual
        z = np.array(measurement, dtype=np.float32)
        y = z - self.H @ self.state
        
        # residual covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # state update
        self.state = self.state + K @ y
        
        # covariance update
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return (float(self.state[0]), float(self.state[1]))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return (float(self.state[2]), float(self.state[3]))


def benchmark_nms(num_boxes: int = 1000, iterations: int = 100):
    """Benchmark NMS performance."""
    import time
    
    # generate random boxes
    boxes = np.random.rand(num_boxes, 4).astype(np.float32)
    boxes[:, 2:] += boxes[:, :2]  # make sure x2 > x1, y2 > y1
    boxes *= 640
    
    scores = np.random.rand(num_boxes).astype(np.float32)
    
    # warmup
    for _ in range(5):
        nms(boxes, scores, 0.45)
    
    # benchmark
    start = time.time()
    for _ in range(iterations):
        nms(boxes, scores, 0.45)
    elapsed = time.time() - start
    
    avg_ms = (elapsed / iterations) * 1000
    
    print(f"NMS Benchmark ({num_boxes} boxes, {iterations} iterations)")
    print(f"  Average time: {avg_ms:.3f} ms")
    print(f"  Using C++: {CPP_AVAILABLE}")
    
    return avg_ms


# test
if __name__ == "__main__":
    # test NMS
    print("Testing NMS...")
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],  # overlaps with first
        [100, 100, 150, 150],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    
    keep = nms(boxes, scores, 0.5)
    print(f"Kept indices: {keep}")
    
    # test IoU
    print("\nTesting IoU...")
    iou = calculate_iou(boxes[0], boxes[1])
    print(f"IoU between box 0 and 1: {iou:.3f}")
    
    # test Kalman
    print("\nTesting Kalman Filter...")
    kf = KalmanFilterCpp(np.array([0, 0, 1, 1]))
    for i in range(5):
        pred = kf.predict()
        print(f"  Prediction {i}: {pred}")
        kf.update(pred + np.random.randn(2) * 0.1)
    
    # benchmark
    print()
    benchmark_nms(num_boxes=100, iterations=50)
