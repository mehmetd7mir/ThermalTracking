# Future Work & Roadmap

## Overview

This document outlines planned features and improvements for the ThermalTracking project.

---

## 🎯 Phase 3: Tracking & Optimization

### 1. Multi-Object Tracking (MOT)

**Goal**: Track detected objects across video frames

**Implementation**: ByteTrack algorithm

```python
# Planned usage
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.track(
    source="video.mp4",
    tracker="bytetrack.yaml",
    persist=True
)
```

**Expected Output**:
- Unique ID for each tracked object
- Trajectory visualization
- Object re-identification after occlusion

---

### 2. Velocity Estimation

**Goal**: Calculate apparent speed of targets in pixels/second

**Method**:
1. Track object centroid across frames
2. Calculate displacement: `Δx, Δy`
3. Estimate velocity: `v = √(Δx² + Δy²) / Δt`

```python
def estimate_velocity(track_history, fps=30):
    """
    Estimate velocity from track history.
    
    Args:
        track_history: List of (x, y) centroids
        fps: Video frame rate
    
    Returns:
        velocity_px_sec: Speed in pixels/second
    """
    if len(track_history) < 2:
        return 0.0
    
    p1 = track_history[-2]
    p2 = track_history[-1]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    distance = math.sqrt(dx**2 + dy**2)
    velocity = distance * fps
    
    return velocity
```

---

### 3. ONNX Export

**Goal**: Export model for deployment on edge devices

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, half=True)
```

**Target Platforms**:
- NVIDIA Jetson (Nano, Xavier)
- Intel OpenVINO
- TensorRT

---

## 🔮 Phase 4: Advanced Features

### 1. Threat Classification

Classify detected objects by threat level:

| Level | Criteria |
|-------|----------|
| **High** | Fast-moving drone near restricted area |
| **Medium** | Unknown aircraft |
| **Low** | Birds, slow-moving objects |

### 2. Alert System

- Real-time notifications
- Integration with external systems
- Logging and reporting

### 3. Web Dashboard

```
┌─────────────────────────────────────────────┐
│           ThermalTracking Dashboard         │
├─────────────────────────────────────────────┤
│  [Live Feed]          │  [Detections]       │
│                       │                     │
│   🔴 Recording        │  Drone: 3           │
│                       │  Bird: 1            │
│                       │  Plane: 0           │
│                       │  Helicopter: 0      │
├───────────────────────┴─────────────────────┤
│  [Track History]      │  [Alerts]           │
│  ID-001: Active       │  ⚠️ New drone @NE   │
│  ID-002: Lost         │  ✅ Bird cleared    │
└─────────────────────────────────────────────┘
```

**Tech Stack**:
- Gradio or Streamlit
- FastAPI backend
- WebSocket for real-time updates

---

## 📅 Timeline

| Phase | Feature | Status | ETA |
|-------|---------|--------|-----|
| 2 | Model Training (50 epochs) | 🔄 In Progress | 1 week |
| 3.1 | ByteTrack Integration | 📋 Planned | 2 weeks |
| 3.2 | Velocity Estimation | 📋 Planned | 2 weeks |
| 3.3 | ONNX Export | 📋 Planned | 1 week |
| 4.1 | Threat Classification | 💡 Idea | TBD |
| 4.2 | Web Dashboard | 💡 Idea | TBD |

---

## 🤝 Contributions Welcome

If you're interested in contributing to any of these features, please open an issue or submit a PR!
