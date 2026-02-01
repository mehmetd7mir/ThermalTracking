# ThermalTracking

**YOLOv8-based Detection and Tracking of Aerial Targets in Thermal Imagery**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> A computer vision system for detecting and classifying aerial targets (drone, plane, helicopter, bird) in thermal/infrared imagery using deep learning.

![Sample Detection](results/sample_predictions/train_batch0.jpg)

---

## Project Overview

This project implements a **Data-Centric AI** approach to build a robust aerial target detection system for thermal imagery. Instead of relying on pre-made datasets, we engineered a custom "MegaSet" by aggregating, cleaning, and augmenting 11+ public thermal datasets.

### Key Features

- **4-Class Detection**: Drone, Plane, Helicopter, Bird
- **80,000+ Training Images**: Augmented thermal dataset
- **73.9% mAP50**: Current best performance
- **Real-time Tracking**: BoTSORT multi-object tracking
- **Edge Deployment**: TensorRT and ONNX export support
- **Alert System**: Telegram, Slack, Discord notifications
- **REST API**: FastAPI integration endpoints
- **Advanced Analytics**: Line counting, zone intrusion, trajectory prediction
- **Swarm Detection**: Detect coordinated drone groups
- **Anomaly Detection**: Loitering, erratic movement detection

---

## Project Structure

```
ThermalTracking/
├── src/
│   ├── analytics/          # Line counting, zone intrusion, trajectory
│   │   ├── line_counter.py
│   │   ├── zone_intrusion.py
│   │   └── trajectory_predictor.py
│   ├── optimization/       # Edge deployment exports
│   │   ├── tensorrt_export.py
│   │   └── onnx_export.py
│   ├── alerts/             # Notification system
│   │   └── alert_system.py
│   ├── api/                # REST API server
│   │   └── api_server.py
│   ├── dashboard/          # Monitoring UI
│   │   └── dashboard.py
│   ├── advanced/           # Swarm and anomaly detection
│   │   ├── swarm_detection.py
│   │   └── anomaly_detector.py
│   ├── cpp_extensions/     # Performance optimizations
│   │   └── cpp_wrapper.py
│   ├── tracking/           # BoTSORT tracker
│   ├── inference/          # Detection scripts
│   └── training/           # Training utilities
├── app.py                  # Gradio web demo
├── notebooks/              # Colab training notebooks
├── configs/                # Configuration files
└── results/                # Training results
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mehmetd7mir/ThermalTracking.git
cd ThermalTracking

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Optional: Install extra features
pip install fastapi uvicorn streamlit requests
```

### Run Detection

```bash
# On an image
python src/inference/detect.py --source image.jpg --weights best.pt

# On a video
python src/inference/detect.py --source video.mp4 --weights best.pt --save

# Web demo
python app.py
```

### Run API Server

```bash
cd src/api
uvicorn api_server:app --reload --port 8000
```

### Run Dashboard

```bash
streamlit run src/dashboard/dashboard.py
```

---

## Advanced Features

### Analytics

```python
from src.analytics.line_counter import LineCounter
from src.analytics.zone_intrusion import ZoneIntrusion
from src.analytics.trajectory_predictor import TrajectoryPredictor

# Count objects crossing a line
counter = LineCounter(line_start=(0, 300), line_end=(640, 300))

# Detect zone intrusions
zone = ZoneIntrusion()
zone.add_zone("restricted", [(100, 100), (400, 100), (400, 400), (100, 400)])

# Predict future positions
predictor = TrajectoryPredictor(method="polynomial")
```

### Alert System

```python
from src.alerts.alert_system import AlertSystem

alerts = AlertSystem()
alerts.add_telegram(token="BOT_TOKEN", chat_id="CHAT_ID")
alerts.add_slack(webhook_url="SLACK_WEBHOOK")

alerts.send(
    level="critical",
    title="Drone Detected",
    message="Unauthorized drone in restricted zone"
)
```

### Edge Deployment

```python
from src.optimization.tensorrt_export import TensorRTExporter
from src.optimization.onnx_export import ONNXExporter

# Export to TensorRT (NVIDIA GPUs)
exporter = TensorRTExporter("best.pt")
engine_path = exporter.export(precision="fp16")

# Export to ONNX (cross-platform)
onnx_exporter = ONNXExporter("best.pt")
onnx_path = onnx_exporter.export()
```

### Swarm Detection

```python
from src.advanced.swarm_detection import SwarmDetector

detector = SwarmDetector(min_group_size=3)
groups = detector.detect(drone_positions)

for group in groups:
    print(f"Swarm: {len(group.member_ids)} drones, formation: {group.formation}")
```

---

## Training Results

### Dataset Statistics

| Class | Instances | Percentage |
|-------|-----------|------------|
| Drone | 42,679 | 53.4% |
| Bird | 13,014 | 16.3% |
| Helicopter | 12,423 | 15.5% |
| Plane | 11,573 | 14.5% |
| **Total** | **79,689** | **100%** |

### Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | 73.9% |
| mAP50-95 | 39.6% |
| Precision | 76.4% |
| Recall | 72.3% |

### Configuration

- **Model**: YOLOv8m (25.8M parameters)
- **Image Size**: 640x640
- **Tracker**: BoTSORT

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Deep Learning** | PyTorch, YOLOv8 (Ultralytics) |
| **Tracking** | BoTSORT |
| **API** | FastAPI |
| **Dashboard** | Streamlit, Gradio |
| **Alerts** | Telegram, Slack, Discord |
| **Edge** | TensorRT, ONNX Runtime |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Detect objects in image |
| POST | `/detect/base64` | Detect from base64 image |
| POST | `/stream/start` | Start video processing |
| GET | `/stream/{id}` | Get stream status |
| GET | `/stats` | Get statistics |
| GET | `/classes` | Get class names |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Mehmet Demir** - [@mehmetd7mir](https://github.com/mehmetd7mir)

Project Link: [https://github.com/mehmetd7mir/ThermalTracking](https://github.com/mehmetd7mir/ThermalTracking)

---

<p align="center">
  <b>Built for Defense & Aerospace Applications</b>
</p>
