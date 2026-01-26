# Model Training Details

## Overview

We train YOLOv8m (Medium) on our custom MegaSet thermal dataset using Google Colab with Tesla T4 GPU.

---

## Training Configuration

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8m |
| Parameters | 25.8M |
| GFLOPs | 79.1 |
| Layers | 169 |

### Hyperparameters

```yaml
# Training
epochs: 50
batch: 16
imgsz: 640
optimizer: SGD (auto)
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005

# Augmentation
mosaic: 1.0
mixup: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.5
erasing: 0.4

# Reproducibility
seed: 0
deterministic: True
```

---

## Training Environment

### Hardware
- **GPU**: Tesla T4 (16GB)
- **Platform**: Google Colab Pro
- **Training Time**: ~40 min/epoch

### Checkpointing Strategy

To prevent data loss from Colab disconnections:

```python
model.train(
    project='/content/drive/MyDrive/MegaSet_Train',
    name='v2_safe_train',
    resume=True  # Resume from last checkpoint
)
```

All weights (`best.pt`, `last.pt`) are saved directly to Google Drive.

---

## Training Progress

### Epoch-by-Epoch Results

| Epoch | mAP50 | mAP50-95 | Precision | Recall | Box Loss | Cls Loss |
|:-----:|:-----:|:--------:|:---------:|:------:|:--------:|:--------:|
| 1 | 57.6% | 28.4% | 58.6% | 55.9% | 1.54 | 1.31 |
| 2 | 67.3% | 33.1% | 71.9% | 64.2% | 1.44 | 0.87 |
| 3 | 67.5% | 32.8% | 73.0% | 63.7% | 1.48 | 0.92 |
| 4 | 69.8% | 36.1% | 70.1% | 65.6% | 1.47 | 0.90 |
| 5 | 70.2% | 36.2% | 72.1% | 67.5% | 1.38 | 0.80 |
| 6 | 72.2% | 37.6% | 74.2% | 68.5% | 1.34 | 0.75 |
| 7 | 71.6% | 37.8% | 75.1% | 69.4% | 1.29 | 0.70 |
| 8 | 73.3% | 38.7% | 75.1% | 70.6% | 1.27 | 0.68 |
| 9 | 73.7% | 39.4% | 75.6% | 72.2% | 1.28 | 0.68 |
| **10** | **73.9%** | **39.6%** | **76.4%** | **72.3%** | **1.26** | **0.66** |

### Observations

1. **Rapid Initial Learning**: mAP50 jumped from 57.6% to 67.3% in first 2 epochs
2. **Steady Improvement**: Consistent gains until epoch 10
3. **Loss Convergence**: Both box and class loss decreasing steadily
4. **No Overfitting**: Validation metrics still improving

---

## Class-wise Performance

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Drone | 78.2% | 75.1% | 76.5% |
| Bird | 72.1% | 68.4% | 69.8% |
| Helicopter | 79.5% | 74.2% | 75.3% |
| Plane | 75.8% | 71.5% | 73.2% |

---

## How to Resume Training

```python
from ultralytics import YOLO

# Load last checkpoint
model = YOLO("path/to/last.pt")

# Resume training
model.train(
    data="data.yaml",
    epochs=50,
    resume=True
)
```

---

## Next Steps

1. Complete training to epoch 50
2. Evaluate on test set
3. Generate confusion matrix
4. Export to ONNX for deployment
