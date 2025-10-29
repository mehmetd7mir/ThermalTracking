# ThermalTracking: YOLOv8-based Detection and Tracking of Aerial Targets in Thermal Imagery

This project is a prototype for detecting, classifying, and tracking aerial targets (drone, plane, helicopter, bird) in thermal (infrared) video streams using YOLOv8.

Instead of relying on a clean, pre-made dataset, this project focuses on a Data-Centric AI approach, emphasizing the critical engineering steps of Data Collection, Data Cleaning, and Data Augmentation.

---

## Project Roadmap

This prototype is being developed in three main phases to address the end-to-end engineering problem:

### Phase 1: "Mega-Set" Data Engineering
No "off-the-shelf" dataset was used. Instead, a custom "Mega-Set" was engineered.

1.  **Data Collection:** Aggregated 11 different, public, and "messy" thermal datasets.
2.  **Data Cleaning:** Wrote the `clean_label.py` script to automate the cleaning process. This script parses all labels (e.g., 'UAV', 'Drone', 'BIRD', '1', '3') and standardizes them into 4 classes: 'drone', 'plane', 'helicopter', 'bird'.
3.  **Data Augmentation:** Used Roboflow to apply 3x augmentations (Blur, Noise, Brightness, etc.) to the 24,000+ raw training images, creating a robust training set of over 80,000+ images (`v2-augmented-3x`).

### Phase 2: Model Training (YOLOv8m)
* **Model:** YOLOv8m (Medium)
* **Dataset:** `v2-augmented-3x` (80,000+ Images)
* **Environment:** Google Colab (Tesla T4 GPU)
* **Checkpointing:** All training outputs (`best.pt`, `last.pt`) are saved directly to Google Drive (`project='/content/drive/...'`) to prevent data loss from Colab runtime disconnections.
* **Reproducibility:** All training runs use `seed=0` and `deterministic=True` for scientifically reproducible results.

### Phase 3: Tracking & Optimization (Future Work)
* **[In Progress] Target Tracking:** Integration with the ByteTrack algorithm.
* **[Planned] Relative Velocity:** Calculating apparent target speed in 'pixels/second'.
* **[Planned] Optimization:** Exporting the final model to ONNX format for inference on edge/embedded systems.

---

## Training Status

The `v2_safe_train` training run is currently in progress.

The table below shows the promising results from the first 5 epochs of the initial `v1` training run. We expect to reproduce these exact results in the new `v2` run due to the `seed=0` setting.

| Epoch | mAP50 | mAP50-95 | cls_loss (Error) |
| :---: | :---: | :---: | :---: |
| 1/50 | 0.659 | 0.346 | 1.269 |
| 2/50 | 0.689 | 0.355 | 0.857 |
| 3/50 | 0.687 | 0.350 | 0.920 |
| 4/50 | 0.710 | 0.369 | 0.889 |
| **5/50** | **0.722** | **0.388** | **0.804** |

Training graphs (`results.png`) will be added here upon completion.

---

## Tech Stack

* Python 3.10+
* YOLOv8 (Ultralytics)
* Roboflow (Data Management & Augmentation)
* OpenCV (Image Processing)
* Google Colab (Tesla T4 GPU Training)
* GitHub (Version Control)

---

## Setup and Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/mehmetd7mir/ThermalTracking.git](https://github.com/mehmetd7mir/ThermalTracking.git)
    cd ThermalTracking
    ```

2.  Create and activate a virtual environment (Recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  To review the training process, see the Colab notebook:
    * `thermal_tracking_test1.ipynb`
