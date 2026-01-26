# Project Overview

## ThermalTracking: Aerial Target Detection in Thermal Imagery

### Problem Statement

Detecting and tracking aerial targets (drones, aircraft, helicopters, birds) in thermal imagery is crucial for:
- **Defense & Security**: Early warning systems, airspace monitoring
- **Counter-UAV**: Drone detection and neutralization
- **Aviation Safety**: Bird strike prevention

### Challenges

1. **Limited Data**: Clean, labeled thermal datasets are scarce
2. **Class Confusion**: Small objects like drones and birds look similar
3. **Real-time Requirements**: Detection must be fast for practical use
4. **Environmental Factors**: Varying thermal signatures across conditions

### Our Solution

We developed a **Data-Centric AI** approach:

1. **MegaSet Engineering**: Combined 11+ public datasets
2. **Robust Cleaning**: Automated label standardization
3. **Heavy Augmentation**: 3x data augmentation for generalization
4. **YOLOv8 Training**: State-of-the-art object detection

### Target Applications

- 🛡️ Military surveillance systems
- 🏢 Critical infrastructure protection
- ✈️ Airport security
- 🔬 Research & Development

---

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Thermal Camera │────▶│    YOLOv8       │────▶│   Detection     │
│   (IR Sensor)   │     │   Detection     │     │   Results       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   ByteTrack     │
                        │   (Tracking)    │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │    Velocity     │
                        │   Estimation    │
                        └─────────────────┘
```

---

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | 73.9% | Mean Average Precision @ IoU 0.5 |
| **mAP50-95** | 39.6% | Mean AP across IoU thresholds |
| **Precision** | 76.4% | True positives / All predictions |
| **Recall** | 72.3% | True positives / All ground truths |

---

## Future Roadmap

See [FUTURE_WORK.md](FUTURE_WORK.md) for detailed plans.
