"""
ThermalTracking - Inference Module
===================================
Thermal görüntü/video üzerinde YOLOv8 ile nesne tespiti yapar.

Kullanım:
    python detect.py --source video.mp4 --weights best.pt
    python detect.py --source image.jpg --weights best.pt
    python detect.py --source 0  # Webcam
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def run_detection(
    source: str,
    weights: str = "best.pt",
    conf: float = 0.25,
    save: bool = True,
    show: bool = False,
    project: str = "runs/detect",
    name: str = "exp"
):
    """
    YOLOv8 ile nesne tespiti yapar.
    
    Args:
        source: Görüntü/video dosyası veya webcam (0)
        weights: Model ağırlık dosyası (.pt)
        conf: Minimum confidence threshold
        save: Sonuçları kaydet
        show: Sonuçları ekranda göster
        project: Sonuçların kaydedileceği klasör
        name: Çalışma adı
    
    Returns:
        results: YOLO sonuçları
    """
    # Model yükle
    print(f"📦 Model yükleniyor: {weights}")
    model = YOLO(weights)
    
    # Tespit yap
    print(f"🔍 Tespit başlıyor: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=show,
        project=project,
        name=name
    )
    
    # İstatistikler
    total_detections = sum(len(r.boxes) for r in results)
    print(f"✅ Tespit tamamlandı!")
    print(f"   Toplam tespit: {total_detections}")
    
    if save:
        print(f"   Sonuçlar: {project}/{name}/")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="ThermalTracking - Thermal görüntülerde hava aracı tespiti"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        required=True,
        help="Görüntü/video dosyası veya webcam (0)"
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        default="best.pt",
        help="Model ağırlık dosyası (.pt)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--save", 
        action="store_true",
        default=True,
        help="Sonuçları kaydet"
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Sonuçları ekranda göster"
    )
    parser.add_argument(
        "--project", 
        type=str, 
        default="runs/detect",
        help="Sonuçların kaydedileceği klasör"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="exp",
        help="Çalışma adı"
    )
    
    args = parser.parse_args()
    
    run_detection(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        save=args.save,
        show=args.show,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()
