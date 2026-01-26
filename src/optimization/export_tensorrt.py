"""
ThermalTracking - TensorRT Export & Optimization
=================================================
YOLOv8 modelini TensorRT Engine'e dönüştürür.

TensorRT Avantajları:
- 2-5x hızlanma
- Düşük latency
- FP16/INT8 precision
- Edge deployment (Jetson)

Kullanım:
    python export_tensorrt.py --weights best.pt --precision fp16
    
Inference:
    python export_tensorrt.py --engine best.engine --source video.mp4
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics not found. Install with: pip install ultralytics")


def export_to_onnx(
    weights: str,
    output_path: Optional[str] = None,
    imgsz: int = 640,
    half: bool = True,
    simplify: bool = True,
    opset: int = 12
) -> str:
    """
    YOLOv8 modelini ONNX formatına dönüştür.
    
    Args:
        weights: Model ağırlık dosyası (.pt)
        output_path: Çıktı yolu
        imgsz: Input size
        half: FP16 precision
        simplify: ONNX simplify
        opset: ONNX opset version
    
    Returns:
        ONNX dosya yolu
    """
    print("=" * 50)
    print("📦 ONNX Export")
    print("=" * 50)
    
    model = YOLO(weights)
    
    # Export
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        opset=opset
    )
    
    if output_path:
        import shutil
        shutil.move(export_path, output_path)
        export_path = output_path
    
    print(f"✅ ONNX export tamamlandı: {export_path}")
    return export_path


def export_to_tensorrt(
    weights: str,
    output_path: Optional[str] = None,
    imgsz: int = 640,
    precision: str = "fp16",
    workspace: int = 4,
    batch: int = 1
) -> str:
    """
    YOLOv8 modelini TensorRT Engine'e dönüştür.
    
    Args:
        weights: Model ağırlık dosyası (.pt)
        output_path: Çıktı yolu
        imgsz: Input size
        precision: "fp32", "fp16", veya "int8"
        workspace: TensorRT workspace GB
        batch: Batch size
    
    Returns:
        Engine dosya yolu
    """
    print("=" * 50)
    print("🚀 TensorRT Export")
    print("=" * 50)
    print(f"   Precision: {precision}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch: {batch}")
    print("=" * 50)
    
    model = YOLO(weights)
    
    # Export options
    half = precision in ["fp16", "int8"]
    int8 = precision == "int8"
    
    try:
        export_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            int8=int8,
            workspace=workspace,
            batch=batch
        )
        
        if output_path:
            import shutil
            shutil.move(export_path, output_path)
            export_path = output_path
        
        print(f"\n✅ TensorRT export tamamlandı: {export_path}")
        return export_path
        
    except Exception as e:
        print(f"❌ TensorRT export hatası: {e}")
        print("\nOlası çözümler:")
        print("1. NVIDIA GPU yüklü olmalı")
        print("2. TensorRT yüklü olmalı: pip install tensorrt")
        print("3. Alternatif: ONNX export deneyin")
        return None


def benchmark_model(
    model_path: str,
    imgsz: int = 640,
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, Any]:
    """
    Model performansını benchmark et.
    
    Args:
        model_path: Model dosyası (.pt, .onnx, veya .engine)
        imgsz: Input size
        iterations: Test iterasyonu
        warmup: Warm-up iterasyonu
    
    Returns:
        Benchmark sonuçları
    """
    print("\n" + "=" * 50)
    print("⏱️ Benchmark")
    print("=" * 50)
    
    model = YOLO(model_path)
    
    # Random input
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warm-up
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model(dummy_input, verbose=False)
    
    # Benchmark
    print(f"Running benchmark ({iterations} iterations)...")
    times = []
    
    for i in range(iterations):
        start = time.time()
        _ = model(dummy_input, verbose=False)
        times.append(time.time() - start)
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{iterations} completed")
    
    # İstatistikler
    times = np.array(times) * 1000  # ms
    
    results = {
        'model': model_path,
        'imgsz': imgsz,
        'iterations': iterations,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'fps': float(1000 / np.mean(times))
    }
    
    print("\n📊 Sonuçlar:")
    print(f"   Model: {Path(model_path).name}")
    print(f"   Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
    print(f"   Min: {results['min_ms']:.2f} ms")
    print(f"   Max: {results['max_ms']:.2f} ms")
    print(f"   FPS: {results['fps']:.1f}")
    print("=" * 50)
    
    return results


def compare_models(
    pt_path: str,
    onnx_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    imgsz: int = 640,
    iterations: int = 50
) -> Dict[str, Dict]:
    """
    Farklı model formatlarını karşılaştır.
    """
    print("\n" + "=" * 60)
    print("📊 Model Karşılaştırma")
    print("=" * 60)
    
    results = {}
    
    # PyTorch
    if pt_path and Path(pt_path).exists():
        print("\n🔵 PyTorch (.pt)")
        results['pytorch'] = benchmark_model(pt_path, imgsz, iterations)
    
    # ONNX
    if onnx_path and Path(onnx_path).exists():
        print("\n🟢 ONNX")
        results['onnx'] = benchmark_model(onnx_path, imgsz, iterations)
    
    # TensorRT
    if engine_path and Path(engine_path).exists():
        print("\n🔴 TensorRT")
        results['tensorrt'] = benchmark_model(engine_path, imgsz, iterations)
    
    # Karşılaştırma tablosu
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("📈 Karşılaştırma Özeti")
        print("-" * 60)
        print(f"{'Format':<15} {'Mean (ms)':<12} {'FPS':<10} {'Speedup':<10}")
        print("-" * 60)
        
        base_fps = results.get('pytorch', {}).get('fps', 1)
        
        for name, data in results.items():
            speedup = data['fps'] / base_fps if base_fps > 0 else 1
            print(f"{name:<15} {data['mean_ms']:<12.2f} {data['fps']:<10.1f} {speedup:<10.2f}x")
        
        print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="ThermalTracking - TensorRT Export & Benchmark"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--weights", type=str, required=True, help="Model weights (.pt)")
    export_parser.add_argument("--format", type=str, default="engine", 
                              choices=["onnx", "engine"], help="Export format")
    export_parser.add_argument("--precision", type=str, default="fp16",
                              choices=["fp32", "fp16", "int8"], help="Precision")
    export_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    export_parser.add_argument("--output", type=str, help="Output path")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark model")
    bench_parser.add_argument("--model", type=str, required=True, help="Model path")
    bench_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    bench_parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("--pt", type=str, required=True, help="PyTorch model")
    compare_parser.add_argument("--onnx", type=str, help="ONNX model")
    compare_parser.add_argument("--engine", type=str, help="TensorRT engine")
    compare_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    if args.command == "export":
        if args.format == "onnx":
            export_to_onnx(args.weights, args.output, args.imgsz)
        else:
            export_to_tensorrt(args.weights, args.output, args.imgsz, args.precision)
    
    elif args.command == "benchmark":
        benchmark_model(args.model, args.imgsz, args.iterations)
    
    elif args.command == "compare":
        compare_models(args.pt, args.onnx, args.engine, args.imgsz)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
