"""
ThermalTracking - CuPy GPU Preprocessing Module
================================================
GPU üzerinde görüntü ön işleme.

CuPy = NumPy'ın GPU versiyonu
- 10-100x hızlı (büyük görüntülerde)
- CUDA kernel yazmadan GPU kullanımı

Özellikler:
- GPU-accelerated normalization
- GPU-accelerated resize
- GPU-accelerated color conversion
- Batch processing

Kurulum:
    pip install cupy-cuda11x  # CUDA 11.x için
    # veya
    pip install cupy-cuda12x  # CUDA 12.x için

Kullanım:
    from gpu_preprocessing import GPUPreprocessor
    
    preprocessor = GPUPreprocessor()
    processed = preprocessor.preprocess(image)
"""

import time
from typing import Tuple, Optional, List
import numpy as np

# CuPy import (GPU yoksa fallback)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy GPU backend kullanılabilir")
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("⚠️ CuPy bulunamadı, CPU fallback kullanılacak")


class GPUPreprocessor:
    """
    GPU-accelerated image preprocessing.
    CuPy kullanarak görüntü işleme operasyonlarını GPU'da çalıştırır.
    """
    
    def __init__(self, device_id: int = 0, use_gpu: bool = True):
        """
        Args:
            device_id: CUDA device ID
            use_gpu: GPU kullan (False ise CPU fallback)
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id
        
        if self.use_gpu:
            cp.cuda.Device(device_id).use()
            print(f"🎮 GPU-{device_id} aktif: {cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode()}")
    
    def to_gpu(self, image: np.ndarray) -> 'cp.ndarray':
        """NumPy array'i GPU'ya transfer et."""
        if self.use_gpu:
            return cp.asarray(image)
        return image
    
    def to_cpu(self, image) -> np.ndarray:
        """GPU array'i CPU'ya transfer et."""
        if self.use_gpu and hasattr(image, 'get'):
            return cp.asnumpy(image)
        return image
    
    def normalize(self, image, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                  std: Tuple[float, ...] = (0.229, 0.224, 0.225)) -> np.ndarray:
        """
        ImageNet normalization (GPU-accelerated).
        
        Args:
            image: Input image (H, W, C) veya (N, H, W, C)
            mean: Channel means
            std: Channel stds
        
        Returns:
            Normalized image
        """
        xp = cp if self.use_gpu else np
        
        # GPU'ya transfer
        img = self.to_gpu(image) if self.use_gpu else image
        
        # Float'a çevir ve [0, 1] aralığına getir
        img = img.astype(xp.float32) / 255.0
        
        # Mean ve std'yi array'e çevir
        mean = xp.array(mean, dtype=xp.float32)
        std = xp.array(std, dtype=xp.float32)
        
        # Normalize
        img = (img - mean) / std
        
        return self.to_cpu(img)
    
    def resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        GPU-accelerated resize (bilinear interpolation).
        
        Args:
            image: Input image (H, W, C)
            target_size: (new_width, new_height)
        
        Returns:
            Resized image
        """
        if not self.use_gpu:
            import cv2
            return cv2.resize(image, target_size)
        
        # CuPy ile resize (scipy.ndimage benzeri)
        from cupyx.scipy import ndimage
        
        img_gpu = self.to_gpu(image)
        
        h, w = image.shape[:2]
        new_w, new_h = target_size
        
        # Scale faktörler
        scale_h = new_h / h
        scale_w = new_w / w
        
        # Her channel için ayrı resize
        if len(img_gpu.shape) == 3:
            resized_channels = []
            for c in range(img_gpu.shape[2]):
                channel = ndimage.zoom(img_gpu[:, :, c], (scale_h, scale_w), order=1)
                resized_channels.append(channel)
            resized = cp.stack(resized_channels, axis=2)
        else:
            resized = ndimage.zoom(img_gpu, (scale_h, scale_w), order=1)
        
        return self.to_cpu(resized)
    
    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        RGB to Grayscale (GPU-accelerated).
        
        Formula: Y = 0.299*R + 0.587*G + 0.114*B
        """
        xp = cp if self.use_gpu else np
        
        img = self.to_gpu(image) if self.use_gpu else image
        img = img.astype(xp.float32)
        
        # Weighted sum
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        
        return self.to_cpu(gray.astype(xp.uint8))
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: float = 0) -> np.ndarray:
        """
        Contrast enhancement (GPU-accelerated).
        
        Formula: output = alpha * input + beta
        
        Args:
            alpha: Contrast multiplier (>1 increases, <1 decreases)
            beta: Brightness offset
        """
        xp = cp if self.use_gpu else np
        
        img = self.to_gpu(image) if self.use_gpu else image
        img = img.astype(xp.float32)
        
        # Enhance
        enhanced = alpha * img + beta
        
        # Clip to valid range
        enhanced = xp.clip(enhanced, 0, 255).astype(xp.uint8)
        
        return self.to_cpu(enhanced)
    
    def gaussian_blur(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Gaussian blur (GPU-accelerated).
        """
        if not self.use_gpu:
            import cv2
            ksize = int(6 * sigma + 1) | 1  # Ensure odd
            return cv2.GaussianBlur(image, (ksize, ksize), sigma)
        
        from cupyx.scipy import ndimage
        
        img_gpu = self.to_gpu(image)
        
        # Her channel için blur
        if len(img_gpu.shape) == 3:
            blurred_channels = []
            for c in range(img_gpu.shape[2]):
                channel = ndimage.gaussian_filter(img_gpu[:, :, c].astype(cp.float32), sigma)
                blurred_channels.append(channel.astype(cp.uint8))
            blurred = cp.stack(blurred_channels, axis=2)
        else:
            blurred = ndimage.gaussian_filter(img_gpu.astype(cp.float32), sigma).astype(cp.uint8)
        
        return self.to_cpu(blurred)
    
    def preprocess_for_yolo(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """
        YOLO için tam preprocessing pipeline.
        
        1. Resize to target size
        2. Normalize to [0, 1]
        3. CHW format (channels-first)
        4. Add batch dimension
        """
        xp = cp if self.use_gpu else np
        
        # GPU'ya transfer
        img = self.to_gpu(image) if self.use_gpu else image
        
        # 1. Resize (şimdilik CPU'da, çünkü küçük operasyon)
        import cv2
        img_cpu = self.to_cpu(img) if self.use_gpu else img
        resized = cv2.resize(img_cpu, target_size)
        img = self.to_gpu(resized) if self.use_gpu else resized
        
        # 2. Normalize [0, 1]
        img = img.astype(xp.float32) / 255.0
        
        # 3. HWC -> CHW
        img = xp.transpose(img, (2, 0, 1))
        
        # 4. Add batch dimension: CHW -> NCHW
        img = xp.expand_dims(img, axis=0)
        
        return self.to_cpu(img)
    
    def batch_preprocess(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """
        Batch preprocessing (GPU-accelerated).
        
        Args:
            images: List of images
            target_size: Target size for all images
        
        Returns:
            Batch tensor (N, C, H, W)
        """
        batch = []
        for img in images:
            processed = self.preprocess_for_yolo(img, target_size)
            batch.append(processed)
        
        xp = cp if self.use_gpu else np
        return xp.concatenate(batch, axis=0)


def benchmark_gpu_vs_cpu(image: np.ndarray, iterations: int = 100):
    """GPU vs CPU preprocessing benchmark."""
    
    print("\n" + "=" * 50)
    print("🔬 GPU vs CPU Benchmark")
    print("=" * 50)
    
    # CPU preprocessing
    cpu_processor = GPUPreprocessor(use_gpu=False)
    
    cpu_times = []
    for _ in range(iterations):
        start = time.time()
        _ = cpu_processor.normalize(image)
        _ = cpu_processor.enhance_contrast(image)
        cpu_times.append(time.time() - start)
    
    cpu_avg = sum(cpu_times) / len(cpu_times) * 1000
    print(f"\n🖥️ CPU: {cpu_avg:.2f}ms ortalama")
    
    # GPU preprocessing (eğer varsa)
    if CUPY_AVAILABLE:
        gpu_processor = GPUPreprocessor(use_gpu=True)
        
        # Warm-up
        for _ in range(10):
            _ = gpu_processor.normalize(image)
        
        gpu_times = []
        for _ in range(iterations):
            start = time.time()
            _ = gpu_processor.normalize(image)
            _ = gpu_processor.enhance_contrast(image)
            cp.cuda.Stream.null.synchronize()  # GPU bitmesini bekle
            gpu_times.append(time.time() - start)
        
        gpu_avg = sum(gpu_times) / len(gpu_times) * 1000
        speedup = cpu_avg / gpu_avg
        
        print(f"🎮 GPU: {gpu_avg:.2f}ms ortalama")
        print(f"⚡ Speedup: {speedup:.1f}x")
    else:
        print("⚠️ CuPy yüklü değil, GPU benchmark atlandı")
    
    print("=" * 50)


# Test
if __name__ == "__main__":
    # Rastgele test görüntüsü
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    print(f"Test görüntüsü: {test_image.shape}")
    
    # Benchmark
    benchmark_gpu_vs_cpu(test_image, iterations=50)
    
    # Preprocessing test
    if CUPY_AVAILABLE:
        processor = GPUPreprocessor(use_gpu=True)
        
        print("\n📊 Preprocessing Test:")
        
        # Normalize
        normalized = processor.normalize(test_image)
        print(f"   Normalize: {normalized.shape}, range [{normalized.min():.2f}, {normalized.max():.2f}]")
        
        # Grayscale
        gray = processor.grayscale(test_image)
        print(f"   Grayscale: {gray.shape}")
        
        # YOLO preprocessing
        yolo_ready = processor.preprocess_for_yolo(test_image)
        print(f"   YOLO ready: {yolo_ready.shape}")
