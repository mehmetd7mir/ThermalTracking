"""
ThermalTracking - CUDA Kernels Python Wrapper
==============================================
Custom CUDA kernel'leri Python'dan çağırmak için wrapper.

Kullanım:
    from cuda_wrapper import CUDAKernels
    
    kernels = CUDAKernels()
    normalized = kernels.normalize(image)
    gray = kernels.grayscale(image)

Not: önce CUDA kernel'leri derlenmeli:
    nvcc -shared -o libkernels.so -Xcompiler -fPIC cuda_kernels.cu
"""

import os
import ctypes
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


class CUDAKernels:
    """Python wrapper for custom CUDA kernels."""
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Args:
            lib_path: Path to compiled CUDA library (libkernels.so)
        """
        self._lib = None
        self._available = False
        
        if lib_path is None:
            # Default path: aynı dizinde
            lib_path = Path(__file__).parent / "libkernels.so"
        
        try:
            self._lib = ctypes.CDLL(str(lib_path))
            self._setup_functions()
            self._available = True
            print(f"✅ CUDA Kernels yüklendi: {lib_path}")
        except OSError as e:
            print(f"⚠️ CUDA library bulunamadı: {e}")
            print("   Kernel'ler derlenmiş olmalı:")
            print("   nvcc -shared -o libkernels.so -Xcompiler -fPIC cuda_kernels.cu")
    
    def _setup_functions(self):
        """Setup ctypes function signatures."""
        
        # cuda_normalize
        self._lib.cuda_normalize.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),   # input
            ctypes.POINTER(ctypes.c_float),    # output
            ctypes.c_int,                      # width
            ctypes.c_int,                      # height
            ctypes.c_int,                      # channels
            ctypes.POINTER(ctypes.c_float),    # mean
            ctypes.POINTER(ctypes.c_float)     # std
        ]
        self._lib.cuda_normalize.restype = None
        
        # cuda_grayscale
        self._lib.cuda_grayscale.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int
        ]
        self._lib.cuda_grayscale.restype = None
        
        # cuda_threshold
        self._lib.cuda_threshold.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ubyte,
            ctypes.c_ubyte
        ]
        self._lib.cuda_threshold.restype = None
        
        # cuda_contrast
        self._lib.cuda_contrast.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float
        ]
        self._lib.cuda_contrast.restype = None
        
        # cuda_get_device_info
        self._lib.cuda_get_device_info.argtypes = []
        self._lib.cuda_get_device_info.restype = None
    
    @property
    def is_available(self) -> bool:
        """CUDA kernel'ler kullanılabilir mi?"""
        return self._available
    
    def get_device_info(self):
        """CUDA device bilgisini yazdır."""
        if not self._available:
            print("❌ CUDA library yüklü değil")
            return
        self._lib.cuda_get_device_info()
    
    def normalize(
        self,
        image: np.ndarray,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> np.ndarray:
        """
        GPU-accelerated ImageNet normalization.
        
        Args:
            image: Input image (H, W, C), uint8
            mean: Channel means
            std: Channel stds
        
        Returns:
            Normalized image (C, H, W), float32
        """
        if not self._available:
            # CPU fallback
            img = image.astype(np.float32) / 255.0
            img = (img - np.array(mean)) / np.array(std)
            return img.transpose(2, 0, 1)
        
        assert image.dtype == np.uint8, "Input must be uint8"
        assert len(image.shape) == 3, "Input must be (H, W, C)"
        
        h, w, c = image.shape
        
        # Output buffer
        output = np.zeros((c, h, w), dtype=np.float32)
        
        # Convert to ctypes
        input_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        mean_ptr = mean_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        std_ptr = std_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        self._lib.cuda_normalize(input_ptr, output_ptr, w, h, c, mean_ptr, std_ptr)
        
        return output
    
    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated RGB to grayscale.
        
        Args:
            image: RGB image (H, W, 3), uint8
        
        Returns:
            Grayscale image (H, W), uint8
        """
        if not self._available:
            # CPU fallback
            return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        assert image.dtype == np.uint8
        assert len(image.shape) == 3 and image.shape[2] == 3
        
        h, w = image.shape[:2]
        output = np.zeros((h, w), dtype=np.uint8)
        
        input_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        self._lib.cuda_grayscale(input_ptr, output_ptr, w, h)
        
        return output
    
    def threshold(
        self,
        image: np.ndarray,
        threshold: int = 128,
        max_value: int = 255
    ) -> np.ndarray:
        """
        GPU-accelerated binary thresholding.
        
        Args:
            image: Grayscale image (H, W), uint8
            threshold: Threshold value
            max_value: Max output value
        
        Returns:
            Binary image (H, W), uint8
        """
        if not self._available:
            return ((image > threshold) * max_value).astype(np.uint8)
        
        assert image.dtype == np.uint8
        assert len(image.shape) == 2
        
        h, w = image.shape
        output = np.zeros((h, w), dtype=np.uint8)
        
        input_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        self._lib.cuda_threshold(
            input_ptr, output_ptr, w, h,
            ctypes.c_ubyte(threshold),
            ctypes.c_ubyte(max_value)
        )
        
        return output
    
    def contrast(
        self,
        image: np.ndarray,
        alpha: float = 1.5,
        beta: float = 0.0
    ) -> np.ndarray:
        """
        GPU-accelerated contrast enhancement.
        
        Args:
            image: Grayscale image (H, W), uint8
            alpha: Contrast multiplier
            beta: Brightness offset
        
        Returns:
            Enhanced image (H, W), uint8
        """
        if not self._available:
            enhanced = alpha * (image.astype(np.float32) - 128) + 128 + beta
            return np.clip(enhanced, 0, 255).astype(np.uint8)
        
        assert image.dtype == np.uint8
        assert len(image.shape) == 2
        
        h, w = image.shape
        output = np.zeros((h, w), dtype=np.uint8)
        
        input_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        self._lib.cuda_contrast(
            input_ptr, output_ptr, w, h,
            ctypes.c_float(alpha),
            ctypes.c_float(beta)
        )
        
        return output


# Compile helper
def compile_kernels(cuda_file: str = "cuda_kernels.cu", output: str = "libkernels.so"):
    """CUDA kernel'leri derle."""
    import subprocess
    
    cuda_path = Path(__file__).parent / cuda_file
    output_path = Path(__file__).parent / output
    
    cmd = f"nvcc -shared -o {output_path} -Xcompiler -fPIC {cuda_path}"
    
    print(f"🔨 Derleniyor: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Derleme başarılı: {output_path}")
    else:
        print(f"❌ Derleme hatası:\n{result.stderr}")
    
    return result.returncode == 0


# Test
if __name__ == "__main__":
    print("=" * 50)
    print("CUDA Kernels Test")
    print("=" * 50)
    
    kernels = CUDAKernels()
    
    if kernels.is_available:
        kernels.get_device_info()
    
    # Test with random image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"\nTest image: {test_img.shape}")
    
    # Normalize
    normalized = kernels.normalize(test_img)
    print(f"Normalized: {normalized.shape}, range [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    # Grayscale
    gray = kernels.grayscale(test_img)
    print(f"Grayscale: {gray.shape}")
    
    # Threshold
    binary = kernels.threshold(gray, threshold=128)
    print(f"Threshold: {binary.shape}, unique values: {np.unique(binary)}")
    
    # Contrast
    enhanced = kernels.contrast(gray, alpha=1.5, beta=10)
    print(f"Contrast: {enhanced.shape}")
    
    print("\n✅ Tüm testler tamamlandı")
