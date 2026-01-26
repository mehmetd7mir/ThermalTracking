"""
ThermalTracking - Velocity Estimation Module
=============================================
Takip edilen nesnelerin hızını hesaplar.

Özellikler:
- Pixel/saniye hesaplama
- Metre/saniye dönüşümü (kamera kalibrasyonu ile)
- Hareket yönü (bearing) hesaplama
- Kalman filter ile smoothing

Kullanım:
    from velocity import VelocityEstimator
    
    estimator = VelocityEstimator(fps=30, scale_factor=0.03)
    velocity = estimator.calculate(track_history)
"""

import math
from collections import deque
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class VelocityData:
    """Hız verisi container'ı"""
    velocity_px_sec: float      # Pixel/saniye
    velocity_m_sec: float       # Metre/saniye
    direction_deg: float        # Hareket yönü (derece)
    smoothed_velocity: float    # Smoothed hız
    acceleration: float         # İvme (px/s²)
    

class KalmanFilter1D:
    """
    Basit 1D Kalman Filter.
    Hız smoothing için kullanılır.
    """
    
    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_estimate = 1.0
    
    def update(self, measurement: float) -> float:
        """Yeni ölçüm ile tahmini güncelle."""
        # Prediction
        prediction = self.estimate
        error_prediction = self.error_estimate + self.process_variance
        
        # Update
        kalman_gain = error_prediction / (error_prediction + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * error_prediction
        
        return self.estimate


class VelocityEstimator:
    """
    Tracking verilerinden hız tahmini yapar.
    
    Özellikler:
    - Multi-frame averaging
    - Kalman filter smoothing
    - İvme hesaplama
    - Yön tahmini
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        scale_factor: float = 0.03,
        history_size: int = 10,
        use_kalman: bool = True
    ):
        """
        Args:
            fps: Video frame rate
            scale_factor: Pixel -> metre dönüşüm faktörü
                         (Örn: 640px genişlik, 20m gerçek genişlik -> 0.03125)
            history_size: Hız ortalaması için saklanacak değer sayısı
            use_kalman: Kalman filter kullan
        """
        self.fps = fps
        self.scale_factor = scale_factor
        self.history_size = history_size
        self.use_kalman = use_kalman
        
        # Her track için ayrı Kalman filter ve history
        self.kalman_filters: Dict[int, KalmanFilter1D] = {}
        self.velocity_history: Dict[int, deque] = {}
        self.last_velocity: Dict[int, float] = {}
    
    def _get_kalman(self, track_id: int) -> KalmanFilter1D:
        """Track için Kalman filter getir veya oluştur."""
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = KalmanFilter1D()
        return self.kalman_filters[track_id]
    
    def _get_history(self, track_id: int) -> deque:
        """Track için velocity history getir veya oluştur."""
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = deque(maxlen=self.history_size)
        return self.velocity_history[track_id]
    
    def calculate_from_points(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        frame_diff: int = 1
    ) -> Tuple[float, float, float]:
        """
        İki nokta arasındaki hızı hesapla.
        
        Args:
            p1: Başlangıç noktası (x, y)
            p2: Bitiş noktası (x, y)
            frame_diff: Frame farkı
        
        Returns:
            (velocity_px_sec, velocity_m_sec, direction_deg)
        """
        if frame_diff == 0:
            frame_diff = 1
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Pixel cinsinden mesafe
        distance_px = math.sqrt(dx**2 + dy**2)
        
        # Pixel/saniye
        velocity_px_sec = (distance_px / frame_diff) * self.fps
        
        # Metre/saniye
        velocity_m_sec = velocity_px_sec * self.scale_factor
        
        # Hareket yönü (0° = sağ, 90° = aşağı)
        direction_deg = math.degrees(math.atan2(dy, dx))
        
        return velocity_px_sec, velocity_m_sec, direction_deg
    
    def calculate(
        self,
        track_id: int,
        track_history: List[Tuple[float, float, int]]
    ) -> Optional[VelocityData]:
        """
        Track history'den hız hesapla.
        
        Args:
            track_id: Track ID
            track_history: [(x, y, frame_num), ...]
        
        Returns:
            VelocityData veya None
        """
        if len(track_history) < 2:
            return None
        
        # Son iki nokta
        p1 = track_history[-2]
        p2 = track_history[-1]
        
        frame_diff = p2[2] - p1[2]
        
        # Temel hız hesabı
        vel_px, vel_m, direction = self.calculate_from_points(
            (p1[0], p1[1]),
            (p2[0], p2[1]),
            frame_diff
        )
        
        # Kalman filter ile smoothing
        smoothed = vel_px
        if self.use_kalman:
            kalman = self._get_kalman(track_id)
            smoothed = kalman.update(vel_px)
        
        # History'ye ekle
        history = self._get_history(track_id)
        history.append(vel_px)
        
        # İvme hesabı
        acceleration = 0.0
        last = self.last_velocity.get(track_id, vel_px)
        acceleration = (vel_px - last) * self.fps
        self.last_velocity[track_id] = vel_px
        
        return VelocityData(
            velocity_px_sec=vel_px,
            velocity_m_sec=vel_m,
            direction_deg=direction,
            smoothed_velocity=smoothed,
            acceleration=acceleration
        )
    
    def get_average_velocity(self, track_id: int) -> float:
        """Track için ortalama hız."""
        history = self.velocity_history.get(track_id, [])
        if not history:
            return 0.0
        return sum(history) / len(history)
    
    def get_max_velocity(self, track_id: int) -> float:
        """Track için maksimum hız."""
        history = self.velocity_history.get(track_id, [])
        if not history:
            return 0.0
        return max(history)
    
    def estimate_scale_factor(
        self,
        known_distance_m: float,
        measured_distance_px: float
    ) -> float:
        """
        Bilinen bir mesafeden scale factor hesapla.
        
        Args:
            known_distance_m: Bilinen gerçek mesafe (metre)
            measured_distance_px: Görüntüdeki mesafe (pixel)
        
        Returns:
            Scale factor (m/px)
        """
        return known_distance_m / measured_distance_px
    
    def clear_track(self, track_id: int):
        """Track verilerini temizle."""
        self.kalman_filters.pop(track_id, None)
        self.velocity_history.pop(track_id, None)
        self.last_velocity.pop(track_id, None)


def direction_to_compass(degrees: float) -> str:
    """Dereceyi pusula yönüne çevir."""
    directions = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
    index = round(degrees / 45) % 8
    return directions[index]


def format_velocity(velocity_data: VelocityData) -> str:
    """Hız verisini formatla."""
    compass = direction_to_compass(velocity_data.direction_deg)
    return (
        f"{velocity_data.velocity_px_sec:.1f} px/s "
        f"({velocity_data.velocity_m_sec:.2f} m/s) "
        f"→ {compass}"
    )


# Test
if __name__ == "__main__":
    # Test verisi
    estimator = VelocityEstimator(fps=30, scale_factor=0.03)
    
    # Simüle track history (hareket eden nesne)
    track_history = [
        (100, 100, 1),
        (110, 105, 2),
        (120, 110, 3),
        (135, 115, 4),
        (150, 120, 5),
    ]
    
    for i in range(2, len(track_history) + 1):
        result = estimator.calculate(1, track_history[:i])
        if result:
            print(f"Frame {i}: {format_velocity(result)}")
