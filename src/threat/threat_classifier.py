"""
ThermalTracking - Threat Classification Module
===============================================
Tespit edilen nesneleri tehdit seviyesine göre sınıflandırır.

Tehdit Seviyeleri:
- HIGH (🔴): Acil müdahale gerektirir
- MEDIUM (🟡): İzleme gerektirir
- LOW (🟢): Normal aktivite

Faktörler:
- Nesne sınıfı (drone > helicopter > plane > bird)
- Hız (hızlı = tehlikeli)
- Mesafe (yakın = tehlikeli)
- Hareket paterni (düzensiz = şüpheli)

Kullanım:
    from threat_classifier import ThreatClassifier
    
    classifier = ThreatClassifier()
    threat = classifier.classify(detection, velocity, distance)
    print(threat.level, threat.score, threat.action)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple
import time


class ThreatLevel(Enum):
    """Tehdit seviyeleri"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class ThreatAssessment:
    """Tehdit değerlendirmesi sonucu"""
    level: ThreatLevel
    score: float           # 0-100 arası tehdit puanı
    icon: str              # Emoji icon
    color: Tuple[int, int, int]  # BGR color
    action: str            # Önerilen aksiyon
    factors: Dict[str, float]    # Puan kaynakları
    timestamp: float       # Değerlendirme zamanı


class ThreatClassifier:
    """
    Threat classification engine.
    
    Nesne tespitlerini analiz ederek tehdit seviyesi belirler.
    """
    
    # Sınıf bazlı temel tehdit puanları
    CLASS_BASE_SCORES = {
        'drone': 70,        # Drone'lar varsayılan olarak yüksek tehdit
        'helicopter': 50,
        'plane': 40,
        'bird': 10,
        'unknown': 30
    }
    
    # Tehdit seviyeleri için eşikler
    THRESHOLDS = {
        'high': 80,
        'medium': 50
    }
    
    # Renkler (BGR)
    COLORS = {
        ThreatLevel.HIGH: (0, 0, 255),     # Kırmızı
        ThreatLevel.MEDIUM: (0, 165, 255),  # Turuncu
        ThreatLevel.LOW: (0, 255, 0),       # Yeşil
        ThreatLevel.UNKNOWN: (128, 128, 128)  # Gri
    }
    
    # Emoji iconlar
    ICONS = {
        ThreatLevel.HIGH: "🔴",
        ThreatLevel.MEDIUM: "🟡",
        ThreatLevel.LOW: "🟢",
        ThreatLevel.UNKNOWN: "⚪"
    }
    
    def __init__(
        self,
        velocity_danger_threshold: float = 30.0,  # m/s
        distance_danger_threshold: float = 500.0,  # m
        restricted_zones: Optional[List[Tuple[float, float, float]]] = None  # [(x, y, radius), ...]
    ):
        """
        Args:
            velocity_danger_threshold: Bu hızın üzerinde tehlikeli sayılır (m/s)
            distance_danger_threshold: Bu mesafenin altında tehlikeli sayılır (m)
            restricted_zones: Yasaklı bölgeler [(x, y, radius), ...]
        """
        self.velocity_danger_threshold = velocity_danger_threshold
        self.distance_danger_threshold = distance_danger_threshold
        self.restricted_zones = restricted_zones or []
        
        # Alarm history
        self.alarm_history: List[ThreatAssessment] = []
    
    def classify(
        self,
        object_class: str,
        velocity: float = 0.0,           # m/s
        distance: float = float('inf'),   # m
        position: Optional[Tuple[float, float]] = None,  # (x, y)
        confidence: float = 1.0,
        trajectory_variance: float = 0.0  # Hareket düzensizliği
    ) -> ThreatAssessment:
        """
        Tek bir nesne için tehdit değerlendirmesi yap.
        
        Args:
            object_class: Nesne sınıfı ('drone', 'bird', vb.)
            velocity: Hız (m/s)
            distance: Mesafe (m)
            position: Konum (x, y)
            confidence: Tespit güveni (0-1)
            trajectory_variance: Hareket düzensizliği
        
        Returns:
            ThreatAssessment
        """
        factors = {}
        
        # 1. Temel sınıf puanı
        base_score = self.CLASS_BASE_SCORES.get(object_class.lower(), 30)
        factors['class'] = base_score
        
        # 2. Hız faktörü
        velocity_score = 0
        if velocity > 0:
            velocity_ratio = velocity / self.velocity_danger_threshold
            velocity_score = min(20, velocity_ratio * 20)
        factors['velocity'] = velocity_score
        
        # 3. Mesafe faktörü
        distance_score = 0
        if distance < float('inf'):
            if distance < self.distance_danger_threshold:
                distance_ratio = 1 - (distance / self.distance_danger_threshold)
                distance_score = distance_ratio * 25
        factors['distance'] = distance_score
        
        # 4. Yasaklı bölge kontrolü
        zone_score = 0
        if position and self.restricted_zones:
            for zone_x, zone_y, zone_radius in self.restricted_zones:
                dist_to_zone = ((position[0] - zone_x)**2 + (position[1] - zone_y)**2)**0.5
                if dist_to_zone < zone_radius:
                    zone_score = 30
                    break
        factors['restricted_zone'] = zone_score
        
        # 5. Hareket paterni (düzensiz hareket şüpheli)
        pattern_score = min(10, trajectory_variance * 5)
        factors['pattern'] = pattern_score
        
        # 6. Güven faktörü (düşük güvenli tespitler puanı azaltır)
        confidence_modifier = 0.7 + (confidence * 0.3)
        
        # Toplam puan
        total_score = sum(factors.values()) * confidence_modifier
        total_score = min(100, max(0, total_score))
        
        # Tehdit seviyesi belirle
        if total_score >= self.THRESHOLDS['high']:
            level = ThreatLevel.HIGH
            action = "IMMEDIATE ALERT: Potential threat detected. Initiate countermeasures."
        elif total_score >= self.THRESHOLDS['medium']:
            level = ThreatLevel.MEDIUM
            action = "MONITOR: Track object closely. Prepare for escalation."
        else:
            level = ThreatLevel.LOW
            action = "LOG: Normal activity. Continue surveillance."
        
        assessment = ThreatAssessment(
            level=level,
            score=total_score,
            icon=self.ICONS[level],
            color=self.COLORS[level],
            action=action,
            factors=factors,
            timestamp=time.time()
        )
        
        # High threat ise alarm history'ye ekle
        if level == ThreatLevel.HIGH:
            self.alarm_history.append(assessment)
        
        return assessment
    
    def classify_batch(
        self,
        detections: List[Dict]
    ) -> List[ThreatAssessment]:
        """
        Birden fazla tespit için batch sınıflandırma.
        
        Args:
            detections: [{'class': str, 'velocity': float, 'distance': float, ...}, ...]
        
        Returns:
            List[ThreatAssessment]
        """
        return [
            self.classify(
                object_class=d.get('class', 'unknown'),
                velocity=d.get('velocity', 0),
                distance=d.get('distance', float('inf')),
                position=d.get('position'),
                confidence=d.get('confidence', 1.0),
                trajectory_variance=d.get('trajectory_variance', 0)
            )
            for d in detections
        ]
    
    def get_highest_threat(
        self,
        assessments: List[ThreatAssessment]
    ) -> Optional[ThreatAssessment]:
        """En yüksek tehdit seviyesini döndür."""
        if not assessments:
            return None
        return max(assessments, key=lambda a: a.score)
    
    def get_threat_summary(
        self,
        assessments: List[ThreatAssessment]
    ) -> Dict:
        """Tehdit özeti oluştur."""
        summary = {
            'total': len(assessments),
            'high': sum(1 for a in assessments if a.level == ThreatLevel.HIGH),
            'medium': sum(1 for a in assessments if a.level == ThreatLevel.MEDIUM),
            'low': sum(1 for a in assessments if a.level == ThreatLevel.LOW),
            'max_score': max((a.score for a in assessments), default=0),
            'avg_score': sum(a.score for a in assessments) / len(assessments) if assessments else 0
        }
        return summary
    
    def format_assessment(self, assessment: ThreatAssessment) -> str:
        """Tehdit değerlendirmesini formatla."""
        return (
            f"{assessment.icon} {assessment.level.value} "
            f"(Score: {assessment.score:.1f})\n"
            f"   Action: {assessment.action}\n"
            f"   Factors: {assessment.factors}"
        )
    
    def add_restricted_zone(self, x: float, y: float, radius: float):
        """Yasaklı bölge ekle."""
        self.restricted_zones.append((x, y, radius))
    
    def clear_alarm_history(self):
        """Alarm geçmişini temizle."""
        self.alarm_history.clear()


class AlertSystem:
    """
    Alert yönetim sistemi.
    
    Threatları toplar, filtreler ve bildirim gönderir.
    """
    
    def __init__(self, cooldown_seconds: float = 5.0):
        """
        Args:
            cooldown_seconds: Aynı nesne için tekrar alarm süresi
        """
        self.cooldown = cooldown_seconds
        self.last_alerts: Dict[int, float] = {}  # {track_id: last_alert_time}
        self.alert_log: List[Dict] = []
    
    def should_alert(self, track_id: int) -> bool:
        """Bu track için alarm verilmeli mi?"""
        now = time.time()
        last = self.last_alerts.get(track_id, 0)
        return (now - last) > self.cooldown
    
    def trigger_alert(
        self,
        track_id: int,
        assessment: ThreatAssessment,
        object_class: str
    ) -> Optional[Dict]:
        """
        Alarm tetikle.
        
        Returns:
            Alert dict veya None (cooldown içindeyse)
        """
        if not self.should_alert(track_id):
            return None
        
        self.last_alerts[track_id] = time.time()
        
        alert = {
            'track_id': track_id,
            'class': object_class,
            'threat_level': assessment.level.value,
            'score': assessment.score,
            'action': assessment.action,
            'timestamp': assessment.timestamp
        }
        
        self.alert_log.append(alert)
        
        # Console output
        print("\n" + "=" * 50)
        print(f"⚠️ ALERT - {assessment.icon} {assessment.level.value}")
        print(f"   Track ID: {track_id}")
        print(f"   Class: {object_class}")
        print(f"   Score: {assessment.score:.1f}")
        print(f"   Action: {assessment.action}")
        print("=" * 50 + "\n")
        
        return alert
    
    def get_recent_alerts(self, seconds: float = 60.0) -> List[Dict]:
        """Son X saniyedeki alarmlar."""
        cutoff = time.time() - seconds
        return [a for a in self.alert_log if a['timestamp'] > cutoff]


# Test
if __name__ == "__main__":
    print("=" * 50)
    print("Threat Classification Test")
    print("=" * 50)
    
    classifier = ThreatClassifier(
        velocity_danger_threshold=30.0,
        distance_danger_threshold=500.0
    )
    
    # Yasaklı bölge ekle (örn: havalimanı)
    classifier.add_restricted_zone(0, 0, 1000)
    
    # Test senaryoları
    test_cases = [
        {'class': 'drone', 'velocity': 50, 'distance': 200, 'position': (100, 100)},
        {'class': 'drone', 'velocity': 10, 'distance': 1000},
        {'class': 'bird', 'velocity': 15, 'distance': 300},
        {'class': 'helicopter', 'velocity': 80, 'distance': 2000},
        {'class': 'plane', 'velocity': 200, 'distance': 5000},
    ]
    
    print("\n📊 Test Sonuçları:\n")
    
    assessments = classifier.classify_batch(test_cases)
    
    for case, assessment in zip(test_cases, assessments):
        print(f"Input: {case}")
        print(f"Result: {classifier.format_assessment(assessment)}")
        print()
    
    # Özet
    summary = classifier.get_threat_summary(assessments)
    print(f"\n📈 Özet: {summary}")
