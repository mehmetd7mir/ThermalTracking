"""
Anomaly Detection Module
-------------------------
Detect unusual behavior patterns in tracked objects.

Types of anomalies detected:
    - Loitering: object stays in area too long
    - Erratic movement: sudden direction changes
    - Speed anomaly: unusual speed patterns
    - Path deviation: unexpected trajectory

Uses combination of rule-based and ML approaches.

Author: Mehmet Demir
"""

import math
import time
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class Anomaly:
    """Detected anomaly record"""
    track_id: int
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high'
    description: str
    position: Tuple[float, float]
    timestamp: float
    metadata: Dict


class LoiteringDetector:
    """
    Detect objects that stay in same area too long.
    
    Loitering is suspicious behavior in security context.
    """
    
    def __init__(
        self,
        loiter_distance: float = 100.0,
        loiter_time: float = 30.0
    ):
        """
        Args:
            loiter_distance: max movement to count as loitering
            loiter_time: seconds before triggering loiter alert
        """
        self.loiter_distance = loiter_distance
        self.loiter_time = loiter_time
        
        # track first position and time for each track
        self.track_anchors: Dict[int, Tuple[float, float, float]] = {}
        
        # set of tracks currently loitering
        self.loitering_tracks: Set[int] = set()
    
    def check(
        self,
        track_id: int,
        position: Tuple[float, float],
        timestamp: float
    ) -> Optional[Anomaly]:
        """
        Check if track is loitering.
        
        Returns Anomaly if loitering detected, None otherwise.
        """
        if track_id not in self.track_anchors:
            # first time seeing this track
            self.track_anchors[track_id] = (position[0], position[1], timestamp)
            return None
        
        anchor = self.track_anchors[track_id]
        anchor_pos = (anchor[0], anchor[1])
        anchor_time = anchor[2]
        
        # calculate distance from anchor
        distance = math.sqrt(
            (position[0] - anchor_pos[0])**2 + 
            (position[1] - anchor_pos[1])**2
        )
        
        if distance > self.loiter_distance:
            # moved too far, reset anchor
            self.track_anchors[track_id] = (position[0], position[1], timestamp)
            self.loitering_tracks.discard(track_id)
            return None
        
        # check time in area
        time_in_area = timestamp - anchor_time
        
        if time_in_area >= self.loiter_time:
            if track_id not in self.loitering_tracks:
                self.loitering_tracks.add(track_id)
                return Anomaly(
                    track_id=track_id,
                    anomaly_type="loitering",
                    severity="medium",
                    description=f"Object loitering for {time_in_area:.1f}s",
                    position=position,
                    timestamp=timestamp,
                    metadata={"duration": time_in_area}
                )
        
        return None
    
    def clear_track(self, track_id: int):
        """Remove track from monitoring."""
        self.track_anchors.pop(track_id, None)
        self.loitering_tracks.discard(track_id)


class ErraticMovementDetector:
    """
    Detect sudden direction changes and erratic movement.
    
    Normal aircraft follow smooth paths. Erratic movement
    might indicate malfunction or evasive behavior.
    """
    
    def __init__(
        self,
        angle_threshold: float = 90.0,
        min_speed: float = 5.0
    ):
        """
        Args:
            angle_threshold: degrees of direction change to count as erratic
            min_speed: minimum speed to consider for direction change
        """
        self.angle_threshold = angle_threshold
        self.min_speed = min_speed
        
        # store recent directions for each track
        self.track_directions: Dict[int, deque] = {}
    
    def check(
        self,
        track_id: int,
        velocity: Tuple[float, float],
        position: Tuple[float, float],
        timestamp: float
    ) -> Optional[Anomaly]:
        """
        Check for erratic movement.
        """
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        if speed < self.min_speed:
            return None  # too slow to judge direction
        
        # calculate direction in degrees
        direction = math.degrees(math.atan2(velocity[1], velocity[0]))
        
        if track_id not in self.track_directions:
            self.track_directions[track_id] = deque(maxlen=10)
        
        directions = self.track_directions[track_id]
        
        if len(directions) > 0:
            prev_direction = directions[-1]
            
            # calculate angle change
            angle_change = abs(direction - prev_direction)
            if angle_change > 180:
                angle_change = 360 - angle_change
            
            if angle_change > self.angle_threshold:
                directions.append(direction)
                return Anomaly(
                    track_id=track_id,
                    anomaly_type="erratic_movement",
                    severity="high" if angle_change > 120 else "medium",
                    description=f"Sudden direction change: {angle_change:.1f} degrees",
                    position=position,
                    timestamp=timestamp,
                    metadata={
                        "angle_change": angle_change,
                        "speed": speed
                    }
                )
        
        directions.append(direction)
        return None
    
    def clear_track(self, track_id: int):
        self.track_directions.pop(track_id, None)


class SpeedAnomalyDetector:
    """
    Detect unusual speed patterns.
    
    This include:
    - Speeds too high for aircraft type
    - Sudden acceleration/deceleration
    - Hovering (speed near zero)
    """
    
    def __init__(
        self,
        max_drone_speed: float = 30.0,
        acceleration_threshold: float = 10.0,
        hover_threshold: float = 2.0
    ):
        """
        Args:
            max_drone_speed: max expected drone speed (m/s)
            acceleration_threshold: max normal acceleration (m/s^2)
            hover_threshold: speed below this counts as hovering
        """
        self.max_drone_speed = max_drone_speed
        self.acceleration_threshold = acceleration_threshold
        self.hover_threshold = hover_threshold
        
        # track previous speeds
        self.track_speeds: Dict[int, deque] = {}
        self.hover_start: Dict[int, float] = {}
    
    def check(
        self,
        track_id: int,
        speed: float,
        class_name: str,
        position: Tuple[float, float],
        timestamp: float
    ) -> Optional[Anomaly]:
        """
        Check for speed anomalies.
        """
        if track_id not in self.track_speeds:
            self.track_speeds[track_id] = deque(maxlen=30)
        
        speeds = self.track_speeds[track_id]
        
        anomaly = None
        
        # check for excessive speed (for drones)
        if class_name == "drone" and speed > self.max_drone_speed:
            anomaly = Anomaly(
                track_id=track_id,
                anomaly_type="speed_excessive",
                severity="high",
                description=f"Drone speed {speed:.1f} m/s exceeds limit",
                position=position,
                timestamp=timestamp,
                metadata={"speed": speed, "limit": self.max_drone_speed}
            )
        
        # check for sudden acceleration
        if len(speeds) > 0:
            prev_speed = speeds[-1]
            acceleration = abs(speed - prev_speed)  # simplified
            
            if acceleration > self.acceleration_threshold:
                anomaly = Anomaly(
                    track_id=track_id,
                    anomaly_type="sudden_acceleration",
                    severity="medium",
                    description=f"Sudden speed change: {acceleration:.1f} m/s",
                    position=position,
                    timestamp=timestamp,
                    metadata={
                        "acceleration": acceleration,
                        "prev_speed": prev_speed,
                        "new_speed": speed
                    }
                )
        
        # check for hovering
        if class_name == "drone" and speed < self.hover_threshold:
            if track_id not in self.hover_start:
                self.hover_start[track_id] = timestamp
            elif timestamp - self.hover_start[track_id] > 5.0:
                hover_time = timestamp - self.hover_start[track_id]
                anomaly = Anomaly(
                    track_id=track_id,
                    anomaly_type="hovering",
                    severity="low",
                    description=f"Drone hovering for {hover_time:.1f}s",
                    position=position,
                    timestamp=timestamp,
                    metadata={"hover_duration": hover_time}
                )
        else:
            self.hover_start.pop(track_id, None)
        
        speeds.append(speed)
        return anomaly
    
    def clear_track(self, track_id: int):
        self.track_speeds.pop(track_id, None)
        self.hover_start.pop(track_id, None)


class AnomalyDetector:
    """
    Main anomaly detection class combining multiple detectors.
    
    Example:
        detector = AnomalyDetector()
        
        # for each tracked object
        anomalies = detector.check(
            track_id=1,
            position=(100, 200),
            velocity=(5, 3),
            class_name="drone",
            timestamp=time.time()
        )
        
        for anomaly in anomalies:
            print(f"Anomaly: {anomaly.anomaly_type}")
    """
    
    def __init__(self):
        self.loitering = LoiteringDetector()
        self.erratic = ErraticMovementDetector()
        self.speed = SpeedAnomalyDetector()
        
        # store all detected anomalies
        self.anomaly_history: List[Anomaly] = []
        self.max_history = 1000
    
    def check(
        self,
        track_id: int,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        class_name: str,
        timestamp: float
    ) -> List[Anomaly]:
        """
        Run all anomaly checks.
        
        Returns list of detected anomalies (may be empty).
        """
        anomalies = []
        
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # loitering check
        loiter = self.loitering.check(track_id, position, timestamp)
        if loiter:
            anomalies.append(loiter)
        
        # erratic movement check
        erratic = self.erratic.check(track_id, velocity, position, timestamp)
        if erratic:
            anomalies.append(erratic)
        
        # speed anomaly check
        speed_anomaly = self.speed.check(
            track_id, speed, class_name, position, timestamp
        )
        if speed_anomaly:
            anomalies.append(speed_anomaly)
        
        # store in history
        for a in anomalies:
            self.anomaly_history.append(a)
            if len(self.anomaly_history) > self.max_history:
                self.anomaly_history.pop(0)
        
        return anomalies
    
    def clear_track(self, track_id: int):
        """Clear all data for track."""
        self.loitering.clear_track(track_id)
        self.erratic.clear_track(track_id)
        self.speed.clear_track(track_id)
    
    def get_recent_anomalies(self, count: int = 10) -> List[Anomaly]:
        """Get most recent anomalies."""
        return self.anomaly_history[-count:]
    
    def get_anomalies_by_track(self, track_id: int) -> List[Anomaly]:
        """Get all anomalies for specific track."""
        return [a for a in self.anomaly_history if a.track_id == track_id]
    
    def get_stats(self) -> Dict:
        """Get anomaly statistics."""
        by_type = {}
        by_severity = {"low": 0, "medium": 0, "high": 0}
        
        for a in self.anomaly_history:
            by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1
            by_severity[a.severity] += 1
        
        return {
            "total": len(self.anomaly_history),
            "by_type": by_type,
            "by_severity": by_severity
        }


# test
if __name__ == "__main__":
    detector = AnomalyDetector()
    
    # simulate loitering
    print("Testing loitering detection...")
    for i in range(40):
        anomalies = detector.check(
            track_id=1,
            position=(100 + i*0.5, 100 + i*0.5),  # small movement
            velocity=(0.5, 0.5),
            class_name="drone",
            timestamp=float(i)
        )
        if anomalies:
            for a in anomalies:
                print(f"  {a.anomaly_type}: {a.description}")
    
    # simulate erratic movement
    print("\nTesting erratic movement detection...")
    velocities = [(10, 0), (10, 0), (-10, 0), (10, 0)]  # sudden reversal
    for i, vel in enumerate(velocities):
        anomalies = detector.check(
            track_id=2,
            position=(200 + i*10, 200),
            velocity=vel,
            class_name="drone",
            timestamp=50.0 + i
        )
        if anomalies:
            for a in anomalies:
                print(f"  {a.anomaly_type}: {a.description}")
    
    print(f"\nStats: {detector.get_stats()}")
