"""
Zone Intrusion Detection Module
--------------------------------
Detect when objects enter or exit defined zones (polygons).

This is usefull for:
    - No-fly zone monitoring
    - Restricted area protection
    - Perimeter security

The module use ray casting algorithm to check if point inside polygon.

Author: Mehmet Demir
"""

from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import time


@dataclass
class IntrusionEvent:
    """Record of zone intrusion"""
    track_id: int
    zone_name: str
    event_type: str  # 'enter' or 'exit'
    timestamp: float
    class_name: str
    position: Tuple[float, float]


class Zone:
    """
    Represent a single zone defined by polygon vertices.
    
    The polygon should be convex for best results.
    Points should be in clockwise or counter-clockwise order.
    """
    
    def __init__(
        self,
        name: str,
        vertices: List[Tuple[int, int]],
        alert_on_enter: bool = True,
        alert_on_exit: bool = False
    ):
        """
        Create new zone.
        
        Args:
            name: name to identify zone
            vertices: list of (x, y) points defining polygon
            alert_on_enter: trigger alert when object enter zone
            alert_on_exit: trigger alert when object exit zone
        """
        self.name = name
        self.vertices = vertices
        self.alert_on_enter = alert_on_enter
        self.alert_on_exit = alert_on_exit
        
        # track which objects currently in zone
        self.objects_inside: Set[int] = set()
        
        # pre compute bounding box for fast rejection
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
    
    def _point_in_polygon(self, x: float, y: float) -> bool:
        """
        Check if point inside polygon using ray casting.
        
        The algorithm cast a ray from point to infinity
        and count how many times it cross polygon edges.
        If odd number of crossings, point is inside.
        """
        # first check bounding box for quick reject
        if x < self.bbox[0] or x > self.bbox[2]:
            return False
        if y < self.bbox[1] or y > self.bbox[3]:
            return False
        
        n = len(self.vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            
            # check if ray cross this edge
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def check_position(
        self,
        track_id: int,
        position: Tuple[float, float]
    ) -> Optional[str]:
        """
        Check if position trigger zone event.
        
        Args:
            track_id: id of tracked object
            position: (x, y) position to check
        
        Returns:
            'enter' if object just entered zone
            'exit' if object just left zone
            None if no change
        """
        x, y = position
        is_inside = self._point_in_polygon(x, y)
        was_inside = track_id in self.objects_inside
        
        if is_inside and not was_inside:
            # object entered zone
            self.objects_inside.add(track_id)
            if self.alert_on_enter:
                return "enter"
        
        elif not is_inside and was_inside:
            # object exited zone
            self.objects_inside.discard(track_id)
            if self.alert_on_exit:
                return "exit"
        
        return None
    
    def get_object_count(self) -> int:
        """Return number of objects currently in zone."""
        return len(self.objects_inside)
    
    def clear(self):
        """Clear all tracked objects."""
        self.objects_inside.clear()


class ZoneIntrusion:
    """
    Monitor multiple zones for intrusion events.
    
    Example:
        detector = ZoneIntrusion()
        
        # add restricted zone
        detector.add_zone(
            name="helipad",
            vertices=[(100, 100), (200, 100), (200, 200), (100, 200)]
        )
        
        # check each detection
        events = detector.check(track_id, position, class_name)
        for event in events:
            print(f"Alert! {event.event_type} in zone {event.zone_name}")
    """
    
    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        self.events: List[IntrusionEvent] = []
        
        # stats
        self.total_intrusions = 0
        self.intrusions_by_zone: Dict[str, int] = defaultdict(int)
        self.intrusions_by_class: Dict[str, int] = defaultdict(int)
    
    def add_zone(
        self,
        name: str,
        vertices: List[Tuple[int, int]],
        alert_on_enter: bool = True,
        alert_on_exit: bool = False
    ):
        """Add zone to monitor."""
        self.zones[name] = Zone(name, vertices, alert_on_enter, alert_on_exit)
    
    def remove_zone(self, name: str):
        """Remove zone from monitoring."""
        if name in self.zones:
            del self.zones[name]
    
    def check(
        self,
        track_id: int,
        position: Tuple[float, float],
        class_name: str = "unknown",
        timestamp: Optional[float] = None
    ) -> List[IntrusionEvent]:
        """
        Check position against all zones.
        
        Returns list of intrusion events (may be empty).
        """
        if timestamp is None:
            timestamp = time.time()
        
        events = []
        
        for zone in self.zones.values():
            event_type = zone.check_position(track_id, position)
            
            if event_type is not None:
                event = IntrusionEvent(
                    track_id=track_id,
                    zone_name=zone.name,
                    event_type=event_type,
                    timestamp=timestamp,
                    class_name=class_name,
                    position=position
                )
                events.append(event)
                self.events.append(event)
                
                # update stats
                if event_type == "enter":
                    self.total_intrusions += 1
                    self.intrusions_by_zone[zone.name] += 1
                    self.intrusions_by_class[class_name] += 1
        
        return events
    
    def get_zone_status(self) -> Dict[str, Dict]:
        """Get current status of all zones."""
        status = {}
        for name, zone in self.zones.items():
            status[name] = {
                "object_count": zone.get_object_count(),
                "objects_inside": list(zone.objects_inside),
                "total_intrusions": self.intrusions_by_zone[name]
            }
        return status
    
    def get_stats(self) -> Dict:
        """Get intrusion statistics."""
        return {
            "total_intrusions": self.total_intrusions,
            "by_zone": dict(self.intrusions_by_zone),
            "by_class": dict(self.intrusions_by_class)
        }
    
    def draw_zones(
        self,
        frame: np.ndarray,
        show_counts: bool = True
    ) -> np.ndarray:
        """
        Draw all zones on frame.
        
        Zones with objects inside are highlighted red.
        """
        import cv2
        
        for name, zone in self.zones.items():
            vertices = np.array(zone.vertices, dtype=np.int32)
            
            # choose color based on occupancy
            if zone.get_object_count() > 0:
                color = (0, 0, 255)  # red when occupied
                thickness = 3
            else:
                color = (0, 255, 0)  # green when empty
                thickness = 2
            
            # draw polygon
            cv2.polylines(frame, [vertices], True, color, thickness)
            
            # fill with transparent color
            overlay = frame.copy()
            if zone.get_object_count() > 0:
                cv2.fillPoly(overlay, [vertices], (0, 0, 100))
            else:
                cv2.fillPoly(overlay, [vertices], (0, 100, 0))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # draw zone name and count
            if show_counts:
                # find center of zone for text
                center_x = sum(v[0] for v in zone.vertices) // len(zone.vertices)
                center_y = sum(v[1] for v in zone.vertices) // len(zone.vertices)
                
                text = f"{name}: {zone.get_object_count()}"
                cv2.putText(
                    frame, text,
                    (center_x - 30, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
        
        return frame
    
    def reset(self):
        """Reset all zones and stats."""
        for zone in self.zones.values():
            zone.clear()
        self.events.clear()
        self.total_intrusions = 0
        self.intrusions_by_zone.clear()
        self.intrusions_by_class.clear()


# test code
if __name__ == "__main__":
    # create detector
    detector = ZoneIntrusion()
    
    # add test zone
    detector.add_zone(
        name="restricted_area",
        vertices=[(100, 100), (300, 100), (300, 300), (100, 300)]
    )
    
    # simulate object moving into zone
    positions = [
        (50, 200),   # outside
        (80, 200),   # outside
        (120, 200),  # inside - should trigger
        (200, 200),  # inside
        (350, 200),  # outside - should trigger exit
    ]
    
    for i, pos in enumerate(positions):
        events = detector.check(
            track_id=1,
            position=pos,
            class_name="drone",
            timestamp=i * 0.033
        )
        for e in events:
            print(f"Event: {e.event_type} zone {e.zone_name}")
    
    print(f"\nStats: {detector.get_stats()}")
