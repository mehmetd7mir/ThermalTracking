"""
Line Counter Module for ThermalTracking
----------------------------------------
This module count objects that crossing a virtual line.
You can use it for counting drones entering or leaving area.

How it works:
    1. Define a line with start and end points
    2. Track object centers
    3. When center cross the line, increment counter

Author: Mehmet Demir
Last update: February 2026
"""

import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class CrossingEvent:
    """Store info about when object cross the line"""
    track_id: int
    direction: str  # 'up' or 'down' (or 'left'/'right' for horizontal)
    timestamp: float
    class_name: str
    confidence: float


class LineCounter:
    """
    Count objects crossing a virtual line.
    
    The line can be horizontal, vertical or diagonal.
    Direction is determined by which side object come from.
    
    Example usage:
        counter = LineCounter(
            line_start=(0, 300),
            line_end=(640, 300)
        )
        
        # for each detection
        crossed = counter.update(track_id, center_point, class_name)
        if crossed:
            print(f"Object {track_id} crossed the line!")
    """
    
    def __init__(
        self,
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
        direction_positive: str = "down"
    ):
        """
        Initialize the line counter.
        
        Args:
            line_start: starting point of line (x, y)
            line_end: ending point of line (x, y)
            direction_positive: which direction counts as "positive" 
                               (entering). Can be 'down', 'up', 'left', 'right'
        """
        self.line_start = line_start
        self.line_end = line_end
        self.direction_positive = direction_positive
        
        # counts for each direction
        self.count_positive = 0  # entering
        self.count_negative = 0  # leaving
        
        # keep track of previous positions for each object
        # we need this to detect when object cross the line
        self.previous_positions: Dict[int, Tuple[float, float]] = {}
        
        # store all crossing events for later analysis
        self.crossing_events: List[CrossingEvent] = []
        
        # track which objects already crossed (prevent double counting)
        self.crossed_ids: set = set()
        
        # class based counting
        self.class_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"positive": 0, "negative": 0}
        )
    
    def _get_line_side(self, point: Tuple[float, float]) -> float:
        """
        Determine which side of line the point is on.
        
        Uses cross product to find side.
        Returns positive or negative value depending on side.
        """
        x, y = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        
        # cross product formula
        # if result > 0: point is on left side
        # if result < 0: point is on right side
        # if result = 0: point is on the line
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        
        return cross
    
    def _check_crossing(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float]
    ) -> Optional[str]:
        """
        Check if object crossed the line between two positions.
        
        Returns direction of crossing or None if no crossing.
        """
        prev_side = self._get_line_side(prev_pos)
        curr_side = self._get_line_side(curr_pos)
        
        # if signs are different, object crossed the line
        if prev_side * curr_side < 0:
            # determine direction based on sign change
            if prev_side > 0:
                return "negative"
            else:
                return "positive"
        
        return None
    
    def update(
        self,
        track_id: int,
        center: Tuple[float, float],
        class_name: str = "unknown",
        confidence: float = 0.0,
        timestamp: float = 0.0
    ) -> Optional[str]:
        """
        Update tracker with new object position.
        
        Args:
            track_id: unique id of tracked object
            center: current center position (x, y)
            class_name: class of object (drone, plane, etc)
            confidence: detection confidence
            timestamp: current time
        
        Returns:
            Direction of crossing if object crossed, None otherwise
        """
        # skip if already counted this object
        if track_id in self.crossed_ids:
            # but still update position for trajectory drawing
            self.previous_positions[track_id] = center
            return None
        
        # check if we have previous position
        if track_id in self.previous_positions:
            prev_pos = self.previous_positions[track_id]
            
            # check for crossing
            direction = self._check_crossing(prev_pos, center)
            
            if direction is not None:
                # object crossed the line!
                self.crossed_ids.add(track_id)
                
                if direction == "positive":
                    self.count_positive += 1
                else:
                    self.count_negative += 1
                
                # update class based counting
                self.class_counts[class_name][direction] += 1
                
                # record the event
                event = CrossingEvent(
                    track_id=track_id,
                    direction=direction,
                    timestamp=timestamp,
                    class_name=class_name,
                    confidence=confidence
                )
                self.crossing_events.append(event)
                
                # update position
                self.previous_positions[track_id] = center
                
                return direction
        
        # update position for next frame
        self.previous_positions[track_id] = center
        return None
    
    def get_counts(self) -> Dict[str, int]:
        """Get current counts."""
        return {
            "positive": self.count_positive,
            "negative": self.count_negative,
            "total": self.count_positive + self.count_negative
        }
    
    def get_class_counts(self) -> Dict[str, Dict[str, int]]:
        """Get counts broken down by class."""
        return dict(self.class_counts)
    
    def reset(self):
        """Reset all counters and state."""
        self.count_positive = 0
        self.count_negative = 0
        self.previous_positions.clear()
        self.crossing_events.clear()
        self.crossed_ids.clear()
        self.class_counts.clear()
    
    def draw_line(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draw the counting line on frame.
        
        Also draws the current counts near the line.
        """
        import cv2
        
        # draw the line
        cv2.line(
            frame,
            self.line_start,
            self.line_end,
            color,
            thickness=2
        )
        
        # calculate midpoint for text placement
        mid_x = (self.line_start[0] + self.line_end[0]) // 2
        mid_y = (self.line_start[1] + self.line_end[1]) // 2
        
        # draw counts
        text = f"IN: {self.count_positive} | OUT: {self.count_negative}"
        cv2.putText(
            frame,
            text,
            (mid_x - 60, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        return frame


class MultiLineCounter:
    """
    Handle multiple counting lines at once.
    
    Usefull when you have multiple entry/exit points.
    """
    
    def __init__(self):
        self.lines: Dict[str, LineCounter] = {}
    
    def add_line(
        self,
        name: str,
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ):
        """Add a new counting line."""
        self.lines[name] = LineCounter(line_start, line_end)
    
    def update_all(
        self,
        track_id: int,
        center: Tuple[float, float],
        class_name: str = "unknown",
        confidence: float = 0.0,
        timestamp: float = 0.0
    ) -> Dict[str, Optional[str]]:
        """
        Update all lines with new position.
        
        Returns dict of line_name -> crossing_direction
        """
        results = {}
        for name, counter in self.lines.items():
            result = counter.update(
                track_id, center, class_name, confidence, timestamp
            )
            if result is not None:
                results[name] = result
        return results
    
    def get_all_counts(self) -> Dict[str, Dict[str, int]]:
        """Get counts from all lines."""
        return {name: counter.get_counts() for name, counter in self.lines.items()}
    
    def draw_all(self, frame: np.ndarray) -> np.ndarray:
        """Draw all counting lines on frame."""
        for counter in self.lines.values():
            frame = counter.draw_line(frame)
        return frame


# simple test
if __name__ == "__main__":
    # create a simple test
    counter = LineCounter(
        line_start=(0, 240),
        line_end=(640, 240)
    )
    
    # simulate object moving across the line
    positions = [
        (320, 200),  # above line
        (320, 230),  # still above
        (320, 250),  # crossed!
        (320, 280),  # below line
    ]
    
    for i, pos in enumerate(positions):
        result = counter.update(
            track_id=1,
            center=pos,
            class_name="drone",
            timestamp=i * 0.033  # 30 fps
        )
        if result:
            print(f"Crossed! Direction: {result}")
    
    print(f"Final counts: {counter.get_counts()}")
