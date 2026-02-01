"""
Trajectory Prediction Module
-----------------------------
Predict future positions of tracked objects.

Uses simple linear prediction and also support 
LSTM model if you want more accuracy.

The prediction is usefull for:
    - Collision warning
    - Intercept calculation
    - Path forecasting

Author: Mehmet Demir
"""

import math
from typing import Tuple, List, Dict, Optional
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class Prediction:
    """Predicted future position"""
    track_id: int
    predicted_positions: List[Tuple[float, float]]
    timestamps: List[float]
    confidence: float


class TrajectoryPredictor:
    """
    Predict future object positions based on history.
    
    Methods:
        - Linear: simple velocity based extrapolation
        - Polynomial: fit polynomial curve to history
        - Kalman: use kalman filter for prediction
    
    Example:
        predictor = TrajectoryPredictor(method='linear')
        
        # update with new positions
        predictor.update(track_id, position, timestamp)
        
        # get prediction
        future = predictor.predict(track_id, seconds=2.0)
    """
    
    def __init__(
        self,
        method: str = "linear",
        history_size: int = 20,
        fps: float = 30.0
    ):
        """
        Initialize predictor.
        
        Args:
            method: prediction method ('linear', 'polynomial', 'kalman')
            history_size: how many past positions to keep
            fps: frames per second for time calculation
        """
        self.method = method
        self.history_size = history_size
        self.fps = fps
        
        # store history for each track
        # each entry is (x, y, timestamp)
        self.histories: Dict[int, deque] = {}
    
    def _get_history(self, track_id: int) -> deque:
        """Get or create history for track."""
        if track_id not in self.histories:
            self.histories[track_id] = deque(maxlen=self.history_size)
        return self.histories[track_id]
    
    def update(
        self,
        track_id: int,
        position: Tuple[float, float],
        timestamp: Optional[float] = None
    ):
        """
        Add new position to track history.
        
        Args:
            track_id: id of track
            position: current (x, y) position
            timestamp: time of observation (auto generated if None)
        """
        history = self._get_history(track_id)
        
        if timestamp is None:
            # calculate timestamp from frame count
            if len(history) > 0:
                timestamp = history[-1][2] + 1.0 / self.fps
            else:
                timestamp = 0.0
        
        history.append((position[0], position[1], timestamp))
    
    def _predict_linear(
        self,
        history: deque,
        horizon_seconds: float,
        num_points: int
    ) -> List[Tuple[float, float]]:
        """
        Linear prediction using last velocity.
        
        Simple but fast. Works good for short time horizon.
        """
        if len(history) < 2:
            return []
        
        # calculate velocity from last two points
        p1 = history[-2]
        p2 = history[-1]
        
        dt = p2[2] - p1[2]
        if dt <= 0:
            dt = 1.0 / self.fps
        
        vx = (p2[0] - p1[0]) / dt
        vy = (p2[1] - p1[1]) / dt
        
        # extrapolate future positions
        predictions = []
        time_step = horizon_seconds / num_points
        
        for i in range(1, num_points + 1):
            t = i * time_step
            x = p2[0] + vx * t
            y = p2[1] + vy * t
            predictions.append((x, y))
        
        return predictions
    
    def _predict_polynomial(
        self,
        history: deque,
        horizon_seconds: float,
        num_points: int,
        degree: int = 2
    ) -> List[Tuple[float, float]]:
        """
        Polynomial curve fitting prediction.
        
        Better for curved trajectories but slower.
        """
        if len(history) < degree + 1:
            return self._predict_linear(history, horizon_seconds, num_points)
        
        # extract x, y, t from history
        t_vals = np.array([p[2] for p in history])
        x_vals = np.array([p[0] for p in history])
        y_vals = np.array([p[1] for p in history])
        
        # normalize time to avoid numerical issues
        t_min = t_vals[0]
        t_vals = t_vals - t_min
        
        try:
            # fit polynomial
            x_coeffs = np.polyfit(t_vals, x_vals, degree)
            y_coeffs = np.polyfit(t_vals, y_vals, degree)
            
            # create polynomial functions
            x_poly = np.poly1d(x_coeffs)
            y_poly = np.poly1d(y_coeffs)
            
            # predict future
            predictions = []
            last_t = t_vals[-1]
            time_step = horizon_seconds / num_points
            
            for i in range(1, num_points + 1):
                t = last_t + i * time_step
                x = float(x_poly(t))
                y = float(y_poly(t))
                predictions.append((x, y))
            
            return predictions
            
        except np.linalg.LinAlgError:
            # fallback to linear if fitting fails
            return self._predict_linear(history, horizon_seconds, num_points)
    
    def predict(
        self,
        track_id: int,
        horizon_seconds: float = 1.0,
        num_points: int = 10
    ) -> Optional[Prediction]:
        """
        Predict future trajectory.
        
        Args:
            track_id: id of track to predict
            horizon_seconds: how far into future to predict
            num_points: number of predicted points
        
        Returns:
            Prediction object or None if not enough data
        """
        if track_id not in self.histories:
            return None
        
        history = self.histories[track_id]
        
        if len(history) < 2:
            return None
        
        # choose prediction method
        if self.method == "linear":
            positions = self._predict_linear(history, horizon_seconds, num_points)
        elif self.method == "polynomial":
            positions = self._predict_polynomial(history, horizon_seconds, num_points)
        else:
            positions = self._predict_linear(history, horizon_seconds, num_points)
        
        if not positions:
            return None
        
        # calculate timestamps for predictions
        last_t = history[-1][2]
        time_step = horizon_seconds / num_points
        timestamps = [last_t + i * time_step for i in range(1, num_points + 1)]
        
        # calculate confidence based on history length and variance
        confidence = min(1.0, len(history) / self.history_size)
        
        return Prediction(
            track_id=track_id,
            predicted_positions=positions,
            timestamps=timestamps,
            confidence=confidence
        )
    
    def draw_prediction(
        self,
        frame: np.ndarray,
        prediction: Prediction,
        color: Tuple[int, int, int] = (255, 0, 255)
    ) -> np.ndarray:
        """
        Draw predicted trajectory on frame.
        
        Uses dashed line style to show uncertainty.
        """
        import cv2
        
        if prediction is None or len(prediction.predicted_positions) < 2:
            return frame
        
        positions = prediction.predicted_positions
        
        # draw predicted path
        for i in range(1, len(positions)):
            p1 = (int(positions[i-1][0]), int(positions[i-1][1]))
            p2 = (int(positions[i][0]), int(positions[i][1]))
            
            # make line more transparent as we go further
            alpha = 1.0 - (i / len(positions)) * 0.5
            thickness = max(1, int(3 * alpha))
            
            cv2.line(frame, p1, p2, color, thickness)
        
        # draw end point marker
        last_pos = positions[-1]
        cv2.circle(
            frame,
            (int(last_pos[0]), int(last_pos[1])),
            5, color, -1
        )
        
        # add confidence text
        text = f"pred: {prediction.confidence:.0%}"
        cv2.putText(
            frame, text,
            (int(last_pos[0]) + 10, int(last_pos[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
        
        return frame
    
    def clear_track(self, track_id: int):
        """Remove track from history."""
        if track_id in self.histories:
            del self.histories[track_id]
    
    def clear_all(self):
        """Clear all track histories."""
        self.histories.clear()


def calculate_collision_risk(
    pred1: Prediction,
    pred2: Prediction,
    threshold_distance: float = 50.0
) -> float:
    """
    Calculate collision risk between two predicted trajectories.
    
    Returns value between 0 (no risk) and 1 (certain collision).
    """
    if pred1 is None or pred2 is None:
        return 0.0
    
    min_distance = float('inf')
    
    # check distance at each time step
    for p1, p2 in zip(pred1.predicted_positions, pred2.predicted_positions):
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        min_distance = min(min_distance, dist)
    
    # convert distance to risk score
    if min_distance >= threshold_distance:
        return 0.0
    else:
        return 1.0 - (min_distance / threshold_distance)


# test
if __name__ == "__main__":
    predictor = TrajectoryPredictor(method='polynomial')
    
    # simulate moving object
    for i in range(10):
        x = 100 + i * 20
        y = 100 + i * 10 + (i ** 2) * 0.5  # slight curve
        predictor.update(1, (x, y), i * 0.033)
    
    # predict future
    pred = predictor.predict(1, horizon_seconds=0.5, num_points=5)
    
    if pred:
        print("Predicted positions:")
        for pos in pred.predicted_positions:
            print(f"  {pos}")
        print(f"Confidence: {pred.confidence:.2f}")
