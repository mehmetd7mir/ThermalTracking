"""
Swarm Detection Module
-----------------------
Detect drone swarms and group behavior patterns.

A swarm is multiple drones flying in coordinated pattern.
This module detect:
    - Group clustering (drones flying together)
    - Formation patterns (line, V shape, grid)
    - Coordinated movement

Used for:
    - Military swarm detection
    - Airspace monitoring
    - Threat assessment

Author: Mehmet Demir
"""

import math
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class SwarmGroup:
    """Represents detected swarm group"""
    group_id: int
    member_ids: List[int]
    center: Tuple[float, float]
    radius: float
    formation: str  # 'cluster', 'line', 'v_shape', 'grid', 'unknown'
    velocity: Tuple[float, float]
    confidence: float


class SwarmDetector:
    """
    Detect and analyze drone swarms.
    
    Uses DBSCAN-like clustering to find groups of drones
    then analyzes their formation and movement patterns.
    
    Example:
        detector = SwarmDetector(min_group_size=3)
        
        # update with track positions
        groups = detector.detect(tracks)
        
        for group in groups:
            print(f"Swarm detected: {len(group.member_ids)} drones")
    """
    
    def __init__(
        self,
        min_group_size: int = 3,
        cluster_distance: float = 100.0,
        velocity_threshold: float = 0.5
    ):
        """
        Initialize detector.
        
        Args:
            min_group_size: minimum drones to count as swarm
            cluster_distance: max distance between cluster members
            velocity_threshold: max velocity difference for group
        """
        self.min_group_size = min_group_size
        self.cluster_distance = cluster_distance
        self.velocity_threshold = velocity_threshold
        
        # track positions over time
        self.track_history: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        
        # detected groups history
        self.group_history: List[List[SwarmGroup]] = []
        
        # group id counter
        self.next_group_id = 0
    
    def update_tracks(
        self,
        tracks: Dict[int, Tuple[float, float]],
        timestamp: float = 0.0
    ):
        """
        Update track positions.
        
        Args:
            tracks: {track_id: (x, y)} current positions
            timestamp: current time
        """
        for track_id, pos in tracks.items():
            self.track_history[track_id].append((pos[0], pos[1], timestamp))
            
            # keep only recent history
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
    
    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Calculate velocity for track."""
        history = self.track_history.get(track_id, [])
        if len(history) < 2:
            return None
        
        p1 = history[-2]
        p2 = history[-1]
        
        dt = p2[2] - p1[2]
        if dt <= 0:
            dt = 1.0
        
        vx = (p2[0] - p1[0]) / dt
        vy = (p2[1] - p1[1]) / dt
        
        return (vx, vy)
    
    def _cluster_tracks(
        self,
        positions: Dict[int, Tuple[float, float]]
    ) -> List[Set[int]]:
        """
        Cluster tracks using simple distance based method.
        
        Similar to DBSCAN but simpler implementation.
        """
        if len(positions) < self.min_group_size:
            return []
        
        track_ids = list(positions.keys())
        visited = set()
        clusters = []
        
        for seed_id in track_ids:
            if seed_id in visited:
                continue
            
            # find neighbors
            cluster = {seed_id}
            queue = [seed_id]
            
            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)
                
                current_pos = positions[current_id]
                
                # find all neighbors within distance
                for other_id in track_ids:
                    if other_id in cluster:
                        continue
                    
                    other_pos = positions[other_id]
                    dist = self._calculate_distance(current_pos, other_pos)
                    
                    if dist <= self.cluster_distance:
                        cluster.add(other_id)
                        if other_id not in visited:
                            queue.append(other_id)
            
            if len(cluster) >= self.min_group_size:
                clusters.append(cluster)
        
        return clusters
    
    def _analyze_formation(
        self,
        member_positions: List[Tuple[float, float]]
    ) -> str:
        """
        Analyze formation pattern of group.
        
        Returns formation type.
        """
        if len(member_positions) < 3:
            return "cluster"
        
        # convert to numpy
        points = np.array(member_positions)
        
        # calculate center
        center = np.mean(points, axis=0)
        
        # center the points
        centered = points - center
        
        # try to fit line (check if points are roughly collinear)
        if len(points) >= 3:
            # use PCA to find main axis
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # ratio of eigenvalues tells us about shape
            ratio = eigenvalues[0] / (eigenvalues[1] + 1e-6)
            
            if ratio < 0.1:
                return "line"
        
        # check for V shape
        if len(points) >= 5:
            # sort by x coordinate
            sorted_idx = np.argsort(points[:, 0])
            sorted_points = points[sorted_idx]
            
            # check if y values form V pattern
            mid = len(sorted_points) // 2
            left_slope = (sorted_points[mid, 1] - sorted_points[0, 1]) / \
                        (sorted_points[mid, 0] - sorted_points[0, 0] + 1e-6)
            right_slope = (sorted_points[-1, 1] - sorted_points[mid, 1]) / \
                         (sorted_points[-1, 0] - sorted_points[mid, 0] + 1e-6)
            
            if left_slope * right_slope < 0:  # opposite signs = V shape
                return "v_shape"
        
        # check for grid pattern
        if len(points) >= 4:
            # check if distances form regular pattern
            distances = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    distances.append(self._calculate_distance(
                        tuple(points[i]), tuple(points[j])
                    ))
            
            # if distances have low variance, might be grid
            std = np.std(distances)
            mean = np.mean(distances)
            if std / mean < 0.3:
                return "grid"
        
        return "cluster"
    
    def detect(
        self,
        current_positions: Dict[int, Tuple[float, float]],
        timestamp: float = 0.0
    ) -> List[SwarmGroup]:
        """
        Detect swarms from current track positions.
        
        Args:
            current_positions: {track_id: (x, y)} for drones only
            timestamp: current time
        
        Returns:
            List of detected swarm groups
        """
        # update history
        self.update_tracks(current_positions, timestamp)
        
        # filter to only drone class if needed
        # (assume all passed positions are drones)
        
        # cluster tracks
        clusters = self._cluster_tracks(current_positions)
        
        groups = []
        
        for cluster in clusters:
            member_ids = list(cluster)
            
            # get positions
            member_positions = [current_positions[mid] for mid in member_ids]
            
            # calculate center
            center_x = sum(p[0] for p in member_positions) / len(member_positions)
            center_y = sum(p[1] for p in member_positions) / len(member_positions)
            center = (center_x, center_y)
            
            # calculate radius (max distance from center)
            radius = max(
                self._calculate_distance(center, pos)
                for pos in member_positions
            )
            
            # analyze formation
            formation = self._analyze_formation(member_positions)
            
            # calculate group velocity (average of members)
            velocities = []
            for mid in member_ids:
                vel = self._get_velocity(mid)
                if vel:
                    velocities.append(vel)
            
            if velocities:
                avg_vx = sum(v[0] for v in velocities) / len(velocities)
                avg_vy = sum(v[1] for v in velocities) / len(velocities)
                group_velocity = (avg_vx, avg_vy)
            else:
                group_velocity = (0.0, 0.0)
            
            # calculate confidence based on group size and formation clarity
            confidence = min(1.0, len(member_ids) / 10.0)
            if formation != "cluster":
                confidence *= 1.2  # boost for clear formation
            confidence = min(1.0, confidence)
            
            group = SwarmGroup(
                group_id=self.next_group_id,
                member_ids=member_ids,
                center=center,
                radius=radius,
                formation=formation,
                velocity=group_velocity,
                confidence=confidence
            )
            
            groups.append(group)
            self.next_group_id += 1
        
        # save to history
        self.group_history.append(groups)
        if len(self.group_history) > 100:
            self.group_history.pop(0)
        
        return groups
    
    def draw_swarms(
        self,
        frame: np.ndarray,
        groups: List[SwarmGroup],
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Draw swarm visualizations on frame.
        """
        import cv2
        
        for group in groups:
            center = (int(group.center[0]), int(group.center[1]))
            radius = int(group.radius)
            
            # draw circle around group
            cv2.circle(frame, center, radius, color, 2)
            
            # draw formation label
            label = f"SWARM: {len(group.member_ids)} ({group.formation})"
            cv2.putText(
                frame, label,
                (center[0] - 50, center[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            
            # draw velocity vector
            if abs(group.velocity[0]) > 0.1 or abs(group.velocity[1]) > 0.1:
                end_x = int(center[0] + group.velocity[0] * 10)
                end_y = int(center[1] + group.velocity[1] * 10)
                cv2.arrowedLine(frame, center, (end_x, end_y), color, 2)
        
        return frame
    
    def get_threat_level(self, groups: List[SwarmGroup]) -> str:
        """
        Assess overall swarm threat level.
        
        Returns: 'none', 'low', 'medium', 'high', 'critical'
        """
        if not groups:
            return "none"
        
        total_drones = sum(len(g.member_ids) for g in groups)
        max_group_size = max(len(g.member_ids) for g in groups)
        
        # coordinated formations are more threatening
        coordinated = any(g.formation != "cluster" for g in groups)
        
        if max_group_size >= 10 or (max_group_size >= 5 and coordinated):
            return "critical"
        elif max_group_size >= 5 or total_drones >= 10:
            return "high"
        elif max_group_size >= 3 or total_drones >= 5:
            return "medium"
        else:
            return "low"


# test
if __name__ == "__main__":
    detector = SwarmDetector(min_group_size=3, cluster_distance=50)
    
    # simulate swarm positions
    positions = {
        1: (100, 100),
        2: (120, 110),
        3: (140, 105),  # group 1
        4: (130, 120),
        5: (110, 115),
        10: (400, 400),  # lone drone
    }
    
    groups = detector.detect(positions, timestamp=1.0)
    
    print(f"Detected {len(groups)} swarm(s)")
    for group in groups:
        print(f"  Group {group.group_id}: {len(group.member_ids)} members")
        print(f"    Formation: {group.formation}")
        print(f"    Center: {group.center}")
    
    threat = detector.get_threat_level(groups)
    print(f"Threat level: {threat}")
