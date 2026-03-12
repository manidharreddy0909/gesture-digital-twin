"""
Motion analysis module for dual-hand tracking.

Responsibilities:
- Maintain short motion histories for selected fingertips.
- Compute per-frame velocity, speed, and simple trajectory metrics.
- Expose data structures that gesture detection and visualization can use
  for dynamic gestures (swipes, circles, drags) and motion trails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FingerTrajectory:
    """
    Motion history for a single fingertip in normalized coordinates.

    Attributes:
        points: List of (x, y, t) triples in normalized space.
        velocities: List of (vx, vy) between consecutive points.
        speeds: Magnitude of velocities, same length as velocities.
    """

    points: List[Tuple[float, float, float]] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)

    def add_point(self, x: float, y: float, t: float, max_len: int) -> None:
        """
        Append a new sample and update velocity/speed. Truncates history
        to at most `max_len` samples.
        """
        if self.points:
            x0, y0, t0 = self.points[-1]
            dt = t - t0
            if dt > 1e-5:
                vx = (x - x0) / dt
                vy = (y - y0) / dt
                speed = math.sqrt(vx * vx + vy * vy)
                self.velocities.append((vx, vy))
                self.speeds.append(speed)
            else:
                self.velocities.append((0.0, 0.0))
                self.speeds.append(0.0)

        self.points.append((x, y, t))

        # Truncate lists to keep them in sync.
        if len(self.points) > max_len:
            excess = len(self.points) - max_len
            self.points = self.points[excess:]
            if self.velocities:
                self.velocities = self.velocities[excess:]
                self.speeds = self.speeds[excess:]

    def net_displacement(self) -> float:
        """Euclidean distance between first and last point."""
        if len(self.points) < 2:
            return 0.0
        x0, y0, _ = self.points[0]
        x1, y1, _ = self.points[-1]
        dx = x1 - x0
        dy = y1 - y0
        return math.sqrt(dx * dx + dy * dy)

    def path_length(self) -> float:
        """Total length of the polyline across all samples."""
        if len(self.points) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(self.points)):
            x0, y0, _ = self.points[i - 1]
            x1, y1, _ = self.points[i]
            dx = x1 - x0
            dy = y1 - y0
            length += math.sqrt(dx * dx + dy * dy)
        return length

    def average_speed(self) -> float:
        """Mean speed across history."""
        if not self.speeds:
            return 0.0
        return sum(self.speeds) / len(self.speeds)


@dataclass
class HandMotionInfo:
    """
    Aggregated motion information for a single hand.

    Attributes:
        hand_id: Index of the hand in the current frame.
        fingertip_trajectories: Mapping from landmark index (tip indices
            like 4, 8, 12, 16, 20) to its FingerTrajectory.
    """

    hand_id: int
    fingertip_trajectories: Dict[int, FingerTrajectory] = field(
        default_factory=dict
    )


class MotionAnalyzer:
    """
    Maintain and update motion histories for fingertips of each hand.

    Typical usage each frame:
        for hand in hands:
            motion_info = analyzer.update_from_landmarks(hand.hand_id, hand.landmarks, now)
    """

    def __init__(
        self,
        history_size: int = 20,
        tracked_tip_indices: Tuple[int, ...] = (4, 8, 12, 16, 20),
    ):
        self.history_size = history_size
        self.tracked_tip_indices = tracked_tip_indices
        # hand_id -> (tip_index -> FingerTrajectory)
        self._trajectories: Dict[int, Dict[int, FingerTrajectory]] = {}

    def update_from_landmarks(
        self,
        hand_id: int,
        landmarks: List,
        timestamp: float,
    ) -> HandMotionInfo:
        """
        Update motion histories for the given hand using its landmarks.

        Args:
            hand_id: Index of the hand in the frame.
            landmarks: Normalized landmarks (0..20).
            timestamp: Current time in seconds.

        Returns:
            HandMotionInfo containing trajectories for tracked fingertips.
        """
        hand_traj = self._trajectories.setdefault(hand_id, {})

        for tip_index in self.tracked_tip_indices:
            lm = landmarks[tip_index]
            x = float(lm.x)
            y = float(lm.y)

            traj = hand_traj.get(tip_index)
            if traj is None:
                traj = FingerTrajectory()
                hand_traj[tip_index] = traj

            traj.add_point(x, y, timestamp, self.history_size)

        return HandMotionInfo(hand_id=hand_id, fingertip_trajectories=hand_traj)


