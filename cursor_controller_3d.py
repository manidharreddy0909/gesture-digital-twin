"""
3D CURSOR CONTROLLER MODULE

Extends 2D cursor filtering to 3D world coordinates with advanced Kalman filtering.

Purpose: Apply motion filtering and prediction to 3D hand coordinates for smooth
cursor/pointer movement in 3D space.

Key Classes:
- KalmanState3D: 3D Constant-Velocity filter (state: [x,y,z,vx,vy,vz])
- EnhancedKalmanState3D: 3D Constant-Acceleration filter (state: [x,y,z,vx,vy,vz,ax,ay,az])
- CursorController3D: Main controller, backward compatible with 2D version

Features:
- Full 9D state space for acceleration model
- Adaptive noise scaling based on motion magnitude
- Independent filter instances per hand
- Fallback to exponential smoothing for 2D mode
- Prediction logging for trajectory visualization
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# 3D KALMAN FILTER STATES
# ============================================================================

@dataclass
class KalmanState3D:
    """
    Constant-Velocity Kalman filter state for 3D motion.

    State vector: [x, y, z, vx, vy, vz] (6D)

    Assumes constant velocity model:
      x(t+1) = x(t) + vx(t) * dt
      y(t+1) = y(t) + vy(t) * dt
      z(t+1) = z(t) + vz(t) * dt
      vx(t+1) = vx(t)
      vy(t+1) = vy(t)
      vz(t+1) = vz(t)
    """
    x: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))  # State vector [x,y,z,vx,vy,vz]
    P: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float32) * 1000)  # Covariance
    dt: float = 0.033  # Time step (seconds, ~30fps)
    q_pos: float = 1.0  # Process noise (position)
    q_vel: float = 0.1  # Process noise (velocity)
    r: float = 1.0     # Measurement noise

    def __post_init__(self):
        """Initialize matrices."""
        self._build_matrices()

    def _build_matrices(self):
        """Build A (transition) and R (measurement noise) matrices."""
        # Transition matrix A (6x6)
        self.A = np.eye(6, dtype=np.float32)
        self.A[0, 3] = self.dt  # x += vx * dt
        self.A[1, 4] = self.dt  # y += vy * dt
        self.A[2, 5] = self.dt  # z += vz * dt

        # Process noise Q (6x6)
        self.Q = np.zeros((6, 6), dtype=np.float32)
        self.Q[0, 0] = self.q_pos  # Position noise
        self.Q[1, 1] = self.q_pos
        self.Q[2, 2] = self.q_pos
        self.Q[3, 3] = self.q_vel  # Velocity noise
        self.Q[4, 4] = self.q_vel
        self.Q[5, 5] = self.q_vel

        # Measurement matrix H (3x6): we measure [x, y, z] only
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Measurement noise R (3x3)
        self.R = np.eye(3, dtype=np.float32) * self.r

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:3].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with measurement.

        Args:
            measurement: [x, y, z] observed coordinates

        Returns:
            Updated position [x, y, z]
        """
        # Innovation
        z = measurement.astype(np.float32)
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:3].copy()


@dataclass
class EnhancedKalmanState3D:
    """
    Constant-Acceleration Kalman filter for 3D motion.

    State vector: [x, y, z, vx, vy, vz, ax, ay, az] (9D)

    Assumes constant acceleration model:
      Same as 6D, plus acceleration terms that slowly decay
    """
    x: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))  # State [x,y,z,vx,vy,vz,ax,ay,az]
    P: np.ndarray = field(default_factory=lambda: np.eye(9, dtype=np.float32) * 1000)  # Covariance (9x9)
    dt: float = 0.033  # Time step
    q_pos: float = 0.1
    q_vel: float = 0.01
    q_acc: float = 0.001  # Acceleration noise (small, decays over time)
    r: float = 1.0     # Measurement noise

    def __post_init__(self):
        """Initialize matrices."""
        self._build_matrices()

    def _build_matrices(self):
        """Build transition and noise matrices."""
        dt = self.dt
        dt2 = dt * dt / 2.0

        # Transition matrix A (9x9)
        self.A = np.eye(9, dtype=np.float32)

        # Position: x += vx*dt + 0.5*ax*dt²
        self.A[0, 3] = dt
        self.A[0, 6] = dt2
        self.A[1, 4] = dt
        self.A[1, 7] = dt2
        self.A[2, 5] = dt
        self.A[2, 8] = dt2

        # Velocity: vx += ax*dt
        self.A[3, 6] = dt
        self.A[4, 7] = dt
        self.A[5, 8] = dt

        # Process noise Q (9x9)
        self.Q = np.zeros((9, 9), dtype=np.float32)
        self.Q[0, 0] = self.q_pos  # Position
        self.Q[1, 1] = self.q_pos
        self.Q[2, 2] = self.q_pos
        self.Q[3, 3] = self.q_vel  # Velocity
        self.Q[4, 4] = self.q_vel
        self.Q[5, 5] = self.q_vel
        self.Q[6, 6] = self.q_acc  # Acceleration
        self.Q[7, 7] = self.q_acc
        self.Q[8, 8] = self.q_acc

        # Measurement matrix H (3x9): measure [x, y, z] only
        self.H = np.zeros((3, 9), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Measurement noise R
        self.R = np.eye(3, dtype=np.float32) * self.r

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:3].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with measurement."""
        z = measurement.astype(np.float32)
        y = z - (self.H @ self.x)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(9, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Get velocity estimate [vx, vy, vz]."""
        return self.x[3:6].copy()

    def get_acceleration(self) -> np.ndarray:
        """Get acceleration estimate [ax, ay, az]."""
        return self.x[6:9].copy()


# ============================================================================
# 3D CURSOR CONTROLLER
# ============================================================================

class CursorController3D:
    """
    Advanced 3D cursor controller with Kalman filtering.

    Maintains separate filter instances for each hand.
    Supports both 2D (backward compatible) and 3D filtering.

    Features:
    - Constant-velocity and constant-acceleration Kalman filters
    - Adaptive noise scaling based on motion magnitude
    - Optional prediction logging
    - Exponential smoothing fallback
    """

    def __init__(self,
                 smoothing_factor: float = 0.25,
                 predictive_factor: float = 0.0,
                 enable_cursor_control: bool = True,
                 use_kalman_filter: bool = False,
                 use_enhanced_kalman: bool = False,
                 adaptive_smoothing: bool = False,
                 kalman_adaptive_noise: bool = True,
                 enable_prediction_logging: bool = False):
        """
        Initialize controller.

        Args:
            smoothing_factor: Exponential smoothing alpha [0, 1]
            predictive_factor: Velocity prediction multiplier
            enable_cursor_control: Enable OS cursor movement
            use_kalman_filter: Use Kalman CV model
            use_enhanced_kalman: Use Kalman CA model (9D) - overrides use_kalman_filter
            adaptive_smoothing: Speed-dependent smoothing
            kalman_adaptive_noise: Speed-dependent measurement noise
            enable_prediction_logging: Log predicted paths
        """
        self.smoothing_factor = smoothing_factor
        self.predictive_factor = predictive_factor
        self.enable_cursor_control = enable_cursor_control
        self.use_kalman = use_kalman_filter or use_enhanced_kalman
        self.use_enhanced_kalman = use_enhanced_kalman
        self.adaptive_smoothing = adaptive_smoothing
        self.kalman_adaptive_noise = kalman_adaptive_noise
        self.enable_prediction_logging = enable_prediction_logging

        # Kalman filters per hand
        self.kalman_states: Dict[int, EnhancedKalmanState3D if use_enhanced_kalman else KalmanState3D] = {}

        # Exponential smoothing state
        self.last_pos: Dict[int, np.ndarray] = {}
        self.last_velocity: Dict[int, np.ndarray] = {}

        # Prediction history
        self.prediction_logs: Dict[int, List[Tuple[float, float, float]]] = {}

        # Frame tracking
        self.last_frame_time: Dict[int, float] = {}

    def update_cursor_3d(self,
                         hand_id: int,
                         x_norm: float, y_norm: float, z_norm: float,
                         timestamp: float,
                         control_this_hand: bool = True) -> Tuple[float, float, float]:
        """
        Update 3D cursor position with filtering.

        Args:
            hand_id: Hand identifier (0=left, 1=right)
            x_norm: Normalized X [0, 1]
            y_norm: Normalized Y [0, 1]
            z_norm: Normalized Z [0, 1]
            timestamp: Current timestamp (seconds)
            control_this_hand: Whether to apply filtering

        Returns:
            Filtered 3D position (x, y, z)
        """
        measurement = np.array([x_norm, y_norm, z_norm], dtype=np.float32)

        # Initialize filter for new hand
        if hand_id not in self.kalman_states:
            if self.use_enhanced_kalman:
                self.kalman_states[hand_id] = EnhancedKalmanState3D()
                self.kalman_states[hand_id].x[0:3] = measurement
            elif self.use_kalman:
                self.kalman_states[hand_id] = KalmanState3D()
                self.kalman_states[hand_id].x[0:3] = measurement
            else:
                self.last_pos[hand_id] = measurement
                self.last_velocity[hand_id] = np.zeros(3, dtype=np.float32)

            self.last_frame_time[hand_id] = timestamp

            if self.enable_prediction_logging:
                self.prediction_logs[hand_id] = []

            return tuple(measurement)

        # Update time step
        dt = timestamp - self.last_frame_time.get(hand_id, timestamp)
        dt = max(0.001, min(0.1, dt))  # Clamp to reasonable range
        self.last_frame_time[hand_id] = timestamp

        if not control_this_hand:
            return tuple(measurement)

        # Apply filtering based on configuration
        if self.use_kalman or self.use_enhanced_kalman:
            return self._update_kalman_3d(hand_id, measurement, dt)
        else:
            return self._update_exponential_3d(hand_id, measurement, dt)

    def _update_kalman_3d(self,
                          hand_id: int,
                          measurement: np.ndarray,
                          dt: float) -> Tuple[float, float, float]:
        """Apply Kalman filter update."""
        state = self.kalman_states[hand_id]

        # Adaptive noise based on motion magnitude
        if self.kalman_adaptive_noise:
            speed = np.linalg.norm(measurement - state.x[:3])
            noise_scale = 1.0 + min(speed * 10, 5.0)
            state.R = np.eye(3, dtype=np.float32) * state.r * noise_scale

        # Update time step
        state.dt = dt
        state._build_matrices()

        # Kalman predict
        state.predict()

        # Kalman update
        filtered_pos = state.update(measurement)

        # Optional prediction logging
        if self.enable_prediction_logging:
            self.prediction_logs[hand_id].append(tuple(filtered_pos))

        return tuple(filtered_pos)

    def _update_exponential_3d(self,
                               hand_id: int,
                               measurement: np.ndarray,
                               dt: float) -> Tuple[float, float, float]:
        """Apply exponential smoothing."""
        alpha = self.smoothing_factor

        # Adaptive smoothing based on speed
        if self.adaptive_smoothing and hand_id in self.last_pos:
            speed = np.linalg.norm(measurement - self.last_pos[hand_id])
            # Higher speed → lower alpha (less smoothing for quick movements)
            alpha = self.smoothing_factor * max(0.5, 1.0 - speed)

        # Exponential smoothing
        if hand_id in self.last_pos:
            smoothed = (1 - alpha) * self.last_pos[hand_id] + alpha * measurement
        else:
            smoothed = measurement

        # Velocity for optional prediction
        if hand_id in self.last_pos:
            velocity = (smoothed - self.last_pos[hand_id]) / max(dt, 0.001)
            self.last_velocity[hand_id] = velocity
        else:
            velocity = np.zeros(3, dtype=np.float32)

        # Apply prediction
        if self.predictive_factor > 0:
            smoothed = smoothed + velocity * self.predictive_factor * dt

        self.last_pos[hand_id] = smoothed

        if self.enable_prediction_logging:
            self.prediction_logs[hand_id].append(tuple(smoothed))

        return tuple(smoothed)

    def get_predicted_path(self, hand_id: int) -> List[Tuple[float, float, float]]:
        """Get logged prediction path for visualization."""
        return self.prediction_logs.get(hand_id, [])

    def clear_prediction_logs(self) -> None:
        """Clear all prediction logs."""
        for hand_id in self.prediction_logs:
            self.prediction_logs[hand_id] = []

    def get_velocity(self, hand_id: int) -> Optional[np.ndarray]:
        """Get current velocity estimate."""
        if hand_id in self.kalman_states and self.use_enhanced_kalman:
            return self.kalman_states[hand_id].get_velocity()
        elif hand_id in self.last_velocity:
            return self.last_velocity[hand_id]
        return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_3d_filtering():
    """Example: 3D Kalman filtering of hand motion."""
    print("\n" + "="*70)
    print("EXAMPLE: 3D CURSOR FILTERING")
    print("="*70)

    # Create controller with enhanced Kalman
    controller = CursorController3D(
        use_enhanced_kalman=True,
        adaptive_smoothing=True,
        kalman_adaptive_noise=True,
        enable_prediction_logging=True
    )

    # Simulate hand trajectory (circular motion in 3D)
    print("\nSimulating circular hand motion in 3D...")
    import math
    for frame in range(120):
        t = frame * 0.033  # 30 fps
        # Circular motion: x,y in circle, z oscillating
        x = 0.5 + 0.2 * math.cos(2 * math.pi * frame / 60)
        y = 0.5 + 0.2 * math.sin(2 * math.pi * frame / 60)
        z = 0.5 + 0.1 * math.sin(4 * math.pi * frame / 60)

        filtered = controller.update_cursor_3d(0, x, y, z, t)

        if frame % 20 == 0:
            print(f"Frame {frame:3d}: Input=({x:.3f}, {y:.3f}, {z:.3f}) "
                  f"→ Filtered=({filtered[0]:.3f}, {filtered[1]:.3f}, {filtered[2]:.3f})")

    print(f"\nLogged {len(controller.get_predicted_path(0))} prediction points")


if __name__ == "__main__":
    example_3d_filtering()
