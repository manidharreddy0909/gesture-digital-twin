"""
Advanced cursor control module for mapping normalized fingertip positions to the OS
cursor with multiple filtering and prediction strategies.

Features:
- Exponential smoothing (simple, fast baseline)
- Kalman filter (constant-velocity motion model with adaptive noise)
- Enhanced Kalman filter with acceleration model (constant-acceleration)
- Velocity-based prediction and adaptive smoothing
- Speed-dependent response tuning
- Per-hand independent filtering
- Predictive path visualization support

Designed to be:
- Hand-agnostic: supports one or two controlling hands.
- Independent of MediaPipe: works purely with normalized coordinates.
- Extensible: supports pluggable filter implementations.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np


def _is_windows() -> bool:
    """Return True if running on Windows."""
    return sys.platform.startswith("win")


def _get_screen_size() -> Tuple[int, int]:
    """
    Query screen resolution.

    On Windows, use the Win32 API; otherwise fall back to a default size
    so the rest of the pipeline can run without OS cursor control.
    """
    if _is_windows():
        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            # On some platforms this call may fail; screen size is still valid.
            pass
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return 1920, 1080


def _set_cursor_pos(x: int, y: int) -> None:
    """Move the OS cursor if on Windows."""
    if _is_windows():
        ctypes.windll.user32.SetCursorPos(int(x), int(y))


@dataclass
class CursorState:
    """
    Per-hand cursor state used for exponential smoothing.

    Attributes:
        smoothed_x / smoothed_y:
            Current smoothed cursor coordinates in screen space.
        last_target_x / last_target_y:
            Last raw (unsmoothed) target coordinates.
        last_timestamp:
            Time of the last update (seconds).
    """

    smoothed_x: float
    smoothed_y: float
    last_target_x: float
    last_target_y: float
    last_timestamp: float


@dataclass
class KalmanState:
    """
    Constant-velocity 2D Kalman filter state.

    The state vector is [x, y, vx, vy]^T in screen coordinates.
    """

    x: np.ndarray  # shape (4, 1)
    P: np.ndarray  # shape (4, 4)
    last_timestamp: float


@dataclass
class EnhancedKalmanState:
    """
    Enhanced constant-acceleration 2D Kalman filter state.

    The state vector is [x, y, vx, vy, ax, ay]^T in screen coordinates.
    This model captures both velocity and acceleration for smoother prediction.
    """

    x: np.ndarray  # shape (6, 1)
    P: np.ndarray  # shape (6, 6)
    last_timestamp: float


@dataclass
class CursorController:
    """
    Convert normalized fingertip coordinates into OS cursor positions.

    Features:
        - Exponential smoothing to reduce jitter.
        - Multiple Kalman filter options (CV-model, CA-model).
        - Velocity-based prediction for reduced lag.
        - Adaptive smoothing based on motion speed.
        - Per-hand control enabling dual-hand or single-hand modes.
        - Predictive path logging for visualization.
    """

    smoothing_factor: float = 0.25
    predictive_factor: float = 0.0  # 0 => no prediction, >0 => velocity amplification
    enable_cursor_control: bool = True
    use_kalman_filter: bool = False
    use_enhanced_kalman: bool = False  # Use const-accel instead of const-vel
    adaptive_smoothing: bool = False
    kalman_adaptive_noise: bool = True  # Adapt Kalman noise based on speed
    enable_prediction_logging: bool = False  # Log predicted cursor paths

    _screen_width: int = field(init=False)
    _screen_height: int = field(init=False)
    _states: Dict[int, CursorState] = field(default_factory=dict, init=False)
    _kalman_states: Dict[int, KalmanState] = field(default_factory=dict, init=False)
    _enhanced_kalman_states: Dict[int, EnhancedKalmanState] = field(default_factory=dict, init=False)
    _prediction_logs: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._screen_width, self._screen_height = _get_screen_size()

    # --- Prediction Logging -----

    def get_predicted_path(self, hand_id: int) -> List[Tuple[float, float]]:
        """Return the logged predicted path for a hand (for visualization)."""
        return self._prediction_logs.get(hand_id, [])

    def clear_prediction_logs(self) -> None:
        """Clear all prediction logs."""
        self._prediction_logs.clear()

    def _log_prediction(self, hand_id: int, x: float, y: float) -> None:
        """Log a predicted cursor position for visualization."""
        if not self.enable_prediction_logging:
            return
        if hand_id not in self._prediction_logs:
            self._prediction_logs[hand_id] = []
        self._prediction_logs[hand_id].append((x, y))
        # Keep only last 30 predictions
        if len(self._prediction_logs[hand_id]) > 30:
            self._prediction_logs[hand_id].pop(0)

    # --- Exponential Smoothing Methods ---------------------------------

    def _predict_target_simple(
        self,
        state: CursorState,
        target_x: float,
        target_y: float,
        timestamp: float,
    ) -> Tuple[float, float]:
        """
        Apply velocity-based prediction (exponential smoothing approach).

        Estimates velocity from last_target -> new_target and extrapolates
        forward by predictive_factor * dt.
        """
        if self.predictive_factor <= 0.0:
            return target_x, target_y

        dt = timestamp - state.last_timestamp
        if dt <= 1e-5:
            return target_x, target_y

        vx = (target_x - state.last_target_x) / dt
        vy = (target_y - state.last_target_y) / dt

        # Advance the target in the direction of motion.
        pred_x = target_x + self.predictive_factor * vx * dt
        pred_y = target_y + self.predictive_factor * vy * dt

        self._log_prediction(state.hand_id if hasattr(state, 'hand_id') else 0, pred_x, pred_y)
        return pred_x, pred_y

    # --- Constant-Velocity Kalman Filter ---------------------------------

    def _kalman_step(
        self,
        hand_id: int,
        meas_x: float,
        meas_y: float,
        timestamp: float,
    ) -> Tuple[float, float]:
        """
        Perform one Kalman filter predict+update step (constant-velocity model).

        State: [x, y, vx, vy]^T
        Measurement: [x_meas, y_meas]^T

        Adaptive noise covariance can adjust based on motion speed to handle
        fast vs. slow movements gracefully.
        """
        z = np.array([[meas_x], [meas_y]], dtype=float)

        # Initialize filter state if needed.
        k_state = self._kalman_states.get(hand_id)
        if k_state is None:
            x0 = np.array([[meas_x], [meas_y], [0.0], [0.0]], dtype=float)
            P0 = np.eye(4, dtype=float) * 1e3  # Large initial uncertainty.
            k_state = KalmanState(x=x0, P=P0, last_timestamp=timestamp)
            self._kalman_states[hand_id] = k_state
            self._log_prediction(hand_id, meas_x, meas_y)
            return meas_x, meas_y

        dt = max(timestamp - k_state.last_timestamp, 1e-4)

        # --- State transition (constant velocity) ---
        A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # --- Measurement matrix ---
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)

        # --- Process noise covariance ---
        q_pos = 1.0
        q_vel = 1.0
        Q = np.diag([q_pos * dt * dt, q_pos * dt * dt, q_vel * dt, q_vel * dt])

        # --- Measurement noise (adaptively scaled based on motion) ---
        R_base = 5.0  # pixels
        if self.kalman_adaptive_noise:
            vx, vy = float(k_state.x[2, 0]), float(k_state.x[3, 0])
            speed = (vx * vx + vy * vy) ** 0.5
            R_scale = 1.0 + min(speed / 500.0, 5.0)
        else:
            R_scale = 1.0
        R = np.eye(2, dtype=float) * (R_base * R_scale)

        # --- Predict ---
        x_pred = A @ k_state.x
        P_pred = A @ k_state.P @ A.T + Q

        # --- Update ---
        y = z - (H @ x_pred)  # Innovation
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_new = x_pred + K @ y
        P_new = (np.eye(4, dtype=float) - K @ H) @ P_pred

        k_state.x = x_new
        k_state.P = P_new
        k_state.last_timestamp = timestamp

        x_est = float(x_new[0, 0])
        y_est = float(x_new[1, 0])
        self._log_prediction(hand_id, x_est, y_est)
        return x_est, y_est

    # --- Enhanced Constant-Acceleration Kalman Filter -------------------

    def _enhanced_kalman_step(
        self,
        hand_id: int,
        meas_x: float,
        meas_y: float,
        timestamp: float,
    ) -> Tuple[float, float]:
        """
        Enhanced Kalman filter with constant-acceleration model.

        State: [x, y, vx, vy, ax, ay]^T
        Measurement: [x_meas, y_meas]^T

        This model better captures accelerated motion and provides smoother
        tracking during non-uniform hand movements.
        """
        z = np.array([[meas_x], [meas_y]], dtype=float)

        # Initialize if needed.
        e_state = self._enhanced_kalman_states.get(hand_id)
        if e_state is None:
            x0 = np.array([[meas_x], [meas_y], [0.0], [0.0], [0.0], [0.0]], dtype=float)
            P0 = np.eye(6, dtype=float) * 1e3
            e_state = EnhancedKalmanState(x=x0, P=P0, last_timestamp=timestamp)
            self._enhanced_kalman_states[hand_id] = e_state
            self._log_prediction(hand_id, meas_x, meas_y)
            return meas_x, meas_y

        dt = max(timestamp - e_state.last_timestamp, 1e-4)

        # --- State transition (constant acceleration) ---
        A = np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt * dt, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt * dt],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # --- Measurement matrix (observe only position) ---
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        # --- Process noise ---
        q_pos = 0.5
        q_vel = 0.1
        q_accel = 0.01
        Q = np.diag(
            [
                q_pos * dt * dt,
                q_pos * dt * dt,
                q_vel * dt,
                q_vel * dt,
                q_accel * dt,
                q_accel * dt,
            ]
        )

        # --- Measurement noise (adaptive) ---
        R_base = 5.0
        if self.kalman_adaptive_noise:
            vx, vy = float(e_state.x[2, 0]), float(e_state.x[3, 0])
            speed = (vx * vx + vy * vy) ** 0.5
            R_scale = 1.0 + min(speed / 500.0, 3.0)
        else:
            R_scale = 1.0
        R = np.eye(2, dtype=float) * (R_base * R_scale)

        # --- Predict ---
        x_pred = A @ e_state.x
        P_pred = A @ e_state.P @ A.T + Q

        # --- Update ---
        y = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((6, 2), dtype=float)
        x_new = x_pred + K @ y
        P_new = (np.eye(6, dtype=float) - K @ H) @ P_pred

        e_state.x = x_new
        e_state.P = P_new
        e_state.last_timestamp = timestamp

        x_est = float(x_new[0, 0])
        y_est = float(x_new[1, 0])
        self._log_prediction(hand_id, x_est, y_est)
        return x_est, y_est

    # --- Public API -----------------------------------------------------------

    def update_cursor(
        self,
        hand_id: int,
        x_norm: float,
        y_norm: float,
        timestamp: float,
        control_this_hand: bool,
    ) -> Tuple[int, int]:
        """
        Update cursor for a given hand.

        Args:
            hand_id: Index of the hand.
            x_norm, y_norm: Normalized fingertip coordinates in [0, 1].
            timestamp: Current time in seconds.
            control_this_hand: If True and OS control is enabled, this hand
                is allowed to move the system cursor.

        Returns:
            (x, y) integer cursor coordinates in screen space.
        """
        # Map normalized coordinates into screen-space target coordinates.
        base_x = x_norm * (self._screen_width - 1)
        base_y = y_norm * (self._screen_height - 1)

        state = self._states.get(hand_id)
        if state is None:
            # Initialize state on first observation.
            sm_x = base_x
            sm_y = base_y
            self._states[hand_id] = CursorState(
                smoothed_x=sm_x,
                smoothed_y=sm_y,
                last_target_x=base_x,
                last_target_y=base_y,
                last_timestamp=timestamp,
            )
        else:
            if self.use_enhanced_kalman:
                # Use enhanced constant-acceleration Kalman filter.
                kx, ky = self._enhanced_kalman_step(hand_id, base_x, base_y, timestamp)
                sm_x, sm_y = kx, ky
            elif self.use_kalman_filter:
                # Use constant-velocity Kalman filter.
                kx, ky = self._kalman_step(hand_id, base_x, base_y, timestamp)
                sm_x, sm_y = kx, ky
            else:
                # Simple exponential smoothing + optional velocity prediction.
                target_x, target_y = self._predict_target_simple(
                    state, base_x, base_y, timestamp
                )

                # Optionally adapt smoothing factor based on speed.
                alpha = self.smoothing_factor
                if self.adaptive_smoothing:
                    dt = max(timestamp - state.last_timestamp, 1e-4)
                    vx = (target_x - state.last_target_x) / dt
                    vy = (target_y - state.last_target_y) / dt
                    speed = (vx * vx + vy * vy) ** 0.5
                    # Faster motion -> lower smoothing (more responsive).
                    alpha = min(max(self.smoothing_factor * (1.0 + speed / 500.0), 0.05), 0.9)

                sm_x = (1.0 - alpha) * state.smoothed_x + alpha * target_x
                sm_y = (1.0 - alpha) * state.smoothed_y + alpha * target_y

                state.smoothed_x = sm_x
                state.smoothed_y = sm_y
                state.last_target_x = target_x
                state.last_target_y = target_y
                state.last_timestamp = timestamp

        ix, iy = int(sm_x), int(sm_y)

        if self.enable_cursor_control and control_this_hand:
            _set_cursor_pos(ix, iy)

        return ix, iy
