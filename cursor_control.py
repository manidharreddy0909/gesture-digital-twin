"""
Cursor control module for mapping normalized hand positions to the Windows
cursor with per-hand exponential smoothing and optional predictive motion.

This module is intentionally independent from MediaPipe and only operates
on normalized coordinates (0.0–1.0), making it reusable for other sensors.
"""

from __future__ import annotations

import sys
import time
import ctypes
from dataclasses import dataclass, field
from typing import Dict, Tuple


def _is_windows() -> bool:
    """Return True if the current platform is Windows."""
    return sys.platform.startswith("win")


def _get_screen_size() -> Tuple[int, int]:
    """
    Query the current screen resolution.

    On Windows, we use the Win32 API. Other platforms fall back to a default
    value so the rest of the pipeline can still run for research purposes.
    """
    if _is_windows():
        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            # Some environments may not allow DPI calls; resolution is still usable.
            pass
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return 1920, 1080


def _set_cursor_pos(x: int, y: int) -> None:
    """Move the OS cursor if running on Windows; otherwise this is a no-op."""
    if _is_windows():
        ctypes.windll.user32.SetCursorPos(int(x), int(y))


@dataclass
class CursorState:
    """
    Internal per-hand cursor state.

    Attributes:
        smoothed_x / smoothed_y: Smoothed screen-space coordinates.
        last_raw_x / last_raw_y: Last un-smoothed target coordinates.
        last_timestamp: Time (seconds) of last update, for velocity estimates.
    """

    smoothed_x: float
    smoothed_y: float
    last_raw_x: float
    last_raw_y: float
    last_timestamp: float


@dataclass
class CursorController:
    """
    Map normalized fingertip coordinates into OS cursor movement.

    Features:
        - Per-hand exponential smoothing to reduce jitter.
        - Optional basic predictive motion using last-frame velocity.
        - Clean separation from hand tracking and gesture logic.
    """

    smoothing_factor: float = 0.25
    predictive_factor: float = 0.0
    enable_cursor_control: bool = True

    _screen_width: int = field(init=False)
    _screen_height: int = field(init=False)
    _states: Dict[int, CursorState] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._screen_width, self._screen_height = _get_screen_size()

    def _apply_prediction(
        self,
        hand_id: int,
        target_x: float,
        target_y: float,
        timestamp: float,
    ) -> Tuple[float, float]:
        """
        Optionally adjust the target position based on a simple velocity
        estimate from the last two frames.
        """
        if self.predictive_factor <= 0.0:
            return target_x, target_y

        state = self._states.get(hand_id)
        if state is None:
            return target_x, target_y

        dt = timestamp - state.last_timestamp
        if dt <= 1e-5:
            return target_x, target_y

        vx = (target_x - state.last_raw_x) / dt
        vy = (target_y - state.last_raw_y) / dt

        target_x += self.predictive_factor * vx
        target_y += self.predictive_factor * vy

        return target_x, target_y

    def update_cursor(
        self,
        hand_id: int,
        x_norm: float,
        y_norm: float,
        timestamp: float,
        control_this_hand: bool,
    ) -> Tuple[int, int]:
        """
        Update the cursor state for a given hand.

        Args:
            hand_id: Index of the hand (0, 1, ...).
            x_norm, y_norm: Normalized fingertip coordinates in [0, 1].
            timestamp: Current time in seconds.
            control_this_hand: If True and OS control is enabled, move
                the system cursor using this hand.

        Returns:
            Integer (x, y) screen coordinates for visualization.
        """
        # Map from normalized to raw screen target.
        target_x = x_norm * (self._screen_width - 1)
        target_y = y_norm * (self._screen_height - 1)

        # Optional prediction step using last-frame velocity.
        target_x, target_y = self._apply_prediction(
            hand_id, target_x, target_y, timestamp
        )

        # Initialize or update exponential smoothing state.
        state = self._states.get(hand_id)
        if state is None:
            smoothed_x = target_x
            smoothed_y = target_y
            self._states[hand_id] = CursorState(
                smoothed_x=smoothed_x,
                smoothed_y=smoothed_y,
                last_raw_x=target_x,
                last_raw_y=target_y,
                last_timestamp=timestamp,
            )
        else:
            alpha = self.smoothing_factor
            smoothed_x = (1.0 - alpha) * state.smoothed_x + alpha * target_x
            smoothed_y = (1.0 - alpha) * state.smoothed_y + alpha * target_y
            state.smoothed_x = smoothed_x
            state.smoothed_y = smoothed_y
            state.last_raw_x = target_x
            state.last_raw_y = target_y
            state.last_timestamp = timestamp

        screen_x = int(smoothed_x)
        screen_y = int(smoothed_y)

        if self.enable_cursor_control and control_this_hand:
            _set_cursor_pos(screen_x, screen_y)

        return screen_x, screen_y

