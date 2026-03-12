"""
Control intelligence layer for robust gesture-driven digital twin control.

Adds:
- Dual mode architecture (real interface vs virtual execution)
- Predictive motion smoothing with latency compensation
- State/context-based gesture interpretation
- Command stability gating (rate limiting / debouncing)
- Lightweight feedback loop events
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional

import numpy as np


class ControlMode(Enum):
    """High-level runtime mode."""

    REAL_INTERFACE = "real_interface"
    VIRTUAL_EXECUTION = "virtual_execution"


@dataclass
class FeedbackEventLite:
    """Small feedback event payload for console/UI callbacks."""

    event: str
    severity: str
    message: str
    data: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class _MotionState:
    """Per-hand motion state for smoothing and prediction."""

    smoothed: np.ndarray
    velocity: np.ndarray
    timestamp: float


class MotionIntelligence:
    """Predictive smoothing with latency compensation."""

    def __init__(
        self,
        smoothing_alpha: float = 0.35,
        deadzone: float = 0.002,
        base_latency_ms: float = 35.0,
        enable_latency_compensation: bool = True,
    ):
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.01, 0.95))
        self.deadzone = max(0.0, deadzone)
        self.base_latency_ms = max(0.0, base_latency_ms)
        self.enable_latency_compensation = enable_latency_compensation
        self.states: Dict[int, _MotionState] = {}
        self.measured_latency_ms: Deque[float] = deque(maxlen=120)

    def add_measured_latency(self, latency_ms: float) -> None:
        """Update observed runtime latency estimate."""
        if latency_ms > 0:
            self.measured_latency_ms.append(float(latency_ms))

    def _effective_latency_ms(self) -> float:
        if not self.measured_latency_ms:
            return self.base_latency_ms
        return 0.5 * self.base_latency_ms + 0.5 * float(np.mean(self.measured_latency_ms))

    def update(self, hand_id: int, raw_pos: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Smooth and predict hand position in normalized 3D coordinates [0,1].
        """
        raw = np.asarray(raw_pos, dtype=np.float64)
        if hand_id not in self.states:
            self.states[hand_id] = _MotionState(
                smoothed=raw.copy(),
                velocity=np.zeros(3, dtype=np.float64),
                timestamp=timestamp,
            )
            return np.clip(raw, 0.0, 1.0)

        state = self.states[hand_id]
        dt = max(1e-3, timestamp - state.timestamp)

        # Exponential smoothing.
        smoothed = self.smoothing_alpha * raw + (1.0 - self.smoothing_alpha) * state.smoothed

        # Deadzone to avoid micro-jitter when near-stationary.
        if np.linalg.norm(smoothed - state.smoothed) < self.deadzone:
            smoothed = state.smoothed.copy()

        velocity = (smoothed - state.smoothed) / dt
        velocity = np.clip(velocity, -3.0, 3.0)  # normalized units/sec

        predicted = smoothed.copy()
        if self.enable_latency_compensation:
            dt_pred = self._effective_latency_ms() / 1000.0
            predicted = smoothed + velocity * dt_pred

        state.smoothed = smoothed
        state.velocity = velocity
        state.timestamp = timestamp
        return np.clip(predicted, 0.0, 1.0)


class StatefulGestureInterpreter:
    """Gesture + mode + context -> action mapping."""

    def __init__(self):
        self._mode_maps: Dict[ControlMode, Dict[str, str]] = {
            ControlMode.REAL_INTERFACE: {
                "pinch": "ui_select",
                "open": "ui_release",
                "swipe_left": "ui_navigate_prev",
                "swipe_right": "ui_navigate_next",
                "circle": "ui_rotate_preview",
                "zoom_in": "ui_scale_preview_up",
                "zoom_out": "ui_scale_preview_down",
            },
            ControlMode.VIRTUAL_EXECUTION: {
                "pinch": "gripper_close",
                "open": "gripper_open",
                "swipe_left": "move_left",
                "swipe_right": "move_right",
                "swipe_up": "move_up",
                "swipe_down": "move_down",
                "circle": "rotate_object",
                "zoom_in": "scale_up",
                "zoom_out": "scale_down",
                "push": "move_forward",
                "pull": "move_backward",
            },
        }

        # Context overrides for task-specific behavior.
        self._context_overrides: Dict[str, Dict[str, str]] = {
            "robot": {
                "move_left": "robot_move_left",
                "move_right": "robot_move_right",
                "move_up": "robot_move_up",
                "move_down": "robot_move_down",
            },
            "objects": {
                "gripper_close": "object_grab",
                "gripper_open": "object_release",
            },
        }

    def resolve(
        self,
        single_gesture: Optional[str],
        two_hand_gesture: Optional[str],
        mode: ControlMode,
        context: str = "default",
    ) -> Optional[str]:
        """Resolve final action; two-hand gesture has priority."""
        source = two_hand_gesture if two_hand_gesture else single_gesture
        if not source:
            return None

        action = self._mode_maps.get(mode, {}).get(source, source)
        if context in self._context_overrides:
            action = self._context_overrides[context].get(action, action)
        return action


class CommandStabilityGate:
    """Rate-limit repeated action dispatch to improve stability."""

    def __init__(self, max_rate_hz: float = 20.0):
        self.min_interval = 1.0 / max(1e-3, max_rate_hz)
        self.last_sent: Dict[str, float] = {}

    def allow(self, action: Optional[str], timestamp: float) -> bool:
        if not action:
            return False
        last = self.last_sent.get(action, 0.0)
        if (timestamp - last) < self.min_interval:
            return False
        self.last_sent[action] = timestamp
        return True


class IntelligentControlLayer:
    """Orchestrates smoothing, interpretation, gating, and feedback."""

    def __init__(
        self,
        smoothing_alpha: float = 0.35,
        deadzone: float = 0.002,
        base_latency_ms: float = 35.0,
        enable_latency_compensation: bool = True,
        max_command_rate_hz: float = 20.0,
    ):
        self.motion = MotionIntelligence(
            smoothing_alpha=smoothing_alpha,
            deadzone=deadzone,
            base_latency_ms=base_latency_ms,
            enable_latency_compensation=enable_latency_compensation,
        )
        self.interpreter = StatefulGestureInterpreter()
        self.gate = CommandStabilityGate(max_rate_hz=max_command_rate_hz)
        self.feedback_events: Deque[FeedbackEventLite] = deque(maxlen=200)

    def filter_and_predict(self, hand_id: int, raw_pos: np.ndarray, timestamp: float) -> np.ndarray:
        """Apply smoothing and latency compensation to hand pose."""
        return self.motion.update(hand_id, raw_pos, timestamp)

    def register_frame_latency(self, latency_ms: float) -> None:
        """Update latency estimate for predictor."""
        self.motion.add_measured_latency(latency_ms)

    def resolve_action(
        self,
        single_gesture: Optional[str],
        two_hand_gesture: Optional[str],
        mode: ControlMode,
        context: str,
        timestamp: float,
    ) -> Optional[str]:
        """Resolve and rate-limit action dispatch."""
        action = self.interpreter.resolve(single_gesture, two_hand_gesture, mode, context)
        return action if self.gate.allow(action, timestamp) else None

    @staticmethod
    def should_execute(mode: ControlMode) -> bool:
        """Only virtual execution mode sends commands to robot/UE/object world."""
        return mode == ControlMode.VIRTUAL_EXECUTION

    def feedback(self, event: str, severity: str, message: str, data: Optional[Dict] = None) -> None:
        """Record feedback event."""
        self.feedback_events.append(
            FeedbackEventLite(
                event=event,
                severity=severity,
                message=message,
                data=data or {},
            )
        )
