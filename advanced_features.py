"""
Feature 6: Advanced Analytics Dashboard
Feature 7: Gesture Feedback System
Feature 8: Gesture Confidence Tracking
Feature 9: Adaptive Frame Rate
Feature 10: Cloud Inference Support

Condensed multi-feature module for analytics, feedback, confidence, FPS, and cloud ML.
"""

from __future__ import annotations

import time
import numpy as np
import threading
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from collections import deque
from enum import Enum
import requests


# ============================================================================
# FEATURE 6: ADVANCED ANALYTICS DASHBOARD
# ============================================================================

@dataclass
class MotionAnalytics:
    """Real-time motion statistics per finger."""
    finger_name: str
    velocities: deque = field(default_factory=lambda: deque(maxlen=60))
    accelerations: deque = field(default_factory=lambda: deque(maxlen=60))
    positions: deque = field(default_factory=lambda: deque(maxlen=60))

    def add_sample(self, vx: float, vy: float, ax: float, ay: float, x: float, y: float) -> None:
        """Add motion sample."""
        speed = (vx*vx + vy*vy) ** 0.5
        accel = (ax*ax + ay*ay) ** 0.5
        self.velocities.append(speed)
        self.accelerations.append(accel)
        self.positions.append((x, y))

    def get_stats(self) -> Dict:
        """Get motion statistics."""
        if not self.velocities:
            return {}

        return {
            "avg_velocity": np.mean(self.velocities),
            "max_velocity": np.max(self.velocities),
            "std_velocity": np.std(self.velocities),
            "avg_acceleration": np.mean(self.accelerations),
            "max_acceleration": np.max(self.accelerations),
        }


class AnalyticsDashboard:
    """Real-time analytics with graphs and statistics."""

    def __init__(self):
        self.per_finger_analytics: Dict[str, MotionAnalytics] = {}
        self.gesture_history: deque = deque(maxlen=100)
        self.session_start = time.time()
        self.frame_times: deque = deque(maxlen=60)  # For FPS

    def update_motion(self, finger_name: str, vx: float, vy: float, ax: float, ay: float, x: float, y: float) -> None:
        """Update motion data."""
        if finger_name not in self.per_finger_analytics:
            self.per_finger_analytics[finger_name] = MotionAnalytics(finger_name)
        self.per_finger_analytics[finger_name].add_sample(vx, vy, ax, ay, x, y)

    def add_gesture(self, gesture: str, confidence: float) -> None:
        """Record detected gesture."""
        self.gesture_history.append((time.time(), gesture, confidence))

    def get_session_statistics(self) -> Dict:
        """Get overall session statistics."""
        elapsed = time.time() - self.session_start
        return {
            "session_duration_sec": elapsed,
            "total_gestures": len(self.gesture_history),
            "average_gesture_confidence": np.mean([g[2] for g in self.gesture_history]) if self.gesture_history else 0.0,
            "most_common_gesture": max(set(g[1] for g in self.gesture_history), key=list(g[1] for g in self.gesture_history).count) if self.gesture_history else "none",
            "fingers_tracked": list(self.per_finger_analytics.keys()),
        }

    def get_motion_graphs_data(self) -> Dict[str, Dict]:
        """Get data for motion graphs."""
        return {
            name: analytics.get_stats()
            for name, analytics in self.per_finger_analytics.items()
        }


# ============================================================================
# FEATURE 7: GESTURE FEEDBACK SYSTEM
# ============================================================================

class FeedbackType(Enum):
    """Types of feedback."""
    VISUAL_HIGHLIGHT = "visual_highlight"
    VISUAL_PULSE = "visual_pulse"
    VISUAL_POPUP = "visual_popup"
    HAPTIC_BUZZ = "haptic_buzz"
    HAPTIC_PATTERN = "haptic_pattern"
    SOUND_BEEP = "sound_beep"
    SOUND_CUSTOM = "sound_custom"


@dataclass
class FeedbackEvent:
    """Single feedback event."""
    feedback_type: FeedbackType
    intensity: float = 0.5  # 0-1
    duration_ms: int = 200
    data: Dict = field(default_factory=dict)


class FeedbackManager:
    """Manage gesture recognition feedback."""

    def __init__(self):
        self.callbacks: Dict[FeedbackType, Callable] = {}
        self.enabled = True
        self.event_queue: List[FeedbackEvent] = []

    def register_feedback(self, feedback_type: FeedbackType, callback: Callable) -> None:
        """Register feedback handler."""
        self.callbacks[feedback_type] = callback

    def trigger_feedback(self, event: FeedbackEvent) -> None:
        """Trigger a feedback event."""
        if not self.enabled:
            return

        self.event_queue.append(event)

        if event.feedback_type in self.callbacks:
            self.callbacks[event.feedback_type](event)

    def provide_gesture_feedback(self, gesture: str, confidence: float) -> None:
        """Provide feedback for recognized gesture."""
        intensity = min(confidence, 1.0)

        # Visual feedback
        self.trigger_feedback(FeedbackEvent(
            feedback_type=FeedbackType.VISUAL_HIGHLIGHT,
            intensity=intensity,
            duration_ms=200,
            data={"gesture": gesture}
        ))

        # Optional haptic (if supported)
        if confidence > 0.7:
            self.trigger_feedback(FeedbackEvent(
                feedback_type=FeedbackType.HAPTIC_BUZZ,
                intensity=intensity,
                duration_ms=100
            ))


# ============================================================================
# FEATURE 8: GESTURE CONFIDENCE TRACKING
# ============================================================================

@dataclass
class ConfidenceStats:
    """Confidence statistics."""
    gesture_name: str
    samples: deque = field(default_factory=lambda: deque(maxlen=100))
    threshold: float = 0.5

    def add_sample(self, confidence: float) -> None:
        """Add confidence sample."""
        self.samples.append(confidence)

    def get_stats(self) -> Dict:
        """Get confidence statistics."""
        if not self.samples:
            return {}

        samples_list = list(self.samples)
        return {
            "avg_confidence": np.mean(samples_list),
            "std_confidence": np.std(samples_list),
            "min_confidence": np.min(samples_list),
            "max_confidence": np.max(samples_list),
            "above_threshold_ratio": sum(1 for s in samples_list if s >= self.threshold) / len(samples_list),
        }


class ConfidenceTracker:
    """Track and manage gesture confidence."""

    def __init__(self, default_threshold: float = 0.5):
        self.stats_per_gesture: Dict[str, ConfidenceStats] = {}
        self.default_threshold = default_threshold
        self.filter_low_confidence = True
        self.min_confidence = 0.3

    def record_confidence(self, gesture: str, confidence: float) -> bool:
        """Record gesture confidence, return True if above threshold."""
        if gesture not in self.stats_per_gesture:
            self.stats_per_gesture[gesture] = ConfidenceStats(
                gesture_name=gesture,
                threshold=self.default_threshold
            )

        self.stats_per_gesture[gesture].add_sample(confidence)

        # Filter low confidence
        if self.filter_low_confidence and confidence < self.min_confidence:
            return False

        return True

    def get_gesture_stats(self, gesture: str) -> Dict:
        """Get confidence stats for gesture."""
        if gesture not in self.stats_per_gesture:
            return {}
        return self.stats_per_gesture[gesture].get_stats()

    def get_gesture_statistics(self, gesture: str) -> Dict:
        """Backward-compatible alias for get_gesture_stats()."""
        return self.get_gesture_stats(gesture)

    def set_gesture_threshold(self, gesture: str, threshold: float) -> None:
        """Set confidence threshold for gesture."""
        if gesture not in self.stats_per_gesture:
            self.stats_per_gesture[gesture] = ConfidenceStats(
                gesture_name=gesture,
                threshold=threshold
            )
        else:
            self.stats_per_gesture[gesture].threshold = threshold


# ============================================================================
# FEATURE 9: ADAPTIVE FRAME RATE
# ============================================================================

class AdaptiveFrameRateController:
    """Dynamically adjust processing frame rate based on system load."""

    def __init__(self, target_fps: int = 30, min_fps: int = 15, max_fps: int = 60):
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.current_fps = target_fps

        self.frame_times: deque = deque(maxlen=60)
        self.cpu_loads: deque = deque(maxlen=30)
        self.adjustment_factor = 1.0

    def record_frame_time(self, elapsed_ms: float) -> None:
        """Record frame processing time."""
        self.frame_times.append(elapsed_ms)

    def get_average_frame_time(self) -> float:
        """Get average frame time in ms."""
        if not self.frame_times:
            return 1000.0 / self.target_fps
        return np.mean(self.frame_times)

    def get_current_fps(self) -> float:
        """Calculate current FPS."""
        avg_time = self.get_average_frame_time()
        if avg_time <= 0:
            return self.target_fps
        return 1000.0 / avg_time

    def measure_cpu_load(self) -> float:
        """Measure system CPU load."""
        try:
            import psutil
            load = psutil.cpu_percent(interval=0.1)
            self.cpu_loads.append(load)
            return load
        except Exception:
            return 50.0  # Assume moderate load if can't measure

    def compute_adaptive_fps(self) -> int:
        """Compute adaptive target FPS based on load."""
        if not self.cpu_loads:
            return self.target_fps

        avg_load = np.mean(self.cpu_loads)

        if avg_load > 80:
            # High load - reduce FPS
            self.adjustment_factor = 0.7
            return max(self.target_fps * 0.7, self.min_fps)
        elif avg_load > 60:
            # Moderate-high load - slight reduction
            self.adjustment_factor = 0.85
            return max(self.target_fps * 0.85, self.min_fps)
        elif avg_load < 40:
            # Low load - increase FPS if possible
            self.adjustment_factor = min(1.2, self.max_fps / self.target_fps)
            return min(int(self.target_fps * 1.2), self.max_fps)

        self.adjustment_factor = 1.0
        return self.target_fps

    def should_skip_frame(self) -> bool:
        """Return True if should skip frame for load reduction."""
        return np.random.random() < (1.0 - self.adjustment_factor)


# ============================================================================
# FEATURE 10: CLOUD INFERENCE SUPPORT
# ============================================================================

class CloudInferenceClient:
    """Client for remote ML inference."""

    def __init__(self, server_url: str, api_key: str = ""):
        self.server_url = server_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.enabled = False
        self.latency_ms: deque = deque(maxlen=30)

    def predict_gesture(self, landmarks: List) -> Optional[Tuple[str, float]]:
        """Send landmarks to cloud for gesture prediction."""
        if not self.enabled or not landmarks:
            return None

        try:
            start = time.time()

            # Prepare payload
            landmark_data = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in landmarks
            ]

            response = self.session.post(
                f"{self.server_url}/predict_gesture",
                json={"landmarks": landmark_data},
                timeout=2.0
            )

            latency = (time.time() - start) * 1000
            self.latency_ms.append(latency)

            if response.status_code == 200:
                data = response.json()
                return (data.get("gesture"), data.get("confidence", 0.0))

            return None
        except Exception as e:
            print(f"[CloudInference] Error: {e}")
            return None

    def get_average_latency(self) -> float:
        """Get average inference latency in ms."""
        return np.mean(self.latency_ms) if self.latency_ms else 0.0

    def connect(self) -> bool:
        """Test connection to cloud service."""
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=5.0
            )
            self.enabled = response.status_code == 200
            if self.enabled:
                print(f"[CloudInference] Connected to {self.server_url}")
            return self.enabled
        except Exception as e:
            print(f"[CloudInference] Connection failed: {e}")
            self.enabled = False
            return False
