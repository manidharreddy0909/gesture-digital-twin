"""
Adaptive Gesture Calibration Module

Dynamically adjusts gesture detection thresholds based on:
- User hand size (wrist-to-fingertip distance)
- Camera distance (inferred from hand landmark spread)
- Environmental factors (lighting, hand speed patterns)
- User gesture speed preferences

Architecture:
- CalibrationProfile: User-specific calibration settings
- CalibrationAnalyzer: Analyzes hand characteristics
- AdaptiveThresholdComputer: Calculates adjusted thresholds
- CalibrationMode: Interactive calibration UI integration
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from hand_tracker import HandLandmarks
from motion_analyzer import HandMotionInfo


@dataclass
class CalibrationProfile:
    """
    User-specific calibration profile.

    Stores baseline measurements and computed threshold adjustments.
    """
    user_id: str
    hand_size_mm: float = 200.0  # Typical adult hand ~200mm wrist to fingertip
    camera_distance_cm: float = 50.0  # Distance from camera to hand
    hand_span_normalized: float = 0.4  # Normalized hand span in image
    dominant_hand: str = "right"

    # Calculated threshold adjustments (multipliers)
    pinch_threshold_multiplier: float = 1.0
    swipe_distance_multiplier: float = 1.0
    finger_angle_multiplier: float = 1.0
    wrist_distance_multiplier: float = 1.0

    # Motion characteristics
    avg_gesture_speed: float = 0.5  # Normalized units/sec
    gesture_speed_variance: float = 0.1

    # Environmental conditions
    lighting_brightness: float = 0.5  # 0-1
    background_complexity: float = 0.3  # 0-1

    # Advanced settings
    custom_overrides: Dict[str, float] = field(default_factory=dict)


class HandCharacteristicsAnalyzer:
    """
    Analyze hand characteristics from tracked landmarks.

    Computes:
    - Hand size and aspect ratio
    - Finger lengths and ratios
    - Typical hand position in frame
    - Hand speed patterns
    """

    def __init__(self, history_size: int = 60):
        self.history_size = history_size
        self.hand_sizes: deque = deque(maxlen=history_size)
        self.hand_speeds: deque = deque(maxlen=history_size)
        self.hand_positions: deque = deque(maxlen=history_size)

    def analyze_hand_size(self, landmarks: List) -> float:
        """
        Calculate normalized hand size.

        Returns ratio of hand span to image dimensions.
        Based on wrist to middle fingertip distance.
        """
        if len(landmarks) < 12:
            return 0.0

        # Wrist: landmark 0
        # Middle fingertip: landmark 12
        wrist_x, wrist_y = landmarks[0].x, landmarks[0].y
        mid_x, mid_y = landmarks[12].x, landmarks[12].y

        span = math.sqrt((mid_x - wrist_x) ** 2 + (mid_y - wrist_y) ** 2)
        self.hand_sizes.append(span)
        return span

    def analyze_finger_lengths(self, landmarks: List) -> Dict[str, float]:
        """Calculate normalized lengths for each finger."""
        if len(landmarks) < 21:
            return {}

        wrist_x, wrist_y = landmarks[0].x, landmarks[0].y

        # Finger tips: 4, 8, 12, 16, 20
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        finger_tips = [4, 8, 12, 16, 20]

        lengths = {}
        for name, tip_idx in zip(finger_names, finger_tips):
            tip_x, tip_y = landmarks[tip_idx].x, landmarks[tip_idx].y
            length = math.sqrt((tip_x - wrist_x) ** 2 + (tip_y - wrist_y) ** 2)
            lengths[name] = length

        return lengths

    def analyze_hand_position(self, landmarks: List) -> Tuple[float, float, float]:
        """Get centroid and spread of hand in frame."""
        if not landmarks:
            return 0.0, 0.0, 0.0

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)

        # Spread = max distance from centroid
        spread = max(
            math.sqrt((x - centroid_x) ** 2 + (y - centroid_y) ** 2)
            for x, y in zip(xs, ys)
        )

        self.hand_positions.append((centroid_x, centroid_y, spread))
        return centroid_x, centroid_y, spread

    def analyze_motion_speed(self, motion: HandMotionInfo) -> float:
        """Calculate average motion speed from trajectory."""
        traj = motion.fingertip_trajectories.get(8)  # Index fingertip
        if not traj or not traj.speeds:
            return 0.0

        avg_speed = sum(traj.speeds) / len(traj.speeds)
        self.hand_speeds.append(avg_speed)
        return avg_speed

    def get_average_hand_size(self) -> float:
        """Get average hand size from history."""
        return sum(self.hand_sizes) / len(self.hand_sizes) if self.hand_sizes else 0.4

    def get_average_gesture_speed(self) -> float:
        """Get average gesture speed from history."""
        return sum(self.hand_speeds) / len(self.hand_speeds) if self.hand_speeds else 0.5

    def infer_camera_distance(self, hand_size: float) -> float:
        """
        Estimate camera distance based on hand size.

        Uses inverse relationship: larger hand size = closer to camera.
        Assumes typical adult hand ~200mm span.
        """
        REFERENCE_HAND_SIZE = 200.0  # mm
        REFERENCE_DISTANCE = 50.0  # cm
        REFERENCE_NORMALIZED_SIZE = 0.35

        if hand_size < 0.01:
            return REFERENCE_DISTANCE

        # Inverse relationship: distance ∝ 1/normalized_size
        estimated_distance = REFERENCE_DISTANCE * (REFERENCE_NORMALIZED_SIZE / max(hand_size, 0.01))
        # Clamp to reasonable range
        return max(20.0, min(200.0, estimated_distance))


class AdaptiveThresholdComputer:
    """
    Compute adaptive gesture detection thresholds.

    Adjusts thresholds from base configuration based on:
    - Hand size (larger hands need larger thresholds)
    - Camera distance (farther = smaller visible movements)
    - Motion patterns (faster gestures need larger thresholds)
    """

    def __init__(self, base_config: Dict[str, float]):
        # Base configuration thresholds
        self.base_config = base_config
        self.profile: Optional[CalibrationProfile] = None

    def set_profile(self, profile: CalibrationProfile) -> None:
        """Set calibration profile for threshold computation."""
        self.profile = profile

    def compute_adaptive_thresholds(self) -> Dict[str, float]:
        """Compute all adaptive thresholds from profile."""
        if self.profile is None:
            return self.base_config.copy()

        thresholds = {}

        # Pinch threshold adjustment based on hand size
        # Larger hands = use larger threshold
        pinch_base = self.base_config.get("pinch_distance_threshold", 0.05)
        size_factor = self.profile.hand_span_normalized / 0.4  # 0.4 is typical
        camera_factor = self.profile.camera_distance_cm / 50.0  # 50cm is typical
        thresholds["pinch_distance_threshold"] = (
            pinch_base * size_factor * camera_factor * self.profile.pinch_threshold_multiplier
        )

        # Swipe distance threshold
        swipe_base = self.base_config.get("swipe_distance_threshold", 0.20)
        thresholds["swipe_distance_threshold"] = (
            swipe_base * size_factor * camera_factor * self.profile.swipe_distance_multiplier
        )

        # Finger angle threshold (slight adjustment for hand size)
        angle_base = self.base_config.get("finger_angle_threshold_deg", 160.0)
        # Angle doesn't change much, but confidence thresholds do
        thresholds["finger_angle_threshold_deg"] = angle_base

        # Wrist distance margin
        wrist_base = self.base_config.get("wrist_distance_margin", 0.015)
        thresholds["wrist_distance_margin"] = (
            wrist_base * size_factor * camera_factor * self.profile.wrist_distance_multiplier
        )

        # Motion-speed based thresholds
        speed_factor = self.profile.avg_gesture_speed / 0.5  # 0.5 is typical
        thresholds["motion_speed_multiplier"] = speed_factor

        # Apply custom overrides
        for key, value in self.profile.custom_overrides.items():
            if key in thresholds:
                thresholds[key] = value

        return thresholds

    def get_threshold(self, threshold_name: str) -> float:
        """Get single adaptive threshold."""
        thresholds = self.compute_adaptive_thresholds()
        return thresholds.get(threshold_name, self.base_config.get(threshold_name, 0.0))


class CalibrationMode:
    """
    Interactive calibration mode.

    Guides user through calibration process:
    1. Relax hand - capture baseline size
    2. Perform test pinches
    3. Perform test swipes
    4. Adjust based on recognition accuracy
    """

    def __init__(self):
        self.calibration_steps = [
            "show_relaxed_hand",
            "test_pinches",
            "test_swipes",
            "test_circles",
            "finalize"
        ]
        self.current_step: int = 0
        self.step_data: Dict = {}
        self.analyzer = HandCharacteristicsAnalyzer()

    def start_calibration(self, user_id: str = "default") -> None:
        """Start calibration sequence."""
        self.current_step = 0
        self.step_data = {
            "user_id": user_id,
            "hand_sizes": [],
            "landmark_data": [],
        }

    def process_calibration_frame(
        self,
        hands: List[HandLandmarks],
        motions: Dict[int, HandMotionInfo]
    ) -> Tuple[str, str]:
        """
        Process frame during calibration.

        Returns:
            (step_name, instruction_text)
        """
        if self.current_step >= len(self.calibration_steps):
            return "complete", "Calibration complete!"

        step = self.calibration_steps[self.current_step]

        if step == "show_relaxed_hand":
            if len(hands) > 0:
                hand = hands[0]
                size = self.analyzer.analyze_hand_size(hand.landmarks)
                self.step_data["hand_sizes"].append(size)

                if len(self.step_data["hand_sizes"]) >= 30:
                    # Got enough samples
                    self.current_step += 1
                    return "test_pinches", "Now perform pinch gestures. Pinch and release 5 times."

            return "show_relaxed_hand", "Show your relaxed hand to camera. Hand will be measured..."

        elif step == "test_pinches":
            # Record pinch data for validation
            if len(hands) > 0:
                self.step_data["landmark_data"].append(hands[0].landmarks)

            if len(self.step_data["landmark_data"]) >= 50:
                self.current_step += 1
                return "test_swipes", "Now perform swipe gestures. Swipe left, right, up, down."

            return "test_pinches", f"Performing pinch test... ({len(self.step_data['landmark_data'])}/50)"

        elif step == "test_swipes":
            if len(hands) > 0 and 0 in motions:
                motion = motions[0]
                # Record motion data
                self.step_data["landmark_data"].append(hands[0].landmarks)

            if len(self.step_data["landmark_data"]) >= 100:
                self.current_step += 1
                return "test_circles", "Now perform circular motions. Make 5 circles."

            return "test_swipes", f"Performing swipe test... ({len(self.step_data['landmark_data'])}/100)"

        elif step == "test_circles":
            if len(hands) > 0 and 0 in motions:
                self.step_data["landmark_data"].append(hands[0].landmarks)

            if len(self.step_data["landmark_data"]) >= 150:
                self.current_step += 1
                return "finalize", "Calibration complete! Computing profile..."

            return "test_circles", f"Performing circle test... ({len(self.step_data['landmark_data'])}/150)"

        elif step == "finalize":
            self.current_step += 1
            return "complete", "Calibration successful!"

        return "unknown", "Unknown calibration step"

    def create_profile(self) -> CalibrationProfile:
        """Create calibration profile from collected data."""
        # Compute statistics
        avg_hand_size = self.analyzer.get_average_hand_size()
        camera_distance = self.analyzer.infer_camera_distance(avg_hand_size)

        profile = CalibrationProfile(
            user_id=self.step_data.get("user_id", "default"),
            hand_span_normalized=avg_hand_size,
            camera_distance_cm=camera_distance,
            avg_gesture_speed=self.analyzer.get_average_gesture_speed(),
        )

        return profile

    def is_complete(self) -> bool:
        """Check if calibration is complete."""
        return self.current_step >= len(self.calibration_steps)


class CalibrationManager:
    """
    Manages multiple calibration profiles and applies them.

    Stores profiles per user and loads/saves from disk.
    """

    def __init__(self, profiles_dir: str = "calibration_profiles"):
        self.profiles_dir = profiles_dir
        self.profiles: Dict[str, CalibrationProfile] = {}
        self.active_profile: Optional[CalibrationProfile] = None

    def save_profile(self, profile: CalibrationProfile) -> None:
        """Save calibration profile to disk."""
        import json
        from pathlib import Path

        Path(self.profiles_dir).mkdir(exist_ok=True)

        profile_path = Path(self.profiles_dir) / f"{profile.user_id}.json"
        profile_dict = {
            "user_id": profile.user_id,
            "hand_size_mm": profile.hand_size_mm,
            "camera_distance_cm": profile.camera_distance_cm,
            "hand_span_normalized": profile.hand_span_normalized,
            "dominant_hand": profile.dominant_hand,
            "pinch_threshold_multiplier": profile.pinch_threshold_multiplier,
            "swipe_distance_multiplier": profile.swipe_distance_multiplier,
            "finger_angle_multiplier": profile.finger_angle_multiplier,
            "wrist_distance_multiplier": profile.wrist_distance_multiplier,
            "avg_gesture_speed": profile.avg_gesture_speed,
            "gesture_speed_variance": profile.gesture_speed_variance,
            "custom_overrides": profile.custom_overrides,
        }

        with open(profile_path, "w") as f:
            json.dump(profile_dict, f, indent=2)

        print(f"[CalibrationManager] Saved profile for {profile.user_id}")

    def load_profile(self, user_id: str) -> Optional[CalibrationProfile]:
        """Load calibration profile from disk."""
        import json
        from pathlib import Path

        profile_path = Path(self.profiles_dir) / f"{user_id}.json"

        if not profile_path.exists():
            return None

        with open(profile_path, "r") as f:
            data = json.load(f)

        profile = CalibrationProfile(**data)
        self.profiles[user_id] = profile
        return profile

    def set_active_profile(self, user_id: str) -> bool:
        """Activate a profile by user ID."""
        if user_id not in self.profiles:
            profile = self.load_profile(user_id)
            if profile is None:
                return False
            self.profiles[user_id] = profile

        self.active_profile = self.profiles[user_id]
        return True
