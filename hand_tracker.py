"""
Low-level hand tracking module built on MediaPipe HandLandmarker (Tasks API).

Responsibilities:
- Download and manage the hand landmark model.
- Run hand detection for 1–2 hands per frame.
- Output normalized landmarks and per-hand metadata.
- Handle image mirroring and color space conversion.

Higher-level concerns (cursor control, gestures, visualization) live
in dedicated modules so this file stays focused on vision I/O.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ----------------- Model paths & download helper ----------------- #

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(MODULE_DIR, "models")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")


def download_hand_landmarker_model(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Ensure the HandLandmarker .task model exists locally.

    If the file is missing it is downloaded from the official MediaPipe
    model repository into the project-local `models/` directory.
    """
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[HandTracker] Downloading hand_landmarker model to '{model_path}'...")

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:  # pragma: no cover - network errors are environment-specific
        raise RuntimeError(
            "Failed to download MediaPipe hand_landmarker model.\n"
            f"URL: {MODEL_URL}\nError: {exc}\n"
            "Download this file manually and place it at:\n"
            f"  {model_path}"
        ) from exc

    print("[HandTracker] Model download complete.")
    return model_path


# ----------------- Hand connections (skeleton) ----------------- #

# Re-declaration of MediaPipe's canonical hand connections so we do not
# depend on deprecated mp.solutions APIs.
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm bridges
    (5, 9), (9, 13), (13, 17),
]


# ----------------- Public configuration & data types ----------------- #

@dataclass
class HandTrackerConfig:
    """
    Configuration shared across tracking, cursor control, and visualization.

    The same config instance can be passed into other modules so that
    system-level tuning is centralized in one object.
    """

    model_path: str = DEFAULT_MODEL_PATH
    max_num_hands: int = 2

    # MediaPipe Task confidence thresholds.
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Global behavioural options.
    smoothing_factor: float = 0.25  # Used by cursor_control for position smoothing.
    mirror_image: bool = True
    enable_cursor_control: bool = True


@dataclass
class HandLandmarks:
    """
    Per-hand data returned from the tracker.

    Attributes:
        hand_id: Index within the current frame (0..N-1).
        handedness: "Left", "Right", or "Unknown" as reported by MediaPipe.
        landmarks: List of normalized landmarks (x, y, z in [0,1] / metric).
        index_finger_tip: Normalized (x, y) of landmark 8 for convenience.
    """

    hand_id: int
    handedness: str
    landmarks: List
    index_finger_tip: Tuple[float, float]


# ----------------- HandTracker implementation ----------------- #

class HandTracker:
    """
    Wrapper around MediaPipe HandLandmarker for real-time webcam frames.

    This class deliberately focuses on:
    - Preparing frames (mirroring, BGR→RGB).
    - Running the HandLandmarker Task.
    - Packaging results into convenient Python dataclasses.

    It does *not* perform cursor control or gesture recognition.
    """

    def __init__(self, config: HandTrackerConfig):
        self.config = config

        model_path = download_hand_landmarker_model(config.model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)

        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )

        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def close(self) -> None:
        """Release any native resources held by the MediaPipe Task."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def _timestamp_ms(self) -> int:
        """Monotonic timestamp for VIDEO mode, in milliseconds."""
        return int((time.time() - self._start_time) * 1000)

    def process(self, frame_bgr: "cv2.Mat") -> Tuple["cv2.Mat", List[HandLandmarks]]:
        """
        Run hand detection on a BGR frame from OpenCV.

        Args:
            frame_bgr: Raw frame as captured from cv2.VideoCapture.

        Returns:
            processed_frame: BGR frame after mirroring (if enabled).
            hands: List of HandLandmarks for each detected hand.
        """
        frame = frame_bgr.copy()

        # Mirror horizontally for "mirror view" interaction if requested.
        if self.config.mirror_image:
            frame = cv2.flip(frame, 1)

        # Convert BGR (OpenCV) to RGB (MediaPipe).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        result = self.landmarker.detect_for_video(
            mp_image, self._timestamp_ms()
        )

        hands: List[HandLandmarks] = []

        if result.hand_landmarks:
            # First, pair raw landmarks with handedness labels from MediaPipe.
            raw: List[Tuple[str, List]] = []
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                handed_label = "Unknown"
                if result.handedness and i < len(result.handedness):
                    categories = result.handedness[i]
                    if categories:
                        handed_label = categories[0].category_name or "Unknown"
                raw.append((handed_label, hand_landmarks))

            # CRITICAL FIX: Apply mirror flip FIRST before sorting
            # This ensures hand_id assignment matches the corrected handedness
            if self.config.mirror_image:
                raw_flipped = []
                for handed_label, hand_landmarks in raw:
                    # Flip the handedness label to match mirrored visual appearance
                    if handed_label.lower() == "left":
                        handed_label = "Right"
                    elif handed_label.lower() == "right":
                        handed_label = "Left"
                    raw_flipped.append((handed_label, hand_landmarks))
                raw = raw_flipped

            # Now sort so that "Left" gets hand_id 0 and "Right" gets hand_id 1
            # This happens AFTER mirror correction, so hand_ids are stable
            def sort_key(item: Tuple[str, List]) -> int:
                label = item[0].lower()
                if label == "left":
                    return 0
                if label == "right":
                    return 1
                return 2

            raw_sorted = sorted(raw, key=sort_key)

            for hand_id, (handed_label, hand_landmarks) in enumerate(raw_sorted):
                index_tip = hand_landmarks[8]
                hands.append(
                    HandLandmarks(
                        hand_id=hand_id,
                        handedness=handed_label,
                        landmarks=hand_landmarks,
                        index_finger_tip=(float(index_tip.x), float(index_tip.y)),
                    )
                )

        return frame, hands

