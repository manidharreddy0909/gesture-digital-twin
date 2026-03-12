"""
High-quality visualization layer for dual-hand tracking.

Responsibilities:
- Draw landmarks and skeletal connections with color-coded fingers.
- Annotate each hand with ID, handedness, and gesture label.
- Show per-hand cursor positions.
- Display overall FPS.

This module only consumes already-processed information, so it can be
swapped for more advanced rendering backends (OpenGL, Unreal, etc.).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from hand_tracker import HandLandmarks, HAND_CONNECTIONS
from gesture_detector import GestureResult


class Visualizer:
    """
    Render tracking, gesture, and cursor information onto video frames.

    All coordinates are assumed to be in the same mirrored / preprocessed
    space that the `HandTracker` returns.
    """

    def __init__(self) -> None:
        # Distinct colors per finger for improved visual parsing.
        self._finger_colors = {
            "thumb": (0, 0, 255),       # Red
            "index": (0, 255, 0),       # Green
            "middle": (255, 0, 0),      # Blue
            "ring": (0, 255, 255),      # Yellow
            "pinky": (255, 0, 255),     # Magenta
        }

    # ---- Public API ----

    def draw(
        self,
        frame_bgr: "cv2.Mat",
        hands: List[HandLandmarks],
        gestures: Dict[int, GestureResult],
        cursor_positions: Dict[int, Tuple[int, int]],
        fps: float,
    ) -> "cv2.Mat":
        """
        Draw all overlays for the current frame.

        Args:
            frame_bgr: Input BGR frame (modified and returned).
            hands: List of HandLandmarks returned from HandTracker.
            gestures: Mapping hand_id -> GestureResult.
            cursor_positions: Mapping hand_id -> (screen_x, screen_y).
            fps: Current frames-per-second estimate.
        """
        frame = frame_bgr

        for hand in hands:
            gesture = gestures.get(hand.hand_id)
            cursor_pos = cursor_positions.get(hand.hand_id)
            self._draw_hand(frame, hand, gesture, cursor_pos)

        self._draw_fps(frame, fps)
        return frame

    # ---- Internal helpers ----

    def _draw_hand(
        self,
        frame_bgr: "cv2.Mat",
        hand: HandLandmarks,
        gesture: GestureResult | None,
        cursor_pos: Tuple[int, int] | None,
    ) -> None:
        """Draw one hand's skeleton, landmarks, and textual labels."""
        h, w, _ = frame_bgr.shape

        # Convert normalized landmarks into pixel coordinates once.
        points: List[Tuple[int, int]] = []
        for lm in hand.landmarks:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            points.append((x_px, y_px))

        # Draw skeletal connections with a neutral color.
        for start_idx, end_idx in HAND_CONNECTIONS:
            x1, y1 = points[start_idx]
            x2, y2 = points[end_idx]
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (80, 255, 80), 2, cv2.LINE_AA)

        # Draw landmarks with per-finger color coding.
        for idx, (x, y) in enumerate(points):
            color = self._color_for_landmark(idx)
            radius = 5 if idx == 8 else 4
            cv2.circle(frame_bgr, (x, y), radius, color, -1, cv2.LINE_AA)

        # Textual annotations near the wrist landmark (0).
        wrist_x, wrist_y = points[0]
        label_lines = [
            f"Hand {hand.hand_id} ({hand.handedness})",
        ]
        if gesture is not None:
            label_lines.append(f"Gesture: {gesture.gesture}")

        if cursor_pos is not None:
            cx, cy = cursor_pos
            label_lines.append(f"Cursor: {cx}, {cy}")

        for i, text in enumerate(label_lines):
            y_text = wrist_y - 10 - i * 20
            y_text = max(20, y_text)
            cv2.putText(
                frame_bgr,
                text,
                (wrist_x + 10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _color_for_landmark(self, idx: int) -> Tuple[int, int, int]:
        """
        Return a BGR color for the given landmark index according to finger group.
        """
        if idx in (0, 1, 2, 3, 4):
            return self._finger_colors["thumb"]
        if idx in (5, 6, 7, 8):
            return self._finger_colors["index"]
        if idx in (9, 10, 11, 12):
            return self._finger_colors["middle"]
        if idx in (13, 14, 15, 16):
            return self._finger_colors["ring"]
        return self._finger_colors["pinky"]

    @staticmethod
    def _draw_fps(frame_bgr: "cv2.Mat", fps: float) -> None:
        """Render the FPS counter in the bottom-left corner."""
        h, _w, _ = frame_bgr.shape
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame_bgr,
            text,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

