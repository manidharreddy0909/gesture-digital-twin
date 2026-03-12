"""
Visualization module for dual-hand motion and gesture tracking.

Responsibilities:
- Draw landmarks and skeletal connections with per-finger color coding.
- Overlay gesture labels, hand IDs, handedness, and cursor positions.
- Visualize fingertip motion trails (especially index fingertip).
- Show FPS and optional global two-hand gesture label.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from hand_tracker import HandLandmarks, HAND_CONNECTIONS
from gesture_detector import GestureResult, TwoHandGestureResult
from motion_analyzer import HandMotionInfo, FingerTrajectory


class Visualizer:
    """
    High-level drawing wrapper operating purely on already-computed state.

    This class intentionally does not depend on configuration modules; instead,
    callers can decide which overlays to render (e.g. whether to include
    motion trails or not) before calling into it.
    """

    def __init__(self) -> None:
        # Distinct BGR colors per finger group.
        self._finger_colors = {
            "thumb": (0, 0, 255),
            "index": (0, 255, 0),
            "middle": (255, 0, 0),
            "ring": (0, 255, 255),
            "pinky": (255, 0, 255),
        }
        # Whether to draw motion trails (can be toggled by callers).
        self.enable_motion_trails: bool = True

    def draw(
        self,
        frame_bgr: "cv2.Mat",
        hands: List[HandLandmarks],
        gestures: Dict[int, GestureResult],
        cursor_positions: Dict[int, Tuple[int, int]],
        motions: Dict[int, HandMotionInfo],
        fps: float,
        two_hand_gesture: Optional[TwoHandGestureResult] = None,
    ) -> "cv2.Mat":
        """
        Render all overlays on the provided frame.
        """
        frame = frame_bgr

        for hand in hands:
            gesture = gestures.get(hand.hand_id)
            cursor_pos = cursor_positions.get(hand.hand_id)
            motion = motions.get(hand.hand_id)
            self._draw_hand(frame, hand, gesture, cursor_pos, motion)

        self._draw_fps(frame, fps)

        if two_hand_gesture is not None:
            self._draw_two_hand_gesture(frame, two_hand_gesture)

        return frame

    # ---- Internal drawing helpers ----

    def _draw_hand(
        self,
        frame_bgr: "cv2.Mat",
        hand: HandLandmarks,
        gesture: Optional[GestureResult],
        cursor_pos: Optional[Tuple[int, int]],
        motion: Optional[HandMotionInfo],
    ) -> None:
        """Draw one hand's skeleton, landmarks, labels, and trails."""
        h, w, _ = frame_bgr.shape

        # Precompute pixel coordinates for all landmarks.
        points: List[Tuple[int, int]] = []
        for lm in hand.landmarks:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            points.append((x_px, y_px))

        # Draw skeletal connections.
        for start_idx, end_idx in HAND_CONNECTIONS:
            x1, y1 = points[start_idx]
            x2, y2 = points[end_idx]
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (80, 255, 80), 2, cv2.LINE_AA)

        # Draw individual landmarks with per-finger color.
        for idx, (x, y) in enumerate(points):
            color = self._color_for_landmark(idx)
            radius = 5 if idx == 8 else 4
            cv2.circle(frame_bgr, (x, y), radius, color, -1, cv2.LINE_AA)

        # Draw fingertip trail for index finger, color-coded by speed.
        if self.enable_motion_trails and motion is not None:
            traj = motion.fingertip_trajectories.get(8)
            if traj is not None and len(traj.points) > 1:
                self._draw_trail(frame_bgr, traj, w, h)

        # Text overlays near wrist.
        wrist_x, wrist_y = points[0]
        label_lines = [f"Hand {hand.hand_id} ({hand.handedness})"]
        if gesture is not None:
            label_lines.append(f"Gesture: {gesture.gesture}")
        if cursor_pos is not None:
            cx, cy = cursor_pos
            label_lines.append(f"Cursor: {cx}, {cy}")

        for i, text in enumerate(label_lines):
            y_text = max(20, wrist_y - 10 - i * 18)
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

    def _draw_trail(
        self,
        frame_bgr: "cv2.Mat",
        traj: FingerTrajectory,
        w: int,
        h: int,
    ) -> None:
        """
        Draw a motion trail for a fingertip where color encodes speed
        (blue = slow, red = fast).
        """
        if len(traj.points) < 2 or not traj.velocities:
            return

        # Normalize speeds to [0, 1] for color mapping.
        speeds = traj.speeds
        max_speed = max(speeds) if speeds else 0.0
        min_speed = min(speeds) if speeds else 0.0
        span = max(max_speed - min_speed, 1e-6)

        for i in range(1, len(traj.points)):
            x0, y0, _ = traj.points[i - 1]
            x1, y1, _ = traj.points[i]
            sx0, sy0 = int(x0 * w), int(y0 * h)
            sx1, sy1 = int(x1 * w), int(y1 * h)

            speed = speeds[i - 1] if i - 1 < len(speeds) else speeds[-1]
            norm = (speed - min_speed) / span

            # Map speed to color: slow (blue) -> fast (red).
            r = int(255 * norm)
            b = int(255 * (1.0 - norm))
            g = 50
            cv2.line(
                frame_bgr,
                (sx0, sy0),
                (sx1, sy1),
                (b, g, r),
                2,
                cv2.LINE_AA,
            )

    def _color_for_landmark(self, idx: int) -> Tuple[int, int, int]:
        """Return BGR color per finger group for a given landmark index."""
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
        """Draw FPS label in the bottom-left corner."""
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

    @staticmethod
    def _draw_two_hand_gesture(
        frame_bgr: "cv2.Mat",
        gesture: TwoHandGestureResult,
    ) -> None:
        """Draw a global two-hand gesture label at the top of the frame."""
        h, w, _ = frame_bgr.shape
        text = f"Two-hand: {gesture.gesture}"
        if gesture.extra:
            text += f" ({gesture.extra})"
        cv2.putText(
            frame_bgr,
            text,
            (int(0.25 * w), 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

