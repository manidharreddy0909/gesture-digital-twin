"""
Main entry point for the dual‑hand tracking project.

This script wires together the modular components:

- hand_tracker:     low‑level MediaPipe hand detection.
- motion_analyzer:  fingertip motion history, velocity, trajectories.
- gesture_detector: per‑hand and two‑hand gesture recognition.
- cursor_controller:per‑hand cursor smoothing and OS cursor control.
- visualizer:       high‑quality rendering, labels, trails, and FPS.

Run with:  python main.py
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import cv2

from config import CursorAIConfig
from hand_tracker import HandTracker, HandTrackerConfig, HandLandmarks
from motion_analyzer import MotionAnalyzer, HandMotionInfo
from gesture_detector import GestureDetector, GestureResult, TwoHandGestureResult
from cursor_controller import CursorController
from visualizer import Visualizer
from integration import FrameConsumer, PrintConsumer


def main() -> None:
    """
    Initialize subsystems and run the real‑time webcam loop.

    Pipeline per frame:
      1) Capture frame from webcam.
      2) Run MediaPipe HandLandmarker via HandTracker.
      3) For each detected hand:
           - detect gesture,
           - update cursor state (with smoothing),
           - collect data for visualization.
      4) Render landmarks, gestures, cursor info, and FPS.
      5) Quit cleanly when 'q' is pressed.
    """
    # Global configuration for tracking, cursor, gestures, and motion.
    app_cfg = CursorAIConfig()

    tracker_cfg = HandTrackerConfig(
        max_num_hands=app_cfg.max_num_hands,
        min_hand_detection_confidence=app_cfg.min_detection_confidence,
        min_hand_presence_confidence=app_cfg.min_presence_confidence,
        min_tracking_confidence=app_cfg.min_tracking_confidence,
        smoothing_factor=app_cfg.cursor_smoothing,
        mirror_image=app_cfg.mirror_image,
        enable_cursor_control=app_cfg.enable_cursor_control,
    )

    tracker = HandTracker(tracker_cfg)
    motion_analyzer = MotionAnalyzer(history_size=app_cfg.motion_history_size)
    gesture_detector = GestureDetector(
        pinch_distance_threshold=app_cfg.pinch_distance_threshold,
        swipe_distance_threshold=app_cfg.swipe_distance_threshold,
        history_size=app_cfg.motion_history_size,
        extended_angle_threshold_deg=app_cfg.finger_angle_threshold_deg,
        wrist_distance_margin=app_cfg.wrist_distance_margin,
        circle_min_path=app_cfg.circle_min_path,
        circle_close_distance=app_cfg.circle_close_distance,
    )
    cursor_controller = CursorController(
        smoothing_factor=app_cfg.cursor_smoothing,
        predictive_factor=app_cfg.cursor_predictive,
        enable_cursor_control=app_cfg.enable_cursor_control,
        use_kalman_filter=app_cfg.use_kalman_filter,
        adaptive_smoothing=app_cfg.adaptive_smoothing,
    )
    visualizer = Visualizer()
    visualizer.enable_motion_trails = app_cfg.enable_motion_trails

    # Optional external consumers (e.g., robots, Unreal, etc.).
    consumers: list[FrameConsumer] = [PrintConsumer(enabled=False)]

    # Open default webcam; CAP_DSHOW tends to be stable on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: could not open webcam (index 0).")
        tracker.close()
        return

    print("Press 'q' in the video window to quit.")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            frame_t0 = time.perf_counter()

            # Step 1–2: hand detection (includes optional mirroring).
            t_det0 = time.perf_counter()
            processed_frame, hands = tracker.process(frame)
            t_det1 = time.perf_counter()

            now = time.time()

            # Per‑hand motion, gesture, and cursor computation.
            motions: Dict[int, HandMotionInfo] = {}
            gestures: Dict[int, GestureResult] = {}
            cursor_positions: Dict[int, Tuple[int, int]] = {}

            for hand in hands:
                # Decide which hand controls the primary cursor based on config.
                is_primary = (app_cfg.primary_cursor_hand.lower() == "left" and hand.handedness.lower() == "left") or (
                    app_cfg.primary_cursor_hand.lower() == "right" and hand.handedness.lower() == "right"
                )

                # Motion analysis (can be disabled for performance).
                motion_info: HandMotionInfo | None = None
                if app_cfg.enable_motion_analysis:
                    motion_info = motion_analyzer.update_from_landmarks(
                        hand_id=hand.hand_id,
                        landmarks=hand.landmarks,
                        timestamp=now,
                    )
                    motions[hand.hand_id] = motion_info

                # Gesture detection (static + dynamic) can be toggled.
                if app_cfg.enable_gesture_detection:
                    gesture = gesture_detector.detect(
                        hand_id=hand.hand_id,
                        handedness=hand.handedness,
                        landmarks=hand.landmarks,
                        timestamp=now,
                        motion=motion_info,
                    )
                    gestures[hand.hand_id] = gesture

                # Per‑hand cursor smoothing and prediction.
                cursor_pos = cursor_controller.update_cursor(
                    hand_id=hand.hand_id,
                    x_norm=hand.index_finger_tip[0],
                    y_norm=hand.index_finger_tip[1],
                    timestamp=now,
                    control_this_hand=is_primary,
                )
                cursor_positions[hand.hand_id] = cursor_pos

            # Two‑hand gesture (optional global descriptor).
            two_hand_gesture: TwoHandGestureResult | None = None
            if app_cfg.enable_two_hand_gestures:
                two_hand_gesture = gesture_detector.detect_two_hand_gesture(
                    hands=hands,
                    motions=motions,
                )

            # FPS computation.
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time

            # Rendering.
            t_vis0 = time.perf_counter()
            annotated = visualizer.draw(
                frame_bgr=processed_frame,
                hands=hands,
                gestures=gestures,
                cursor_positions=cursor_positions,
                motions=motions,
                fps=fps,
                two_hand_gesture=two_hand_gesture,
            )
            t_vis1 = time.perf_counter()

            # Optional per-frame profiling.
            if app_cfg.enable_profiling:
                t_frame = time.perf_counter() - frame_t0
                t_det = t_det1 - t_det0
                t_vis = t_vis1 - t_vis0
                print(
                    f"Frame {t_frame*1000:.1f} ms | detect {t_det*1000:.1f} ms | "
                    f"viz {t_vis*1000:.1f} ms | fps~{fps:.1f}"
                )

            # Notify any external consumers (robots, Unreal, etc.).
            for consumer in consumers:
                consumer.consume_frame_state(
                    hands=hands,
                    motions=motions,
                    gestures=gestures,
                    cursor_positions=cursor_positions,
                    two_hand_gesture=two_hand_gesture,
                )

            cv2.imshow("Cursor AI Tracker", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()