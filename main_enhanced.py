"""
Enhanced main entry point for the dual-hand tracking project with all advanced features.

Integrates:
- Advanced cursor prediction (Kalman filters)
- ML gesture detection framework
- Advanced two-hand gestures
- Performance profiling
- Virtual target integration
- PyQt6 GUI (optional)
- Multi-threading support for performance

Pipeline:
  1. Capture from webcam
  2. Hand detection via MediaPipe
  3. Motion analysis
  4. Gesture detection (heuristic + ML)
  5. Cursor control with advanced filtering
  6. Virtual target integration (if enabled)
  7. Visualization + GUI (if enabled)
"""

from __future__ import annotations

import time
import sys
import threading
from typing import Dict, Tuple, Optional

import cv2

from config import CursorAIConfig
from hand_tracker import HandTracker, HandTrackerConfig, HandLandmarks
from motion_analyzer import MotionAnalyzer, HandMotionInfo
from gesture_detector import GestureDetector, GestureResult, TwoHandGestureResult
from cursor_controller import CursorController
from visualizer import Visualizer
from integration import FrameConsumer, PrintConsumer
from performance_profiler import PipelineModuleController
from gesture_ml import GestureModelManager
from advanced_integration import CommandMapper, SocketBackend, HTTPBackend, UnrealEngine5Backend

# Optional 3D/AR/VR pipeline (for advanced users - SESSION 2 FINAL)
try:
    from camera_calibration import CameraCalibrator, CoordinateTransformer
    from cursor_controller_3d import CursorController3D
    from robot_controller import RobotArmUR5, HandToArmMapper
    from object_manipulator import ObjectManipulator, Object3D
    from unreal_bridge import UnrealPythonAPIBridge, HandSkeletonConverter, SkeletalMeshUpdate
    HAS_3D_MODULES = True
except ImportError as e:
    HAS_3D_MODULES = False
    if False:  # Set to True for debugging
        print(f"[INFO] 3D modules not available: {e}")

# Optional GUI
try:
    from gui_pyqt6 import create_pyqt_gui, GUIConfig as PyQtGUIConfig
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def main() -> None:
    """
    Initialize all subsystems and run the real-time webcam loop.

    Pipeline per frame:
      1) Capture frame from webcam.
      2) Run MediaPipe HandLandmarker via HandTracker.
      3) Motion analysis (optional, controllable).
      4) Gesture detection (heuristic + ML optional).
      5) Cursor control with advanced filtering.
      6) Integration to virtual targets (optional).
      7) Visualization and optional GUI rendering.
      8) Profiling and statistics collection.
    """
    # Global configuration
    app_cfg = CursorAIConfig()

    # Initialize performance controller
    perf_controller = PipelineModuleController()
    if app_cfg.enable_profiling:
        print("[INFO] Performance profiling enabled")

    # Initialize hand tracker
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

    # Motion analyzer
    motion_analyzer = MotionAnalyzer(history_size=app_cfg.motion_history_size)

    # Gesture detector with heuristic rules
    gesture_detector = GestureDetector(
        pinch_distance_threshold=app_cfg.pinch_distance_threshold,
        swipe_distance_threshold=app_cfg.swipe_distance_threshold,
        history_size=app_cfg.motion_history_size,
        extended_angle_threshold_deg=app_cfg.finger_angle_threshold_deg,
        wrist_distance_margin=app_cfg.wrist_distance_margin,
        circle_min_path=app_cfg.circle_min_path,
        circle_close_distance=app_cfg.circle_close_distance,
    )

    # ML gesture framework (optional)
    gesture_ml: Optional[GestureModelManager] = None
    if app_cfg.enable_ml_gestures:
        gesture_ml = GestureModelManager(app_cfg.ml_models_dir)
        gesture_ml.initialize_default_models()
        print("[INFO] ML gesture framework initialized")

    # Advanced cursor controller
    cursor_controller = CursorController(
        smoothing_factor=app_cfg.cursor_smoothing,
        predictive_factor=app_cfg.cursor_predictive,
        enable_cursor_control=app_cfg.enable_cursor_control,
        use_kalman_filter=app_cfg.use_kalman_filter,
        use_enhanced_kalman=app_cfg.use_enhanced_kalman,
        adaptive_smoothing=app_cfg.adaptive_smoothing,
        kalman_adaptive_noise=app_cfg.kalman_adaptive_noise,
        enable_prediction_logging=app_cfg.enable_cursor_prediction_logging,
    )

    # Visualizer
    visualizer = Visualizer()
    visualizer.enable_motion_trails = app_cfg.enable_motion_trails

    # Virtual target integration (optional)
    integration_backend = None
    command_mapper = CommandMapper()

    if app_cfg.enable_integration:
        if app_cfg.integration_backend.lower() == "socket":
            integration_backend = SocketBackend(*app_cfg.integration_endpoint.split(":"))
        elif app_cfg.integration_backend.lower() == "http":
            integration_backend = HTTPBackend(f"http://{app_cfg.integration_endpoint}/api/commands")
        elif app_cfg.integration_backend.lower() == "ue5":
            integration_backend = UnrealEngine5Backend(f"http://{app_cfg.integration_endpoint}/api/skeletal")

        if integration_backend:
            integration_backend.connect()
            print(f"[INFO] Integration backend connected: {app_cfg.integration_backend}")

            # Bind some example gestures
            command_mapper.bind_gesture("pinch", "grab_object")
            command_mapper.bind_gesture("open", "release_object")
            command_mapper.bind_two_hand_gesture("zoom_in", "scale_up")

    # Optional external consumers
    consumers: list[FrameConsumer] = [PrintConsumer(enabled=False)]

    # PyQt6 GUI (optional, requires PyQt6 installation)
    gui = None
    if app_cfg.enable_gui and GUI_AVAILABLE:
        gui_cfg = PyQtGUIConfig()
        gui = create_pyqt_gui(gui_cfg)
        print("[INFO] PyQt6 GUI enabled")

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        tracker.close()
        return

    print("Press 'q' in the video window to quit.")
    print("Press 'p' to toggle profiling.")
    print("Press 'm' to toggle ML gestures.")

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            frame_t0 = time.perf_counter()
            frame_count += 1

            # --- Hand detection ---
            with perf_controller.profiler.measure("hand_tracking"):
                processed_frame, hands = tracker.process(frame)

            now = time.time()

            # --- Motion and gesture analysis ---
            motions: Dict[int, HandMotionInfo] = {}
            gestures: Dict[int, GestureResult] = {}
            cursor_positions: Dict[int, Tuple[int, int]] = {}

            for hand in hands:
                is_primary = (
                    app_cfg.primary_cursor_hand.lower() == hand.handedness.lower()
                )

                # Motion analysis
                motion_info: Optional[HandMotionInfo] = None
                if app_cfg.enable_motion_analysis and perf_controller.is_enabled("motion_analysis"):
                    with perf_controller.profiler.measure("motion_analysis"):
                        motion_info = motion_analyzer.update_from_landmarks(
                            hand_id=hand.hand_id,
                            landmarks=hand.landmarks,
                            timestamp=now,
                        )
                        motions[hand.hand_id] = motion_info

                # Gesture detection (heuristic)
                if app_cfg.enable_gesture_detection and perf_controller.is_enabled("static_gesture_detection"):
                    with perf_controller.profiler.measure("gesture_detection"):
                        gesture = gesture_detector.detect(
                            hand_id=hand.hand_id,
                            handedness=hand.handedness,
                            landmarks=hand.landmarks,
                            timestamp=now,
                            motion=motion_info,
                        )
                        gestures[hand.hand_id] = gesture

                        # ML-based gesture refinement (optional)
                        if (
                            app_cfg.enable_ml_gestures
                            and gesture_ml is not None
                            and perf_controller.is_enabled("ml_gesture_prediction")
                        ):
                            # Try static prediction
                            ml_static = gesture_ml.predict_static(hand.landmarks)
                            if ml_static and ml_static.confidence > gesture.confidence:
                                gesture.gesture = ml_static.label
                                gesture.confidence = ml_static.confidence

                            # Try dynamic prediction
                            if motion_info:
                                ml_dynamic = gesture_ml.predict_dynamic(motion_info)
                                if ml_dynamic and ml_dynamic.confidence > gesture.confidence:
                                    gesture.gesture = ml_dynamic.label
                                    gesture.confidence = ml_dynamic.confidence

                # Cursor control with advanced filtering
                if perf_controller.is_enabled("cursor_control"):
                    with perf_controller.profiler.measure("cursor_control"):
                        cursor_pos = cursor_controller.update_cursor(
                            hand_id=hand.hand_id,
                            x_norm=hand.index_finger_tip[0],
                            y_norm=hand.index_finger_tip[1],
                            timestamp=now,
                            control_this_hand=is_primary,
                        )
                        cursor_positions[hand.hand_id] = cursor_pos

            # --- Two-hand gestures ---
            two_hand_gesture: Optional[TwoHandGestureResult] = None
            if app_cfg.enable_two_hand_gestures and perf_controller.is_enabled("two_hand_gestures"):
                with perf_controller.profiler.measure("two_hand_gestures"):
                    two_hand_gesture = gesture_detector.detect_two_hand_gesture(hands, motions)

            # --- Virtual target integration ---
            if integration_backend is not None:
                for gesture in gestures.values():
                    commands = command_mapper.map_gesture(gesture, now)
                    for cmd in commands:
                        integration_backend.send_command(cmd)

                if two_hand_gesture:
                    commands = command_mapper.map_two_hand_gesture(two_hand_gesture, now)
                    for cmd in commands:
                        integration_backend.send_command(cmd)

            # --- FPS and timing ---
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time
            perf_controller.fps_tracker.update()

            # --- Visualization ---
            if perf_controller.is_enabled("visualization"):
                with perf_controller.profiler.measure("visualization"):
                    annotated = visualizer.draw(
                        frame_bgr=processed_frame,
                        hands=hands,
                        gestures=gestures,
                        cursor_positions=cursor_positions,
                        motions=motions,
                        fps=fps,
                        two_hand_gesture=two_hand_gesture,
                    )

            # Display frame
            cv2.imshow("Cursor AI Tracker - Advanced", annotated if "annotated" in locals() else processed_frame)

            # --- External consumers ---
            for consumer in consumers:
                consumer.consume_frame_state(
                    hands=hands,
                    motions=motions,
                    gestures=gestures,
                    cursor_positions=cursor_positions,
                    two_hand_gesture=two_hand_gesture,
                )

            # --- Optional profiling output ---
            frame_time = time.perf_counter() - frame_t0
            if app_cfg.enable_profiling and frame_count % 30 == 0:
                print(f"Frame {frame_count}: {frame_time*1000:.1f}ms | FPS: {fps:.1f}")
                print(f"  Hands: {len(hands)}")
                if gestures:
                    print(f"  Gestures: {[g.gesture for g in gestures.values()]}")

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                enabled = perf_controller.modules["profiling"]
                print(f"[Profiling toggled: {not enabled}]")
            elif key == ord("m"):
                if gesture_ml:
                    enabled = perf_controller.toggle_module("ml_gesture_prediction")
                    print(f"[ML Gestures toggled: {enabled}]")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

        if integration_backend:
            integration_backend.disconnect()

        if app_cfg.enable_profiling:
            print("\n" + perf_controller.get_performance_report())

        print("Exited gracefully.")


if __name__ == "__main__":
    main()
