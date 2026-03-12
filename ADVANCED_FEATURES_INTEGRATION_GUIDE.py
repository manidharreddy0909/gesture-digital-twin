"""
================================================================================
ADVANCED FEATURES INTEGRATION GUIDE
================================================================================

Complete guide for integrating all 10 optional features into the Cursor AI system.

Features Implemented:
1. Gesture Dataset Recording & Playback     (gesture_dataset.py)
2. Adaptive Gesture Calibration             (gesture_calibration.py)
3. Multi-Cursor / Multi-Hand OS Control     (multi_cursor_controller.py)
4. Gesture Macros & Automation              (gesture_macros.py)
5. Cross-Platform Cursor Support            (cursor_platform.py)
6. Advanced Motion Analytics Dashboard      (advanced_features.py)
7. User Feedback & Haptics                  (advanced_features.py)
8. Gesture Confidence / Uncertainty Tracking (advanced_features.py)
9. Dynamic Frame Rate Adaptation             (advanced_features.py)
10. Cloud / Networked ML Models             (advanced_features.py)

================================================================================
USAGE EXAMPLE: COMPLETE SYSTEM WITH ALL FEATURES
================================================================================
"""

# Example main.py integration showing all 10 features:

import time
import cv2
from config import CursorAIConfig
from hand_tracker import HandTracker, HandTrackerConfig, HandLandmarks
from motion_analyzer import MotionAnalyzer, HandMotionInfo
from gesture_detector import GestureDetector, GestureResult

# Feature imports
from gesture_dataset import DatasetRecorder, DatasetManager, GestureDataset
from gesture_calibration import CalibrationMode, CalibrationManager, AdaptiveThresholdComputer
from multi_cursor_controller import MultiCursorController, CursorMode, WindowMapper
from gesture_macros import MacroExecutor, ProfileManager, ActionType, GestureAction, create_default_profiles
from cursor_platform import CrossPlatformCursorController
from advanced_features import (
    AnalyticsDashboard, FeedbackManager, FeedbackType, FeedbackEvent,
    ConfidenceTracker, AdaptiveFrameRateController, CloudInferenceClient
)


def example_complete_system():
    """
    Example: Complete system integration with all 10 features.

    Demonstrates:
    - Recording gestures and creating datasets
    - Calibrating for individual users
    - Controlling multiple cursors
    - Mapping gestures to actions
    - Cross-platform compatibility
    - Real-time analytics
    - Feedback during gestures
    - Confidence filtering
    - Adaptive FPS for performance
    - Cloud inference fallback
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    cfg = CursorAIConfig(
        max_num_hands=2,
        enable_motion_analysis=True,
        enable_gesture_detection=True,
        enable_two_hand_gestures=True,
    )

    # Core trackers
    tracker = HandTracker(HandTrackerConfig(
        max_num_hands=cfg.max_num_hands,
        mirror_image=cfg.mirror_image,
    ))
    motion_analyzer = MotionAnalyzer(history_size=cfg.motion_history_size)
    gesture_detector = GestureDetector()

    # ========================================================================
    # FEATURE 1: GESTURE DATASET RECORDING
    # ========================================================================

    dataset_recorder = DatasetRecorder()
    dataset_manager = DatasetManager(storage_dir="datasets")
    dataset = GestureDataset()

    # Later when user records:
    # dataset_recorder.start_recording("pinch", hand_used="left")
    # record_frame(hands, now)
    # record = dataset_recorder.stop_recording()
    # dataset.add_record(record)

    # ========================================================================
    # FEATURE 2: ADAPTIVE GESTURE CALIBRATION
    # ========================================================================

    calibration_mode = CalibrationMode()
    calibration_manager = CalibrationManager(profiles_dir="calibration_profiles")

    # Start calibration:
    # calibration_mode.start_calibration(user_id="user1")
    # During frames: calibration_mode.process_calibration_frame(hands, motions)
    # When done: profile = calibration_mode.create_profile()
    # calibration_manager.save_profile(profile)

    # Create adaptive threshold computer
    adaptive_computer = AdaptiveThresholdComputer(
        base_config={
            "pinch_distance_threshold": 0.05,
            "swipe_distance_threshold": 0.20,
            "finger_angle_threshold_deg": 160.0,
            "wrist_distance_margin": 0.015,
        }
    )

    # Apply calibration if exists
    user_profile = calibration_manager.load_profile("user1")
    if user_profile:
        adaptive_computer.set_profile(user_profile)
        adaptive_thresholds = adaptive_computer.compute_adaptive_thresholds()
        print(f"Adaptive thresholds: {adaptive_thresholds}")

    # ========================================================================
    # FEATURE 3: MULTI-CURSOR CONTROL
    # ========================================================================

    multi_cursor = MultiCursorController(mode=CursorMode.DUAL_INDEPENDENT)

    # Feature 5: Cross-Platform cursor
    platform_cursor = CrossPlatformCursorController()

    # ========================================================================
    # FEATURE 4: GESTURE MACROS & AUTOMATION
    # ========================================================================

    macro_executor = MacroExecutor()
    profile_manager = ProfileManager(profiles_dir="gesture_profiles")

    # Load default profiles
    default_profiles = create_default_profiles()
    for name, profile in default_profiles.items():
        profile_manager.save_profile(profile)

    # Load active profile
    active_profile = profile_manager.load_profile("browser")

    # ========================================================================
    # FEATURE 6: ADVANCED ANALYTICS
    # ========================================================================

    analytics_dashboard = AnalyticsDashboard()

    # ========================================================================
    # FEATURE 7: FEEDBACK SYSTEM
    # ========================================================================

    feedback_manager = FeedbackManager()

    # Example: Register custom feedback handlers
    def visual_highlight(event):
        """Draw highlight on gesture recognized."""
        print(f"[Feedback] Visual: {event.data.get('gesture')} ({event.intensity:.1f})")

    def haptic_buzz(event):
        """Trigger haptic feedback."""
        print(f"[Feedback] Haptic buzz: {event.intensity:.1f}")

    feedback_manager.register_feedback(FeedbackType.VISUAL_HIGHLIGHT, visual_highlight)
    feedback_manager.register_feedback(FeedbackType.HAPTIC_BUZZ, haptic_buzz)

    # ========================================================================
    # FEATURE 8: CONFIDENCE TRACKING
    # ========================================================================

    confidence_tracker = ConfidenceTracker(default_threshold=0.5)

    # ========================================================================
    # FEATURE 9: ADAPTIVE FRAME RATE
    # ========================================================================

    frame_rate_controller = AdaptiveFrameRateController(target_fps=30, min_fps=15, max_fps=60)

    # ========================================================================
    # FEATURE 10: CLOUD INFERENCE (Optional)
    # ========================================================================

    cloud_client = CloudInferenceClient(
        server_url="http://localhost:8000",  # Your ML server
        api_key="your-api-key"
    )
    # Try to connect (optional)
    cloud_client.connect()

    # ========================================================================
    # WINDOW MAPPING
    # ========================================================================

    window_mapper = WindowMapper(screen_width=1920, screen_height=1080)
    window_mapper.bind_gesture_to_region("left", "pinch", "open_file")
    window_mapper.bind_gesture_to_region("center", "swipe_right", "previous_page")
    window_mapper.bind_gesture_to_region("right", "swipe_left", "next_page")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Press 'q' to quit, 'r' to record gesture, 'c' for calibration, 'm' to cycle cursor mode")

    recording = False
    calibrating = False
    frame_count = 0

    try:
        while True:
            # Handle adaptive FPS
            if frame_rate_controller.should_skip_frame():
                continue

            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            # Hand detection
            processed_frame, hands = tracker.process(frame)

            now = time.time()
            motions = {}
            gestures = {}
            cursor_positions = {}

            # ================================================================
            # PER-HAND PROCESSING WITH ALL FEATURES
            # ================================================================

            for hand in hands:
                # Motion analysis
                motion_info = motion_analyzer.update_from_landmarks(
                    hand_id=hand.hand_id,
                    landmarks=hand.landmarks,
                    timestamp=now,
                )
                motions[hand.hand_id] = motion_info

                # Update analytics (Feature 6)
                # Extract velocity from motion_info...
                # analytics_dashboard.update_motion(f"hand_{hand.hand_id}_index", vx, vy, ax, ay, x, y)

                # Gesture detection with adaptive thresholds (Feature 2)
                gesture = gesture_detector.detect(
                    hand_id=hand.hand_id,
                    handedness=hand.handedness,
                    landmarks=hand.landmarks,
                    timestamp=now,
                    motion=motion_info,
                )
                gestures[hand.hand_id] = gesture

                # Track confidence (Feature 8)
                confidence_ok = confidence_tracker.record_confidence(
                    gesture.gesture,
                    gesture.confidence
                )

                if confidence_ok:
                    # Provide feedback (Feature 7)
                    feedback_manager.provide_gesture_feedback(
                        gesture.gesture,
                        gesture.confidence
                    )

                    # Record gesture if recording (Feature 1)
                    if recording:
                        dataset_recorder.record_frame(hands, now)

                    # Execute macro if mapped (Feature 4)
                    if active_profile:
                        actions = active_profile.get_actions_for_gesture(gesture.gesture)
                        for action in actions:
                            if macro_executor.execute_action(action):
                                print(f"[Macro] Executed: {action.action_type.value}")

                    # Multi-cursor control (Feature 3)
                    cursor_pos = multi_cursor.update_cursor(
                        hand_id=hand.hand_id,
                        handedness=hand.handedness,
                        x_norm=hand.index_finger_tip[0],
                        y_norm=hand.index_finger_tip[1],
                        timestamp=now,
                    )

                    # Cross-platform cursor move (Feature 5)
                    if cursor_pos:
                        platform_cursor.move_cursor(cursor_pos[0], cursor_pos[1])
                        cursor_positions[hand.hand_id] = cursor_pos

                    # Try cloud inference if available (Feature 10)
                    if cloud_client.enabled:
                        cloud_result = cloud_client.predict_gesture(hand.landmarks)
                        if cloud_result and cloud_result[1] > gesture.confidence:
                            gesture.gesture = cloud_result[0]
                            gesture.confidence = cloud_result[1]

            # ================================================================
            # CALIBRATION MODE
            # ================================================================

            if calibrating:
                step, instruction = calibration_mode.process_calibration_frame(hands, motions)
                if calibration_mode.is_complete():
                    profile = calibration_mode.create_profile()
                    calibration_manager.save_profile(profile)
                    calibrating = False
                    print(f"Calibration complete! Profile saved.")

            # ================================================================
            # VISUALIZATION
            # ================================================================

            # Draw gestures and analytics
            for hand in hands:
                if hand.hand_id in gestures:
                    gesture = gestures[hand.hand_id]
                    cv2.putText(
                        processed_frame,
                        f"{gesture.gesture} ({gesture.confidence:.2f})",
                        (10, 30 + hand.hand_id * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            # Draw virtual cursors (Feature 3)
            for vcursor in multi_cursor.get_all_virtual_cursors():
                if vcursor.is_visible:
                    cv2.circle(processed_frame, (int(vcursor.x), int(vcursor.y)), vcursor.radius, vcursor.color, -1)

            # Display session stats (Feature 6)
            stats = analytics_dashboard.get_session_statistics()
            cv2.putText(
                processed_frame,
                f"Gestures: {stats.get('total_gestures', 0)} | FPS: {frame_rate_controller.get_current_fps():.1f}",
                (10, processed_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            cv2.imshow("Advanced Cursor AI System", processed_frame)

            # ================================================================
            # FRAME RATE ADAPTATION (Feature 9)
            # ================================================================

            frame_time = (time.perf_counter() - frame_start) * 1000  # ms
            frame_rate_controller.record_frame_time(frame_time)
            frame_rate_controller.measure_cpu_load()
            adaptive_fps = frame_rate_controller.compute_adaptive_fps()

            frame_count += 1

            # ================================================================
            # KEYBOARD CONTROLS
            # ================================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                # Toggle recording
                if not recording:
                    dataset_recorder.start_recording("custom_gesture", hand_used="left")
                    recording = True
                    print("Recording started...")
                else:
                    record = dataset_recorder.stop_recording()
                    if record:
                        dataset.add_record(record)
                        print(f"Recording saved: {record.gesture_name}")
                    recording = False

            elif key == ord("c"):
                # Start calibration
                calibrating = True
                calibration_mode.start_calibration(user_id="user1")
                print("Calibration started...")

            elif key == ord("m"):
                # Cycle cursor mode
                mode_list = [
                    CursorMode.SINGLE_LEFT,
                    CursorMode.SINGLE_RIGHT,
                    CursorMode.DUAL_INDEPENDENT,
                    CursorMode.DUAL_COLLABORATIVE,
                ]
                current_idx = mode_list.index(multi_cursor.mode)
                next_idx = (current_idx + 1) % len(mode_list)
                multi_cursor.set_mode(mode_list[next_idx])

            elif key == ord("s"):
                # Save dataset
                dataset_manager.save_json(dataset, "my_dataset")
                print("Dataset saved!")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


# ============================================================================
# FEATURE CONFIGURATION REFERENCE
# ============================================================================

FEATURE_CONFIG_REFERENCE = """
CONFIGURATION OPTIONS FOR ALL 10 FEATURES

1. DATASET RECORDING
   - storage_dir: Where to save recorded gestures
   - max_dataset_size: Maximum gestures per dataset
   - auto_save: Auto-save recordings

2. ADAPTIVE CALIBRATION
   - calibration_mode: "auto" or "manual"
   - recalibrate_frequency: How often to recalibrate (frames)
   - hand_size_smoothing: Smoothing factor

3. MULTI-CURSOR
   - cursor_mode: "single_left", "single_right", "dual_independent", "dual_collaborative"
   - virtual_cursor_visible: Show virtual cursors
   - primary_hand: Which hand is primary

4. GESTURE MACROS
   - active_profile: Which gesture profile to use
   - macro_cooldown_ms: Minimum time between actions
   - enable_macros: Toggle macros on/off

5. CROSS-PLATFORM
   - platform_auto_detect: Auto-detect OS
   - use_xdotool: For Linux (if available)

6. ANALYTICS
   - analytics_enabled: Enable/disable dashboard
   - record_motion_history: Store motion data
   - analytics_buffer_size: How many frames to buffer

7. FEEDBACK
   - feedback_enabled: Enable/disable feedback
   - visual_feedback: Show visual responses
   - haptic_feedback: Enable haptics (if device supports)

8. CONFIDENCE
   - min_confidence_threshold: Minimum to accept gesture
   - filter_low_confidence: Ignore low-confidence gestures
   - per_gesture_thresholds: Customize per gesture

9. ADAPTIVE FPS
   - target_fps: Desired frame rate
   - min_fps: Minimum acceptable FPS
   - max_fps: Maximum FPS
   - cpu_load_threshold: When to reduce FPS

10. CLOUD INFERENCE
    - cloud_enabled: Enable remote ML
    - cloud_server_url: ML server endpoint
    - cloud_fallback: Fall back to local if cloud fails
    - cloud_timeout_ms: Request timeout
"""

# ============================================================================
# TUNING RECOMMENDATIONS
# ============================================================================

TUNING_GUIDE = """
TUNING & OPTIMIZATION GUIDE

GESTURE DETECTION THRESHOLDS:
- pinch_threshold: Increase for less sensitive pinch (default: 0.05)
- swipe_threshold: Increase for more deliberate swipes (default: 0.20)
- finger_angle: Decrease for more forgiving finger detection (default: 160°)

PERFORMANCE:
- Use adaptive FPS for variable load
- Disable cloud inference if latency > 100ms
- Lower motion_history_size for faster processing

ACCURACY:
- Run calibration for each user
- Use confidence filtering to ignore false positives
- Adjust per-gesture thresholds based on usage patterns

USER EXPERIENCE:
- Enable feedback for gesture recognition confirmation
- Use macro profiles for task-specific setups
- Record gestures for custom training sets
"""


if __name__ == "__main__":
    print(__doc__)
    print(FEATURE_CONFIG_REFERENCE)
    print(TUNING_GUIDE)
    # Uncomment to run the complete system:
    # example_complete_system()
