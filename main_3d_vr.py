"""
MAIN 3D/AR/VR ORCHESTRATION PIPELINE

Integrates all 5 core 3D/AR/VR modules into a unified, configurable pipeline.

Orchestrates:
- Camera calibration and 2D→3D coordinate transformation
- Hand tracking and 3D gesture detection
- 3D Kalman filtering for smooth motion
- Robot arm control or 3D object manipulation
- Unreal Engine 5 skeletal animation
- Real-time analytics and profiling

Features:
- Clean separation of concerns (each module independent)
- Flexible feature activation via config
- Real-time performance monitoring
- Graceful fallbacks and error handling
- Keyboard controls for runtime configuration
"""

from __future__ import annotations

import time
import sys
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path

# Try imports with graceful fallbacks
try:
    from config import CursorAIConfig
    from hand_tracker import HandTracker, HandTrackerConfig
    from motion_analyzer import MotionAnalyzer
    from gesture_detector import GestureDetector
    from cursor_controller_3d import CursorController3D
    from visualizer import Visualizer
    print("[main_3d_vr] Core modules loaded successfully")
except ImportError as e:
    print(f"[main_3d_vr] Core modules not available: {e}")
    sys.exit(1)

# Optional 3D/AR/VR modules
try:
    from camera_calibration import CameraCalibrator, CoordinateTransformer, DepthEstimator
    CAMERA_CALIB_AVAILABLE = True
except ImportError:
    CAMERA_CALIB_AVAILABLE = False
    print("[main_3d_vr] Camera calibration not available")

try:
    from robot_controller import RobotArmUR5, HandToArmMapper, RobotSafetyLimits
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("[main_3d_vr] Robot control not available")

try:
    from object_manipulator import ObjectManipulator, Object3D, BasicPhysics
    OBJECT_MANIP_AVAILABLE = True
except ImportError:
    OBJECT_MANIP_AVAILABLE = False
    print("[main_3d_vr] Object manipulation not available")

try:
    from unreal_bridge import (
        UnrealPythonAPIBridge,
        UnrealWebSocketBridge,
        HandSkeletonConverter,
        SkeletalMeshUpdate,
        UnrealFeedbackEvent,
    )
    UNREAL_AVAILABLE = True
except ImportError:
    UNREAL_AVAILABLE = False
    print("[main_3d_vr] Unreal Engine bridge not available")

try:
    from control_intelligence import ControlMode, IntelligentControlLayer
    CONTROL_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONTROL_INTELLIGENCE_AVAILABLE = False
    print("[main_3d_vr] Control intelligence layer not available")
    from enum import Enum

    class ControlMode(Enum):
        REAL_INTERFACE = "real_interface"
        VIRTUAL_EXECUTION = "virtual_execution"


# ============================================================================
# 3D VR PIPELINE STATE
# ============================================================================

class Pipeline3DState:
    """Manages state of 3D pipeline."""

    def __init__(self):
        """Initialize pipeline state."""
        self.running = True
        self.profiling_enabled = False
        self.robot_enabled = False
        self.object_manipulation_enabled = False
        self.unreal_enabled = False
        self.frame_count = 0
        self.total_latency_ms = 0.0
        self.fps_history: List[float] = []
        self.control_mode: ControlMode = ControlMode.REAL_INTERFACE
        self.control_context: str = "default"
        self.last_action: Optional[str] = None

    def update_fps(self, fps: float) -> None:
        """Update FPS history."""
        self.fps_history.append(fps)
        if len(self.fps_history) > 300:
            self.fps_history.pop(0)

    def get_average_fps(self) -> float:
        """Get average FPS over history."""
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

    def print_status(self) -> None:
        """Print pipeline status."""
        avg_fps = self.get_average_fps()
        print(f"\n[Pipeline] Frame {self.frame_count} | FPS: {avg_fps:.1f} | Latency: {self.total_latency_ms:.1f}ms")
        print(f"  Profiling: {self.profiling_enabled} | Robot: {self.robot_enabled} | "
              f"Objects: {self.object_manipulation_enabled} | UE5: {self.unreal_enabled}")
        print(f"  Mode: {self.control_mode.value} | Context: {self.control_context} | Last action: {self.last_action}")


# ============================================================================
# MAIN 3D PIPELINE
# ============================================================================

def main_3d_vr(cfg: Optional[CursorAIConfig] = None) -> None:
    """
    Main 3D/AR/VR pipeline orchestration.

    Pipeline flow:
    1. Initialize all modules (with feature flags)
    2. Capture frames and detect hands
    3. Transform to 3D world coordinates
    4. Apply 3D gesture recognition
    5. Route to robot/objects/UE5 based on configuration
    6. Update physics and visualizations
    7. Display real-time feedback
    """

    print("\n" + "="*70)
    print("GESTURE DIGITAL TWIN - 3D/AR/VR PIPELINE")
    print("="*70)

    # Load configuration
    cfg = cfg or CursorAIConfig()
    state = Pipeline3DState()
    state.control_mode = (
        ControlMode.VIRTUAL_EXECUTION
        if getattr(cfg, "control_mode", "real_interface").lower() == "virtual_execution"
        else ControlMode.REAL_INTERFACE
    )
    state.control_context = getattr(cfg, "control_context", "default")

    # Module availability check
    print("\n[Initialization] Checking module availability...")
    print(f"  Camera Calibration: {'✓' if CAMERA_CALIB_AVAILABLE else '✗'}")
    print(f"  Robot Control: {'✓' if ROBOT_AVAILABLE else '✗'}")
    print(f"  Object Manipulation: {'✓' if OBJECT_MANIP_AVAILABLE else '✗'}")
    print(f"  Unreal Engine: {'✓' if UNREAL_AVAILABLE else '✗'}")
    print(f"  Control Intelligence: {'✓' if CONTROL_INTELLIGENCE_AVAILABLE else '✗'}")

    # Initialize core modules
    print("\n[Initialization] Loading core modules...")
    tracker_cfg = HandTrackerConfig(
        max_num_hands=cfg.max_num_hands,
        mirror_image=cfg.mirror_image,
        enable_cursor_control=cfg.enable_cursor_control,
    )
    tracker = HandTracker(tracker_cfg)
    motion_analyzer = MotionAnalyzer(history_size=cfg.motion_history_size)
    gesture_detector = GestureDetector(
        pinch_distance_threshold=cfg.pinch_distance_threshold,
        swipe_distance_threshold=cfg.swipe_distance_threshold,
    )
    cursor_3d = CursorController3D(
        use_enhanced_kalman=cfg.use_enhanced_kalman,
        adaptive_smoothing=cfg.adaptive_smoothing,
    )
    visualizer = Visualizer()
    control_layer: Optional[IntelligentControlLayer] = None
    if CONTROL_INTELLIGENCE_AVAILABLE and getattr(cfg, "enable_control_intelligence", True):
        control_layer = IntelligentControlLayer(
            smoothing_alpha=getattr(cfg, "stability_smoothing_alpha", 0.35),
            deadzone=getattr(cfg, "stability_deadzone", 0.002),
            base_latency_ms=getattr(cfg, "base_pipeline_latency_ms", 35.0),
            enable_latency_compensation=getattr(cfg, "enable_latency_compensation", True),
            max_command_rate_hz=getattr(cfg, "max_command_rate_hz", 20.0),
        )
        print("[Initialization] Control intelligence enabled")

    # Initialize optional 3D modules
    transformer: Optional[CoordinateTransformer] = None
    robot: Optional[RobotArmUR5] = None
    robot_mapper: Optional[HandToArmMapper] = None
    manipulator: Optional[ObjectManipulator] = None
    unreal_bridge = None

    if CAMERA_CALIB_AVAILABLE:
        print("[Initialization] Loading camera calibration...")
        try:
            calibration = CameraCalibrator.load_calibration("calibration.json")
            transformer = CoordinateTransformer(calibration, depth_scale=1.0)
            print("  ✓ Camera calibration loaded")
        except FileNotFoundError:
            print("  ! Calibration file not found - using estimated")
            calibration = CameraCalibrator.estimate_from_image_size(1920, 1080)
            transformer = CoordinateTransformer(calibration)

    if ROBOT_AVAILABLE and cfg.enable_robot_control:
        print("[Initialization] Initializing robot control...")
        try:
            robot = RobotArmUR5(cfg.robot_ip_address if hasattr(cfg, 'robot_ip_address') else "192.168.1.100")
            if robot.connect():
                robot_mapper = HandToArmMapper()
                robot_mapper.max_velocity = float(getattr(cfg, "robot_max_velocity", robot_mapper.max_velocity))
                robot.set_safety_limits(
                    RobotSafetyLimits(
                        workspace_min=np.array(getattr(cfg, "robot_workspace_min_xyz", (0.10, -0.80, 0.10))),
                        workspace_max=np.array(getattr(cfg, "robot_workspace_max_xyz", (0.85, 0.80, 1.00))),
                        max_step_m=float(getattr(cfg, "robot_max_step_m", 0.05)),
                        max_velocity_mps=float(getattr(cfg, "robot_max_velocity", 0.5)),
                        stop_on_out_of_workspace=bool(getattr(cfg, "robot_stop_on_out_of_workspace", True)),
                    )
                )
                state.robot_enabled = True
                print("  ✓ Robot connected")
            else:
                print("  ! Robot connection failed - disabling")
                robot = None
        except Exception as e:
            print(f"  ! Robot initialization error: {e}")

    if OBJECT_MANIP_AVAILABLE and getattr(cfg, "enable_3d_world", False):
        print("[Initialization] Initializing object manipulation...")
        try:
            manipulator = ObjectManipulator()
            # Create sample objects
            cube = Object3D(id="cube", position=np.array([0.5, 0.5, 0.5]))
            manipulator.add_object(cube)
            state.object_manipulation_enabled = True
            print("  ✓ Object manipulation initialized")
        except Exception as e:
            print(f"  ! Object manipulation error: {e}")

    if UNREAL_AVAILABLE and hasattr(cfg, 'enable_ue5_integration') and cfg.enable_ue5_integration:
        print("[Initialization] Initializing Unreal Engine bridge...")
        try:
            if getattr(cfg, "unreal_use_websocket", False):
                endpoint = getattr(cfg, "unreal_websocket_endpoint", "ws://localhost:8765")
                unreal_bridge = UnrealWebSocketBridge(endpoint)
                unreal_bridge.allow_short_lived_fallback = bool(
                    getattr(cfg, "unreal_allow_short_lived_fallback", False)
                )
            else:
                unreal_bridge = UnrealPythonAPIBridge(getattr(cfg, "unreal_project_path", ""))
            if unreal_bridge.connect():
                state.unreal_enabled = True
                print("  ✓ Unreal Engine connected")
            else:
                print("  ! Unreal Engine connection failed - disabling")
                unreal_bridge = None
        except Exception as e:
            print(f"  ! Unreal Engine error: {e}")

    # Open webcam
    print("\n[Initialization] Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    print("\n[Ready] Pipeline ready. Keyboard controls:")
    print("  Q - Quit")
    print("  P - Toggle profiling")
    print("  R - Toggle robot control")
    print("  O - Toggle object manipulation")
    print("  U - Toggle Unreal Engine")
    print("  V - Toggle mode (real_interface / virtual_execution)")
    print("  C - Cycle control context")
    print("  S - Print status")

    prev_time = time.time()

    # Main loop
    try:
        while state.running:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            state.frame_count += 1

            # Hand detection
            processed_frame, hands = tracker.process(frame)

            # Transform to 3D if calibrated
            hands_3d = None
            if transformer and hands:
                hands_3d = []
                for hand in hands:
                    # Store original hand data
                    hand_3d = hand
                    # Add 3D landmarks
                    if hasattr(hand, 'landmarks'):
                        landmarks_3d = transformer.transform_landmarks(hand.landmarks)
                        hand_3d.landmarks_3d = landmarks_3d
                    hands_3d.append(hand_3d)

            now = time.time()

            # Motion analysis / gesture detection / cursor control
            motions: Dict[int, object] = {}
            gestures: Dict[int, object] = {}
            cursor_positions: Dict[int, Tuple[float, float, float]] = {}

            for hand in hands:
                if cfg.enable_motion_analysis:
                    motion_info = motion_analyzer.update_from_landmarks(
                        hand_id=hand.hand_id,
                        landmarks=hand.landmarks,
                        timestamp=now,
                    )
                    motions[hand.hand_id] = motion_info

                if cfg.enable_gesture_detection:
                    gesture = gesture_detector.detect(
                        hand_id=hand.hand_id,
                        handedness=hand.handedness,
                        landmarks=hand.landmarks,
                        timestamp=now,
                        motion=motions.get(hand.hand_id),
                    )
                    gestures[hand.hand_id] = gesture

                raw_norm = np.array(
                    [
                        hand.index_finger_tip[0],
                        hand.index_finger_tip[1],
                        float(hand.landmarks[8].z) if len(hand.landmarks) > 8 and hasattr(hand.landmarks[8], "z") else 0.5,
                    ],
                    dtype=np.float64,
                )
                filtered_norm = (
                    control_layer.filter_and_predict(hand.hand_id, raw_norm, now)
                    if control_layer
                    else raw_norm
                )

                cursor_pos = cursor_3d.update_cursor_3d(
                    hand_id=hand.hand_id,
                    x_norm=float(filtered_norm[0]),
                    y_norm=float(filtered_norm[1]),
                    z_norm=float(filtered_norm[2]),
                    timestamp=now,
                )
                # Visualizer expects 2D cursor coordinates.
                cursor_positions[hand.hand_id] = (cursor_pos[0], cursor_pos[1])

            # Two-hand gesture for stateful interpretation.
            two_hand_gesture = (
                gesture_detector.detect_two_hand_gesture(hands, motions)
                if cfg.enable_two_hand_gestures
                else None
            )

            first_gesture = list(gestures.values())[0] if gestures else None
            single_label = first_gesture.gesture if first_gesture else None
            two_label = two_hand_gesture.gesture if two_hand_gesture else None

            action = None
            if control_layer:
                runtime_context = state.control_context
                if runtime_context == "default":
                    if state.robot_enabled:
                        runtime_context = "robot"
                    elif state.object_manipulation_enabled:
                        runtime_context = "objects"
                    else:
                        runtime_context = "ui"
                action = control_layer.resolve_action(
                    single_gesture=single_label,
                    two_hand_gesture=two_label,
                    mode=state.control_mode,
                    context=runtime_context,
                    timestamp=now,
                )
            else:
                action = two_label if two_label else single_label
            state.last_action = action

            can_execute = control_layer.should_execute(state.control_mode) if control_layer else True
            if control_layer and action and not can_execute:
                control_layer.feedback(
                    "interface_preview",
                    "info",
                    f"Preview action in interface mode: {action}",
                )

            # Route to robot
            if can_execute and state.robot_enabled and robot and robot_mapper and hands_3d and len(hands_3d) > 0:
                try:
                    left_hand = next((h for h in hands_3d if h.handedness.lower() == "left"), hands_3d[0])
                    right_hand = next((h for h in hands_3d if h.handedness.lower() == "right"), None)
                    target = robot_mapper.map_hand_to_arm_target(
                        left_hand,
                        right_hand,
                        gestures.get(left_hand.hand_id),
                    )
                    if target:
                        if action in ("gripper_close", "object_grab"):
                            from robot_controller import GripperCommand

                            target.gripper_command = GripperCommand.CLOSE
                        elif action in ("gripper_open", "object_release"):
                            from robot_controller import GripperCommand

                            target.gripper_command = GripperCommand.OPEN

                        target = robot_mapper.smooth_trajectory(target, dt=0.033)
                        move_ok = robot.move_to_position(target)
                        if control_layer and not move_ok:
                            control_layer.feedback("robot_move_failed", "warn", "Robot move command failed")

                        robot_state = robot.get_current_state()
                        if control_layer and any(abs(j) > 0.95 * np.pi for j in robot_state.joint_angles):
                            control_layer.feedback("joint_limit_near", "warn", "Robot joint near limit")
                except Exception as e:
                    if control_layer:
                        control_layer.feedback("robot_error", "error", str(e))
                    if state.profiling_enabled:
                        print(f"[Robot] Error: {e}")

            # Route to object manipulation
            if can_execute and state.object_manipulation_enabled and manipulator and hands_3d:
                try:
                    hand_pos = np.array(
                        hands_3d[0].landmarks_3d[8] if hasattr(hands_3d[0], "landmarks_3d") else hands_3d[0].index_finger_tip
                    )
                    manipulator.select_object_at(hand_pos)

                    gesture_for_objects = single_label
                    action_to_gesture = {
                        "object_grab": "pinch",
                        "object_release": "open",
                        "rotate_object": "circle",
                        "move_left": "swipe_left",
                        "move_right": "swipe_right",
                        "move_up": "swipe_up",
                        "move_down": "swipe_down",
                    }
                    if action in action_to_gesture:
                        gesture_for_objects = action_to_gesture[action]

                    if gesture_for_objects:
                        manipulator.apply_gesture(
                            gesture_for_objects,
                            hand_position=hand_pos,
                            two_hand_gesture=two_label,
                        )
                    manipulator.update_hand_position(hand_pos, hold_fixed=True)
                    manipulator.update_physics(dt=0.033, damping=getattr(cfg, "physics_damping", 0.95))

                    # Mirror manipulated object transform into Unreal actor.
                    if state.unreal_enabled and unreal_bridge:
                        target_obj = manipulator.selected_object
                        if target_obj is None and manipulator.world_objects:
                            target_obj = next(iter(manipulator.world_objects.values()))
                        if target_obj is not None:
                            actor_name = getattr(cfg, "unreal_object_actor_name", "ManipulatedObject")
                            pos = tuple(float(v) for v in target_obj.position)
                            rot_deg = tuple(float(v) for v in np.degrees(target_obj.rotation))
                            scale = tuple(float(v) for v in target_obj.scale)
                            moved = unreal_bridge.move_actor(actor_name, pos, rot_deg)
                            scaled = unreal_bridge.set_actor_scale(actor_name, scale)
                            if control_layer and (not moved or not scaled):
                                control_layer.feedback(
                                    "unreal_object_sync_failed",
                                    "warn",
                                    f"UE object sync failed for actor '{actor_name}'",
                                )

                    # Feedback: detect collisions in scene.
                    if control_layer:
                        objects = list(manipulator.world_objects.values())
                        for i in range(len(objects)):
                            for j in range(i + 1, len(objects)):
                                if BasicPhysics.check_collision_aabb(objects[i], objects[j]):
                                    control_layer.feedback(
                                        "object_collision",
                                        "info",
                                        f"Collision: {objects[i].id} <-> {objects[j].id}",
                                    )
                except Exception as e:
                    if control_layer:
                        control_layer.feedback("object_error", "error", str(e))
                    if state.profiling_enabled:
                        print(f"[Objects] Error: {e}")

            # Route to Unreal Engine
            if can_execute and state.unreal_enabled and unreal_bridge and hands_3d:
                try:
                    for hand in hands_3d:
                        if hasattr(hand, "landmarks_3d"):
                            transforms = HandSkeletonConverter.landmarks_to_bone_transforms(
                                hand.landmarks_3d,
                                hand_side=hand.handedness,
                            )
                            update = SkeletalMeshUpdate(
                                hand_id=hand.hand_id,
                                hand_side=hand.handedness,
                                bone_transforms=transforms,
                                timestamp=now,
                            )
                            ok = unreal_bridge.send_skeletal_update(update)
                            if control_layer and not ok:
                                control_layer.feedback("unreal_send_failed", "warn", "UE skeletal update failed")
                except Exception as e:
                    if control_layer:
                        control_layer.feedback("unreal_error", "error", str(e))
                    if state.profiling_enabled:
                        print(f"[Unreal] Error: {e}")

            # Bidirectional feedback loop: control layer <-> Unreal bridge.
            if state.unreal_enabled and unreal_bridge and control_layer:
                # Push local control/safety feedback to Unreal.
                while control_layer.feedback_events:
                    evt = control_layer.feedback_events.popleft()
                    unreal_bridge.send_feedback_event(
                        UnrealFeedbackEvent(
                            event_type=evt.event,
                            severity=evt.severity,
                            message=evt.message,
                            data=evt.data,
                            timestamp=evt.timestamp,
                        )
                    )

                # Pull Unreal-side feedback (if any) back into operator console.
                incoming_events = unreal_bridge.poll_feedback_events()
                for incoming in incoming_events:
                    if incoming.severity.lower() in ("warn", "warning", "error"):
                        print(f"[UE Feedback] {incoming.severity.upper()}: {incoming.message}")

            # Visualization
            annotated = visualizer.draw(
                frame_bgr=processed_frame,
                hands=hands,
                gestures=gestures,
                cursor_positions=cursor_positions,
                motions=motions,
                fps=1.0 / (now - prev_time) if now > prev_time else 0,
                two_hand_gesture=two_hand_gesture,
            )

            cv2.imshow("3D/AR/VR Pipeline", annotated)

            # Update timing
            frame_time = time.perf_counter() - frame_start
            state.total_latency_ms = frame_time * 1000
            if control_layer:
                control_layer.register_frame_latency(state.total_latency_ms)
            state.update_fps(1.0 / (now - prev_time) if now > prev_time else 0)
            prev_time = now

            # Profiling output
            if state.profiling_enabled and state.frame_count % 30 == 0:
                state.print_status()

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                state.running = False
            elif key == ord('p'):
                state.profiling_enabled = not state.profiling_enabled
                print(f"[Pipeline] Profiling: {state.profiling_enabled}")
            elif key == ord('r') and robot:
                state.robot_enabled = not state.robot_enabled
                print(f"[Pipeline] Robot: {state.robot_enabled}")
            elif key == ord('o') and manipulator:
                state.object_manipulation_enabled = not state.object_manipulation_enabled
                print(f"[Pipeline] Objects: {state.object_manipulation_enabled}")
            elif key == ord('u') and unreal_bridge:
                state.unreal_enabled = not state.unreal_enabled
                print(f"[Pipeline] Unreal: {state.unreal_enabled}")
            elif key == ord('v'):
                state.control_mode = (
                    ControlMode.REAL_INTERFACE
                    if state.control_mode == ControlMode.VIRTUAL_EXECUTION
                    else ControlMode.VIRTUAL_EXECUTION
                )
                print(f"[Pipeline] Mode: {state.control_mode.value}")
            elif key == ord('c'):
                contexts = ["default", "robot", "objects", "ui"]
                try:
                    idx = contexts.index(state.control_context)
                except ValueError:
                    idx = 0
                state.control_context = contexts[(idx + 1) % len(contexts)]
                print(f"[Pipeline] Context: {state.control_context}")
            elif key == ord('s'):
                state.print_status()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        if robot:
            robot.stop()
        if unreal_bridge:
            unreal_bridge.disconnect()

        print("\n[Pipeline] Exited gracefully")
        state.print_status()


if __name__ == "__main__":
    main_3d_vr()
