"""
EXAMPLE 2: ROBOTIC ARM CONTROL WITH GESTURES

Demonstrates gesture-controlled manipulation of a collaborative robot arm (UR5)
using inverse kinematics and hand motion mapping.

Features:
- Hand position → robot end-effector control
- Gesture-driven gripper open/close
- Inverse kinematics solver validation (ikpy)
- Safety checking: joint limits, reachability
- Real-time performance monitoring
- Trajectory smoothing for robotic motion

Usage:
  python example_robot_arm_control.py

Controls (simulated):
  - Move left hand: Robot end-effector follows
  - Pinch gesture: Close gripper
  - Open hand: Open gripper
  - Both hands: Two-hand commands (future)
"""

import sys
import time
import numpy as np
from typing import Dict, List, Optional

# Try imports
try:
    from robot_controller import RobotArmUR5, HandToArmMapper, RobotTarget, GripperCommand
    ROBOT_MODULES_AVAILABLE = True
except ImportError:
    ROBOT_MODULES_AVAILABLE = False
    print("[Example] Robot modules not available - ensure robot_controller.py in path")

try:
    from camera_calibration import CameraCalibrator, CoordinateTransformer
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    print("[Example] Calibration module not available")


def simulate_hand_position(frame: int, gesture_type: str) -> Dict:
    """Simulate hand motion for demo."""
    t = frame * 0.033  # 30 fps
    import math

    if gesture_type == "reach_forward":
        # Extended reach forward
        x = 0.5 + 0.25 * (frame % 60) / 60.0
        y = 0.5
        z = 0.5
        return {
            "position": (x, y, z),
            "gesture": "reach",
            "confidence": 0.85
        }

    elif gesture_type == "circular_motion":
        # Circular end-effector motion
        radius = 0.15
        x = 0.6 + radius * math.cos(t * 1.5)
        y = 0.5 + radius * math.sin(t * 1.5)
        z = 0.5
        return {
            "position": (x, y, z),
            "gesture": "circle",
            "confidence": 0.8
        }

    elif gesture_type == "grasp_sequence":
        # Motion to grasp, close gripper, retract
        phase = (frame % 90) // 30
        if phase == 0:  # Move to object
            x = 0.4 + 0.1 * (frame % 30) / 30.0
            y = 0.3 + 0.1 * (frame % 30) / 30.0
            z = 0.5
            gesture = "reach"
        elif phase == 1:  # Grasp (pinch)
            x = 0.5
            y = 0.4
            z = 0.5
            gesture = "pinch"
        else:  # Retract
            x = 0.4 - 0.1 * ((frame % 30) / 30.0)
            y = 0.3 - 0.1 * ((frame % 30) / 30.0)
            z = 0.5
            gesture = "open"

        return {
            "position": (x, y, z),
            "gesture": gesture,
            "confidence": 0.82
        }

    return {"position": (0.5, 0.5, 0.5), "gesture": "neutral", "confidence": 0.5}


def check_reachability(robot: 'RobotArmUR5', position: np.ndarray) -> bool:
    """Check if position is within robot's reachable workspace."""
    # UR5 reachability: roughly 850mm reach from base
    # Workspace: x:[-0.85, 0.85], y:[-0.85, 0.85], z:[0.1, 1.0] (rough)
    dist_xy = np.sqrt(position[0]**2 + position[1]**2)
    return (dist_xy < 0.85 and 0.1 < position[2] < 1.0)


def example_robot_arm_control():
    """Run UR5 robot arm control example."""
    if not ROBOT_MODULES_AVAILABLE:
        print("[Example] Cannot run without robot modules")
        return

    print("\n" + "="*70)
    print("EXAMPLE 2: ROBOTIC ARM CONTROL WITH GESTURES")
    print("="*70)

    # Initialize robot
    print("\n[Initialization] Setting up UR5 robot...")
    robot = RobotArmUR5(ip_address="192.168.1.100", port=30003)
    if not robot.connect():
        print("[Example] Robot connection failed (OK for simulation)")

    # Initialize hand-to-arm mapper
    print("[Initialization] Creating hand-to-arm mapper...")
    mapper = HandToArmMapper(position_scale=1.0, max_velocity=0.5)

    # Camera calibration (optional, for 3D projection)
    transformer = None
    if CALIBRATION_AVAILABLE:
        print("[Initialization] Loading camera calibration...")
        calibration = CameraCalibrator.estimate_from_image_size(1920, 1080, fov_degrees=60.0)
        transformer = CoordinateTransformer(calibration, depth_scale=1.0)
        print(f"  Camera: {calibration.width}x{calibration.height}, FOV=60°")

    # Simulation parameters
    num_frames = 300  # 10 seconds @ 30fps
    gesture_sequence = ["reach_forward", "circular_motion", "grasp_sequence", "reach_forward"]
    current_gesture_idx = 0
    gesture_frames_per = num_frames // len(gesture_sequence)

    # Statistics
    stats = {
        "frames_processed": 0,
        "total_latencies": [],
        "ik_solutions": 0,
        "ik_failures": 0,
        "reachable": 0,
        "unreachable": 0,
        "gripper_commands": {"open": 0, "close": 0},
        "joint_limit_violations": 0,
    }

    print("\n[Running] Simulating UR5 robot control...")
    print(f"  Duration: {num_frames * 0.033:.1f}s ({num_frames} frames)")
    print(f"  Gesture sequence: {' → '.join(gesture_sequence)}\n")

    # Main simulation loop
    for frame in range(num_frames):
        frame_start = time.perf_counter()

        # Determine current gesture
        gesture_idx = (frame // gesture_frames_per) % len(gesture_sequence)
        current_gesture = gesture_sequence[gesture_idx]

        # Simulate hand input
        hand_data = simulate_hand_position(frame % gesture_frames_per, current_gesture)
        hand_pos = np.array(hand_data["position"])

        # Transform to 3D world if calibration available
        if transformer:
            hand_3d = transformer.normalized_to_3d(hand_pos[0], hand_pos[1], hand_pos[2])
            hand_pos_world = np.array(hand_3d)
        else:
            hand_pos_world = hand_pos * 0.8  # Scale normalized to approximate world coords

        # Check reachability
        is_reachable = check_reachability(robot, hand_pos_world)
        if is_reachable:
            stats["reachable"] += 1
        else:
            stats["unreachable"] += 1

        # Create mock hand object for mapper
        class MockHand:
            def __init__(self, pos_world, confidence):
                # 21 landmarks: index tip is landmark 8
                # Pad with zeros for other joints
                self.landmarks = [(0, 0, 0)] * 8 + [pos_world] + [(0, 0, 0)] * 12
                self.landmarks_3d = self.landmarks
                self.handedness = "Left"

        left_hand = MockHand(hand_pos_world, hand_data["confidence"])

        # Map hand to robot target
        target = mapper.map_hand_to_arm_target(left_hand, None)

        if target and is_reachable:
            # Smooth trajectory
            target = mapper.smooth_trajectory(target, dt=0.033)

            # Attempt movement
            success = robot.move_to_position(target)

            if success:
                stats["ik_solutions"] += 1

                # Track gripper commands
                if target.gripper_command:
                    cmd_str = target.gripper_command.value
                    if cmd_str not in stats["gripper_commands"]:
                        stats["gripper_commands"][cmd_str] = 0
                    stats["gripper_commands"][cmd_str] += 1
            else:
                stats["ik_failures"] += 1
        elif not is_reachable:
            stats["ik_failures"] += 1

        # Record statistics
        latency = (time.perf_counter() - frame_start) * 1000
        stats["frames_processed"] = frame + 1
        stats["total_latencies"].append(latency)

        # Progress output
        if frame % 60 == 0:
            avg_lat = np.mean(stats["total_latencies"][-60:])
            reach_pct = 100 * stats["reachable"] / max(1, frame + 1)
            print(f"  Frame {frame:3d}: Gesture={current_gesture:18} Reach={reach_pct:5.1f}% "
                  f"IK_OK={stats['ik_solutions']:3d} IK_Fail={stats['ik_failures']:3d} "
                  f"Latency={avg_lat:5.2f}ms")

    # Print results
    print("\n[Results Summary]")
    print("="*70)
    print(f"Total frames processed: {stats['frames_processed']}")
    print(f"Average latency: {np.mean(stats['total_latencies']):.2f}ms")
    print(f"Min/Max latency: {np.min(stats['total_latencies']):.2f}ms / {np.max(stats['total_latencies']):.2f}ms")
    print(f"FPS achieved: {1000 / np.mean(stats['total_latencies']):.1f}")

    print(f"\n[Reachability Analysis]")
    print(f"  Reachable targets: {stats['reachable']} ({100*stats['reachable']/stats['frames_processed']:.1f}%)")
    print(f"  Unreachable targets: {stats['unreachable']} ({100*stats['unreachable']/stats['frames_processed']:.1f}%)")

    print(f"\n[IK Solver Performance]")
    print(f"  Successful IK solutions: {stats['ik_solutions']}")
    print(f"  IK failures/unreachable: {stats['ik_failures']}")
    ik_success_rate = 100 * stats['ik_solutions'] / max(1, stats['ik_solutions'] + stats['ik_failures'])
    print(f"  Success rate: {ik_success_rate:.1f}%")

    print(f"\n[Gripper Commands]")
    for cmd, count in stats["gripper_commands"].items():
        print(f"  {cmd:10} : {count:3d} times")

    # Final robot state
    print(f"\n[Final Robot State]")
    final_state = robot.get_current_state()
    print(f"  Joint angles: {[f'{j:.2f}' for j in final_state.joint_angles]}")
    print(f"  TCP position: ({final_state.tcp_position[0]:.3f}, "
          f"{final_state.tcp_position[1]:.3f}, {final_state.tcp_position[2]:.3f})")
    print(f"  Gripper: {'OPEN' if final_state.gripper_open else 'CLOSED'}")

    print("\n[Complete] UR5 robot control example finished")


if __name__ == "__main__":
    example_robot_arm_control()
