"""
EXAMPLE 3: UNREAL ENGINE 5 INTEGRATION WITH HAND SKELETON ANIMATION

Demonstrates real-time skeletal mesh animation controlled by hand gestures.

Features:
- Direct UE5 Python API connection (with WebSocket fallback)
- 21-bone hand skeleton animation (MediaPipe → UE5 skeleton mapping)
- Gesture-driven gripper control
- Real-time bone transform updates
- Bridge latency monitoring
- Dual-hand skeleton rendering (left + right hands)
- Performance profiling (FPS, bone updates/frame, network latency)

Usage:
  python example_unreal_integration.py

Requirements:
  - Unreal Engine 5.0+ with Python plugin enabled
  - (or) WebSocket endpoint at ws://localhost:8765 for remote UE5

Controls (simulated):
  - Hand reach gesture: Bones animate in sequence
  - Grab/pinch gesture: Gripper closes
  - Open hand gesture: Gripper opens
"""

import sys
import time
import numpy as np
from typing import Dict, List, Optional
import json

# Try imports
try:
    from unreal_bridge import (
        UnrealPythonAPIBridge,
        UnrealWebSocketBridge,
        SkeletalMeshUpdate,
        BoneTransform,
        HandSkeletonConverter,
        LANDMARK_TO_BONE_NAME
    )
    UNREAL_MODULES_AVAILABLE = True
except ImportError:
    UNREAL_MODULES_AVAILABLE = False
    print("[Example] Unreal modules not available - ensure unreal_bridge.py in path")


def simulate_hand_landmarks(frame: int, gesture_type: str) -> Dict:
    """Simulate 21 hand landmarks for MediaPipe skeleton."""
    t = frame * 0.033  # 30 fps
    import math

    # Base hand position (wrist)
    wrist_base = np.array([0.5, 0.5, 0.5])

    if gesture_type == "reach_forward":
        # Hand reaches forward
        offset = np.array([0.3 * ((frame % 60) / 60.0), 0.0, 0.0])
        wrist = wrist_base + offset

    elif gesture_type == "wave":
        # Hand waves side to side
        offset = np.array([
            0.2 * math.sin(t * 2.0),
            0.1 * math.cos(t * 3.0),
            0.05 * math.sin(t * 1.5)
        ])
        wrist = wrist_base + offset

    elif gesture_type == "grab_and_release":
        # Sequence: reach, grab (pinch), release
        phase = (frame % 90) // 30
        if phase == 0:  # Reach
            offset = np.array([0.2, 0.0, 0.0])
            gesture = "reach"
        elif phase == 1:  # Grab
            offset = np.array([0.2, 0.0, 0.0])
            gesture = "pinch"
        else:  # Release
            offset = np.array([0.1, 0.0, 0.0])
            gesture = "open"
        wrist = wrist_base + offset

    else:
        wrist = wrist_base
        gesture = "neutral"

    # Generate 21 landmarks: wrist (0) + 5 fingers * 4 joints (1-20)
    # Simple model: each finger extends outward from wrist
    landmarks_3d = [tuple(wrist)]  # Landmark 0: wrist

    # Thumb (landmarks 1-4)
    for i in range(1, 5):
        tip_offset = np.array([0.05, -0.08, 0.02]) * (i / 4.0)
        landmarks_3d.append(tuple(wrist + tip_offset))

    # Index (landmarks 5-8)
    for i in range(5, 9):
        tip_offset = np.array([0.08, 0.01, 0.02]) * ((i - 4) / 4.0)
        landmarks_3d.append(tuple(wrist + tip_offset))

    # Middle (landmarks 9-12)
    for i in range(9, 13):
        tip_offset = np.array([0.08, 0.04, 0.02]) * ((i - 8) / 4.0)
        landmarks_3d.append(tuple(wrist + tip_offset))

    # Ring (landmarks 13-16)
    for i in range(13, 17):
        tip_offset = np.array([0.07, 0.07, 0.01]) * ((i - 12) / 4.0)
        landmarks_3d.append(tuple(wrist + tip_offset))

    # Pinky (landmarks 17-20)
    for i in range(17, 21):
        tip_offset = np.array([0.06, 0.09, 0.0]) * ((i - 16) / 4.0)
        landmarks_3d.append(tuple(wrist + tip_offset))

    return {
        "landmarks_3d": landmarks_3d,
        "gesture": gesture if gesture_type == "grab_and_release" else gesture_type,
        "confidence": 0.85,
        "handedness": "Left"
    }


def example_unreal_integration():
    """Run Unreal Engine 5 integration example."""
    if not UNREAL_MODULES_AVAILABLE:
        print("[Example] Cannot run without Unreal modules")
        return

    print("\n" + "="*70)
    print("EXAMPLE 3: UNREAL ENGINE 5 INTEGRATION")
    print("="*70)

    # Initialize Unreal bridge
    print("\n[Initialization] Setting up Unreal Engine bridge...")
    bridge = None

    # Try Python API first
    try:
        bridge = UnrealPythonAPIBridge(project_path="/path/to/ue5/project")
        if not bridge.connect():
            print("  Python API connection failed, trying WebSocket...")
            bridge = None
    except Exception as e:
        print(f"  Python API unavailable: {e}")

    # Fallback to WebSocket
    if bridge is None:
        print("[Initialization] Using WebSocket fallback...")
        bridge = UnrealWebSocketBridge(endpoint="ws://localhost:8765")
        if not bridge.connect():
            print("  WebSocket connection failed (OK for demo, using simulation)")
            bridge = None

    # If no connection, use simulation mode
    use_simulation = bridge is None
    if use_simulation:
        print("  [Demo Mode] No actual bridge connection, simulating updates")

    # Initialize hand skeleton converter
    print("[Initialization] Creating hand skeleton converter...")
    converter = HandSkeletonConverter()

    # Simulation parameters
    num_frames = 300  # 10 seconds @ 30fps
    gesture_sequence = ["reach_forward", "wave", "grab_and_release", "reach_forward"]
    current_gesture_idx = 0
    gesture_frames_per = num_frames // len(gesture_sequence)

    # Statistics
    stats = {
        "frames_processed": 0,
        "total_latencies": [],
        "skeleton_updates": 0,
        "gripper_commands": {"open": 0, "close": 0},
        "bridge_latencies": [],
        "errors": 0,
    }

    print("\n[Running] Simulating UE5 skeletal animation...")
    print(f"  Duration: {num_frames * 0.033:.1f}s ({num_frames} frames)")
    print(f"  Hand count: 2 (Left + Right)")
    print(f"  Bones per hand: 21")
    print(f"  Gesture sequence: {' → '.join(gesture_sequence)}\n")

    # Main simulation loop
    for frame in range(num_frames):
        frame_start = time.perf_counter()

        # Determine current gesture
        gesture_idx = (frame // gesture_frames_per) % len(gesture_sequence)
        current_gesture = gesture_sequence[gesture_idx]

        # Simulate left hand
        left_hand_data = simulate_hand_landmarks(frame % gesture_frames_per, current_gesture)
        left_landmarks_3d = left_hand_data["landmarks_3d"]

        # Simulate right hand (mirror)
        right_hand_data = simulate_hand_landmarks(frame % gesture_frames_per, current_gesture)
        right_landmarks_3d = [
            (-x, y, z) for x, y, z in right_hand_data["landmarks_3d"]
        ]

        try:
            # Convert landmarks to bone transforms
            left_transforms = converter.landmarks_to_bone_transforms(left_landmarks_3d, "Left")
            right_transforms = converter.landmarks_to_bone_transforms(right_landmarks_3d, "Right")

            # Create skeletal mesh updates
            left_update = SkeletalMeshUpdate(
                hand_id=0,
                hand_side="Left",
                bone_transforms=left_transforms,
                timestamp=time.time()
            )

            right_update = SkeletalMeshUpdate(
                hand_id=1,
                hand_side="Right",
                bone_transforms=right_transforms,
                timestamp=time.time()
            )

            # Send updates (if bridge available)
            if bridge and not use_simulation:
                bridge_start = time.perf_counter()

                try:
                    success_left = bridge.send_skeletal_update(left_update)
                    success_right = bridge.send_skeletal_update(right_update)

                    bridge_latency = (time.perf_counter() - bridge_start) * 1000
                    stats["bridge_latencies"].append(bridge_latency)

                    if success_left and success_right:
                        stats["skeleton_updates"] += 2
                    else:
                        stats["errors"] += 1

                except Exception as e:
                    stats["errors"] += 1

            # Handle gripper commands based on gesture
            if left_hand_data["gesture"] == "pinch":
                stats["gripper_commands"]["close"] += 1
                if bridge and not use_simulation:
                    # Send gripper close command (would be implemented in real bridge)
                    pass

            elif left_hand_data["gesture"] == "open":
                stats["gripper_commands"]["open"] += 1
                if bridge and not use_simulation:
                    # Send gripper open command
                    pass

        except Exception as e:
            print(f"  [Error Frame {frame}] {e}")
            stats["errors"] += 1

        # Record statistics
        latency = (time.perf_counter() - frame_start) * 1000
        stats["frames_processed"] = frame + 1
        stats["total_latencies"].append(latency)

        # Progress output
        if frame % 60 == 0:
            avg_lat = np.mean(stats["total_latencies"][-60:])
            bone_updates = stats["skeleton_updates"]
            print(f"  Frame {frame:3d}: Gesture={current_gesture:18} "
                  f"Updates={bone_updates:4d} Latency={avg_lat:5.2f}ms "
                  f"Errors={stats['errors']:2d}")

    # Print results
    print("\n[Results Summary]")
    print("="*70)
    print(f"Total frames processed: {stats['frames_processed']}")
    print(f"Average frame latency: {np.mean(stats['total_latencies']):.2f}ms")
    print(f"Min/Max frame latency: {np.min(stats['total_latencies']):.2f}ms / "
          f"{np.max(stats['total_latencies']):.2f}ms")
    print(f"FPS achieved: {1000 / np.mean(stats['total_latencies']):.1f}")

    print(f"\n[Skeletal Animation Performance]")
    print(f"  Total skeletal updates sent: {stats['skeleton_updates']}")
    print(f"  Expected updates: {num_frames * 2} (2 hands per frame)")
    print(f"  Update success rate: {100 * stats['skeleton_updates'] / max(1, num_frames * 2):.1f}%")

    if stats["bridge_latencies"]:
        print(f"\n[Bridge Latency (Network Only)]")
        print(f"  Average bridge latency: {np.mean(stats['bridge_latencies']):.2f}ms")
        print(f"  Min/Max bridge latency: {np.min(stats['bridge_latencies']):.2f}ms / "
              f"{np.max(stats['bridge_latencies']):.2f}ms")

    print(f"\n[Gesture Command Summary]")
    for cmd, count in stats["gripper_commands"].items():
        print(f"  {cmd:10} : {count:4d} times")

    print(f"\n[Bone Mapping Reference]")
    print(f"  Total bones per hand: 21")
    print(f"  Mapping used: MediaPipe 21-point hand skeleton → UE5 hand skeleton")
    print(f"  Sample bones:")
    sample_bones = list(LANDMARK_TO_BONE_NAME.items())[:5]
    for landmark_idx, bone_name in sample_bones:
        print(f"    Landmark {landmark_idx:2d} → {bone_name}")
    print(f"    ... ({21 - len(sample_bones)} more bones)")

    if stats["errors"] > 0:
        print(f"\n[Errors Encountered]")
        print(f"  Total errors: {stats['errors']}")
        print(f"  Error rate: {100 * stats['errors'] / stats['frames_processed']:.2f}%")

    if use_simulation:
        print(f"\n[Note] Demo ran in simulation mode (no actual UE5 connection)")
        print(f"  To connect to real UE5:")
        print(f"    1. Ensure UE5 5.0+ with Python plugin enabled")
        print(f"    2. Run editor with Python enabled")
        print(f"    3. Bridge will auto-detect and connect")

    print("\n[Complete] Unreal Engine 5 integration example finished")


if __name__ == "__main__":
    example_unreal_integration()
