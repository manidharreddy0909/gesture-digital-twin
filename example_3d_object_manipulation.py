"""
EXAMPLE 1: 3D OBJECT MANIPULATION WITH GESTURES

Demonstrates gesture-controlled manipulation of 3D objects with physics simulation.

Features:
- Interactive 3D visualizations using matplotlib
- Gesture-driven object transforms (rotate, scale, translate)
- Physics simulation with velocity and damping
- Real-time statistics and performance monitoring

Usage:
  python example_3d_object_manipulation.py

Controls:
  - Make pinch gesture to grab objects
  - Swipe to move objects
  - Circular gesture to rotate
  - Two-hand zoom to scale
"""

import sys
import time
import numpy as np
from typing import Dict, List, Optional

# Try imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Example] Matplotlib not available - install with: pip install matplotlib")

try:
    from camera_calibration import CameraCalibrator, CoordinateTransformer, DepthEstimator
    from object_manipulator import ObjectManipulator, Object3D
    from cursor_controller_3d import CursorController3D
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("[Example] Required modules not available")


def simulate_hand_gesture(gesture_type: str, frame: int) -> Dict:
    """Simulate hand motion for demo."""
    t = frame * 0.033  # 30 fps
    import math

    if gesture_type == "pinch":
        # Swipe motion simulating pinch gesture
        x = 0.5 + 0.15 * math.sin(t * 2)
        y = 0.5 + 0.15 * math.cos(t * 2)
        z = 0.5 + 0.05 * math.sin(t * 4)
        return {"position": (x, y, z), "gesture": "pinch", "confidence": 0.8}

    elif gesture_type == "swipe":
        # Horizontal swipe
        x = 0.3 + 0.3 * (frame % 60) / 60.0
        y = 0.5
        z = 0.5
        return {"position": (x, y, z), "gesture": "swipe_right", "confidence": 0.7}

    elif gesture_type == "circle":
        # Circular motion
        radius = 0.15
        x = 0.5 + radius * math.cos(t)
        y = 0.5 + radius * math.sin(t)
        z = 0.5
        return {"position": (x, y, z), "gesture": "circle", "confidence": 0.75}

    return {"position": (0.5, 0.5, 0.5), "gesture": "neutral", "confidence": 0.5}


def example_3d_object_manipulation():
    """Run 3D object manipulation example."""
    if not MODULES_AVAILABLE:
        print("[Example] Cannot run without required modules")
        return

    print("\n" + "="*70)
    print("EXAMPLE 1: 3D OBJECT MANIPULATION WITH GESTURES")
    print("="*70)

    # Initialize camera calibration
    print("\n[Initialization] Setting up camera calibration...")
    calibration = CameraCalibrator.estimate_from_image_size(1920, 1080, fov_degrees=60.0)
    transformer = CoordinateTransformer(calibration, depth_scale=1.0)
    print(f"  Camera: {calibration.width}x{calibration.height}, F={calibration.fx:.0f}")

    # Initialize object manipulator
    print("[Initialization] Creating 3D scene...")
    manipulator = ObjectManipulator()

    # Create objects
    cube = Object3D(id="cube", position=np.array([0.0, 0.0, 0.0]), scale=np.array([0.05, 0.05, 0.05]))
    sphere = Object3D(id="sphere", position=np.array([-0.3, 0.0, 0.0]), scale=np.array([0.04, 0.04, 0.04]))
    cylinder = Object3D(id="cylinder", position=np.array([0.3, 0.0, 0.0]), scale=np.array([0.03, 0.08, 0.03]))

    manipulator.add_object(cube)
    manipulator.add_object(sphere)
    manipulator.add_object(cylinder)
    print(f"  Created {len(manipulator.world_objects)} objects")

    # Initialize 3D Kalman filter
    print("[Initialization] Setting up 3D motion filtering...")
    cursor_3d = CursorController3D(
        use_enhanced_kalman=True,
        adaptive_smoothing=True,
        kalman_adaptive_noise=True
    )

    # Simulation parameters
    num_frames = 300  # 10 seconds @ 30fps
    gesture_sequence = ["pinch", "swipe", "pinch", "circle", "swipe"]
    current_gesture_idx = 0
    gesture_frames_per = num_frames // len(gesture_sequence)

    # Statistics
    stats = {
        "frames_processed": 0,
        "avg_latency_ms": 0.0,
        "total_latencies": [],
        "gestures_detected": {},
    }

    print("\n[Running] Simulating 3D gesture interaction...")
    print(f"  Duration: {num_frames * 0.033:.1f}s ({num_frames} frames)")
    print(f"  Gesture sequence: {' → '.join(gesture_sequence)}\n")

    # Main simulation loop
    for frame in range(num_frames):
        frame_start = time.perf_counter()

        # Determine current gesture
        gesture_idx = (frame // gesture_frames_per) % len(gesture_sequence)
        current_gesture = gesture_sequence[gesture_idx]

        # Simulate hand input
        hand_data = simulate_hand_gesture(current_gesture, frame % gesture_frames_per)
        hand_pos = np.array(hand_data["position"])

        # Transform to 3D world
        hand_3d = transformer.normalized_to_3d(hand_pos[0], hand_pos[1], hand_pos[2])

        # Filter motion with Kalman
        filtered_pos = cursor_3d.update_cursor_3d(
            hand_id=0,
            x_norm=hand_pos[0],
            y_norm=hand_pos[1],
            z_norm=hand_pos[2],
            timestamp=time.time()
        )
        filtered_3d = np.array(filtered_pos)

        # Select object if near
        manipulator.select_object_at(filtered_3d, radius=0.15)

        # Apply gesture
        manipulator.apply_gesture(
            hand_data["gesture"],
            hand_position=filtered_3d
        )

        # Update grabbed object
        if manipulator.selected_object:
            manipulator.update_grab(filtered_3d, hold_fixed=True)

        # Update physics
        manipulator.update_physics(dt=0.033, damping=0.9)

        # Record statistics
        latency = (time.perf_counter() - frame_start) * 1000
        stats["frames_processed"] = frame + 1
        stats["total_latencies"].append(latency)
        if hand_data["gesture"] not in stats["gestures_detected"]:
            stats["gestures_detected"][hand_data["gesture"]] = 0
        stats["gestures_detected"][hand_data["gesture"]] += 1

        # Progress output
        if frame % 60 == 0:
            avg_lat = np.mean(stats["total_latencies"][-60:])
            obj_info = f" [Selected: {manipulator.selected_object.id}]" if manipulator.selected_object else ""
            print(f"  Frame {frame:3d}: Gesture={current_gesture:10} Latency={avg_lat:5.2f}ms{obj_info}")

    # Print results
    print("\n[Results Summary]")
    print("="*70)
    print(f"Total frames processed: {stats['frames_processed']}")
    print(f"Average latency: {np.mean(stats['total_latencies']):.2f}ms")
    print(f"Min/Max latency: {np.min(stats['total_latencies']):.2f}ms / {np.max(stats['total_latencies']):.2f}ms")
    print(f"FPS achieved: {1000 / np.mean(stats['total_latencies']):.1f}")
    print(f"\nGestures detected:")
    for gesture, count in stats["gestures_detected"].items():
        print(f"  {gesture:15} : {count:3d} times")

    print(f"\nFinal object states:")
    transforms = manipulator.get_all_transforms()
    for obj_id, transform in transforms.items():
        pos = transform["position"]
        scale = transform["scale"]
        print(f"  {obj_id:10} : pos=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}) "
              f"scale=({scale[0]:5.2f}, {scale[1]:5.2f}, {scale[2]:5.2f})")

    # Visualization (if matplotlib available)
    if MATPLOTLIB_AVAILABLE:
        print("\n[Visualization] Creating 3D plot...")
        plot_3d_scene(manipulator.world_objects)

    print("\n[Complete] 3D object manipulation example finished")


def plot_3d_scene(objects: Dict[str, Object3D]) -> None:
    """Plot 3D scene using matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot objects
    colors = {"cube": 'r', "sphere": 'b', "cylinder": 'g'}
    for obj_id, obj in objects.items():
        pos = obj.position
        scale = obj.scale
        color = colors.get(obj_id, 'k')

        # Plot as cube
        size = np.mean(scale)
        ax.scatter([pos[0]], [pos[1]], [pos[2]], c=color, s=300, marker='o', label=obj_id)

        # Add text label
        ax.text(pos[0], pos[1], pos[2] + 0.05, obj_id, fontsize=10)

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.2, 0.8])
    ax.legend()
    ax.set_title('3D Object Manipulation Scene')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_3d_object_manipulation()
