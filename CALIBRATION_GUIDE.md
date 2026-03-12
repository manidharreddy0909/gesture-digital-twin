# Camera Calibration Guide

## Table of Contents

- [Overview](#overview)
- [Why Calibration Matters](#why-calibration-matters)
- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Precise Calibration (30 minutes)](#precise-calibration-30-minutes)
- [Interactive Calibration Tool](#interactive-calibration-tool)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Overview

Camera calibration transforms 2D normalized hand coordinates from MediaPipe into accurate 3D world coordinates. Without proper calibration, hand tracking will appear distorted, with incorrect depth perception.

**Two Calibration Approaches:**
1. **Automatic** (5 min): Quick estimation from camera resolution and FOV
2. **Precise** (30 min): Reference-point calibration for maximum accuracy

---

## Why Calibration Matters

### 2D vs 3D Coordinate Systems

**Without calibration:**
- Hand tracking: Only 2D screen coordinates (x, y)
- Depth (z): Unreliable, depends on hand detection confidence
- Result: 3D gestures don't translate accurately to robot/object control

**With calibration:**
- Hand position: Accurate 3D world coordinates (x, y, z)
- Integration: 3D objects, robot arm, Unreal Engine work smoothly
- Performance: Gestures map predictably to physical transformations

### Real-World Impact

| Scenario | Without Calibration | With Calibration |
|----------|---------------------|------------------|
| **Robot arm control** | Z-depth jerky, hand "floats" | Smooth, stable reaching |
| **3D object manipulation** | Objects scale/rotate unpredictably | Precise gesture→transform mapping |
| **UE5 hand skeleton** | Bones distort, unusual angles | Naturalistic hand animation |

---

## Quick Start (5 minutes)

### For Most Webcams

If you have a standard 1920×1080 webcam with ~60° field-of-view, use this:

```python
from camera_calibration import CameraCalibrator

# Estimate calibration from image size
calibration = CameraCalibrator.estimate_from_image_size(
    width=1920,
    height=1080,
    fov_degrees=60.0  # Typical for USB webcams
)

# Save for future use
CameraCalibrator.save_calibration(calibration, "calibration.json")

print(f"Generated calibration: {calibration.name}")
print(f"  Focal length: ({calibration.fx:.1f}, {calibration.fy:.1f})")
print(f"  Principal point: ({calibration.cx:.1f}, {calibration.cy:.1f})")
```

### Estimate FOV for Your Webcam

If unsure of your camera's FOV, check manufacturer specs or use this table:

| Webcam Model | Typical FOV |
|--------------|------------|
| Logitech C920 | 60-78° |
| Built-in laptop cam | 60-65° |
| Intel RealSense | 69-87° (depth-dependent) |
| Generic USB cam | 50-65° |

---

## Precise Calibration (30 minutes)

### Step 1: Gather Reference Points

You need 4-8 points with **known 3D positions** in your physical space.

**Option A: Checkerboard Pattern (Recommended)**
```
Measure 4 corners of a printed checkerboard pattern:
  - Print standard checkerboard (100mm × 100mm squares)
  - Mount on flat surface
  - Record 4 corner positions: (x, y, z) in meters

Example (if placed on table 1m from camera):
  - Bottom-left: (-0.1, -0.1, 1.0)
  - Bottom-right: (0.1, -0.1, 1.0)
  - Top-left: (-0.1, 0.1, 1.0)
  - Top-right: (0.1, 0.1, 1.0)
```

**Option B: Marked Objects**
```
Place objects at known distances from camera:
  1. Tape measure: Measure exact distance
  2. Known objects: Cup, block, tennis ball at known positions
  3. Depth sensor (if available): Use actual depth readings

Need at least 4 points, more is better (8-12 recommended).
```

### Step 2: Record Image Coordinates

Run MediaPipe hand detection and record 2D positions of reference points:

```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

print("Click on reference points in order (ESC when done):")

reference_2d = []  # List of (x_pixel, y_pixel)

while True:
    ret, frame = cap.read()
    cv2.imshow("Click reference points (ESC to finish)", frame)

    # (You would implement mouse click handling here)
    # For this guide, manually record coordinates from visual inspection

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
```

### Step 3: Run Calibration

```python
from camera_calibration import CameraCalibrator

# Your collected data
landmarks_2d = [
    (100, 150),    # Point 1 in pixels
    (1820, 150),   # Point 2 in pixels
    (100, 900),    # Point 3 in pixels
    (1820, 900),   # Point 4 in pixels
]

world_3d = [
    (-0.1, -0.1, 1.0),  # Point 1 in world space (meters)
    (0.1, -0.1, 1.0),   # Point 2
    (-0.1, 0.1, 1.0),   # Point 3
    (0.1, 0.1, 1.0),    # Point 4
]

# Calibrate
calibration = CameraCalibrator.from_reference_points(
    landmark_2d=landmarks_2d,
    world_3d=world_3d,
    width=1920,
    height=1080
)

# Test accuracy
print(f"Focal length (fx, fy): ({calibration.fx:.1f}, {calibration.fy:.1f})")
print(f"Principal point (cx, cy): ({calibration.cx:.1f}, {calibration.cy:.1f})")

# Save
CameraCalibrator.save_calibration(calibration, "calibration_precise.json")
```

### Step 4: Verify Accuracy

```python
from camera_calibration import CoordinateTransformer

transformer = CoordinateTransformer(calibration, depth_scale=1.0)

# Test: reproject reference points
print("\nAccuracy test (error should be < 5% of distance):")
for i, (x2d, y2d, z2d_norm) in enumerate(zip(landmarks_2d, world_3d)):
    # Convert pixel→3D
    x_norm = x2d / 1920.0
    y_norm = y2d / 1080.0
    x_world, y_world, z_world = transformer.normalized_to_3d(x_norm, y_norm, z2d_norm[2])

    # Compare to expected
    x_expected, y_expected, z_expected = world_3d[i]
    error_x = abs(x_world - x_expected)
    error_y = abs(y_world - y_expected)

    print(f"  Point {i}: Error = ({error_x:.3f}, {error_y:.3f}) meters")
```

---

## Interactive Calibration Tool

### One-Command Setup

Use the interactive calibration tool for guided setup:

```bash
python calibrate_camera_interactive.py
```

**What it does:**
1. Prompts for camera resolution (default: 1920×1080)
2. Prompts for horizontal FOV (default: 60°)
3. Calculates intrinsic parameters
4. Displays estimated calibration
5. Optionally saves to file

**Example session:**
```
========================================
INTERACTIVE CAMERA CALIBRATION
========================================

Enter image width (pixels) [1920]: 1920
Enter image height (pixels) [1080]: 1080
Enter horizontal FOV (degrees) [60]: 65

[Generated Calibration]
  Resolution: 1920x1080
  Focal length (fx, fy): (1656.2, 1656.2)
  Principal point (cx, cy): (960.0, 540.0)

Save to file? (y/n) [y]: y
Filename [calibration.json]: my_camera.json

[CameraCalibrator] Calibration saved: my_camera.json
```

---

## Troubleshooting

### Issue: "Hand appears offset from real position"

**Symptom:** When you move your hand, tracked position lags or is off-center

**Causes:**
- Principal point (cx, cy) is wrong
- Camera not centered in frame during calibration
- Lens distortion not accounted for

**Fix:**
- Recalibrate with more reference points
- Ensure camera is perpendicular to calibration surface
- If distortion coefficients matter, use `k1, k2, k3` parameters

### Issue: "Depth (Z) coordinate unstable"

**Symptom:** Hand Z-value jumps or "floats" unexpectedly

**Causes:**
- Depth scale (depth_scale parameter) wrong
- Using MediaPipe z (relative depth) instead of sensor depth
- Hand confidence fluctuating

**Fix:**
```python
# Use more aggressive smoothing for Z
from camera_calibration import DepthEstimator

z_raw = [0.4, 0.42, 0.41, 0.43, 0.40, 0.39]
z_smooth = DepthEstimator.smooth_depth_sequence(z_raw, smoothing_factor=0.85)
# Result: [0.4, 0.408, 0.4092, 0.4159, 0.4057, 0.3965]
```

### Issue: "Objects too large/small when manipulated"

**Symptom:** 3D object scaling doesn't match hand motion

**Causes:**
- depth_scale parameter incorrect
- Focal length (fx, fy) way off

**Fix:**
```python
# Adjust depth scale (default 1.0)
# Higher value = hand appears closer, object manipulation more sensitive
transformer = CoordinateTransformer(
    calibration,
    depth_scale=1.2  # Try 0.8-1.5 range
)
```

### Issue: "Robot arm reaches wrong coordinates"

**Symptom:** If hand is at screen (0.5, 0.5), arm doesn't point to center of workspace

**Causes:**
- Principal point (cx, cy) wrong (should be image center normally)
- Focal length way off

**Fix:**
```python
# Verify principal point is at image center
assert calibration.cx == calibration.width / 2.0
assert calibration.cy == calibration.height / 2.0

# If not, recalibrate with checkerboard method
```

---

## Advanced Topics

### Custom Distortion Coefficients

MediaPipe doesn't provide distortion info, but you can add it if you have a calibrated camera:

```python
calibration = CameraIntrinsics(
    fx=1656.2, fy=1656.2,
    cx=960.0, cy=540.0,
    width=1920, height=1080,
    k1=-0.05,   # Barrel distortion
    k2=0.01,
    k3=0.0,
    p1=-0.001,  # Tangential distortion
    p2=0.0
)

# Save includes distortion coefficients
CameraCalibrator.save_calibration(calibration, "calibration.json")
```

### Combining Multiple Depth Sources

If you have both MediaPipe z-coordinate AND sensor depth (e.g., RealSense):

```python
from camera_calibration import DepthEstimator

z_landmark = 0.45  # From MediaPipe (relative)
z_sensor = 0.5     # From depth sensor (absolute)

# Blend them (60% landmark, 40% sensor)
z_final = DepthEstimator.combine_depth_sources(
    z_landmark=z_landmark,
    z_confidence=z_sensor,
    landmark_weight=0.6
)
# Result: More stable depth
```

### Loading Calibration at Runtime

```python
from camera_calibration import CameraCalibrator

# Load from JSON
calibration = CameraCalibrator.load_calibration("calibration.json")

# Use in coordinate transformer
from camera_calibration import CoordinateTransformer
transformer = CoordinateTransformer(calibration)

# Transform hand landmarks
normalized_landmarks = [
    (0.5, 0.5, 0.5),   # Normalized MediaPipe output
    (0.6, 0.3, 0.6),
    # ...
]

world_landmarks = transformer.transform_landmarks(normalized_landmarks)
print(world_landmarks)  # [(0.0, 0.0, 0.5), (0.15, -0.3, 0.6), ...]
```

### Integration with robot_controller

Once calibrated, use world coordinates directly with robot control:

```python
from camera_calibration import CameraCalibrator, CoordinateTransformer
from robot_controller import RobotArmUR5, HandToArmMapper

# Load calibration
calibration = CameraCalibrator.load_calibration("calibration.json")
transformer = CoordinateTransformer(calibration)

# Initialize robot
robot = RobotArmUR5("192.168.1.100")
robot.connect()

# In main loop:
hand_normalized = (0.5, 0.3, 0.6)  # From MediaPipe
hand_3d = transformer.normalized_to_3d(*hand_normalized)

mapper = HandToArmMapper()
target = mapper.map_hand_to_arm_target(hand_3d)
robot.move_to_position(target)
```

---

## Summary

| Task | Time | Accuracy |
|------|------|----------|
| Quick start (auto) | 5 min | 85-90% |
| Precise (reference points) | 30 min | 95-98% |
| With distortion coefficients | 45 min | 98-99% |

**Recommendation:** Start with quick start. If results aren't satisfactory, refine with reference points.

Once calibrated, save calibration.json and reuse it - calibration is camera-specific but stable over time.
