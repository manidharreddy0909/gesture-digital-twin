# 3D/AR/VR API Reference Guide

Complete API documentation for gesture digital twin 3D extensions.

---

## Module Overview

| Module | Purpose | Key Class |
|--------|---------|-----------|
| camera_calibration | 2D→3D coordinate transformation | `CoordinateTransformer` |
| cursor_controller_3d | 3D motion filtering | `CursorController3D` |
| robot_controller | 6-DOF arm inverse kinematics | `RobotArmUR5` |
| object_manipulator | Gesture-driven 3D object control | `ObjectManipulator` |
| unreal_bridge | Unreal Engine 5 integration | `UnrealPythonAPIBridge` |

---

## camera_calibration Module

### CameraIntrinsics (Dataclass)

Camera intrinsic parameters for 3D projection.

```python
@dataclass
class CameraIntrinsics:
    fx: float                    # Focal length X (pixels)
    fy: float                    # Focal length Y (pixels)
    cx: float                    # Principal point X (pixels)
    cy: float                    # Principal point Y (pixels)
    width: int                   # Image width
    height: int                  # Image height
    k1, k2, k3: float           # Radial distortion
    p1, p2: float               # Tangential distortion
    name: str                    # Calibration name

    # Methods:
    def get_camera_matrix() -> np.ndarray  # Returns 3×3 K matrix
    def get_distortion_coeffs() -> np.ndarray  # Returns [k1,k2,p1,p2,k3]
    def to_dict() -> Dict               # Serialize to JSON
    @staticmethod
    def from_dict(data) -> CameraIntrinsics  # Load from JSON
```

### CoordinateTransformer (Class)

Transform 2D normalized coordinates to 3D world space.

```python
class CoordinateTransformer:
    def __init__(intrinsics, depth_scale=1.0, default_z=0.5)
        """
        Args:
            intrinsics: CameraIntrinsics object
            depth_scale: Scale factor for Z (default 1.0)
            default_z: Default Z if not provided (0-1 normalized)
        """

    def normalized_to_3d(x_norm, y_norm, z_norm=None) -> (float, float, float)
        """Convert normalized 2D to 3D world coordinates.

        Args:
            x_norm, y_norm: In range [0, 1]
            z_norm: Optional depth in [0, 1]

        Returns:
            (x_world, y_world, z_world)
        """

    def transform_landmarks(landmarks: List[(float,float,float)],
                           use_z_from_landmarks=True) -> List[(float,float,float)]
        """Transform 21 MediaPipe landmarks to 3D world."""
```

### CameraCalibrator (Static Methods)

```python
class CameraCalibrator:
    @staticmethod
    def estimate_from_image_size(width: int, height: int,
                                fov_degrees: float = 60.0) -> CameraIntrinsics
        """Quick calibration from image size and assumed FOV."""

    @staticmethod
    def from_reference_points(landmark_2d: List[(float,float)],
                             world_3d: List[(float,float,float)],
                             width: int, height: int) -> CameraIntrinsics
        """Precise calibration using 4+ reference points."""

    @staticmethod
    def save_calibration(intrinsics: CameraIntrinsics, filepath: str)
        """Save to JSON."""

    @staticmethod
    def load_calibration(filepath: str) -> CameraIntrinsics
        """Load from JSON."""
```

---

## cursor_controller_3d Module

### CursorController3D (Class)

```python
class CursorController3D:
    def __init__(use_enhanced_kalman=True, adaptive_smoothing=True,
                kalman_adaptive_noise=True)
        """3D Kalman-filtered cursor controller.

        Args:
            use_enhanced_kalman: Use 9D state (x,y,z,vx,vy,vz,ax,ay,az) vs 6D
            adaptive_smoothing: Adjust Kalman noise based on motion
            kalman_adaptive_noise: Scale Q matrix by motion magnitude
        """

    def update_cursor_3d(hand_id: int, x_norm: float, y_norm: float,
                        z_norm: float, timestamp: float,
                        control_this_hand: bool = True) -> (float, float, float)
        """Update cursor position for one hand.

        Args:
            hand_id: 0 for left, 1 for right
            x_norm, y_norm, z_norm: Normalized coordinates [0, 1]
            timestamp: Frame timestamp (seconds)
            control_this_hand: If False, don't update this hand

        Returns:
            (x_filtered, y_filtered, z_filtered): Smoothed position

        Note: Maintains per-hand Kalman filter state internally.
        """

    def get_cursor_state(hand_id: int) -> Dict
        """Get current filtered state for hand."""
```

### KalmanState3D vs EnhancedKalmanState3D

- **KalmanState3D**: 6D state [x, y, z, vx, vy, vz]
- **EnhancedKalmanState3D**: 9D state [x, y, z, vx, vy, vz, ax, ay, az]

Enhanced tracks acceleration for smoother prediction.

---

## robot_controller Module

### RobotArm (Abstract Base Class)

```python
class RobotArm(ABC):
    @abstractmethod
    def forward_kinematics(joint_angles: List[float]) ->
            (np.ndarray, np.ndarray)
        """Joint angles → (position, orientation)."""

    @abstractmethod
    def inverse_kinematics(position: np.ndarray,
                          orientation: Optional[np.ndarray] = None) ->
            Optional[List[float]]
        """(Position, orientation) → joint angles or None if unreachable."""

    @abstractmethod
    def move_to_position(target: RobotTarget) -> bool
        """Move arm to target position."""

    @abstractmethod
    def get_current_state() -> RobotState
        """Get current joint angles and TCP position."""

    @abstractmethod
    def stop() -> bool
        """Emergency stop."""
```

### RobotArmUR5 (UR5 Implementation)

```python
class RobotArmUR5(RobotArm):
    def __init__(ip_address: str = "192.168.1.100", port: int = 30003)
        """Connect to UR5 at given IP:port via RTDE."""

    # Inherits all abstract methods from RobotArm
    # Plus:
    def connect() -> bool
        """Establish RTDE connection."""

    def _solve_ik_geometric(position: np.ndarray) -> List[float]
        """Fallback geometric IK if ikpy not available."""
```

### RobotTarget (Dataclass)

```python
@dataclass
class RobotTarget:
    position: np.ndarray               # [x, y, z] in meters
    orientation: np.ndarray = [0,0,0]  # [rx, ry, rz] in radians
    gripper_command: Optional[GripperCommand] = None
    gripper_force: float = 50.0        # 0-100

    def to_dict() -> Dict              # Serialize to JSON
```

### HandToArmMapper (Class)

Maps hand landmarks to robot arm targets with trajectory smoothing.

```python
class HandToArmMapper:
    def __init__(position_scale: float = 1.0, max_velocity: float = 0.5)
        """Initialize mapper.

        Args:
            position_scale: Scale hand coordinates to robot space
            max_velocity: Max arm velocity (m/s)
        """

    def map_hand_to_arm_target(left_hand, right_hand,
                              left_gesture=None, right_gesture=None) ->
            Optional[RobotTarget]
        """Convert hand data to robot target.

        Mapping:
        - Left index finger tip → end effector position
        - Right hand orientation → end effector orientation
        - Pinch gesture → gripper close
        - Open gesture → gripper open
        """

    def smooth_trajectory(target: RobotTarget, dt: float = 0.033) ->
            RobotTarget
        """Smooth target trajectory respecting velocity limits."""
```

---

## object_manipulator Module

### Object3D (Dataclass)

3D object with physics state.

```python
@dataclass
class Object3D:
    id: str
    position: np.ndarray = [0,0,0]        # [x, y, z]
    rotation: np.ndarray = [0,0,0]        # [rx, ry, rz] radians
    scale: np.ndarray = [1,1,1]           # [sx, sy, sz]
    velocity: np.ndarray = [0,0,0]        # [vx, vy, vz]
    angular_velocity: np.ndarray = [0,0,0]
    mass: float = 1.0
    interaction_mode: InteractionMode = IDLE

    def get_bounds() -> (np.ndarray, np.ndarray)  # Min/max AABB points
    def distance_to_point(point: np.ndarray) -> float
```

### ObjectManipulator (Class)

```python
class ObjectManipulator:
    def __init__(world_objects: Dict[str, Object3D] = None)

    def add_object(obj: Object3D)
        """Add object to scene."""

    def select_object_at(position: np.ndarray, radius: float = 0.1) ->
            Optional[Object3D]
        """Find nearest object within radius."""

    def apply_gesture(gesture_name: str, hand_position: np.ndarray = None,
                     hand_motion: np.ndarray = None,
                     two_hand_gesture: str = None)
        """Apply gesture to selected object.

        Single-hand:
            "pinch" → grab object
            "open" → release
            "swipe_left", "swipe_right" → translate X
            "swipe_up", "swipe_down" → translate Y
            "circle" → rotate around Z

        Two-hand:
            "zoom_in" → scale up
            "zoom_out" → scale down
            "rotate_cw", "rotate_ccw" → rotate
            "mirror" → mirror across axis
            "push", "pull" → translate Z
        """

    def update_grab(hand_position: np.ndarray, hold_fixed: bool = True)
        """Update grabbed object position."""

    def update_physics(dt: float, damping: float = 0.95,
                      gravity: Optional[np.ndarray] = None)
        """Update physics for all objects."""

    def get_all_transforms() -> Dict
        """Return all object transforms as serializable dict."""
```

### BasicPhysics (Static Methods)

```python
class BasicPhysics:
    @staticmethod
    def apply_physics(obj: Object3D, dt: float, damping: float = 0.95,
                     gravity: Optional[np.ndarray] = None) -> Object3D
        """Apply physics-based motion update."""

    @staticmethod
    def check_collision_aabb(obj1: Object3D, obj2: Object3D) -> bool
        """Check AABB collision."""

    @staticmethod
    def resolve_collision(obj1: Object3D, obj2: Object3D)
        """Separate colliding objects."""
```

---

## unreal_bridge Module

### UnrealPythonAPIBridge (Class)

Direct connection to UE5 Python API.

```python
class UnrealPythonAPIBridge(UnrealBackend):
    def __init__(project_path: str = None)

    def connect() -> bool
        """Connect to running UE5 editor."""

    def send_skeletal_update(update: SkeletalMeshUpdate) -> bool
        """Update hand skeleton 21 bones."""

    def move_actor(actor_name: str, position: (float,float,float),
                  rotation: (float,float,float) = (0,0,0)) -> bool
        """Move object in UE5 world."""

    def set_actor_scale(actor_name: str,
                       scale: (float,float,float) -> bool
        """Scale object."""
```

### SkeletalMeshUpdate (Dataclass)

```python
@dataclass
class SkeletalMeshUpdate:
    hand_id: int
    hand_side: str               # "Left" or "Right"
    bone_transforms: List[BoneTransform]
    timestamp: float = 0.0

    def to_json() -> str         # Serialize for WebSocket
```

### HandSkeletonConverter (Static Methods)

```python
class HandSkeletonConverter:
    @staticmethod
    def landmarks_to_bone_transforms(landmarks_3d: List[(float,float,float)],
                                     hand_side: str = "Left") ->
            List[BoneTransform]
        """Convert 21 MediaPipe landmarks to 21 bone transforms."""

    @staticmethod
    def compute_joint_angles(landmarks_3d: List[(float,float,float)]) ->
            Dict[str, (float,float,float)]
        """Compute joint angles from landmarks."""

# Bone mapping reference:
LANDMARK_TO_BONE_NAME = {
    0: "hand_wrist",
    1-4: "hand_thumb_01-04",
    5-8: "hand_index_01-04",
    # ... etc (21 total bones)
}
```

---

## main_3d_vr Module

### Pipeline3DState (Class)

```python
class Pipeline3DState:
    def __init__(config)

    def initialize() -> bool
        """Initialize all 3D subsystems."""

    def process_hands(hands: List[Hand])
        """Main pipeline: detect → transform → gesture → control."""

    def update_performance_monitor()
        """Track FPS, latency per stage."""
```

---

## Configuration Options

```python
# In config.py, enable/disable features:

enable_3d_world = False              # Master 3D enable
enable_robot_control = False         # Gesture→robot arm
enable_ue5_integration = False       # Gesture→UE5 skeleton
enable_3d_visualization = False      # Matplotlib 3D plots

camera_calibration_file = "calibration.json"
camera_auto_calibrate = True
camera_fov_degrees = 60.0

robot_ip = "192.168.1.100"
robot_port = 30003
robot_max_velocity = 0.5
robot_enable_collision_check = False

unreal_project_path = "/path/to/ue5/project"
unreal_use_websocket = False
unreal_websocket_endpoint = "ws://localhost:8765"

kalman_3d_adaptive_noise = True
depth_smoothing_factor = 0.7
physics_damping = 0.95
```

---

## Common Usage Patterns

### Pattern 1: Robot Control Only

```python
# config.py
enable_robot_control = True
enable_ue5_integration = False
enable_3d_objects = False

# Then use:
from main_3d_vr import main_3d_vr
main_3d_vr()
```

### Pattern 2: 3D Object Manipulation

```python
import numpy as np
from object_manipulator import ObjectManipulator, Object3D

manipulator = ObjectManipulator()

# Create objects
cube = Object3D(id="cube", position=np.array([0, 0, 0]))
manipulator.add_object(cube)

# Gesture control
manipulator.select_object_at(np.array([0, 0, 0]))
manipulator.apply_gesture("pinch")
manipulator.apply_gesture("swipe_left")
manipulator.update_grab(np.array([0.1, 0, 0]))
manipulator.update_physics(dt=0.033)
```

### Pattern 3: Camera Calibration Integration

```python
from camera_calibration import CameraCalibrator, CoordinateTransformer

# Load or create calibration
cal = CameraCalibrator.load_calibration("calibration.json")
transformer = CoordinateTransformer(cal, depth_scale=1.0)

# Transform MediaPipe output to world coordinates
hand_norm = (0.5, 0.3, 0.6)  # From MediaPipe
hand_3d = transformer.normalized_to_3d(*hand_norm)
# Returns: (0.0, -0.3, 0.6) in meters
```

### Pattern 4: Real-time Hand Skeleton Animation

```python
from unreal_bridge import UnrealPythonAPIBridge, HandSkeletonConverter

bridge = UnrealPythonAPIBridge()
bridge.connect()

converter = HandSkeletonConverter()

# In main loop:
landmarks_3d = [...]  # 21 landmarks from camera
transforms = converter.landmarks_to_bone_transforms(landmarks_3d, "Left")

from unreal_bridge import SkeletalMeshUpdate
update = SkeletalMeshUpdate(0, "Left", transforms)
bridge.send_skeletal_update(update)
```

---

## Performance Benchmarks

| Operation | Typical Latency |
|-----------|-----------------|
| Camera calibration transform | 0.5-1ms |
| 3D Kalman filter update | 2-3ms |
| IK solver (ikpy) | 5-15ms per joint |
| Object physics update | 1-2ms |
| UE5 skeletal update (Python API) | 1-5ms |
| UE5 skeletal update (WebSocket) | 10-20ms |

**Total system latency (all features):** 20-50ms per frame @ 20-30 FPS

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Import error: `No module named 'camera_calibration'` | Ensure .py files in same directory as main_3d_vr.py |
| Robot IK fails | Check position in meters, not mm; verify workspace |
| UE5 bones not moving | Verify bone names exactly match `LANDMARK_TO_BONE_NAME` |
| Janky 3D object motion | Enable trajectory smoothing, reduce gesture speed |
| High latency (>100ms) | Disable unnecessary features, reduce frame rate, profile |

---

## Next Steps

1. Start with camera calibration: `CALIBRATION_GUIDE.md`
2. Choose feature: robot vs objects vs UE5
3. Follow corresponding setup guide
4. Run example script
5. Integrate into main system
6. Tune performance parameters in config.py

For production use, maintain 30+ FPS and <50ms latency by:
- Using conditional imports (disable unused features)
- Running on dedicated GPU
- Tuning Kalman smoothing parameters
- Batching multiple updates
