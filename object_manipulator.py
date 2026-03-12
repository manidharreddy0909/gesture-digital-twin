"""
3D OBJECT MANIPULATOR MODULE

Maps gestures to 3D object transformations (rotate, translate, scale).

Purpose: Enable gesture-driven manipulation of 3D objects in virtual or AR/VR environments.

Key Classes:
- Object3D: Represents a 3D object with position, rotation, scale, physics state
- ObjectManipulator: Gesture-to-object-transformation mapper
- ConvexPhysics: Simple physics simulation (velocity, damping, collision detection)

Gesture Mapping:
Single-Hand:
  Pinch on object → Grab (select, lock position)
  Open hand → Release (allow physics)
  Swipe left/right → Translate X
  Swipe up/down → Translate Y
  Circle → Rotate around Z axis

Two-Hand:
  Zoom in/out → Scale up/down
  Rotate CW/CCW → Rotate object around all axes
  Mirror → Mirror object geometry
  Push/Pull → Translate Z (depth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


# ============================================================================
# 3D OBJECT REPRESENTATIONS
# ============================================================================

class InteractionMode(Enum):
    """Object interaction modes."""
    IDLE = "idle"            # No interaction
    SELECTED = "selected"    # Object selected but not grabbed
    GRABBED = "grabbed"      # Object being held
    DRAGGED = "dragged"      # Object being dragged


@dataclass
class Object3D:
    """
    3D object with transformation and physics state.

    Attributes:
        id: Unique identifier
        position: [x, y, z] world coordinates
        rotation: [rx, ry, rz] euler angles in radians
        scale: [sx, sy, sz] scale factors
        velocity: [vx, vy, vz] linear velocity
        angular_velocity: [wx, wy, wz] angular velocity
        mass: Object mass (for physics)
    """
    id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    interaction_mode: InteractionMode = InteractionMode.IDLE

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get AABB (axis-aligned bounding box) min and max."""
        half_size = 0.1 * self.scale
        min_pt = self.position - half_size
        max_pt = self.position + half_size
        return min_pt, max_pt

    def distance_to_point(self, point: np.ndarray) -> float:
        """Compute distance from object center to point."""
        return np.linalg.norm(self.position - point)


# ============================================================================
# SIMPLE PHYSICS ENGINE
# ============================================================================

class BasicPhysics:
    """
    Simple physics simulation for object motion.

    Features:
    - Velocity-based motion (x += v*dt)
    - Linear damping (velocity *= damping)
    - Angular momentum  (rotation += angular_velocity * dt)
    - Gravity (optional)
    """

    @staticmethod
    def apply_physics(obj: Object3D, dt: float,
                     damping: float = 0.95,
                     gravity: Optional[np.ndarray] = None) -> Object3D:
        """
        Update object physics for one timestep.

        Args:
            obj: Object to update
            dt: Time delta (seconds)
            damping: Velocity damping factor [0, 1]
            gravity: Gravity vector or None

        Returns:
            Updated object
        """
        # Position update (linear motion)
        obj.position += obj.velocity * dt

        # Rotation update (angular motion) - simple Euler angles
        obj.rotation += obj.angular_velocity * dt
        obj.rotation = obj.rotation % (2 * np.pi)  # Normalize

        # Apply gravity
        if gravity is not None:
            obj.velocity += gravity * dt

        # Apply damping
        obj.velocity *= damping
        obj.angular_velocity *= damping

        # Stop when nearly stationary (prevents infinite jitter)
        if np.linalg.norm(obj.velocity) < 0.001:
            obj.velocity[:] = 0
        if np.linalg.norm(obj.angular_velocity) < 0.001:
            obj.angular_velocity[:] = 0

        return obj

    @staticmethod
    def check_collision_aabb(obj1: Object3D, obj2: Object3D) -> bool:
        """Check if two objects collide (AABB)."""
        min1, max1 = obj1.get_bounds()
        min2, max2 = obj2.get_bounds()

        return (min1[0] <= max2[0] and max1[0] >= min2[0] and
                min1[1] <= max2[1] and max1[1] >= min2[1] and
                min1[2] <= max2[2] and max1[2] >= min2[2])

    @staticmethod
    def resolve_collision(obj1: Object3D, obj2: Object3D) -> None:
        """Simple collision response (separate objects)."""
        delta = obj2.position - obj1.position
        distance = np.linalg.norm(delta)

        if distance > 0:
            direction = delta / distance
            min1, max1 = obj1.get_bounds()
            min2, max2 = obj2.get_bounds()

            size1 = (max1 - min1) / 2
            size2 = (max2 - min2) / 2
            overlap = np.linalg.norm(size1) + np.linalg.norm(size2) - distance

            if overlap > 0:
                separation = direction * overlap / 2
                obj1.position -= separation
                obj2.position += separation


# ============================================================================
# OBJECT MANIPULATOR
# ============================================================================

class ObjectManipulator:
    """
    Maps gestures to 3D object transformations.

    Maintains:
    - Selected object
    - Interaction mode
    - Gesture-to-object mapping
    """

    def __init__(self, world_objects: Optional[Dict[str, Object3D]] = None):
        """
        Initialize manipulator.

        Args:
            world_objects: Dictionary of objects in scene
        """
        self.world_objects: Dict[str, Object3D] = world_objects or {}
        self.selected_object: Optional[Object3D] = None
        self.grab_point: Optional[np.ndarray] = None  # Offset from object center

        # Gesture tracking
        self.last_gesture: Optional[str] = None
        self.gesture_start_pos: Optional[np.ndarray] = None

    def add_object(self, obj: Object3D) -> None:
        """Add object to scene."""
        self.world_objects[obj.id] = obj

    def select_object_at(self, position: np.ndarray, radius: float = 0.1) -> Optional[Object3D]:
        """
        Find and select object near position.

        Args:
            position: Test position [x, y, z]
            radius: Selection radius

        Returns:
            Selected object or None
        """
        closest_obj = None
        closest_dist = radius

        for obj in self.world_objects.values():
            dist = obj.distance_to_point(position)
            if dist < closest_dist:
                closest_obj = obj
                closest_dist = dist

        if closest_obj:
            self.selected_object = closest_obj
            self.grab_point = closest_obj.position - position
            closest_obj.interaction_mode = InteractionMode.SELECTED
            print(f"[ObjectManipulator] Selected {closest_obj.id}")

        return closest_obj

    def apply_gesture(self,
                     gesture_name: str,
                     hand_position: Optional[np.ndarray] = None,
                     hand_motion: Optional[np.ndarray] = None,
                     two_hand_gesture: Optional[str] = None) -> None:
        """
        Apply gesture to selected object.

        Args:
            gesture_name: Gesture type (pinch, swipe_left, circle, etc.)
            hand_position: Hand position for grab/drag
            hand_motion: Hand velocity for motion inference
            two_hand_gesture: Two-hand gesture (zoom, rotate, etc.)
        """
        if not self.selected_object:
            return

        obj = self.selected_object

        # Single-hand gestures
        if gesture_name == "pinch":
            obj.interaction_mode = InteractionMode.GRABBED
            if hand_position is not None:
                self.grab_point = obj.position - hand_position

        elif gesture_name == "open":
            if obj.interaction_mode == InteractionMode.GRABBED:
                obj.interaction_mode = InteractionMode.IDLE

        elif gesture_name == "swipe_left":
            obj.velocity[0] -= hand_motion[0] if hand_motion is not None else 0.5

        elif gesture_name == "swipe_right":
            obj.velocity[0] += hand_motion[0] if hand_motion is not None else 0.5

        elif gesture_name == "swipe_up":
            obj.velocity[1] += hand_motion[1] if hand_motion is not None else 0.5

        elif gesture_name == "swipe_down":
            obj.velocity[1] -= hand_motion[1] if hand_motion is not None else 0.5

        elif gesture_name == "circle":
            # Rotate around Z axis
            rotation_speed = 2.0
            obj.angular_velocity[2] += rotation_speed

        # Two-hand gestures
        if two_hand_gesture == "zoom_in":
            obj.scale *= 1.1
            obj.scale = np.clip(obj.scale, 0.5, 5.0)

        elif two_hand_gesture == "zoom_out":
            obj.scale *= 0.9
            obj.scale = np.clip(obj.scale, 0.5, 5.0)

        elif two_hand_gesture in ["rotate_cw", "rotate_ccw"]:
            direction = 1.0 if two_hand_gesture == "rotate_cw" else -1.0
            obj.angular_velocity += np.array([0, direction * 2.0, 0])

        elif two_hand_gesture == "mirror":
            obj.scale[0] *= -1  # Mirror X axis

        elif two_hand_gesture == "push":
            obj.velocity[2] -= 0.5

        elif two_hand_gesture == "pull":
            obj.velocity[2] += 0.5

    def update_grab(self, hand_position: np.ndarray,
                   hold_fixed: bool = True) -> None:
        """
        Update grabbed object position to follow hand.

        Args:
            hand_position: Current hand position
            hold_fixed: If True, object sticks to hand exactly
        """
        if not self.selected_object or self.selected_object.interaction_mode != InteractionMode.GRABBED:
            return

        obj = self.selected_object

        if hold_fixed:
            # Object tracks hand exactly
            target_pos = hand_position + self.grab_point
            obj.position = target_pos
            obj.velocity[:] = 0
        else:
            # Object follows hand with inertia
            target_pos = hand_position + self.grab_point
            delta = target_pos - obj.position
            obj.velocity = delta / 0.033  # Assume ~30fps

    def update_hand_position(self, hand_position: np.ndarray,
                             hold_fixed: bool = True) -> None:
        """Backward-compatible alias used by the main pipeline."""
        self.update_grab(hand_position, hold_fixed=hold_fixed)
    def update_physics(self, dt: float, damping: float = 0.95,
                      gravity: Optional[np.ndarray] = None) -> None:
        """Update all objects in scene."""
        for obj in self.world_objects.values():
            if obj.interaction_mode != InteractionMode.GRABBED:
                BasicPhysics.apply_physics(obj, dt, damping, gravity)

        # Collision detection
        objects = list(self.world_objects.values())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if BasicPhysics.check_collision_aabb(objects[i], objects[j]):
                    BasicPhysics.resolve_collision(objects[i], objects[j])

    def get_all_transforms(self) -> Dict[str, Dict]:
        """Get all object transforms as serializable dict."""
        transforms = {}
        for obj_id, obj in self.world_objects.items():
            transforms[obj_id] = {
                "position": obj.position.tolist(),
                "rotation": obj.rotation.tolist(),
                "scale": obj.scale.tolist(),
            }
        return transforms

    def release_all(self) -> None:
        """Release all grabbed objects."""
        for obj in self.world_objects.values():
            if obj.interaction_mode == InteractionMode.GRABBED:
                obj.interaction_mode = InteractionMode.IDLE


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_3d_object_manipulation():
    """Example: Manipulate 3D objects with gestures."""
    print("\n" + "="*70)
    print("EXAMPLE: 3D OBJECT MANIPULATION")
    print("="*70)

    # Create scene
    manipulator = ObjectManipulator()

    # Add objects
    cube = Object3D(id="cube", position=np.array([0.5, 0.5, 0.5]))
    sphere = Object3D(id="sphere", position=np.array([0.3, 0.3, 0.3]))
    manipulator.add_object(cube)
    manipulator.add_object(sphere)

    print(f"\nCreated {len(manipulator.world_objects)} objects")

    # Simulate interaction
    print("\nSimulating gestures...")

    # Select cube
    manipulator.select_object_at(np.array([0.5, 0.5, 0.5]))

    # Apply gestures
    manipulator.apply_gesture("pinch", hand_position=np.array([0.5, 0.5, 0.5]))
    print(f"Cube mode: {manipulator.selected_object.interaction_mode.value}")

    manipulator.apply_gesture("swipe_left", hand_motion=np.array([0.1, 0, 0]))
    print(f"Cube velocity: {manipulator.selected_object.velocity}")

    manipulator.apply_gesture("zoom_in", two_hand_gesture="zoom_in")
    print(f"Cube scale: {manipulator.selected_object.scale}")

    # Update physics
    for frame in range(10):
        manipulator.update_physics(dt=0.033, damping=0.9)

    print("\nObject manipulation example complete")
    transforms = manipulator.get_all_transforms()
    print(f"Final transforms: {transforms}")


if __name__ == "__main__":
    example_3d_object_manipulation()
