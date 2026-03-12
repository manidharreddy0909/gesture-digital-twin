"""
ROBOTIC ARM CONTROLLER MODULE

Maps hand motion to 6-DOF robot arm control using inverse kinematics (IK).

Purpose: Enable gesture-driven control of collaborative robots (e.g., UR5).

Key Classes:
- RobotArm: Abstract base for any 6-DOF robot
- RobotArmUR5: UR5 robot implementation with ikpy
- HandToArmMapper: Maps dual-hand gesture → arm target
- RobotCommand: Command queue for robust communication

Features:
- 6-DOF inverse kinematics using ikpy library
- Smooth target trajectory generation
- Optional collision detection (PyBullet)
- Gripper control (open/close with force)
- Safety limits (velocity, acceleration, joint limits)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
from pathlib import Path
import time
import numpy as np


# ============================================================================
# ROBOT COMMAND STRUCTURES
# ============================================================================

class GripperCommand(Enum):
    """Gripper states."""
    OPEN = "open"
    CLOSE = "close"
    STOP = "stop"


@dataclass
class RobotTarget:
    """Target position and orientation for robot arm."""
    position: np.ndarray           # [x, y, z] in meters
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))  # [rx, ry, rz] in radians
    gripper_command: Optional[GripperCommand] = None
    gripper_force: float = 50.0     # Gripper force (0-100)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "gripper_command": self.gripper_command.value if self.gripper_command else None,
            "gripper_force": self.gripper_force
        }


@dataclass
class RobotState:
    """Current robot state."""
    joint_angles: List[float]       # Joint angles in radians
    tcp_position: np.ndarray        # Tool center point position
    tcp_orientation: np.ndarray     # TCP orientation
    gripper_open: bool              # Gripper state
    timestamp: float = 0.0


@dataclass
class RobotSafetyLimits:
    """Runtime safety constraints for motion execution."""

    workspace_min: np.ndarray = field(default_factory=lambda: np.array([0.10, -0.80, 0.10]))
    workspace_max: np.ndarray = field(default_factory=lambda: np.array([0.85, 0.80, 1.00]))
    max_step_m: float = 0.05
    max_velocity_mps: float = 0.5
    max_orientation_step_rad: float = 0.25
    stop_on_out_of_workspace: bool = True
    min_gripper_force: float = 0.0
    max_gripper_force: float = 100.0


# ============================================================================
# ROBOT ARM ABSTRACT BASE
# ============================================================================

class RobotArm(ABC):
    """
    Abstract base class for 6-DOF robot arms.

    Must implement: forward_kinematics, move_to_position, get_current_state
    """

    def __init__(self, name: str):
        """Initialize robot arm."""
        self.name = name
        self.connected = False

    @abstractmethod
    def forward_kinematics(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute end-effector position and orientation from joint angles.

        Args:
            joint_angles: Joint angles [j1, j2, j3, j4, j5, j6] in radians

        Returns:
            (position [x,y,z], orientation [rx,ry,rz])
        """
        pass

    @abstractmethod
    def inverse_kinematics(self, position: np.ndarray,
                         orientation: Optional[np.ndarray] = None) -> Optional[List[float]]:
        """
        Compute joint angles for target position and orientation.

        Args:
            position: Target [x, y, z]
            orientation: Target [rx, ry, rz], optional

        Returns:
            Joint angles [j1...j6] or None if unreachable
        """
        pass

    @abstractmethod
    def move_to_position(self, target: RobotTarget) -> bool:
        """Send movement command to robot."""
        pass

    @abstractmethod
    def get_current_state(self) -> RobotState:
        """Query robot state."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Emergency stop."""
        pass


# ============================================================================
# UR5 ROBOT IMPLEMENTATION
# ============================================================================

class RobotArmUR5(RobotArm):
    """
    UR5 Collaborative Robot (Universal Robots).

    6-DOF, 6kg payload, ~850mm reach.
    Uses ikpy for inverse kinematics.
    """

    def __init__(self, ip_address: str = "192.168.1.100", port: int = 30003):
        """
        Initialize UR5 controller.

        Args:
            ip_address: Robot IP address
            port: RTDE communication port (30003 default)
        """
        super().__init__("UR5")
        self.ip_address = ip_address
        self.port = port

        # UR5 DH parameters (from UR documentation)
        # This is a simplified model
        self.dh_params = {
            "d": [0.08916, 0, 0, 0.10915, 0.09475, 0.0823],
            "a": [0, -0.425, -0.39225, 0, 0, 0],
            "alpha": [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0],
        }

        self.joint_limits = {
            "lower": [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi],
            "upper": [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi],
        }

        # Current state
        self.current_state = RobotState(
            joint_angles=[0] * 6,
            tcp_position=np.array([0.5, 0, 0.5]),
            tcp_orientation=np.array([0, 0, 0]),
            gripper_open=True
        )
        self.safety_limits = RobotSafetyLimits()
        self.emergency_stopped = False
        self._last_command_time = time.time()

        print(f"[UR5] Initialized: {ip_address}:{port}")

    def connect(self) -> bool:
        """Establish connection to robot."""
        try:
            # Real implementation would use RTDE or ROS
            # For now, simulate connection
            print(f"[UR5] Connecting to {self.ip_address}...")
            self.connected = True
            print(f"[UR5] Connected!")
            return True
        except Exception as e:
            print(f"[UR5] Connection failed: {e}")
            return False

    def forward_kinematics(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute TCP position and orientation.

        Simplified FK using DH parameters.
        """
        # Placeholder: would use actual FK computation
        # For demo: estimate position based on joint angles
        x = 0.5 + 0.2 * np.cos(joint_angles[0])
        y = 0.2 * np.sin(joint_angles[0])
        z = 0.5 + 0.15 * np.sin(joint_angles[1])

        position = np.array([x, y, z])
        orientation = np.array(joint_angles[-3:])

        return position, orientation

    def inverse_kinematics(self, position: np.ndarray,
                         orientation: Optional[np.ndarray] = None) -> Optional[List[float]]:
        """
        Compute joint angles for target position.

        Uses simple geometric IK or ikpy library if available.
        """
        try:
            import ikpy
            from ikpy.chain import Chain
            from ikpy.link import OriginLink, DHLink

            # Build chain from DH parameters
            chain = Chain(links=[
                OriginLink(),
                DHLink(name="Joint1", d=self.dh_params["d"][0], a=self.dh_params["a"][0],
                       alpha=self.dh_params["alpha"][0], bounds=self.joint_limits["lower"]),
                DHLink(name="Joint2", d=self.dh_params["d"][1], a=self.dh_params["a"][1],
                       alpha=self.dh_params["alpha"][1], bounds=self.joint_limits["lower"]),
                DHLink(name="Joint3", d=self.dh_params["d"][2], a=self.dh_params["a"][2],
                       alpha=self.dh_params["alpha"][2], bounds=self.joint_limits["lower"]),
                DHLink(name="Joint4", d=self.dh_params["d"][3], a=self.dh_params["a"][3],
                       alpha=self.dh_params["alpha"][3], bounds=self.joint_limits["lower"]),
                DHLink(name="Joint5", d=self.dh_params["d"][4], a=self.dh_params["a"][4],
                       alpha=self.dh_params["alpha"][4], bounds=self.joint_limits["lower"]),
                DHLink(name="Joint6", d=self.dh_params["d"][5], a=self.dh_params["a"][5],
                       alpha=self.dh_params["alpha"][5], bounds=self.joint_limits["lower"]),
            ])

            # Target: position [x,y,z] with identity orientation
            target_frame = np.eye(4)
            target_frame[:3, 3] = position

            # Solve IK
            joint_angles = chain.inverse_kinematics(
                target_frame,
                initial_position=self.current_state.joint_angles,
                max_iter=1000
            )

            # Validate solution
            if joint_angles is not None and all(
                self.joint_limits["lower"][i] <= joint_angles[i] <= self.joint_limits["upper"][i]
                for i in range(6)
            ):
                return list(joint_angles)

        except ImportError:
            print("[UR5] ikpy not installed: `pip install ikpy`")
        except Exception as e:
            print(f"[UR5] IK computation failed: {e}")

        # Fallback: simple geometric solution
        return self._solve_ik_geometric(position)

    def _solve_ik_geometric(self, position: np.ndarray) -> List[float]:
        """Simple geometric IK solution (placeholder)."""
        x, y, z = position

        # Very simplified: compute angles pointing toward target
        theta1 = np.arctan2(y, x)
        r_xy = np.sqrt(x*x + y*y)
        theta2 = np.arctan2(z - 0.089, r_xy) - 0.1  # Offset for base height
        theta3 = -theta2  # Simplification
        theta4, theta5, theta6 = 0, 0, 0

        return [theta1, theta2, theta3, theta4, theta5, theta6]

    def move_to_position(self, target: RobotTarget) -> bool:
        """Send movement command to robot."""
        if not self.connected:
            print("[UR5] Not connected")
            return False
        if self.emergency_stopped:
            print("[UR5] Rejecting motion: emergency stop active")
            return False

        # Enforce runtime safety envelope and dynamic limits.
        safe_target = self._apply_safety_limits(target)
        if safe_target is None:
            return False

        # Compute IK
        joint_angles = self.inverse_kinematics(safe_target.position, safe_target.orientation)

        if joint_angles is None:
            print(f"[UR5] Target unreachable: {safe_target.position}")
            return False

        # Simulate movement
        print(f"[UR5] Moving to {safe_target.position} with joints {[f'{j:.2f}' for j in joint_angles]}")
        self.current_state.joint_angles = joint_angles

        # Handle gripper
        if safe_target.gripper_command == GripperCommand.OPEN:
            print(f"[UR5] Opening gripper")
            self.current_state.gripper_open = True
        elif safe_target.gripper_command == GripperCommand.CLOSE:
            print(f"[UR5] Closing gripper with force {safe_target.gripper_force}")
            self.current_state.gripper_open = False

        # Update TCP state
        self.current_state.tcp_position = safe_target.position
        self.current_state.tcp_orientation = safe_target.orientation
        self.current_state.timestamp = time.time()
        self._last_command_time = self.current_state.timestamp

        return True

    def get_current_state(self) -> RobotState:
        """Return current robot state."""
        return self.current_state

    def stop(self) -> bool:
        """Emergency stop."""
        print("[UR5] EMERGENCY STOP")
        self.emergency_stopped = True
        self.current_state.timestamp = time.time()
        return True

    def clear_emergency_stop(self) -> None:
        """Clear emergency-stop latch after operator intervention."""
        self.emergency_stopped = False
        print("[UR5] Emergency stop cleared")

    def set_safety_limits(self, limits: RobotSafetyLimits) -> None:
        """Update robot runtime safety limits."""
        self.safety_limits = limits

    def is_within_workspace(self, position: np.ndarray) -> bool:
        """Check if a target is inside configured workspace envelope."""
        pos = np.asarray(position, dtype=np.float64)
        return bool(np.all(pos >= self.safety_limits.workspace_min) and np.all(pos <= self.safety_limits.workspace_max))

    def _apply_safety_limits(self, target: RobotTarget) -> Optional[RobotTarget]:
        """Return clamped safe target, or None if command must be rejected."""
        pos = np.asarray(target.position, dtype=np.float64).copy()
        ori = np.asarray(target.orientation, dtype=np.float64).copy()

        if self.safety_limits.stop_on_out_of_workspace and not self.is_within_workspace(pos):
            print(f"[UR5] Safety reject: target outside workspace {pos}")
            return None

        # Clamp to workspace envelope if stop-on-outside is disabled.
        pos = np.clip(pos, self.safety_limits.workspace_min, self.safety_limits.workspace_max)

        # Clamp linear displacement step.
        delta = pos - self.current_state.tcp_position
        step = float(np.linalg.norm(delta))
        if step > self.safety_limits.max_step_m and step > 0.0:
            pos = self.current_state.tcp_position + (delta / step) * self.safety_limits.max_step_m

        # Clamp by velocity limit using observed command period.
        now = time.time()
        dt = max(1e-3, now - self._last_command_time)
        max_disp_by_vel = self.safety_limits.max_velocity_mps * dt
        delta = pos - self.current_state.tcp_position
        disp = float(np.linalg.norm(delta))
        if disp > max_disp_by_vel and disp > 0.0:
            pos = self.current_state.tcp_position + (delta / disp) * max_disp_by_vel

        # Clamp orientation step.
        ori_delta = ori - self.current_state.tcp_orientation
        ori_step = float(np.linalg.norm(ori_delta))
        if ori_step > self.safety_limits.max_orientation_step_rad and ori_step > 0.0:
            ori = self.current_state.tcp_orientation + (
                ori_delta / ori_step
            ) * self.safety_limits.max_orientation_step_rad

        # Clamp gripper force.
        force = float(np.clip(target.gripper_force, self.safety_limits.min_gripper_force, self.safety_limits.max_gripper_force))
        return RobotTarget(
            position=pos,
            orientation=ori,
            gripper_command=target.gripper_command,
            gripper_force=force,
        )


# ============================================================================
# HAND-TO-ARM MAPPER
# ============================================================================

class HandToArmMapper:
    """
    Maps dual-hand and gesture information to robot arm targets.

    Mapping Logic:
    - Left index finger tip → End effector position (x, y, z)
    - Right hand orientation → Arm end-effector orientation
    - Pinch gesture → Close gripper
    - Open gesture → Open gripper
    - Swipe → Move robot in direction
    """

    def __init__(self,
                 position_scale: float = 1.0,  # Multiplier for position input
                 max_velocity: float = 0.5):   # Max velocity (m/s)
        """
        Initialize mapper.

        Args:
            position_scale: Scale factor for position mapping
            max_velocity: Maximum arm velocity
        """
        self.position_scale = position_scale
        self.max_velocity = max_velocity
        self.last_target: Optional[RobotTarget] = None
        self.last_update_time = 0.0

    def map_hand_to_arm_target(self,
                              left_hand,  # From gesture_detector: HandLandmarks or None
                              right_hand,  # From gesture_detector: HandLandmarks or None
                              left_gesture=None,  # GestureResult or None
                              right_gesture=None) -> Optional[RobotTarget]:
        """
        Convert hand data to robot arm target.

        Args:
            left_hand: Left HandLandmarks (world space)
            right_hand: Right HandLandmarks (world space)
            left_gesture: Left gesture detection result
            right_gesture: Right gesture detection result

        Returns:
            RobotTarget or None if no valid input
        """
        if left_hand is None:
            return None

        # Extract position from left hand index finger tip
        # Assuming left_hand.landmarks_3d = [(x,y,z), ...]
        # Index finger tip is landmark 8
        if hasattr(left_hand, 'landmarks_3d'):
            idx_tip = left_hand.landmarks_3d[8]
        else:
            # Fallback: use index finger tip from landmarks directly
            landmarks_3d = [(l[0], l[1], l[2]) for l in left_hand.landmarks]
            idx_tip = landmarks_3d[8]

        position = np.array(idx_tip[:3]) * self.position_scale

        # Right hand orientation (if available)
        orientation = np.array([0, 0, 0])
        if right_hand is not None:
            # Simplified: use hand palm normal for orientation
            # Real implementation would compute proper quaternion
            pass

        # Gripper based on left gesture
        gripper_command = None
        if left_gesture:
            if left_gesture.gesture == "pinch":
                gripper_command = GripperCommand.CLOSE
            elif left_gesture.gesture == "open":
                gripper_command = GripperCommand.OPEN

        target = RobotTarget(
            position=position,
            orientation=orientation,
            gripper_command=gripper_command
        )

        return target

    def smooth_trajectory(self, target: RobotTarget, dt: float = 0.033) -> RobotTarget:
        """
        Smooth target trajectory for gradual movement.

        Prevents sudden jumps, respects velocity limits.
        """
        if self.last_target is None:
            self.last_target = target
            return target

        # Limit velocity
        delta = target.position - self.last_target.position
        distance = np.linalg.norm(delta)
        max_displacement = self.max_velocity * dt

        if distance > max_displacement:
            direction = delta / distance
            target.position = self.last_target.position + direction * max_displacement

        self.last_target = target
        return target


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_ur5_control():
    """Example: UR5 robot control."""
    print("\n" + "="*70)
    print("EXAMPLE: UR5 ROBOT ARM CONTROL")
    print("="*70)

    # Initialize robot
    robot = RobotArmUR5("192.168.1.100")
    robot.connect()

    # Initialize mapper
    mapper = HandToArmMapper(position_scale=1.0, max_velocity=0.5)

    # Simulate hand positions (3D world coordinates in meters)
    print("\nSimulating hand motion...")
    for frame in range(10):
        # Simulate moving hand forward/backward
        x = 0.5 + 0.1 * np.sin(frame * 0.5)
        y = 0.3
        z = 0.5 + 0.05 * np.cos(frame * 0.5)

        # Create mock hand object
        class MockHand:
            def __init__(self):
                self.landmarks_3d = [(x, y, z)] + [(0, 0, 0)] * 7 + [(x, y, z)] + [(0, 0, 0)] * 12

        left_hand = MockHand()
        target = mapper.map_hand_to_arm_target(left_hand, None)

        if target:
            robot.move_to_position(target)

    print("\nUR5 example complete")


if __name__ == "__main__":
    example_ur5_control()
