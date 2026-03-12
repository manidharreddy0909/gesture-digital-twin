"""
Advanced integration API for virtual robots, Unreal Engine, AR/VR, and other targets.

This module extends the basic FrameConsumer protocol with:
- Gesture-to-command mapping system
- Server/client backend support (socket, HTTP, shared memory)
- Unreal Engine integration (skeletal mesh animation)
- Robot arm control (cartesian coordinates, joint angles)
- Custom action binding system

Architecture:
- CommandMapper: Maps gestures/poses to high-level commands
- IntegrationBackend: Base class for different communication protocols
- SocketBackend: TCP/UDP socket communication
- HTTPBackend: HTTP/REST API integration
- UnrealEngine5Backend: Specialized for UE5 skeletal mesh control
- RobotArmBackend: Specialized for robot arm control
"""

from __future__ import annotations

import json
import socket
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum

from hand_tracker import HandLandmarks
from motion_analyzer import HandMotionInfo
from gesture_detector import GestureResult, TwoHandGestureResult


class CommandType(Enum):
    """Types of commands that can be sent to virtual targets."""
    CURSOR_MOVE = "cursor_move"
    GESTURE_DETECTED = "gesture_detected"
    HAND_POSE = "hand_pose"
    SKELETAL_ANIMATION = "skeletal_animation"
    OBJECT_MANIPULATION = "object_manipulation"
    ROBOT_MOVE = "robot_move"
    CUSTOM = "custom"


@dataclass
class Command:
    """A command to send to a virtual target system."""

    command_type: CommandType
    timestamp: float
    payload: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize command to JSON."""
        return json.dumps({
            "type": self.command_type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
        })

    @staticmethod
    def from_json(json_str: str) -> Command:
        """Deserialize command from JSON."""
        data = json.loads(json_str)
        return Command(
            command_type=CommandType(data["type"]),
            timestamp=data["timestamp"],
            payload=data["payload"],
        )


class CommandMapper:
    """
    Map hand gestures/poses to high-level commands.

    Provides a flexible system for binding gestures to actions.
    """

    def __init__(self):
        # gesture_name -> list of (action_name, callback) pairs
        self.gesture_actions: Dict[str, List[Tuple[str, Callable]]] = {}
        # pose_name -> action
        self.pose_actions: Dict[str, str] = {}

    def bind_gesture(
        self,
        gesture_name: str,
        action_name: str,
        callback: Callable[[GestureResult], Optional[Command]] = None,
    ) -> None:
        """Bind a gesture to an action with optional callback."""
        if gesture_name not in self.gesture_actions:
            self.gesture_actions[gesture_name] = []

        if callback is None:
            # Default callback: generate basic command
            callback = lambda g: Command(
                command_type=CommandType.GESTURE_DETECTED,
                timestamp=0.0,
                payload={"gesture": gesture_name, "action": action_name},
            )

        self.gesture_actions[gesture_name].append((action_name, callback))

    def bind_two_hand_gesture(
        self,
        gesture_name: str,
        action_name: str,
        callback: Callable[[TwoHandGestureResult], Optional[Command]] = None,
    ) -> None:
        """Bind a two-hand gesture to an action."""
        if callback is None:
            callback = lambda g: Command(
                command_type=CommandType.GESTURE_DETECTED,
                timestamp=0.0,
                payload={"gesture": gesture_name, "action": action_name, "data": g.extra},
            )
        self.gesture_actions[gesture_name] = [(action_name, callback)]

    def map_gesture(self, gesture: GestureResult, timestamp: float) -> List[Command]:
        """Map a gesture result to commands."""
        commands = []
        callbacks = self.gesture_actions.get(gesture.gesture, [])

        for action_name, callback in callbacks:
            try:
                cmd = callback(gesture)
                if cmd is not None:
                    cmd.timestamp = timestamp
                    commands.append(cmd)
            except Exception as e:
                print(f"Error in gesture callback: {e}")

        return commands

    def map_two_hand_gesture(
        self,
        gesture: TwoHandGestureResult,
        timestamp: float,
    ) -> List[Command]:
        """Map a two-hand gesture to commands."""
        commands = []
        callbacks = self.gesture_actions.get(gesture.gesture, [])

        for action_name, callback in callbacks:
            try:
                cmd = callback(gesture)
                if cmd is not None:
                    cmd.timestamp = timestamp
                    commands.append(cmd)
            except Exception as e:
                print(f"Error in two-hand gesture callback: {e}")

        return commands


class IntegrationBackend(ABC):
    """Base class for integration backends (socket, HTTP, Unreal, etc.)."""

    def __init__(self, name: str):
        self.name = name
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to target system."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from target system."""
        pass

    @abstractmethod
    def send_command(self, command: Command) -> bool:
        """Send a command to the target system."""
        pass

    @abstractmethod
    def send_hand_data(
        self,
        hands: List[HandLandmarks],
        gestures: Dict[int, GestureResult],
    ) -> bool:
        """Send hand tracking data to target."""
        pass


class SocketBackend(IntegrationBackend):
    """
    TCP socket backend for remote systems.

    Sends commands via TCP to a listening server.
    """

    def __init__(self, host: str = "localhost", port: int = 5000):
        super().__init__("socket")
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None

    def connect(self) -> bool:
        """Connect to remote server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((self.host, self.port))
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to {self.host}:{self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Close connection."""
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
        self.connected = False

    def send_command(self, command: Command) -> bool:
        """Send command via socket."""
        if not self.connected or self.socket is None:
            return False

        try:
            data = command.to_json().encode('utf-8') + b'\n'
            self.socket.sendall(data)
            return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            self.connected = False
            return False

    def send_hand_data(
        self,
        hands: List[HandLandmarks],
        gestures: Dict[int, GestureResult],
    ) -> bool:
        """Send hand data payload."""
        if not self.connected:
            return False

        try:
            payload = {
                "num_hands": len(hands),
                "hands": [
                    {
                        "id": h.hand_id,
                        "handedness": h.handedness,
                        "index_tip": h.index_finger_tip,
                    }
                    for h in hands
                ],
                "gestures": [
                    {
                        "hand_id": g.hand_id,
                        "gesture": g.gesture,
                        "confidence": g.confidence,
                    }
                    for g in gestures.values()
                ],
            }

            cmd = Command(
                command_type=CommandType.HAND_POSE,
                timestamp=0.0,
                payload=payload,
            )
            return self.send_command(cmd)
        except Exception as e:
            print(f"Error sending hand data: {e}")
            return False


class HTTPBackend(IntegrationBackend):
    """
    HTTP/REST backend for web-based systems and Unreal Engine REST API.

    Posts commands to a specified HTTP endpoint.
    """

    def __init__(self, endpoint_url: str = "http://localhost:8000/api/commands"):
        super().__init__("http")
        self.endpoint_url = endpoint_url

    def connect(self) -> bool:
        """HTTP backend doesn't require persistent connection."""
        self.connected = True
        return True

    def disconnect(self) -> None:
        """HTTP backend cleanup."""
        self.connected = False

    def send_command(self, command: Command) -> bool:
        """Send command via HTTP POST."""
        try:
            import urllib.request
            data = command.to_json().encode('utf-8')
            req = urllib.request.Request(
                self.endpoint_url,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=2.0) as response:
                return response.status == 200
        except Exception as e:
            print(f"Failed to send HTTP command: {e}")
            return False

    def send_hand_data(
        self,
        hands: List[HandLandmarks],
        gestures: Dict[int, GestureResult],
    ) -> bool:
        """Send hand data via HTTP."""
        payload = {
            "num_hands": len(hands),
            "hands": [
                {
                    "id": h.hand_id,
                    "handedness": h.handedness,
                    "index_tip": h.index_finger_tip,
                }
                for h in hands
            ],
        }

        cmd = Command(
            command_type=CommandType.HAND_POSE,
            timestamp=0.0,
            payload=payload,
        )
        return self.send_command(cmd)


class UnrealEngine5Backend(IntegrationBackend):
    """
    Specialized backend for Unreal Engine 5 skeletal mesh control.

    Maps hand pose to skeletal mesh bone rotations.
    """

    def __init__(self, endpoint_url: str = "http://localhost:8000/api/skeletal"):
        super().__init__("ue5_skeletal")
        self.endpoint_url = endpoint_url
        # Map MediaPipe landmark indices to UE5 bone names
        self.landmark_to_bone = self._create_bone_mapping()

    def _create_bone_mapping(self) -> Dict[int, str]:
        """Create mapping from MediaPipe landmarks to UE5 skeleton bones."""
        return {
            # Thumb (4 -> tip)
            1: "hand_thumb_01",
            2: "hand_thumb_02",
            3: "hand_thumb_03",
            4: "hand_thumb_04",
            # Index (8 -> tip)
            5: "hand_index_01",
            6: "hand_index_02",
            7: "hand_index_03",
            8: "hand_index_04",
            # Middle
            9: "hand_middle_01",
            10: "hand_middle_02",
            11: "hand_middle_03",
            12: "hand_middle_04",
            # Ring
            13: "hand_ring_01",
            14: "hand_ring_02",
            15: "hand_ring_03",
            16: "hand_ring_04",
            # Pinky
            17: "hand_pinky_01",
            18: "hand_pinky_02",
            19: "hand_pinky_03",
            20: "hand_pinky_04",
        }

    def connect(self) -> bool:
        """Verify connection to UE5."""
        self.connected = True
        return True

    def disconnect(self) -> None:
        """Cleanup."""
        self.connected = False

    def send_command(self, command: Command) -> bool:
        """Send skeletal animation command to UE5."""
        try:
            import urllib.request
            data = command.to_json().encode('utf-8')
            req = urllib.request.Request(
                self.endpoint_url,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=2.0) as response:
                return response.status == 200
        except Exception as e:
            print(f"Failed to send UE5 command: {e}")
            return False

    def send_hand_data(
        self,
        hands: List[HandLandmarks],
        gestures: Dict[int, GestureResult],
    ) -> bool:
        """Send hand landmarks as skeletal bone transforms to UE5."""
        for hand in hands:
            bone_transforms = {}

            for lm_idx, bone_name in self.landmark_to_bone.items():
                if lm_idx < len(hand.landmarks):
                    lm = hand.landmarks[lm_idx]
                    # Convert normalized coordinates to bone rotation (simplified)
                    bone_transforms[bone_name] = {
                        "position": [lm.x, lm.y, lm.z],
                        "rotation": [0.0, 0.0, 0.0],  # Default; can be computed from joint angles
                    }

            cmd = Command(
                command_type=CommandType.SKELETAL_ANIMATION,
                timestamp=0.0,
                payload={
                    "hand_id": hand.hand_id,
                    "handedness": hand.handedness,
                    "bone_transforms": bone_transforms,
                },
            )
            self.send_command(cmd)

        return True


class RobotArmBackend(IntegrationBackend):
    """
    Specialized backend for robot arm control.

    Converts hand pose to cartesian coordinates or joint angles.
    """

    def __init__(self, socket_or_url: str = "localhost:5000"):
        super().__init__("robot_arm")
        self.socket_or_url = socket_or_url
        self.http_backend = HTTPBackend(f"http://{socket_or_url}/api/robot")

    def connect(self) -> bool:
        """Connect to robot."""
        return self.http_backend.connect()

    def disconnect(self) -> None:
        """Disconnect from robot."""
        self.http_backend.disconnect()

    def send_command(self, command: Command) -> bool:
        """Send command to robot."""
        return self.http_backend.send_command(command)

    def send_hand_data(
        self,
        hands: List[HandLandmarks],
        gestures: Dict[int, GestureResult],
    ) -> bool:
        """Send hand data mapped to robot endpoint."""
        for hand in hands:
            # Use index fingertip as end-effector target
            x, y = hand.index_finger_tip

            cmd = Command(
                command_type=CommandType.ROBOT_MOVE,
                timestamp=0.0,
                payload={
                    "end_effector": {
                        "x": x,
                        "y": y,
                        "z": 0.5,  # Default height
                    },
                    "hand_id": hand.hand_id,
                },
            )
            self.send_command(cmd)

        return True
