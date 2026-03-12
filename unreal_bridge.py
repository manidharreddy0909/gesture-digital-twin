"""
UNREAL ENGINE 5 BRIDGE MODULE

Real-time integration with Unreal Engine 5 for skeletal mesh animation and object manipulation.

Dual approach:
1. Direct UE5 Python API (when running in editor/with Python enabled)
2. WebSocket fallback (for PIE or networked execution)

Maps MediaPipe 21-point hand skeleton to UE5 skeletal mesh bone hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Any
import json
import numpy as np
from enum import Enum
from collections import deque
import time


# ============================================================================
# BONE MAPPING
# ============================================================================

# MediaPipe hand landmarks → UE5 skeletal mesh bones
LANDMARK_TO_BONE_NAME = {
    # Wrist
    0: "hand_wrist",

    # Thumb (1-4)
    1: "hand_thumb_01",
    2: "hand_thumb_02",
    3: "hand_thumb_03",
    4: "hand_thumb_04",

    # Index (5-8)
    5: "hand_index_01",
    6: "hand_index_02",
    7: "hand_index_03",
    8: "hand_index_04",

    # Middle (9-12)
    9: "hand_middle_01",
    10: "hand_middle_02",
    11: "hand_middle_03",
    12: "hand_middle_04",

    # Ring (13-16)
    13: "hand_ring_01",
    14: "hand_ring_02",
    15: "hand_ring_03",
    16: "hand_ring_04",

    # Pinky (17-20)
    17: "hand_pinky_01",
    18: "hand_pinky_02",
    19: "hand_pinky_03",
    20: "hand_pinky_04",
}


# ============================================================================
# COMMAND STRUCTURES
# ============================================================================

@dataclass
class BoneTransform:
    """Transform for a single bone."""
    bone_name: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]  # Euler angles in degrees


@dataclass
class SkeletalMeshUpdate:
    """Update for skeletal mesh."""
    hand_id: int
    hand_side: str              # "Left" or "Right"
    bone_transforms: List[BoneTransform]
    timestamp: float = 0.0

    def to_json(self) -> str:
        """Serialize to JSON for WebSocket transmission."""
        data = {
            "type": "skeletal_mesh_update",
            "hand_id": self.hand_id,
            "hand_side": self.hand_side,
            "timestamp": self.timestamp,
            "bones": [
                {
                    "name": bt.bone_name,
                    "position": bt.position,
                    "rotation": bt.rotation
                }
                for bt in self.bone_transforms
            ]
        }
        return json.dumps(data)


@dataclass
class UnrealFeedbackEvent:
    """Feedback event flowing from simulation to control layer."""

    event_type: str
    severity: str
    message: str
    data: Dict
    timestamp: float = 0.0

    def to_json(self) -> str:
        payload = {
            "type": "feedback_event",
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp or time.time(),
        }
        return json.dumps(payload)

    @staticmethod
    def from_dict(data: Dict) -> "UnrealFeedbackEvent":
        return UnrealFeedbackEvent(
            event_type=str(data.get("event_type", "unknown")),
            severity=str(data.get("severity", "info")),
            message=str(data.get("message", "")),
            data=dict(data.get("data", {})),
            timestamp=float(data.get("timestamp", time.time())),
        )


# ============================================================================
# UNREAL ENGINE BACKENDS
# ============================================================================

class UnrealBackend(ABC):
    """Abstract base for Unreal Engine integration."""

    @abstractmethod
    def send_skeletal_update(self, update: SkeletalMeshUpdate) -> bool:
        """Send skeletal mesh update."""
        pass

    @abstractmethod
    def move_actor(self, actor_name: str, position: Tuple[float, float, float],
                  rotation: Tuple[float, float, float]) -> bool:
        """Move actor in world."""
        pass

    @abstractmethod
    def set_actor_scale(self, actor_name: str, scale: Tuple[float, float, float]) -> bool:
        """Set actor scale."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection."""
        pass

    @abstractmethod
    def send_feedback_event(self, event: UnrealFeedbackEvent) -> bool:
        """Send feedback event to Unreal/UI side."""
        pass

    @abstractmethod
    def poll_feedback_events(self) -> List[UnrealFeedbackEvent]:
        """Poll events from Unreal side (if transport supports it)."""
        pass


class UnrealPythonAPIBridge(UnrealBackend):
    """
    Direct integration using UE5 Python API.

    Requires:
    - UE5 editor running with Python plugin enabled
    - Python >= 3.9 matching UE5 Python version
    """

    def __init__(self, project_path: str = None):
        """
        Initialize bridge.

        Args:
            project_path: Path to UE5 project (for context)
        """
        self.project_path = project_path
        self.connected = False
        self.ue_api = None
        self._feedback_queue: deque = deque(maxlen=200)
        self._feedback_callback: Optional[Callable[[UnrealFeedbackEvent], None]] = None

        # Try to import UE5 Python API
        try:
            import unreal
            self.ue_api = unreal
            print("[UnrealPythonAPIBridge] UE5 Python API available")
        except ImportError:
            print("[UnrealPythonAPIBridge] UE5 Python API not available")
            print("  Run from UE5 editor with Python plugin enabled")

    def connect(self) -> bool:
        """Connect to Unreal Editor."""
        if self.ue_api is None:
            print("[UnrealPythonAPIBridge] Cannot connect: UE API not available")
            return False

        try:
            # Check if running in editor
            if self.ue_api.get_editor_world():
                self.connected =True
                print("[UnrealPythonAPIBridge] Connected to UE5 Editor")
                return True
        except Exception as e:
            print(f"[UnrealPythonAPIBridge] Connection failed: {e}")

        return False

    def send_skeletal_update(self, update: SkeletalMeshUpdate) -> bool:
        """Update skeletal mesh bones."""
        if not self.connected or not self.ue_api:
            return False

        try:
            # Get skeletal mesh actor
            world = self.ue_api.get_editor_world()
            actor_name = f"Hand_{update.hand_side}"

            # Find actor (simplified)
            # Real implementation would use ue_api.get_actor() and iterate
            print(f"[UnrealPythonAPIBridge] Updating {actor_name} with {len(update.bone_transforms)} bones")

            for bt in update.bone_transforms:
                # Update bone transform
                # ue_api.set_bone_transform(actor_name, bt.bone_name,
                #                           position=bt.position,
                #                           rotation=bt.rotation)
                pass

            return True

        except Exception as e:
            print(f"[UnrealPythonAPIBridge] Update failed: {e}")
            return False

    def move_actor(self, actor_name: str,
                  position: Tuple[float, float, float],
                  rotation: Tuple[float, float, float] = (0, 0, 0)) -> bool:
        """Move actor in world."""
        if not self.connected or not self.ue_api:
            return False

        try:
            print(f"[UnrealPythonAPIBridge] Moving {actor_name} to {position}")
            # ue_api.set_actor_location(actor_name, position)
            # ue_api.set_actor_rotation(actor_name, rotation)
            return True
        except Exception as e:
            print(f"[UnrealPythonAPIBridge] Move failed: {e}")
            return False

    def set_actor_scale(self, actor_name: str,
                       scale: Tuple[float, float, float]) -> bool:
        """Set actor scale."""
        if not self.connected or not self.ue_api:
            return False

        try:
            print(f"[UnrealPythonAPIBridge] Setting {actor_name} scale to {scale}")
            # ue_api.set_actor_scale(actor_name, scale)
            return True
        except Exception as e:
            print(f"[UnrealPythonAPIBridge] Scale failed: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect."""
        self.connected = False
        print("[UnrealPythonAPIBridge] Disconnected")
        return True

    def send_feedback_event(self, event: UnrealFeedbackEvent) -> bool:
        """Send feedback event into UE editor channel (or queue fallback)."""
        if not self.connected:
            return False
        try:
            # Production path would push to UE message bus/log panel.
            self._feedback_queue.append(event)
            if self._feedback_callback is not None:
                self._feedback_callback(event)
            return True
        except Exception as e:
            print(f"[UnrealPythonAPIBridge] Feedback send failed: {e}")
            return False

    def poll_feedback_events(self) -> List[UnrealFeedbackEvent]:
        """Return queued feedback events."""
        events = list(self._feedback_queue)
        self._feedback_queue.clear()
        return events

    def register_feedback_callback(self, callback: Callable[[UnrealFeedbackEvent], None]) -> None:
        """Register callback for immediate feedback handling."""
        self._feedback_callback = callback


class UnrealWebSocketBridge(UnrealBackend):
    """
    WebSocket-based integration for remote Unreal Engine.

    Sends JSON commands over WebSocket to Unreal plugin.
    """

    def __init__(self, endpoint: str = "ws://localhost:8765"):
        """
        Initialize bridge.

        Args:
            endpoint: WebSocket endpoint URL
        """
        self.endpoint = endpoint
        self.connected = False
        self.websocket = None
        self._feedback_queue: deque = deque(maxlen=400)
        self._ws_sync: Optional[Any] = None
        self._sync_connect = None
        # Opening a new WS connection per message is extremely slow on weak machines.
        # Keep this disabled by default; enable only for debugging compatibility.
        self.allow_short_lived_fallback: bool = False

    def _send_json_sync(self, payload: str) -> bool:
        """Send one JSON payload using persistent sync socket or fallback."""
        if self._ws_sync is not None:
            try:
                self._ws_sync.send(payload)
                return True
            except Exception as e:
                print(f"[UnrealWebSocketBridge] Persistent send failed: {e}")
                self.connected = False
                # Try to re-establish persistent socket once.
                if self._sync_connect is not None:
                    try:
                        self._ws_sync = self._sync_connect(self.endpoint, open_timeout=2)
                        self.connected = True
                        self._ws_sync.send(payload)
                        return True
                    except Exception as reconnect_error:
                        print(f"[UnrealWebSocketBridge] Reconnect failed: {reconnect_error}")
                        self._ws_sync = None

        try:
            import asyncio
            import websockets

            async def send_once():
                async with websockets.connect(self.endpoint) as ws:
                    await ws.send(payload)

            asyncio.run(send_once())
            return True
        except Exception as e:
            print(f"[UnrealWebSocketBridge] Send failed: {e}")
            return False

    def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            print(f"[UnrealWebSocketBridge] Connecting to {self.endpoint}...")
            # Preferred path: persistent sync connection (avoids reconnect per frame).
            try:
                from websockets.sync.client import connect as ws_sync_connect
                self._sync_connect = ws_sync_connect
                self._ws_sync = ws_sync_connect(self.endpoint, open_timeout=2)
                self.connected = True
                print("[UnrealWebSocketBridge] Connected (persistent)")
                return True
            except Exception:
                self._ws_sync = None
                self._sync_connect = None

            # Optional fallback path: short-lived connection probe.
            if not self.allow_short_lived_fallback:
                print("[UnrealWebSocketBridge] Persistent connect unavailable; fallback disabled")
                self.connected = False
                return False

            ok = self._send_json_sync(
                json.dumps(
                    {
                        "type": "feedback_event",
                        "severity": "info",
                        "message": "bridge_connect_probe",
                        "event_type": "bridge",
                        "data": {},
                    }
                )
            )
            self.connected = ok
            if ok:
                print("[UnrealWebSocketBridge] Connected (short-lived fallback)")
                return True
            print("[UnrealWebSocketBridge] Connection probe failed")
            return False

        except ImportError:
            print("[UnrealWebSocketBridge] websockets library not installed")
            print("  pip install websockets")
            return False
        except Exception as e:
            print(f"[UnrealWebSocketBridge] Connection failed: {e}")
            return False

    def _drain_incoming(self, max_messages: int = 16) -> None:
        """Drain available inbound WS messages into feedback queue."""
        if self._ws_sync is None or not self.connected:
            return

        for _ in range(max_messages):
            try:
                message = self._ws_sync.recv(timeout=0)
            except Exception:
                break

            if not message:
                break
            try:
                payload = json.loads(message)
            except Exception:
                continue

            if payload.get("type") == "feedback_event":
                self._feedback_queue.append(UnrealFeedbackEvent.from_dict(payload))

    def send_skeletal_update(self, update: SkeletalMeshUpdate) -> bool:
        """Send skeletal update to Unreal."""
        if not self.connected:
            return False

        ok = self._send_json_sync(update.to_json())
        if ok:
            print("[UnrealWebSocketBridge] Sent skeletal update")
        return ok

    def move_actor(self, actor_name: str,
                  position: Tuple[float, float, float],
                  rotation: Tuple[float, float, float] = (0, 0, 0)) -> bool:
        """Send actor move command."""
        if not self.connected:
            return False

        command = {
            "type": "move_actor",
            "actor_name": actor_name,
            "position": position,
            "rotation": rotation
        }
        return self._send_json_sync(json.dumps(command))

    def set_actor_scale(self, actor_name: str,
                       scale: Tuple[float, float, float]) -> bool:
        """Send actor scale command."""
        if not self.connected:
            return False

        command = {
            "type": "set_actor_scale",
            "actor_name": actor_name,
            "scale": scale
        }
        return self._send_json_sync(json.dumps(command))

    def disconnect(self) -> bool:
        """Disconnect."""
        if self._ws_sync is not None:
            try:
                self._ws_sync.close()
            except Exception:
                pass
        self._ws_sync = None
        self.websocket = None
        self.connected = False
        print("[UnrealWebSocketBridge] Disconnected")
        return True

    def send_feedback_event(self, event: UnrealFeedbackEvent) -> bool:
        """Send structured feedback event over WebSocket transport."""
        if not self.connected:
            return False

        payload = event.to_json()
        ok = self._send_json_sync(payload)
        if ok:
            self._feedback_queue.append(event)
        return ok

    def poll_feedback_events(self) -> List[UnrealFeedbackEvent]:
        """
        Poll received feedback events.

        In real async runtime this should drain inbound websocket messages.
        """
        self._drain_incoming()
        events = list(self._feedback_queue)
        self._feedback_queue.clear()
        return events


# ============================================================================
# HAND SKELETON CONVERTER
# ============================================================================

class HandSkeletonConverter:
    """Convert MediaPipe hand landmarks to UE5 skeletal mesh transforms."""

    @staticmethod
    def landmarks_to_bone_transforms(
        landmarks_3d: List[Tuple[float, float, float]],
        hand_side: str = "Left"
    ) -> List[BoneTransform]:
        """
        Convert 3D landmarks to bone transforms.

        Args:
            landmarks_3d: 21 points in 3D world coordinates
            hand_side: "Left" or "Right"

        Returns:
            List of BoneTransform objects
        """
        transforms = []

        for landmark_idx, landmark_3d in enumerate(landmarks_3d):
            if landmark_idx not in LANDMARK_TO_BONE_NAME:
                continue

            bone_name = LANDMARK_TO_BONE_NAME[landmark_idx]

            # Position: use landmark position directly
            position = landmark_3d

            # Rotation: simplified (would compute from joint angles in production)
            # For demo: use landmark position to estimate orientation
            rotation = (0, 0, 0)  # Placeholder

            transform = BoneTransform(
                bone_name=bone_name,
                position=position,
                rotation=rotation
            )
            transforms.append(transform)

        return transforms

    @staticmethod
    def compute_joint_angles(landmarks_3d: List[Tuple[float, float, float]]) \
            -> Dict[str, Tuple[float, float, float]]:
        """
        Compute joint angles from landmarks (simplified).

        Real implementation would use proper forward kinematics.
        """
        angles = {}

        # Simplified: estimate angles from bone lengths
        for i in range(len(landmarks_3d) - 1):
            p1 = np.array(landmarks_3d[i])
            p2 = np.array(landmarks_3d[i + 1])

            # Vector between points
            v = p2 - p1
            length = np.linalg.norm(v)

            if length > 0:
                # Azimuth angle (XY plane)
                theta = np.arctan2(v[1], v[0])
                # Altitude angle (Z)
                phi = np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2))

                angles[f"joint_{i}"] = (float(np.degrees(theta)), float(np.degrees(phi)), 0)

        return angles


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_bridge_usage():
    """Example: UE5 bridge communication."""
    print("\n" + "="*70)
    print("EXAMPLE: UNREAL ENGINE 5 BRIDGE")
    print("="*70)

    # Try Python API
    bridge = UnrealPythonAPIBridge()
    if not bridge.connect():
        print("\nUsing WebSocket fallback...")
        bridge = UnrealWebSocketBridge("ws://localhost:8765")
        bridge.connect()

    # Simulate hand landmarks
    landmarks_3d = [
        (0.5, 0.5, 0.5),   # Wrist
        (0.55, 0.45, 0.55),  # Thumb 1
        (0.58, 0.40, 0.60),  # Thumb 2
        (0.60, 0.35, 0.65),  # Thumb 3
        (0.62, 0.30, 0.70),  # Thumb tip
        # ... 16 more landmarks
    ] + [(0.5 + 0.1*i%5, 0.5, 0.5 + 0.1*i//5) for i in range(16)]

    # Convert to bone transforms
    transforms = HandSkeletonConverter.landmarks_to_bone_transforms(landmarks_3d, "Left")

    # Create update
    update = SkeletalMeshUpdate(
        hand_id=0,
        hand_side="Left",
        bone_transforms=transforms
    )

    # Send to Unreal
    bridge.send_skeletal_update(update)

    # Move object in scene
    bridge.move_actor("ManipulatedObject", (1.0, 2.0, 3.0))

    print("\nBridge example complete")


if __name__ == "__main__":
    example_bridge_usage()
