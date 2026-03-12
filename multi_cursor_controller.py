"""
Multi-Cursor / Multi-Hand OS Control Module

Supports:
- Independent left and right hand cursor control
- Virtual cursors for UI representation
- Multi-window gesture mapping
- Cross-hand coordination

Architecture:
- CursorMode: Enum for different cursor modes
- VirtualCursor: In-app cursor representation
- MultiCursorController: Manages multiple cursors
- WindowMapper: Maps gestures to window regions
"""

from __future__ import annotations

import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


class CursorMode(Enum):
    """Different cursor operation modes."""
    SINGLE_LEFT = "single_left"  # Left hand controls OS cursor
    SINGLE_RIGHT = "single_right"  # Right hand controls OS cursor
    DUAL_INDEPENDENT = "dual_independent"  # Each hand controls separate virtual cursor
    DUAL_COLLABORATIVE = "dual_collaborative"  # Both hands control one cursor
    GESTURE_ONLY = "gesture_only"  # No cursor movement, gestures only


@dataclass
class CursorPosition:
    """Single cursor position with metadata."""
    x: int
    y: int
    hand_id: int
    timestamp: float
    is_active: bool = True
    confidence: float = 1.0


@dataclass
class VirtualCursor:
    """
    In-application virtual cursor representation.

    Separate from OS cursor, useful for testing without
    moving actual mouse cursor.
    """
    cursor_id: int = 0
    hand_id: int = 0
    x: float = 0.0
    y: float = 0.0
    is_visible: bool = True
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR (green)
    radius: int = 10
    trail: deque = field(default_factory=lambda: deque(maxlen=20))

    def update_position(self, x: float, y: float) -> None:
        """Update cursor position and add to trail."""
        self.trail.append((self.x, self.y))
        self.x = x
        self.y = y

    def get_trail_points(self) -> List[Tuple[float, float]]:
        """Get cursor trail for visualization."""
        return list(self.trail)


class MultiCursorController:
    """
    manages multiple independent cursors.

    Supports:
    - Dual cursor mode (left + right hand)
    - Single cursor mode (one hand chosen)
    - Gesture-only mode (no cursor movement)
    - Cross-hand coordination
    """

    def __init__(self, mode: CursorMode = CursorMode.SINGLE_LEFT):
        self.mode = mode
        self.cursors: Dict[int, VirtualCursor] = {}
        self.active_cursor: Optional[int] = None
        self.screen_width: int = 1920
        self.screen_height: int = 1080
        self.primary_hand: str = "left"

        # Track mouse state
        self._last_os_cursor_pos: Optional[Tuple[int, int]] = None

        # Cross-hand coordination
        self.last_left_pos: Optional[Tuple[float, float]] = None
        self.last_right_pos: Optional[Tuple[float, float]] = None
        self.hand_distance: float = 0.0

    def set_mode(self, mode: CursorMode) -> None:
        """Change cursor mode."""
        self.mode = mode
        print(f"[MultiCursorController] Switched to mode: {mode.value}")

    def update_cursor(
        self,
        hand_id: int,
        handedness: str,
        x_norm: float,
        y_norm: float,
        timestamp: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Update cursor position for a hand.

        Args:
            hand_id: MediaPipe hand ID (0, 1, ...)
            handedness: "Left" or "Right"
            x_norm, y_norm: Normalized coordinates [0, 1]
            timestamp: Current time

        Returns:
            (x_screen, y_screen) OS cursor position or None
        """
        # Convert normalized to screen coordinates
        x_screen = int(x_norm * (self.screen_width - 1))
        y_screen = int(y_norm * (self.screen_height - 1))

        # Handle based on mode
        if self.mode == CursorMode.SINGLE_LEFT:
            if handedness.lower() == "left":
                self.active_cursor = hand_id
                return self._move_os_cursor(x_screen, y_screen)
            return None

        elif self.mode == CursorMode.SINGLE_RIGHT:
            if handedness.lower() == "right":
                self.active_cursor = hand_id
                return self._move_os_cursor(x_screen, y_screen)
            return None

        elif self.mode == CursorMode.DUAL_INDEPENDENT:
            # Update virtual cursor for this hand
            if hand_id not in self.cursors:
                color = (0, 255, 0) if handedness.lower() == "left" else (255, 0, 0)
                self.cursors[hand_id] = VirtualCursor(
                    cursor_id=hand_id,
                    hand_id=hand_id,
                    color=color
                )

            cursor = self.cursors[hand_id]
            cursor.update_position(float(x_screen), float(y_screen))

            # Move OS cursor if this is the active/primary hand
            if handedness.lower() == self.primary_hand.lower():
                return self._move_os_cursor(x_screen, y_screen)
            return (x_screen, y_screen)

        elif self.mode == CursorMode.DUAL_COLLABORATIVE:
            # Both hands contribute to single cursor movement
            if hand_id == 0:
                self.last_left_pos = (x_norm, y_norm)
            else:
                self.last_right_pos = (x_norm, y_norm)

            # Merge positions if both hands present
            if self.last_left_pos and self.last_right_pos:
                avg_x = (self.last_left_pos[0] + self.last_right_pos[0]) / 2.0
                avg_y = (self.last_left_pos[1] + self.last_right_pos[1]) / 2.0
                x_merged = int(avg_x * (self.screen_width - 1))
                y_merged = int(avg_y * (self.screen_height - 1))
                return self._move_os_cursor(x_merged, y_merged)
            elif self.last_left_pos or self.last_right_pos:
                return self._move_os_cursor(x_screen, y_screen)

            return None

        elif self.mode == CursorMode.GESTURE_ONLY:
            # No cursor movement
            return None

        return None

    def _move_os_cursor(self, x: int, y: int) -> Tuple[int, int]:
        """Move actual OS cursor (if supported)."""
        # Platform-specific cursor movement is handled by cursor_controller
        # This is just position tracking
        self._last_os_cursor_pos = (x, y)
        return (x, y)

    def get_virtual_cursor(self, hand_id: int) -> Optional[VirtualCursor]:
        """Get virtual cursor for visualization."""
        return self.cursors.get(hand_id)

    def get_all_virtual_cursors(self) -> List[VirtualCursor]:
        """Get all active virtual cursors."""
        return list(self.cursors.values())

    def compute_hand_distance(self, left_pos: Optional[Tuple[float, float]],
                             right_pos: Optional[Tuple[float, float]]) -> float:
        """Compute distance between two hands (normalized coordinates)."""
        if not left_pos or not right_pos:
            return 0.0

        dx = right_pos[0] - left_pos[0]
        dy = right_pos[1] - left_pos[1]
        return (dx * dx + dy * dy) ** 0.5

    def is_dual_mode(self) -> bool:
        """Check if in dual-cursor mode."""
        return self.mode in [CursorMode.DUAL_INDEPENDENT, CursorMode.DUAL_COLLABORATIVE]


class WindowMapper:
    """
    Map gesture regions/windows for multi-display support.

    Allows different gestures in different screen regions
    to trigger different actions.
    """

    @dataclass
    class ScreenRegion:
        """Define a screen region for gesture mapping."""
        name: str
        x_min: float  # Normalized [0, 1]
        y_min: float
        x_max: float
        y_max: float
        gestures: Dict[str, str] = field(default_factory=dict)  # gesture -> action

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.regions: Dict[str, WindowMapper.ScreenRegion] = {}

        # Create default regions
        self._create_default_regions()

    def _create_default_regions(self) -> None:
        """Create default screen regions."""
        # Left third
        self.regions["left"] = WindowMapper.ScreenRegion(
            name="left",
            x_min=0.0, y_min=0.0,
            x_max=0.33, y_max=1.0
        )

        # Center third
        self.regions["center"] = WindowMapper.ScreenRegion(
            name="center",
            x_min=0.33, y_min=0.0,
            x_max=0.67, y_max=1.0
        )

        # Right third
        self.regions["right"] = WindowMapper.ScreenRegion(
            name="right",
            x_min=0.67, y_min=0.0,
            x_max=1.0, y_max=1.0
        )

    def add_region(self, region: ScreenRegion) -> None:
        """Add a new screen region."""
        self.regions[region.name] = region

    def get_region_at(self, x_norm: float, y_norm: float) -> Optional[str]:
        """Get the region name containing normalized coordinates."""
        for region in self.regions.values():
            if (region.x_min <= x_norm <= region.x_max and
                region.y_min <= y_norm <= region.y_max):
                return region.name
        return None

    def map_gesture(self, gesture: str, x_norm: float, y_norm: float) -> Optional[str]:
        """Map gesture to action based on screen region."""
        region_name = self.get_region_at(x_norm, y_norm)
        if region_name is None:
            return None

        region = self.regions.get(region_name)
        if region is None:
            return None
        return region.gestures.get(gesture)

    def bind_gesture_to_region(self, region_name: str, gesture: str, action: str) -> bool:
        """Bind a gesture to an action in a specific region."""
        region = self.regions.get(region_name)
        if region is None:
            return False

        region.gestures[gesture] = action
        return True

    def list_regions(self) -> List[str]:
        """List all defined regions."""
        return list(self.regions.keys())


class CursorHistory:
    """Track cursor movement history for analytics and playback."""

    def __init__(self, max_history: int = 600):  # 600 frames at 60fps = 10 seconds
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)

    def add_position(self, pos: CursorPosition) -> None:
        """Add cursor position to history."""
        self.history.append(pos)

    def get_history(self) -> List[CursorPosition]:
        """Get all recorded positions."""
        return list(self.history)

    def get_last_n_frames(self, n: int) -> List[CursorPosition]:
        """Get last N frames of history."""
        return list(self.history)[-n:]

    def compute_statistics(self) -> Dict[str, float]:
        """Compute cursor movement statistics."""
        if len(self.history) < 2:
            return {}

        positions = list(self.history)
        times = [p.timestamp for p in positions]

        # Calculate velocities
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i].x - positions[i-1].x
            dy = positions[i].y - positions[i-1].y
            dt = times[i] - times[i-1]

            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                speed = (vx*vx + vy*vy) ** 0.5
                velocities.append(speed)

        if not velocities:
            return {}

        return {
            "avg_speed": sum(velocities) / len(velocities),
            "max_speed": max(velocities),
            "min_speed": min(velocities),
            "total_distance": sum(
                ((positions[i].x - positions[i-1].x)**2 +
                  (positions[i].y - positions[i-1].y)**2)**0.5
                for i in range(1, len(positions))
            ),
        }

    def clear(self) -> None:
        """Clear history."""
        self.history.clear()
