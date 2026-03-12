"""
Integration hooks for virtual / digital twin / robotics targets.

This module defines a minimal plugin-style API that can be used to forward
hand, motion, gesture, and cursor information to external systems such as:

- robot arms
- Unreal Engine / game engines
- AR/VR applications

The base class can be subclassed to send data over sockets, shared memory,
HTTP, etc., without modifying the core tracking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Protocol, runtime_checkable

from hand_tracker import HandLandmarks
from motion_analyzer import HandMotionInfo
from gesture_detector import GestureResult, TwoHandGestureResult


@runtime_checkable
class FrameConsumer(Protocol):
    """
    Protocol/base interface for external consumers.

    Implementations can forward the current frame state to any target system.
    """

    def consume_frame_state(
        self,
        hands: List[HandLandmarks],
        motions: Dict[int, HandMotionInfo],
        gestures: Dict[int, GestureResult],
        cursor_positions: Dict[int, Tuple[int, int]],
        two_hand_gesture: TwoHandGestureResult | None,
    ) -> None:
        ...


@dataclass
class PrintConsumer:
    """
    Simple reference implementation of FrameConsumer that logs key state
    to the console. This is mainly useful when prototyping new mappings
    to robots or virtual environments.
    """

    enabled: bool = False

    def consume_frame_state(
        self,
        hands: List[HandLandmarks],
        motions: Dict[int, HandMotionInfo],
        gestures: Dict[int, GestureResult],
        cursor_positions: Dict[int, Tuple[int, int]],
        two_hand_gesture: TwoHandGestureResult | None,
    ) -> None:
        if not self.enabled:
            return

        summary = []
        for hand in hands:
            g = gestures.get(hand.hand_id)
            cur = cursor_positions.get(hand.hand_id)
            summary.append(
                f"Hand {hand.hand_id}({hand.handedness}) "
                f"gesture={g.gesture if g else 'none'} "
                f"cursor={cur}"
            )

        if two_hand_gesture is not None:
            summary.append(f"2H:{two_hand_gesture.gesture}")

        print(" | ".join(summary))

