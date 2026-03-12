"""
Gesture Macros & Automation Module

Maps gestures to OS actions:
- Mouse clicks (left, right, middle, double)
- Scrolling (up, down, left, right)
- Keyboard shortcuts
- Application switching
- Custom command execution

Architecture:
- ActionType: Enum of supported actions
- GestureAction: Single automated action
- GestureProfile: Collection of gesture-to-action mappings
- MacroExecutor: Executes actions on gesture recognition
- ProfileManager: Save/load gesture profiles
"""

from __future__ import annotations

import json
import subprocess
import time
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from pathlib import Path


class ActionType(Enum):
    """Types of actions that can be automated."""
    MOUSE_CLICK_LEFT = "mouse_click_left"
    MOUSE_CLICK_RIGHT = "mouse_click_right"
    MOUSE_CLICK_MIDDLE = "mouse_click_middle"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_SCROLL_UP = "mouse_scroll_up"
    MOUSE_SCROLL_DOWN = "mouse_scroll_down"
    MOUSE_SCROLL_LEFT = "mouse_scroll_left"
    MOUSE_SCROLL_RIGHT = "mouse_scroll_right"
    KEYBOARD_KEY = "keyboard_key"  # Single key
    KEYBOARD_SHORTCUT = "keyboard_shortcut"  # Key combination
    APP_LAUNCH = "app_launch"  # Launch application
    APP_SWITCH = "app_switch"  # Switch between apps
    CUSTOM_COMMAND = "custom_command"  # Run custom command
    SYSTEM_ACTION = "system_action"  # System actions (screenshot, etc.)


@dataclass
class GestureAction:
    """
    Single automated action triggered by a gesture.

    Attributes:
        gesture_name: Gesture label (e.g., "pinch", "swipe_left")
        action_type: What action to perform
        action_value: Parameter for the action (key name, app path, etc.)
        hand_filter: Optional filter ("left", "right", "any")
        confidence_threshold: Minimum confidence to trigger
        cooldown_ms: Milliseconds to wait before next trigger
        enabled: Whether this mapping is active
    """
    gesture_name: str
    action_type: ActionType
    action_value: str = ""
    hand_filter: str = "any"
    confidence_threshold: float = 0.5
    cooldown_ms: int = 200
    enabled: bool = True

    def matches_gesture(
        self,
        gesture: str,
        hand: str,
        confidence: float
    ) -> bool:
        """Check if this action matches a gesture."""
        return (
            self.gesture_name == gesture
            and (self.hand_filter == "any" or self.hand_filter.lower() == hand.lower())
            and confidence >= self.confidence_threshold
            and self.enabled
        )


@dataclass
class GestureProfile:
    """Collection of gesture-to-action mappings (like a preset)."""
    name: str
    description: str = ""
    actions: List[GestureAction] = None
    created_date: str = ""
    last_modified: str = ""
    is_default: bool = False

    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if not self.created_date:
            from datetime import datetime
            self.created_date = datetime.now().isoformat()

    def add_action(self, action: GestureAction) -> None:
        """Add gesture-to-action mapping."""
        self.actions.append(action)

    def remove_action(self, gesture_name: str) -> None:
        """Remove all actions for a gesture."""
        self.actions = [a for a in self.actions if a.gesture_name != gesture_name]

    def get_actions_for_gesture(self, gesture_name: str) -> List[GestureAction]:
        """Get all actions mapped to a gesture."""
        return [a for a in self.actions if a.gesture_name == gesture_name]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "created_date": self.created_date,
            "last_modified": self.last_modified,
            "is_default": self.is_default,
            "actions": [
                {
                    "gesture_name": a.gesture_name,
                    "action_type": a.action_type.value,
                    "action_value": a.action_value,
                    "hand_filter": a.hand_filter,
                    "confidence_threshold": a.confidence_threshold,
                    "cooldown_ms": a.cooldown_ms,
                    "enabled": a.enabled,
                }
                for a in self.actions
            ],
        }

    @staticmethod
    def from_dict(data: Dict) -> GestureProfile:
        """Create GestureProfile from dictionary."""
        actions = [
            GestureAction(
                gesture_name=a["gesture_name"],
                action_type=ActionType(a["action_type"]),
                action_value=a.get("action_value", ""),
                hand_filter=a.get("hand_filter", "any"),
                confidence_threshold=a.get("confidence_threshold", 0.5),
                cooldown_ms=a.get("cooldown_ms", 200),
                enabled=a.get("enabled", True),
            )
            for a in data.get("actions", [])
        ]

        return GestureProfile(
            name=data["name"],
            description=data.get("description", ""),
            actions=actions,
            created_date=data.get("created_date", ""),
            last_modified=data.get("last_modified", ""),
            is_default=data.get("is_default", False),
        )


class MacroExecutor:
    """
    Execute automated actions from gestures.

    Handles:
    - Mouse operations (clicks, scroll)
    - Keyboard input
    - Application control
    - Custom command execution
    - Cooldown tracking to prevent accidental triggers
    """

    def __init__(self):
        self.last_action_time: Dict[str, float] = {}
        self.callbacks: Dict[ActionType, Callable] = {}

    def register_callback(self, action_type: ActionType, callback: Callable) -> None:
        """Register custom callback for action type."""
        self.callbacks[action_type] = callback

    def execute_action(self, action: GestureAction) -> bool:
        """
        Execute a gesture action.

        Returns True if executed, False if cooled down or invalid.
        """
        # Check cooldown
        key = f"{action.gesture_name}_{action.hand_filter}"
        now = time.time()
        last_time = self.last_action_time.get(key, 0.0)
        cooldown_sec = action.cooldown_ms / 1000.0

        if now - last_time < cooldown_sec:
            return False  # Still in cooldown

        self.last_action_time[key] = now

        # Check for custom callback
        if action.action_type in self.callbacks:
            return self.callbacks[action.action_type](action.action_value)

        # Built-in action handling
        if action.action_type == ActionType.MOUSE_CLICK_LEFT:
            return self._mouse_click_left()
        elif action.action_type == ActionType.MOUSE_CLICK_RIGHT:
            return self._mouse_click_right()
        elif action.action_type == ActionType.MOUSE_CLICK_MIDDLE:
            return self._mouse_click_middle()
        elif action.action_type == ActionType.MOUSE_DOUBLE_CLICK:
            return self._mouse_double_click()
        elif action.action_type == ActionType.MOUSE_SCROLL_UP:
            return self._mouse_scroll(action.action_value or "5")
        elif action.action_type == ActionType.MOUSE_SCROLL_DOWN:
            return self._mouse_scroll(-(int(action.action_value or "5")))
        elif action.action_type == ActionType.KEYBOARD_KEY:
            return self._keyboard_press(action.action_value)
        elif action.action_type == ActionType.KEYBOARD_SHORTCUT:
            return self._keyboard_shortcut(action.action_value)
        elif action.action_type == ActionType.APP_LAUNCH:
            return self._app_launch(action.action_value)
        elif action.action_type == ActionType.CUSTOM_COMMAND:
            return self._execute_command(action.action_value)

        return False

    # --- Mouse operations ---

    def _mouse_click_left(self) -> bool:
        """Click left mouse button."""
        try:
            import pyautogui
            pyautogui.click(button='left')
            return True
        except Exception as e:
            print(f"Error clicking mouse: {e}")
            return False

    def _mouse_click_right(self) -> bool:
        """Click right mouse button."""
        try:
            import pyautogui
            pyautogui.click(button='right')
            return True
        except Exception as e:
            print(f"Error right-clicking: {e}")
            return False

    def _mouse_click_middle(self) -> bool:
        """Click middle mouse button."""
        try:
            import pyautogui
            pyautogui.click(button='middle')
            return True
        except Exception as e:
            print(f"Error middle-clicking: {e}")
            return False

    def _mouse_double_click(self) -> bool:
        """Double-click mouse."""
        try:
            import pyautogui
            pyautogui.doubleClick()
            return True
        except Exception as e:
            print(f"Error double-clicking: {e}")
            return False

    def _mouse_scroll(self, amount: int) -> bool:
        """Scroll mouse wheel."""
        try:
            import pyautogui
            pyautogui.scroll(amount)
            return True
        except Exception as e:
            print(f"Error scrolling: {e}")
            return False

    # --- Keyboard operations ---

    def _keyboard_press(self, key: str) -> bool:
        """Press a single key."""
        try:
            import pyautogui
            pyautogui.press(key.lower())
            return True
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
            return False

    def _keyboard_shortcut(self, shortcut: str) -> bool:
        """Execute keyboard shortcut (e.g., "ctrl+c")."""
        try:
            import pyautogui
            keys = shortcut.split('+')
            pyautogui.hotkey(*[k.strip().lower() for k in keys])
            return True
        except Exception as e:
            print(f"Error executing shortcut {shortcut}: {e}")
            return False

    # --- Application control ---

    def _app_launch(self, app_path: str) -> bool:
        """Launch an application."""
        try:
            subprocess.Popen(app_path, shell=True)
            return True
        except Exception as e:
            print(f"Error launching app {app_path}: {e}")
            return False

    # --- Custom command ---

    def _execute_command(self, command: str) -> bool:
        """Execute a custom command."""
        try:
            subprocess.Popen(command, shell=True)
            return True
        except Exception as e:
            print(f"Error executing command {command}: {e}")
            return False


class ProfileManager:
    """Manage gesture macro profiles."""

    def __init__(self, profiles_dir: str = "gesture_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.profiles: Dict[str, GestureProfile] = {}

    def save_profile(self, profile: GestureProfile) -> bool:
        """Save profile to JSON file."""
        try:
            profile_path = self.profiles_dir / f"{profile.name}.json"

            with open(profile_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

            print(f"[ProfileManager] Saved profile: {profile.name}")
            return True
        except Exception as e:
            print(f"[ProfileManager] Error saving profile: {e}")
            return False

    def load_profile(self, name: str) -> Optional[GestureProfile]:
        """Load profile from JSON file."""
        try:
            profile_path = self.profiles_dir / f"{name}.json"

            if not profile_path.exists():
                return None

            with open(profile_path, "r") as f:
                data = json.load(f)

            profile = GestureProfile.from_dict(data)
            self.profiles[name] = profile
            print(f"[ProfileManager] Loaded profile: {name}")
            return profile
        except Exception as e:
            print(f"[ProfileManager] Error loading profile: {e}")
            return None

    def list_profiles(self) -> List[str]:
        """List all available profiles."""
        return [f.stem for f in self.profiles_dir.glob("*.json")]

    def get_profile(self, name: str) -> Optional[GestureProfile]:
        """Get profile by name (load if not cached)."""
        if name in self.profiles:
            return self.profiles[name]
        return self.load_profile(name)

    def delete_profile(self, name: str) -> bool:
        """Delete a profile."""
        try:
            profile_path = self.profiles_dir / f"{name}.json"
            if profile_path.exists():
                profile_path.unlink()
                if name in self.profiles:
                    del self.profiles[name]
                print(f"[ProfileManager] Deleted profile: {name}")
                return True
            return False
        except Exception as e:
            print(f"[ProfileManager] Error deleting profile: {e}")
            return False


def create_default_profiles() -> Dict[str, GestureProfile]:
    """Create default gesture profiles for common use cases."""

    # Browser profile
    browser_profile = GestureProfile(
        name="browser",
        description="Gestures for web browsing",
        is_default=False
    )
    browser_profile.add_action(GestureAction(
        gesture_name="swipe_left",
        action_type=ActionType.KEYBOARD_SHORTCUT,
        action_value="alt+right",  # Forward
        hand_filter="any",
        confidence_threshold=0.6
    ))
    browser_profile.add_action(GestureAction(
        gesture_name="swipe_right",
        action_type=ActionType.KEYBOARD_SHORTCUT,
        action_value="alt+left",  # Back
        hand_filter="any",
        confidence_threshold=0.6
    ))
    browser_profile.add_action(GestureAction(
        gesture_name="pinch",
        action_type=ActionType.MOUSE_CLICK_LEFT,
        hand_filter="any",
        confidence_threshold=0.7
    ))

    # Presentation profile
    presentation_profile = GestureProfile(
        name="presentation",
        description="Gestures for presentations",
        is_default=False
    )
    presentation_profile.add_action(GestureAction(
        gesture_name="swipe_right",
        action_type=ActionType.KEYBOARD_KEY,
        action_value="left",  # Previous slide
        hand_filter="any",
        confidence_threshold=0.6
    ))
    presentation_profile.add_action(GestureAction(
        gesture_name="swipe_left",
        action_type=ActionType.KEYBOARD_KEY,
        action_value="right",  # Next slide
        hand_filter="any",
        confidence_threshold=0.6
    ))

    # Media profile
    media_profile = GestureProfile(
        name="media",
        description="Gestures for media control",
        is_default=False
    )
    media_profile.add_action(GestureAction(
        gesture_name="pinch",
        action_type=ActionType.KEYBOARD_KEY,
        action_value="space",  # Play/pause
        hand_filter="any",
        confidence_threshold=0.7
    ))
    media_profile.add_action(GestureAction(
        gesture_name="swipe_left",
        action_type=ActionType.KEYBOARD_KEY,
        action_value="n",  # Next track
        hand_filter="any",
        confidence_threshold=0.6
    ))
    media_profile.add_action(GestureAction(
        gesture_name="swipe_right",
        action_type=ActionType.KEYBOARD_KEY,
        action_value="b",  # Previous track
        hand_filter="any",
        confidence_threshold=0.6
    ))

    return {
        "browser": browser_profile,
        "presentation": presentation_profile,
        "media": media_profile,
    }
