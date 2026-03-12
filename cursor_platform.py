"""
Cross-Platform Cursor Support Module

Abstracts cursor control across Windows, macOS, and Linux.

Supports:
- Mouse movement
- Mouse clicks (left, right, middle)
- Scroll wheel
- Modifier keys

Platform detection and appropriate library usage:
- Windows: ctypes (built-in)
- macOS: pyobjc
- Linux: Xlib or xdotool
"""

from __future__ import annotations

import sys
import platform
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class CursorPlatform(ABC):
    """Abstract base class for platform-specific cursor control."""

    @abstractmethod
    def move_cursor(self, x: int, y: int) -> bool:
        """Move cursor to position."""
        pass

    @abstractmethod
    def get_cursor_pos(self) -> Optional[Tuple[int, int]]:
        """Get current cursor position."""
        pass

    @abstractmethod
    def click(self, button: str = "left") -> bool:
        """Click mouse button."""
        pass

    @abstractmethod
    def double_click(self) -> bool:
        """Double-click mouse."""
        pass

    @abstractmethod
    def scroll(self, amount: int) -> bool:
        """Scroll mouse wheel."""
        pass


class WindowsCursorPlatform(CursorPlatform):
    """Windows cursor control using ctypes."""

    def __init__(self):
        import ctypes
        self.user32 = ctypes.windll.user32
        try:
            self.user32.SetProcessDPIAware()
        except Exception:
            pass

    def move_cursor(self, x: int, y: int) -> bool:
        """Move cursor to position."""
        try:
            self.user32.SetCursorPos(int(x), int(y))
            return True
        except Exception as e:
            print(f"[WindowsCursor] Error moving cursor: {e}")
            return False

    def get_cursor_pos(self) -> Optional[Tuple[int, int]]:
        """Get current cursor position."""
        try:
            import ctypes
            pos = ctypes.wintypes.POINT()
            self.user32.GetCursorPos(ctypes.byref(pos))
            return (pos.x, pos.y)
        except Exception as e:
            print(f"[WindowsCursor] Error getting position: {e}")
            return None

    def click(self, button: str = "left") -> bool:
        """Click mouse button."""
        try:
            import pyautogui
            pyautogui.click(button=button)
            return True
        except Exception as e:
            print(f"[WindowsCursor] Error clicking: {e}")
            return False

    def double_click(self) -> bool:
        """Double-click mouse."""
        try:
            import pyautogui
            pyautogui.doubleClick()
            return True
        except Exception as e:
            print(f"[WindowsCursor] Error double-clicking: {e}")
            return False

    def scroll(self, amount: int) -> bool:
        """Scroll mouse wheel."""
        try:
            import pyautogui
            pyautogui.scroll(amount)
            return True
        except Exception as e:
            print(f"[WindowsCursor] Error scrolling: {e}")
            return False


class MacCursorPlatform(CursorPlatform):
    """macOS cursor control using pyobjc."""

    def __init__(self):
        try:
            from PyObjCTools.MachSignals import signals
            from Foundation import NSScreen, NSEvent
            self.NSScreen = NSScreen
            self.NSEvent = NSEvent
        except ImportError:
            print("[MacCursor] pyobjc not installed. Install: pip install pyobjc")

    def move_cursor(self, x: int, y: int) -> bool:
        """Move cursor to position."""
        try:
            from quartz import CoreGraphics
            pos = CoreGraphics.CGPointMake(x, y)
            CoreGraphics.CGDisplayMoveCursorToPoint(0, pos)
            return True
        except Exception as e:
            print(f"[MacCursor] Error moving cursor: {e}")
            return False

    def get_cursor_pos(self) -> Optional[Tuple[int, int]]:
        """Get current cursor position."""
        try:
            from quartz import CoreGraphics
            pos = CoreGraphics.CGEventGetLocation(
                CoreGraphics.CGEventCreate(None)
            )
            return (int(pos.x), int(pos.y))
        except Exception as e:
            print(f"[MacCursor] Error getting position: {e}")
            return None

    def click(self, button: str = "left") -> bool:
        """Click mouse button."""
        try:
            import pyautogui
            pyautogui.click(button=button)
            return True
        except Exception as e:
            print(f"[MacCursor] Error clicking: {e}")
            return False

    def double_click(self) -> bool:
        """Double-click mouse."""
        try:
            import pyautogui
            pyautogui.doubleClick()
            return True
        except Exception as e:
            print(f"[MacCursor] Error double-clicking: {e}")
            return False

    def scroll(self, amount: int) -> bool:
        """Scroll mouse wheel."""
        try:
            import pyautogui
            pyautogui.scroll(amount)
            return True
        except Exception as e:
            print(f"[MacCursor] Error scrolling: {e}")
            return False


class LinuxCursorPlatform(CursorPlatform):
    """Linux cursor control using xdotool/Xlib."""

    def __init__(self):
        self.use_xdotool = self._check_xdotool()
        if not self.use_xdotool:
            try:
                from Xlib import display
                self.display = display.Display()
                self.screen = self.display.screen()
            except ImportError:
                print("[LinuxCursor] Xlib not installed. Install: pip install python-xlib")

    def _check_xdotool(self) -> bool:
        """Check if xdotool is available."""
        import subprocess
        try:
            subprocess.run(["which", "xdotool"], check=True, capture_output=True)
            return True
        except Exception:
            return False

    def move_cursor(self, x: int, y: int) -> bool:
        """Move cursor to position."""
        if self.use_xdotool:
            try:
                import subprocess
                subprocess.run(["xdotool", "mousemove", str(x), str(y)])
                return True
            except Exception as e:
                print(f"[LinuxCursor] Error with xdotool: {e}")
                return False
        else:
            try:
                from Xlib import X
                self.display.screen().root.warp_pointer(x, y)
                self.display.sync()
                return True
            except Exception as e:
                print(f"[LinuxCursor] Error with Xlib: {e}")
                return False

    def get_cursor_pos(self) -> Optional[Tuple[int, int]]:
        """Get current cursor position."""
        try:
            from Xlib import display
            d = display.Display()
            coords = d.screen().root.query_pointer()
            return (coords.root_x, coords.root_y)
        except Exception as e:
            print(f"[LinuxCursor] Error getting position: {e}")
            return None

    def click(self, button: str = "left") -> bool:
        """Click mouse button."""
        if self.use_xdotool:
            try:
                import subprocess
                button_num = {"left": "1", "right": "3", "middle": "2"}.get(button, "1")
                subprocess.run(["xdotool", "click", button_num])
                return True
            except Exception as e:
                print(f"[LinuxCursor] Click error: {e}")
                return False
        else:
            try:
                import pyautogui
                pyautogui.click(button=button)
                return True
            except Exception as e:
                print(f"[LinuxCursor] Click error: {e}")
                return False

    def double_click(self) -> bool:
        """Double-click mouse."""
        if self.use_xdotool:
            try:
                import subprocess
                subprocess.run(["xdotool", "click", "1", "click", "1"])
                return True
            except Exception as e:
                print(f"[LinuxCursor] Double-click error: {e}")
                return False
        else:
            try:
                import pyautogui
                pyautogui.doubleClick()
                return True
            except Exception as e:
                print(f"[LinuxCursor] Double-click error: {e}")
                return False

    def scroll(self, amount: int) -> bool:
        """Scroll mouse wheel."""
        try:
            import pyautogui
            pyautogui.scroll(amount)
            return True
        except Exception as e:
            print(f"[LinuxCursor] Scroll error: {e}")
            return False


class CrossPlatformCursorController:
    """Platform-agnostic cursor controller."""

    def __init__(self):
        self.platform_name = platform.system()
        self.platform = self._detect_platform()
        print(f"[CrossPlatformCursor] Detected platform: {self.platform_name}")

    def _detect_platform(self) -> CursorPlatform:
        """Detect OS and create appropriate platform handler."""
        system = platform.system().lower()

        if system == "windows":
            return WindowsCursorPlatform()
        elif system == "darwin":  # macOS
            return MacCursorPlatform()
        elif system == "linux":
            return LinuxCursorPlatform()
        else:
            print(f"[CrossPlatformCursor] Unknown platform: {system}")
            # Fallback to Windows-like behavior
            return WindowsCursorPlatform()

    def move_cursor(self, x: int, y: int) -> bool:
        """Move cursor to position."""
        return self.platform.move_cursor(int(x), int(y))

    def get_cursor_pos(self) -> Optional[Tuple[int, int]]:
        """Get current cursor position."""
        return self.platform.get_cursor_pos()

    def click(self, button: str = "left") -> bool:
        """Click mouse button."""
        return self.platform.click(button.lower())

    def double_click(self) -> bool:
        """Double-click mouse."""
        return self.platform.double_click()

    def scroll(self, amount: int) -> bool:
        """Scroll mouse wheel."""
        return self.platform.scroll(amount)
