"""
Performance optimization and profiling module.

Provides:
- Per-module timing profiling
- Selective module activation for performance tuning
- Multi-threading support for pipeline stages
- FPS monitoring and statistics
- Bottleneck identification
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from collections import deque
from enum import Enum


class ModuleState(Enum):
    """State of a module in the pipeline."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    PROFILING = "profiling"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a module or full pipeline."""

    module_name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    last_call_time_ms: float = 0.0

    @property
    def average_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        return self.total_time_ms / max(1, self.total_calls)

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.module_name}: "
            f"calls={self.total_calls} "
            f"avg={self.average_time_ms:.2f}ms "
            f"min={self.min_time_ms:.2f}ms "
            f"max={self.max_time_ms:.2f}ms"
        )


@dataclass
class FPSTracker:
    """Track frames per second over a sliding window."""

    window_size: int = 60
    _timestamps: deque = field(default_factory=deque)
    _fps_history: deque = field(default_factory=deque)

    def update(self) -> float:
        """Record a frame and return current FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)

        # Keep only window_size timestamps
        if len(self._timestamps) > self.window_size:
            self._timestamps.popleft()

        # Calculate FPS from time span
        if len(self._timestamps) < 2:
            fps = 0.0
        else:
            time_span = self._timestamps[-1] - self._timestamps[0]
            if time_span > 0:
                fps = (len(self._timestamps) - 1) / time_span
            else:
                fps = 0.0

        self._fps_history.append(fps)
        if len(self._fps_history) > self.window_size:
            self._fps_history.popleft()

        return fps

    def get_average_fps(self) -> float:
        """Get average FPS over history."""
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    def get_min_fps(self) -> float:
        """Get minimum FPS in history."""
        return min(self._fps_history) if self._fps_history else 0.0

    def get_max_fps(self) -> float:
        """Get maximum FPS in history."""
        return max(self._fps_history) if self._fps_history else 0.0


class PerformanceProfiler:
    """
    Profile and manage performance of individual pipeline modules.

    Usage:
        profiler = PerformanceProfiler()
        with profiler.measure("hand_tracking"):
            # run tracking code
    """

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._context_name: Optional[str] = None
        self._context_start: Optional[float] = None

    def measure(self, module_name: str) -> PerformanceProfiler.MeasureContext:
        """Create a context manager for timing a code block."""
        return self.MeasureContext(self, module_name)

    def start_measure(self, module_name: str) -> None:
        """Start timing a module (manual mode)."""
        self._context_name = module_name
        self._context_start = time.perf_counter()

    def end_measure(self) -> float:
        """End timing and return elapsed time in milliseconds."""
        if self._context_start is None:
            return 0.0

        elapsed_ms = (time.perf_counter() - self._context_start) * 1000.0
        if self._context_name:
            self._record_metric(self._context_name, elapsed_ms)

        self._context_start = None
        self._context_name = None
        return elapsed_ms

    def _record_metric(self, module_name: str, time_ms: float) -> None:
        """Record a timing measurement."""
        if module_name not in self.metrics:
            self.metrics[module_name] = PerformanceMetrics(module_name=module_name)

        m = self.metrics[module_name]
        m.total_calls += 1
        m.total_time_ms += time_ms
        m.min_time_ms = min(m.min_time_ms, time_ms)
        m.max_time_ms = max(m.max_time_ms, time_ms)
        m.last_call_time_ms = time_ms

    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all recorded metrics."""
        return self.metrics.copy()

    def get_metric(self, module_name: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a specific module."""
        return self.metrics.get(module_name)

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()

    def report(self) -> str:
        """Generate a human-readable performance report."""
        lines = ["Performance Report:"]
        lines.append("=" * 60)

        if not self.metrics:
            lines.append("No metrics recorded.")
            return "\n".join(lines)

        # Sort by total time
        sorted_metrics = sorted(
            self.metrics.values(),
            key=lambda m: m.total_time_ms,
            reverse=True
        )

        for m in sorted_metrics:
            lines.append(str(m))

        total_time = sum(m.total_time_ms for m in self.metrics.values())
        lines.append("=" * 60)
        lines.append(f"Total pipeline time: {total_time:.2f}ms")

        return "\n".join(lines)

    class MeasureContext:
        """Context manager for profiling."""

        def __init__(self, profiler: PerformanceProfiler, module_name: str):
            self.profiler = profiler
            self.module_name = module_name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            self.profiler._record_metric(self.module_name, elapsed_ms)


class PipelineModuleController:
    """
    Control selective activation/deactivation of pipeline modules.

    Allows enabling/disabling expensive operations (gesture detection, motion
    analysis) to optimize for performance vs. features trade-off.
    """

    def __init__(self):
        self.modules: Dict[str, ModuleState] = {
            "hand_tracking": ModuleState.ENABLED,
            "motion_analysis": ModuleState.ENABLED,
            "static_gesture_detection": ModuleState.ENABLED,
            "dynamic_gesture_detection": ModuleState.ENABLED,
            "two_hand_gestures": ModuleState.ENABLED,
            "ml_gesture_prediction": ModuleState.DISABLED,
            "cursor_control": ModuleState.ENABLED,
            "visualization": ModuleState.ENABLED,
        }

        self.profiler = PerformanceProfiler()
        self.fps_tracker = FPSTracker()

    def enable_module(self, module_name: str) -> None:
        """Enable a module."""
        if module_name in self.modules:
            self.modules[module_name] = ModuleState.ENABLED

    def disable_module(self, module_name: str) -> None:
        """Disable a module."""
        if module_name in self.modules:
            self.modules[module_name] = ModuleState.DISABLED

    def is_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled."""
        return self.modules.get(module_name) == ModuleState.ENABLED

    def toggle_module(self, module_name: str) -> bool:
        """Toggle module state and return new state."""
        state = self.modules.get(module_name)
        if state == ModuleState.ENABLED:
            self.disable_module(module_name)
            return False
        elif state == ModuleState.DISABLED:
            self.enable_module(module_name)
            return True
        return False

    def set_profiling_enabled(self, module_name: str, enabled: bool) -> None:
        """Enable/disable profiling for a specific module."""
        if module_name in self.modules:
            if enabled:
                self.modules[module_name] = ModuleState.PROFILING
            else:
                self.modules[module_name] = ModuleState.ENABLED

    def get_status_report(self) -> str:
        """Get current status of all modules."""
        lines = ["Pipeline Module Status:"]
        lines.append("=" * 50)

        for module_name, state in sorted(self.modules.items()):
            lines.append(f"  {module_name}: {state.value}")

        lines.append("=" * 50)
        lines.append(f"Current FPS: {self.fps_tracker.get_average_fps():.1f}")

        return "\n".join(lines)

    def get_performance_report(self) -> str:
        """Get performance profiling report."""
        return self.profiler.report()


class ThreadedPipelineStage:
    """
    Wrapper for running a pipeline stage in a separate thread.

    Useful for offloading heavy computation (gesture detection, visualization)
    from the main hand-tracking thread.
    """

    def __init__(self, stage_name: str, timeout_ms: float = 100.0):
        self.stage_name = stage_name
        self.timeout_ms = timeout_ms
        self.thread: Optional[threading.Thread] = None
        self.input_queue: Any = None
        self.output_queue: Any = None
        self.running = False
        self._lock = threading.Lock()

    def start(self, processor: Callable[[Any], Any]) -> None:
        """Start the threaded stage."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._worker,
            args=(processor,),
            daemon=True,
            name=f"{self.stage_name}_worker",
        )
        self.thread.start()

    def stop(self) -> None:
        """Stop the threaded stage."""
        with self._lock:
            self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _worker(self, processor: Callable[[Any], Any]) -> None:
        """Worker thread main loop."""
        while self.running:
            # Process incoming data
            if self.input_queue is not None and not self.input_queue.empty():
                try:
                    data = self.input_queue.get(timeout=0.01)
                    result = processor(data)
                    if self.output_queue is not None:
                        self.output_queue.put(result)
                except Exception as e:
                    print(f"Error in {self.stage_name}: {e}")

            time.sleep(0.001)  # Prevent spinning

    def submit_work(self, data: Any) -> None:
        """Submit work to the stage (non-blocking)."""
        if self.input_queue is not None and self.running:
            try:
                self.input_queue.put_nowait(data)
            except Exception:
                pass  # Queue full, skip

    def get_result(self, blocking: bool = False) -> Optional[Any]:
        """Get result from the stage."""
        if self.output_queue is None:
            return None

        try:
            return self.output_queue.get(
                blocking=blocking,
                timeout=self.timeout_ms / 1000.0
            )
        except Exception:
            return None
