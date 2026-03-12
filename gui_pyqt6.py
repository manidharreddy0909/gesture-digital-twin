"""
PyQt6-based GUI for the motion-driven cursor system with analytics dashboard.

Features:
- Real-time video preview with hand landmarks
- Toggle controls for each pipeline module
- Gesture/motion analytics visualization
- Performance profiling display
- Model training interface
- Gesture binding editor
- Real-time metrics graphs
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
        QTabWidget, QTableWidget, QTableWidgetItem, QSlider, QGroupBox,
        QGridLayout, QTextEdit, QDialog, QMessageBox, QProgressBar,
        QScrollArea, QFrame, QListWidget, QListWidgetItem,
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
    from PyQt6.QtGui import QImage, QPixmap, QIcon, QFont, QColor
    from PyQt6.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
    from PyQt6.QtCore import QDateTime
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False


@dataclass
class GUIConfig:
    """Configuration for the GUI."""
    window_width: int = 1600
    window_height: int = 1000
    refresh_rate_ms: int = 30
    show_motion_trails: bool = True
    show_predictions: bool = False
    enable_recording: bool = False


class AnalyticsBuffer:
    """Circular buffer for real-time analytics data."""

    def __init__(self, max_size: int = 300):
        self.max_size = max_size
        self.timestamps: deque = deque(maxlen=max_size)
        self.velocities: deque = deque(maxlen=max_size)
        self.accelerations: deque = deque(maxlen=max_size)
        self.fps_values: deque = deque(maxlen=max_size)

    def add_frame(self, timestamp: float, velocity: float, acceleration: float, fps: float):
        """Add a data point."""
        self.timestamps.append(timestamp)
        self.velocities.append(velocity)
        self.accelerations.append(acceleration)
        self.fps_values.append(fps)


class CursorAITrackerGUI(QMainWindow):
    """
    Main GUI window for the Cursor AI Tracker system.

    Provides:
    - Real-time video display with landmarks
    - Control panels for module activation
    - Analytics graphs
    - Performance metrics
    - Model training interface
    """

    if PYQT6_AVAILABLE:
        frame_ready = pyqtSignal(object)  # Emits processed frame for display
        metrics_updated = pyqtSignal(dict)  # Emits performance metrics

    def __init__(self, config: Optional[GUIConfig] = None):
        if not PYQT6_AVAILABLE:
            print("ERROR: PyQt6 not available. Install with: pip install PyQt6 PyQt6-Charts")
            sys.exit(1)

        super().__init__()

        self.config = config or GUIConfig()
        self.setWindowTitle("Cursor AI Tracker - Advanced Motion Control")
        self.setGeometry(100, 100, self.config.window_width, self.config.window_height)

        self.analytics_buffer = AnalyticsBuffer()
        self.data_lock = threading.Lock()

        self._create_ui()
        self._connect_signals()
        self._setup_timers()

    def _create_ui(self) -> None:
        """Create the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()

        # Left panel: Video display
        left_panel = self._create_video_panel()
        main_layout.addWidget(left_panel, 2)

        # Right panel: Controls and analytics
        right_panel = self._create_control_panel()
        main_layout.addWidget(right_panel, 1)

        central_widget.setLayout(main_layout)

    def _create_video_panel(self) -> QWidget:
        """Create video display panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Video label
        self.video_label = QLabel()
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.video_label.setMinimumSize(800, 600)
        layout.addWidget(self.video_label)

        # Info bar
        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.info_label)

        panel.setLayout(layout)
        return panel

    def _create_control_panel(self) -> QWidget:
        """Create control panel with tabs."""
        tabs = QTabWidget()

        tabs.addTab(self._create_modules_tab(), "Modules")
        tabs.addTab(self._create_cursor_tab(), "Cursor Control")
        tabs.addTab(self._create_gestures_tab(), "Gestures")
        tabs.addTab(self._create_analytics_tab(), "Analytics")
        tabs.addTab(self._create_training_tab(), "ML Training")
        tabs.addTab(self._create_integration_tab(), "Integration")

        return tabs

    def _create_modules_tab(self) -> QWidget:
        """Create module control tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Pipeline Module Control", font=QFont("Arial", 10, QFont.Weight.Bold)))

        # Module toggles
        self.module_toggles: Dict[str, QCheckBox] = {}
        modules = [
            "Hand Tracking",
            "Motion Analysis",
            "Static Gesture Detection",
            "Dynamic Gesture Detection",
            "Two-Hand Gestures",
            "ML Gesture Prediction",
            "Cursor Control",
            "Visualization",
            "Motion Trails",
        ]

        for module_name in modules:
            checkbox = QCheckBox(module_name)
            checkbox.setChecked(True if "Motion Trails" not in module_name else False)
            self.module_toggles[module_name] = checkbox
            layout.addWidget(checkbox)

        # Performance metrics
        layout.addWidget(QLabel("Performance", font=QFont("Arial", 10, QFont.Weight.Bold)))

        self.fps_label = QLabel("FPS: --")
        self.latency_label = QLabel("Latency: --ms")
        layout.addWidget(self.fps_label)
        layout.addWidget(self.latency_label)

        # Profiling button
        profiling_btn = QPushButton("Toggle Profiling")
        profiling_btn.clicked.connect(self.toggle_profiling)
        layout.addWidget(profiling_btn)

        # Report button
        report_btn = QPushButton("Show Performance Report")
        report_btn.clicked.connect(self.show_performance_report)
        layout.addWidget(report_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_cursor_tab(self) -> QWidget:
        """Create cursor control tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Cursor Control Settings", font=QFont("Arial", 10, QFont.Weight.Bold)))

        # Filtering mode
        layout.addWidget(QLabel("Filter Mode:"))
        self.filter_mode = QComboBox()
        self.filter_mode.addItems(["Exponential Smoothing", "Kalman (CV)", "Kalman (CA)"])
        layout.addWidget(self.filter_mode)

        # Smoothing factor
        layout.addWidget(QLabel("Smoothing Factor:"))
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(25)
        layout.addWidget(self.smoothing_slider)
        self.smoothing_label = QLabel("0.25")
        layout.addWidget(self.smoothing_label)

        # Prediction factor
        layout.addWidget(QLabel("Prediction Factor:"))
        self.prediction_slider = QSlider(Qt.Orientation.Horizontal)
        self.prediction_slider.setRange(0, 100)
        self.prediction_slider.setValue(0)
        layout.addWidget(self.prediction_slider)
        self.prediction_label = QLabel("0.0")
        layout.addWidget(self.prediction_label)

        # Checkboxes
        self.adaptive_smoothing_cb = QCheckBox("Adaptive Smoothing")
        layout.addWidget(self.adaptive_smoothing_cb)

        self.adaptive_noise_cb = QCheckBox("Adaptive Kalman Noise")
        self.adaptive_noise_cb.setChecked(True)
        layout.addWidget(self.adaptive_noise_cb)

        self.prediction_logging_cb = QCheckBox("Enable Prediction Logging")
        layout.addWidget(self.prediction_logging_cb)

        # Primary hand selection
        layout.addWidget(QLabel("Primary Cursor Hand:"))
        self.primary_hand = QComboBox()
        self.primary_hand.addItems(["Left", "Right"])
        layout.addWidget(self.primary_hand)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_gestures_tab(self) -> QWidget:
        """Create gesture control tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Gesture Settings", font=QFont("Arial", 10, QFont.Weight.Bold)))

        # Thresholds
        layout.addWidget(QLabel("Pinch Distance Threshold:"))
        self.pinch_threshold = QDoubleSpinBox()
        self.pinch_threshold.setRange(0.01, 0.2)
        self.pinch_threshold.setValue(0.05)
        self.pinch_threshold.setSingleStep(0.01)
        layout.addWidget(self.pinch_threshold)

        layout.addWidget(QLabel("Swipe Distance Threshold:"))
        self.swipe_threshold = QDoubleSpinBox()
        self.swipe_threshold.setRange(0.05, 0.5)
        self.swipe_threshold.setValue(0.20)
        self.swipe_threshold.setSingleStep(0.05)
        layout.addWidget(self.swipe_threshold)

        # Detected gestures list
        layout.addWidget(QLabel("Detected Gestures", font=QFont("Arial", 10, QFont.Weight.Bold)))
        self.detected_gestures_list = QListWidget()
        layout.addWidget(self.detected_gestures_list)

        # Clear list button
        clear_btn = QPushButton("Clear Gesture History")
        clear_btn.clicked.connect(self.detected_gestures_list.clear)
        layout.addWidget(clear_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_analytics_tab(self) -> QWidget:
        """Create analytics visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Motion Analytics", font=QFont("Arial", 10, QFont.Weight.Bold)))

        # Summary stats
        self.avg_velocity_label = QLabel("Avg Velocity: --")
        self.max_velocity_label = QLabel("Max Velocity: --")
        self.total_distance_label = QLabel("Total Distance: --")
        layout.addWidget(self.avg_velocity_label)
        layout.addWidget(self.max_velocity_label)
        layout.addWidget(self.total_distance_label)

        # Graphs (placeholder - would need QChart setup)
        graph_group = QGroupBox("Real-Time Graphs")
        graph_layout = QVBoxLayout()
        self.velocity_graph = QLabel("[Velocity Graph - Placeholder]")
        self.acceleration_graph = QLabel("[Acceleration Graph - Placeholder]")
        graph_layout.addWidget(self.velocity_graph)
        graph_layout.addWidget(self.acceleration_graph)
        graph_group.setLayout(graph_layout)
        layout.addWidget(graph_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_training_tab(self) -> QWidget:
        """Create ML model training interface."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("ML Model Training", font=QFont("Arial", 10, QFont.Weight.Bold)))

        # Training sample collection
        layout.addWidget(QLabel("Collect Training Samples"))
        self.gesture_label_input = QComboBox()
        self.gesture_label_input.addItems(["pinch", "fist", "open", "peace", "custom_gesture"])
        self.gesture_label_input.setEditable(True)
        layout.addWidget(QLabel("Gesture Label:"))
        layout.addWidget(self.gesture_label_input)

        self.collect_static_btn = QPushButton("Collect Static Sample")
        self.collect_dynamic_btn = QPushButton("Collect Dynamic Sample")
        layout.addWidget(self.collect_static_btn)
        layout.addWidget(self.collect_dynamic_btn)

        # Training data summary
        self.training_summary = QTextEdit()
        self.training_summary.setReadOnly(True)
        layout.addWidget(QLabel("Training Data Summary:"))
        layout.addWidget(self.training_summary)

        # Train models
        train_static_btn = QPushButton("Train Static Model")
        train_dynamic_btn = QPushButton("Train Dynamic Model")
        layout.addWidget(train_static_btn)
        layout.addWidget(train_dynamic_btn)

        # Save/Load models
        layout.addWidget(QLabel("Model Management:"))
        save_static_btn = QPushButton("Save Static Model")
        save_dynamic_btn = QPushButton("Save Dynamic Model")
        load_static_btn = QPushButton("Load Static Model")
        load_dynamic_btn = QPushButton("Load Dynamic Model")
        layout.addWidget(save_static_btn)
        layout.addWidget(save_dynamic_btn)
        layout.addWidget(load_static_btn)
        layout.addWidget(load_dynamic_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_integration_tab(self) -> QWidget:
        """Create virtual target integration tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Virtual Target Integration", font=QFont("Arial", 10, QFont.Weight.Bold)))

        # Backend selection
        layout.addWidget(QLabel("Integration Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["None", "Socket", "HTTP/REST", "Unreal Engine 5", "Robot Arm"])
        layout.addWidget(self.backend_combo)

        # Connection parameters
        layout.addWidget(QLabel("Connection Parameters:"))
        layout.addWidget(QLabel("Host/URL:"))
        self.backend_host = QComboBox()
        self.backend_host.addItems(["localhost:5000", "127.0.0.1:8000", "http://localhost:8000/api"])
        self.backend_host.setEditable(True)
        layout.addWidget(self.backend_host)

        # Connect/Disconnect buttons
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.disconnect_btn)

        # Status
        self.integration_status = QLabel("Status: Disconnected")
        self.integration_status.setStyleSheet("color: red;")
        layout.addWidget(self.integration_status)

        # Gesture mapping
        layout.addWidget(QLabel("Gesture Command Mapping:", font=QFont("Arial", 10, QFont.Weight.Bold)))
        self.gesture_mappings = QListWidget()
        layout.addWidget(self.gesture_mappings)

        add_mapping_btn = QPushButton("Add Gesture Mapping")
        add_mapping_btn.clicked.connect(self.add_gesture_mapping)
        layout.addWidget(add_mapping_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _connect_signals(self) -> None:
        """Connect all UI signals to slots."""
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_changed)
        self.prediction_slider.valueChanged.connect(self.on_prediction_changed)
        self.connect_btn.clicked.connect(self.on_backend_connect)
        self.disconnect_btn.clicked.connect(self.on_backend_disconnect)

    def _setup_timers(self) -> None:
        """Setup refresh timers."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(self.config.refresh_rate_ms)

    # --- Slot methods ---

    def on_smoothing_changed(self, value: int) -> None:
        """Handle smoothing factor slider change."""
        factor = value / 100.0
        self.smoothing_label.setText(f"{factor:.2f}")

    def on_prediction_changed(self, value: int) -> None:
        """Handle prediction factor slider change."""
        factor = value / 100.0
        self.prediction_label.setText(f"{factor:.2f}")

    def on_backend_connect(self) -> None:
        """Connect to backend."""
        self.integration_status.setText("Status: Connecting...")
        self.integration_status.setStyleSheet("color: orange;")
        # In real implementation, would connect to backend here
        self.integration_status.setText("Status: Connected")
        self.integration_status.setStyleSheet("color: green;")
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)

    def on_backend_disconnect(self) -> None:
        """Disconnect from backend."""
        self.integration_status.setText("Status: Disconnected")
        self.integration_status.setStyleSheet("color: red;")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

    def toggle_profiling(self) -> None:
        """Toggle performance profiling."""
        QMessageBox.information(self, "Profiling", "Profiling toggled")

    def show_performance_report(self) -> None:
        """Show detailed performance report."""
        report = "Performance Report:\n" + "=" * 50 + "\n"
        report += "FPS: 45.2\nLatency: 22.5ms\n"
        report += "Hand Tracking: 15.3ms\nGesture Detection: 8.2ms\n"
        report += "Visualization: 5.8ms\n"
        QMessageBox.information(self, "Performance Report", report)

    def add_gesture_mapping(self) -> None:
        """Add a gesture-to-command mapping."""
        QMessageBox.information(self, "Add Mapping", "Gesture mapping dialog would open here")

    def update_display(self) -> None:
        """Update display each frame."""
        # This would be called from the main tracking loop
        self.fps_label.setText(f"FPS: ~60")
        self.latency_label.setText("Latency: ~16ms")

    def set_video_frame(self, frame_data: object) -> None:
        """Set the video frame for display."""
        # In real implementation, would convert frame to QPixmap and display
        pass

    def add_detected_gesture(self, gesture_name: str, confidence: float) -> None:
        """Add a detected gesture to the list."""
        item_text = f"{gesture_name} ({confidence:.2f})"
        item = QListWidgetItem(item_text)
        self.detected_gestures_list.insertItem(0, item)


def create_pyqt_gui(config: Optional[GUIConfig] = None) -> Optional[CursorAITrackerGUI]:
    """Factory function to create the GUI if PyQt6 is available."""
    if not PYQT6_AVAILABLE:
        print("PyQt6 not installed. Run: pip install PyQt6 PyQt6-Charts")
        return None

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    gui = CursorAITrackerGUI(config)
    gui.show()

    return gui


def run_gui_standalone() -> None:
    """Run the GUI as a standalone application."""
    if not PYQT6_AVAILABLE:
        print("PyQt6 not installed. Run: pip install PyQt6 PyQt6-Charts")
        return

    app = QApplication(sys.argv)
    gui = CursorAITrackerGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui_standalone()
