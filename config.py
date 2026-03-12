"""
Configuration module for the Cursor AI Tracker project.

All tunable parameters for tracking, cursor control, motion analysis, and
gesture detection live here so that experiments can be run by editing a
single file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CursorAIConfig:
    """
    Global configuration for the Cursor AI Tracker system.

    This dataclass is intentionally flat and simple so it can be easily
    serialized, logged, or replaced by a config file in the future.
    """

    # --- Hand tracking / MediaPipe ---
    # Maximum number of hands to detect per frame.
    max_num_hands: int = 2
    # Minimum confidence for initial hand detection.
    min_detection_confidence: float = 0.5
    # Minimum confidence to consider a hand "present" in the frame.
    min_presence_confidence: float = 0.5
    # Minimum confidence required to keep tracking an already detected hand.
    min_tracking_confidence: float = 0.5
    # If True, horizontally mirror the webcam image to behave like a mirror.
    mirror_image: bool = True

    # --- Cursor control ---
    # Exponential smoothing factor for cursor position (0..1, lower = smoother).
    cursor_smoothing: float = 0.25
    # Factor for simple velocity-based prediction (0 disables prediction).
    cursor_predictive: float = 0.0
    # If False, no OS cursor movement will occur (visualization only).
    enable_cursor_control: bool = True
    # If True, use a Kalman filter instead of plain exponential smoothing.
    use_kalman_filter: bool = False
    # If True, adapt smoothing/prediction strength based on hand speed.
    adaptive_smoothing: bool = False
    # Preferred primary cursor hand ("left" or "right").
    primary_cursor_hand: str = "left"

    # --- Motion analysis ---
    # Number of historical samples to keep per fingertip for trajectory analysis.
    motion_history_size: int = 20
    # Thresholds (in normalized units / sec) to categorize motion speed.
    low_speed_threshold: float = 0.2
    high_speed_threshold: float = 0.8

    # --- Gesture thresholds ---
    # Maximum thumb-index distance for a pinch in normalized image space.
    pinch_distance_threshold: float = 0.05
    # Minimum fingertip displacement for a motion to qualify as a swipe.
    swipe_distance_threshold: float = 0.20
    # Minimum path length for a circular motion gesture.
    circle_min_path: float = 0.6
    # Maximum start-end distance for a circular motion (circle should "close").
    circle_close_distance: float = 0.05
    # Minimum joint angle (in degrees) to consider a finger "straight"/extended.
    finger_angle_threshold_deg: float = 160.0
    # Extra required distance between wrist-TIP vs wrist-PIP to count as extended.
    wrist_distance_margin: float = 0.015

    # --- Feature toggles / profiling ---
    # Enable or disable heavy motion analysis for performance tuning.
    enable_motion_analysis: bool = True
    # Enable or disable gesture detection (static + dynamic).
    enable_gesture_detection: bool = True
    # Enable or disable two-hand gesture detection (zoom/separate/rotate).
    enable_two_hand_gestures: bool = True
    # If True, draw fingertip motion trails.
    enable_motion_trails: bool = True
    # If True, collect and display basic profiling information per frame.
    enable_profiling: bool = False
    # Enable ML-based gesture prediction (requires trained models).
    enable_ml_gestures: bool = False
    # Enable prediction logging for cursor visualization.
    enable_cursor_prediction_logging: bool = False
    # Enable PyQt6 GUI (requires PyQt6 installation).
    enable_gui: bool = False
    # Enable virtual target integration.
    enable_integration: bool = False

    # --- ML/Advanced features ---
    # Use enhanced Kalman filter (constant-acceleration model).
    use_enhanced_kalman: bool = False
    # Adapt Kalman filter measurement noise based on motion speed.
    kalman_adaptive_noise: bool = True
    # Path to pre-trained gesture models directory.
    ml_models_dir: str = "models/gestures"

    # --- Integration settings ---
    # Integration backend: "socket", "http", "ue5", "robot", or "none"
    integration_backend: str = "none"
    # Backend connection address (host:port or URL)
    integration_endpoint: str = "localhost:5000"

    # --- 3D/AR/VR FEATURES ---
    # Master enable for 3D world-space coordinates and gesture processing
    enable_3d_world: bool = True
    # Enable gesture-driven 6-DOF robot arm control
    enable_robot_control: bool = False
    # Enable real-time Unreal Engine 5 skeletal mesh animation
    enable_ue5_integration: bool = True
    # Enable matplotlib 3D visualization (optional, performance impact)
    enable_3d_visualization: bool = False

    # --- CAMERA CALIBRATION ---
    # Path to camera calibration file
    camera_calibration_file: str = "calibration.json"
    # Auto-calibrate from image size if file not found
    camera_auto_calibrate: bool = True
    # Assumed field-of-view for auto-calibration (degrees)
    camera_fov_degrees: float = 60.0
    # Depth scale factor for world coordinates (1.0 = meters)
    camera_depth_scale: float = 1.0

    # --- ROBOT CONFIGURATION ---
    # UR5 robot IP address
    robot_ip_address: str = "192.168.1.100"
    # RTDE port for UR5 communication
    robot_port: int = 30003
    # Connection timeout in seconds
    robot_connection_timeout: float = 5.0
    # Maximum arm velocity (m/s), conservative for first live tests
    robot_max_velocity: float = 0.15
    # Maximum pose displacement per command (meters), conservative
    robot_max_step_m: float = 0.01
    # Robot workspace envelope min corner (meters)
    robot_workspace_min_xyz: tuple[float, float, float] = (0.20, -0.30, 0.20)
    # Robot workspace envelope max corner (meters)
    robot_workspace_max_xyz: tuple[float, float, float] = (0.65, 0.30, 0.70)
    # If True, reject out-of-workspace targets instead of clamping
    robot_stop_on_out_of_workspace: bool = True
    # Enable PyBullet collision detection (optional, performance impact)
    robot_enable_collision_check: bool = False

    # --- UNREAL ENGINE CONFIGURATION ---
    # Path to UE5 project directory
    unreal_project_path: str = ""
    # Use WebSocket if Python API unavailable (for remote UE5)
    unreal_use_websocket: bool = True
    # WebSocket endpoint for remote UE5 connections
    unreal_websocket_endpoint: str = "ws://localhost:8765"
    # WebSocket connection timeout (seconds)
    unreal_websocket_timeout: float = 5.0
    # If True, allow slow short-lived websocket fallback when persistent mode is unavailable.
    unreal_allow_short_lived_fallback: bool = False
    # Actor name that receives object-manipulation transforms.
    unreal_object_actor_name: str = "ManipulatedObject"

    # --- 3D PIPELINE PARAMETERS ---
    # Enable adaptive Kalman noise scaling based on motion magnitude
    kalman_3d_adaptive_noise: bool = True
    # Exponential smoothing factor for Z-depth (0-1, lower = smoother)
    depth_smoothing_factor: float = 0.7
    # Physics damping factor for 3D objects (0.9 = 10% energy loss per frame)
    physics_damping: float = 0.95
    # Enable profiling of 3D pipeline stages (latency per module)
    enable_3d_profiling: bool = False

    # --- CONTROL INTELLIGENCE LAYER ---
    # Dual-mode architecture: "real_interface" or "virtual_execution"
    control_mode: str = "real_interface"
    # Runtime context for state-based mapping
    control_context: str = "robot"
    # Master toggle for predictive smoothing + stateful action mapping
    enable_control_intelligence: bool = True
    # Exponential smoothing alpha: S(t) = a*X(t) + (1-a)*S(t-1)
    stability_smoothing_alpha: float = 0.35
    # Ignore tiny coordinate motion below this threshold
    stability_deadzone: float = 0.002
    # Enable velocity-based latency compensation
    enable_latency_compensation: bool = True
    # Baseline CV->execution latency estimate used by predictor
    base_pipeline_latency_ms: float = 35.0
    # Max command dispatch rate (prevents command spam/jitter)
    max_command_rate_hz: float = 20.0
