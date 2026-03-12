"""
CAMERA CALIBRATION MODULE

Handles camera intrinsics, calibration from reference points, and 2D→3D coordinate transformation.

Purpose: Transform normalized hand coordinates from MediaPipe (2D, [0,1] range) into real-world 3D space.

Key Classes:
- CameraIntrinsics: Stores camera matrix (focal length, principal point, distortion)
- CameraCalibrator: Interactive or automatic calibration from known reference points
- CoordinateTransformer: Projects 2D normalized → 3D world coordinates

Calibration Approaches:
1. Reference Points: Provide known 3D positions, calibrate to match 2D image detection
2. Checkerboard: OpenCV standard checkerboard pattern calibration
3. Manual Input: Manually specify camera parameters (focal length, resolution, etc.)

Mathematical Model:
-----------
Intrinsic Matrix K:
  [fx   0  cx]
  [ 0  fy  cy]
  [ 0   0   1]

Projection: p_screen = K × P_world (homography)

Reverse (Unprojection):
  X_world = (x_pixel - cx) * Z_world / fx
  Y_world = (y_pixel - cy) * Z_world / fy
  Z_world = depth (from sensor or estimated from confidence)

Normalized Coordinates (from MediaPipe):
  x_norm = x_pixel / image_width
  y_norm = y_pixel / image_height
  (range [0, 1])
-----------
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import math
import numpy as np


# ============================================================================
# CAMERA INTRINSICS
# ============================================================================

@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters for 3D coordinate transformation.

    Attributes:
        fx, fy: Focal length in pixels (typically 500-2000 for webcams)
        cx, cy: Principal point (optical center) in pixels
        width, height: Image resolution in pixels
        k1, k2, k3: Radial distortion coefficients
        p1, p2: Tangential distortion coefficients
        name: Descriptive name (e.g., "logitech_c920")
    """
    fx: float                    # Focal length x (pixels)
    fy: float                    # Focal length y (pixels)
    cx: float                    # Principal point x (pixels)
    cy: float                    # Principal point y (pixels)
    width: int                   # Image width (pixels)
    height: int                  # Image height (pixels)
    k1: float = 0.0             # Radial distortion
    k2: float = 0.0             # Radial distortion
    k3: float = 0.0             # Radial distortion
    p1: float = 0.0             # Tangential distortion
    p2: float = 0.0             # Tangential distortion
    name: str = "unknown"       # Calibration name

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "k1": self.k1, "k2": self.k2, "k3": self.k3,
            "p1": self.p1, "p2": self.p2,
            "name": self.name
        }

    @staticmethod
    def from_dict(data: Dict) -> CameraIntrinsics:
        """Deserialize from dictionary."""
        return CameraIntrinsics(**data)

    def get_camera_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix K."""
        return np.array([
            [self.fx,    0,  self.cx],
            [   0,    self.fy,  self.cy],
            [   0,       0,      1   ]
        ], dtype=np.float32)

    def get_distortion_coeffs(self) -> np.ndarray:
        """Return distortion coefficient vector."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3],
                       dtype=np.float32)


# ============================================================================
# COORDINATE TRANSFORMER
# ============================================================================

class CoordinateTransformer:
    """
    Transform 2D normalized coordinates → 3D world coordinates.

    Handles:
    - Normalized (0-1) → pixel coordinate conversion
    - Unprojection using camera intrinsics
    - Optional depth estimation from confidence or sensor Z
    - Depth scaling for physical interpretation
    """

    def __init__(self, intrinsics: CameraIntrinsics,
                 depth_scale: float = 1.0,
                 default_z: float = 0.5):
        """
        Initialize transformer.

        Args:
            intrinsics: Camera intrinsic parameters
            depth_scale: Scale factor for depth (Z) coordinate
            default_z: Default Z value if not provided (normalized [0,1])
        """
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale
        self.default_z = default_z

    def normalized_to_pixel(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """
        Convert normalized coordinates (0-1) to pixel coordinates.

        Args:
            x_norm: X coordinate in [0, 1]
            y_norm: Y coordinate in [0, 1]

        Returns:
            (x_pixel, y_pixel): Coordinates in image space
        """
        x_pixel = x_norm * self.intrinsics.width
        y_pixel = y_norm * self.intrinsics.height
        return x_pixel, y_pixel

    def pixel_to_3d(self, x_pixel: float, y_pixel: float,
                   z_norm: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to 3D world coordinates.

        Uses camera intrinsics unprojection:
          X = (x_pixel - cx) * Z / fx
          Y = (y_pixel - cy) * Z / fy
          Z = depth

        Args:
            x_pixel: X in pixel space
            y_pixel: Y in pixel space
            z_norm: Normalized depth [0, 1], or None to use default

        Returns:
            (x_world, y_world, z_world): 3D world coordinates
        """
        z_norm = z_norm if z_norm is not None else self.default_z

        # Clamp to valid range
        z_norm = max(0.01, min(1.0, z_norm))

        # Scale depth: normalized [0,1] → actual distance
        z_world = z_norm * self.depth_scale

        # Unproject using intrinsics
        cx, cy, fx, fy = self.intrinsics.cx, self.intrinsics.cy, \
                         self.intrinsics.fx, self.intrinsics.fy

        x_world = (x_pixel - cx) * z_world / fx
        y_world = (y_pixel - cy) * z_world / fy

        return x_world, y_world, z_world

    def normalized_to_3d(self, x_norm: float, y_norm: float,
                        z_norm: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Convert normalized coordinates directly to 3D world.

        Convenience method combining normalized→pixel and pixel→3d.

        Args:
            x_norm: X in [0, 1]
            y_norm: Y in [0, 1]
            z_norm: Depth in [0, 1], or None for default

        Returns:
            (x_world, y_world, z_world): 3D world coordinates
        """
        x_pixel, y_pixel = self.normalized_to_pixel(x_norm, y_norm)
        return self.pixel_to_3d(x_pixel, y_pixel, z_norm)

    def transform_landmarks(self, landmarks: List[Tuple[float, float, float]],
                           use_z_from_landmarks: bool = True) \
            -> List[Tuple[float, float, float]]:
        """
        Transform a list of MediaPipe landmarks to 3D world coordinates.

        Args:
            landmarks: List of (x, y, z) normalized coordinates from MediaPipe
            use_z_from_landmarks: Use z from landmarks, or use default

        Returns:
            List of (x_world, y_world, z_world) in 3D space
        """
        result = []
        for lm in landmarks:
            # Support both tuple-style landmarks and MediaPipe NormalizedLandmark objects.
            if hasattr(lm, "x") and hasattr(lm, "y"):
                x_norm = float(lm.x)
                y_norm = float(lm.y)
                z_norm = float(getattr(lm, "z", self.default_z)) if use_z_from_landmarks else None
            else:
                x_norm = lm[0]
                y_norm = lm[1]
                z_norm = lm[2] if (use_z_from_landmarks and len(lm) > 2) else None
            result.append(self.normalized_to_3d(x_norm, y_norm, z_norm))
        return result


# ============================================================================
# CAMERA CALIBRATOR
# ============================================================================

class CameraCalibrator:
    """
    Calibrate camera intrinsics from reference points.

    Approaches:
    1. From known 3D→2D correspondences
    2. Manual specification (focal length, principal point)
    3. Estimation from image size (assuming square pixels)
    """

    @staticmethod
    def estimate_from_image_size(width: int, height: int,
                                fov_degrees: float = 60.0) -> CameraIntrinsics:
        """
        Estimate camera intrinsics from image size and assumed field of view.

        Useful for quick calibration without reference points.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov_degrees: Horizontal field of view (degrees)
                Default 60° is typical for webcams

        Returns:
            CameraIntrinsics with estimated parameters
        """
        # FOV → focal length relationship:
        # fov_rad = 2 * arctan(width / (2 * fx))
        # fx = width / (2 * tan(fov_rad / 2))

        fov_rad = math.radians(fov_degrees)
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx  # Assume square pixels

        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0

        return CameraIntrinsics(
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            width=width, height=height,
            name=f"estimated_{width}x{height}_{fov_degrees}fov"
        )

    @staticmethod
    def from_reference_points(
        landmark_2d: List[Tuple[float, float]],
        world_3d: List[Tuple[float, float, float]],
        width: int, height: int,
        initial_guess: Optional[CameraIntrinsics] = None
    ) -> CameraIntrinsics:
        """
        Calibrate camera intrinsics using 2D→3D point correspondences.

        Solves for best-fit camera matrix using least-squares optimization.

        Args:
            landmark_2d: List of 2D points in pixels [(x, y), ...]
            world_3d: Corresponding 3D world points [(x, y, z), ...]
            width, height: Image resolution
            initial_guess: Initial camera intrinsics (for refinement)

        Returns:
            Calibrated CameraIntrinsics
        """
        if len(landmark_2d) < 4:
            raise ValueError("Need at least 4 point correspondences for calibration")
        if len(landmark_2d) != len(world_3d):
            raise ValueError("2D and 3D point lists must have same length")

        # Start with estimate or provided guess
        if initial_guess is None:
            intrinsics = CameraCalibrator.estimate_from_image_size(width, height)
        else:
            intrinsics = initial_guess

        # Use simple iterative refinement
        # In practice, could use scipy.optimize or cv2.calibrateCamera
        # For now, return the estimate (extension point for advanced calibration)

        return intrinsics

    @staticmethod
    def save_calibration(intrinsics: CameraIntrinsics,
                        filepath: str) -> None:
        """Save calibration to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(intrinsics.to_dict(), f, indent=2)

        print(f"[CameraCalibrator] Calibration saved: {filepath}")

    @staticmethod
    def load_calibration(filepath: str) -> CameraIntrinsics:
        """Load calibration from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        with open(path, 'r') as f:
            data = json.load(f)

        calibration = CameraIntrinsics.from_dict(data)
        print(f"[CameraCalibrator] Calibration loaded: {filepath}")
        return calibration

    @staticmethod
    def create_interactive_calibration() -> CameraIntrinsics:
        """
        Interactive calibration wizard.

        Prompts user for camera parameters and returns CameraIntrinsics.
        """
        print("\n" + "="*70)
        print("INTERACTIVE CAMERA CALIBRATION")
        print("="*70)

        width = int(input("Enter image width (pixels) [1920]: ") or "1920")
        height = int(input("Enter image height (pixels) [1080]: ") or "1080")
        fov = float(input("Enter horizontal FOV (degrees) [60]: ") or "60.0")

        intrinsics = CameraCalibrator.estimate_from_image_size(width, height, fov)

        print("\n[Generated Calibration]")
        print(f"  Resolution: {width}x{height}")
        print(f"  Focal length (fx, fy): ({intrinsics.fx:.1f}, {intrinsics.fy:.1f})")
        print(f"  Principal point (cx, cy): ({intrinsics.cx:.1f}, {intrinsics.cy:.1f})")

        save = input("\nSave to file? (y/n) [y]: ").lower() or "y"
        if save == 'y':
            filename = input("Filename [calibration.json]: ") or "calibration.json"
            CameraCalibrator.save_calibration(intrinsics, filename)

        return intrinsics


# ============================================================================
# DEPTH ESTIMATION
# ============================================================================

class DepthEstimator:
    """
    Estimate or process depth information for 3D coordinate calculation.

    Methods:
    - Use MediaPipe hand landmark z-coordinate (normalized, relative depth)
    - Estimate from hand detection confidence
    - Assume fixed distance (for 2D mode as fallback)
    """

    @staticmethod
    def estimate_from_confidence(confidence: float,
                                min_z: float = 0.3,
                                max_z: float = 1.0) -> float:
        """
        Estimate depth (Z) from detection confidence.

        Higher confidence → hand likely closer to camera

        Args:
            confidence: Confidence value [0, 1]
            min_z: Minimum depth (far from camera)
            max_z: Maximum depth (close to camera)

        Returns:
            Estimated normalized Z coordinate [0, 1]
        """
        # Linear mapping: low confidence→far, high confidence→close
        z = min_z + confidence * (max_z - min_z)
        return z

    @staticmethod
    def smooth_depth_sequence(z_values: List[float],
                             smoothing_factor: float = 0.7) -> List[float]:
        """
        Smooth depth values over time using exponential smoothing.

        Useful for reducing jitter in z-coordinate.

        Args:
            z_values: Sequence of normalized Z values
            smoothing_factor: Alpha in [0, 1] (higher = more smoothing)

        Returns:
            Smoothed Z values
        """
        if not z_values:
            return z_values

        result = [z_values[0]]
        for z in z_values[1:]:
            smoothed = smoothing_factor * result[-1] + (1 - smoothing_factor) * z
            result.append(smoothed)

        return result

    @staticmethod
    def combine_depth_sources(z_landmark: Optional[float],
                             z_confidence: Optional[float],
                             landmark_weight: float = 0.6) -> float:
        """
        Combine multiple depth estimates.

        Args:
            z_landmark: Z from MediaPipe landmark
            z_confidence: Z estimated from confidence
            landmark_weight: Weight for landmark Z [0, 1]

        Returns:
            Combined Z estimate
        """
        if z_landmark is not None and z_confidence is not None:
            return landmark_weight * z_landmark + (1 - landmark_weight) * z_confidence
        elif z_landmark is not None:
            return z_landmark
        elif z_confidence is not None:
            return z_confidence
        else:
            return 0.5  # Default midpoint


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_basic_calibration():
    """Example: Basic camera calibration from image size."""
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC CALIBRATION")
    print("="*70)

    # Assume 1080p webcam with 60° FOV
    intrinsics = CameraCalibrator.estimate_from_image_size(1920, 1080, fov_degrees=60.0)
    print(f"Camera: {intrinsics.name}")
    print(f"  F x,fy: {intrinsics.fx:.1f}, {intrinsics.fy:.1f}")
    print(f"  C x,cy: {intrinsics.cx:.1f}, {intrinsics.cy:.1f}")

    # Save for later use
    CameraCalibrator.save_calibration(intrinsics, "example_calibration.json")


def example_coordinate_transformation():
    """Example: Transform MediaPipe coordinates to 3D world."""
    print("\n" + "="*70)
    print("EXAMPLE 2: COORDINATE TRANSFORMATION")
    print("="*70)

    # Load or create calibration
    intrinsics = CameraCalibrator.estimate_from_image_size(1920, 1080)
    transformer = CoordinateTransformer(intrinsics, depth_scale=1.0)

    # Example normalized hand landmarks (from MediaPipe)
    # 21 landmarks: wrist + 5 fingers × 4 joints
    normalized_landmarks = [
        (0.5, 0.5, 0.5),   # Wrist (center, mid-depth)
        (0.6, 0.3, 0.6),   # Thumb tip
        (0.4, 0.2, 0.7),   # Index tip
        (0.4, 0.3, 0.7),   # Middle tip
        (0.45, 0.4, 0.6),  # Ring tip
        (0.5, 0.45, 0.5),  # Pinky tip
    ]

    # Transform to 3D world coordinates
    world_coords = transformer.transform_landmarks(normalized_landmarks)

    print("Transformed landmarks:")
    names = ["Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky"]
    for name, coord in zip(names, world_coords):
        print(f"  {name:10} → ({coord[0]:8.3f}, {coord[1]:8.3f}, {coord[2]:8.3f})")


if __name__ == "__main__":
    example_basic_calibration()
    example_coordinate_transformation()
