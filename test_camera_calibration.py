"""
Unit tests for camera calibration module.

Validates:
- FOV to focal length conversion
- Coordinate transformations (normalized→pixel→3D)
- Calibration serialization/deserialization
- Landmark transformation accuracy
"""

import numpy as np
import sys

try:
    from camera_calibration import (
        CameraIntrinsics, CameraCalibrator, CoordinateTransformer, DepthEstimator
    )
except ImportError:
    print("[Error] camera_calibration module not found")
    sys.exit(1)


def test_estimate_from_image_size():
    """Test FOV-based calibration estimation."""
    print("\n[Test 1] Estimate calibration from image size")

    cal = CameraCalibrator.estimate_from_image_size(1920, 1080, fov_degrees=60.0)

    assert cal.width == 1920, "Width mismatch"
    assert cal.height == 1080, "Height mismatch"
    assert cal.cx == 960.0, "Principal point X should be at center"
    assert cal.cy == 540.0, "Principal point Y should be at center"
    assert abs(cal.fx - cal.fy) < 1.0, "Focal lengths should be equal (square pixels)"
    assert cal.fx > 500 and cal.fx < 3000, "Focal length should be reasonable (500-3000 pixels)"

    print(f"  ✓ Generated calibration:")
    print(f"    - Focal (fx, fy): ({cal.fx:.1f}, {cal.fy:.1f})")
    print(f"    - Principal point (cx, cy): ({cal.cx:.1f}, {cal.cy:.1f})")


def test_normalized_to_pixel():
    """Test normalized→pixel coordinate conversion."""
    print("\n[Test 2] Normalized to pixel conversion")

    cal = CameraIntrinsics(1920, 1080, 960, 540, 1920, 1080)
    transformer = CoordinateTransformer(cal)

    # Test center
    x_pix, y_pix = transformer.normalized_to_pixel(0.5, 0.5)
    assert x_pix == 960.0 and y_pix == 540.0, "Center should map to principal point"

    # Test corner
    x_pix, y_pix = transformer.normalized_to_pixel(0.0, 0.0)
    assert x_pix == 0.0 and y_pix == 0.0, "Top-left corner mismatch"

    x_pix, y_pix = transformer.normalized_to_pixel(1.0, 1.0)
    assert x_pix == 1920.0 and y_pix == 1080.0, "Bottom-right corner mismatch"

    print("  ✓ All coordinate conversions correct")


def test_pixel_to_3d():
    """Test pixel→3D unprojection."""
    print("\n[Test 3] Pixel to 3D unprojection")

    cal = CameraIntrinsics(1000, 1000, 500, 500, 1000, 1000)
    transformer = CoordinateTransformer(cal, depth_scale=1.0)

    # Unproject principal point at Z=1.0
    x, y, z = transformer.pixel_to_3d(500, 500, z_norm=1.0)
    assert abs(x) < 0.01 and abs(y) < 0.01, "Should be near origin at principal point"

    # Check scaling
    x, y, z = transformer.pixel_to_3d(1500, 500, z_norm=1.0)
    expected_x = (1500 - 500) * 1.0 / 1000  # (pixel - cx) * Z / fx
    assert abs(x - expected_x) < 0.01, f"X calculation error: {x} vs {expected_x}"

    print("  ✓ Unprojection calculations correct")


def test_normalized_to_3d():
    """Test full normalized→3D pipeline."""
    print("\n[Test 4] Normalized to 3D transformation")

    cal = CameraCalibrator.estimate_from_image_size(1920, 1080, 60.0)
    transformer = CoordinateTransformer(cal, depth_scale=1.0)

    # Center point at depth 0.5
    x, y, z = transformer.normalized_to_3d(0.5, 0.5, z_norm=0.5)
    assert abs(x) < 0.01 and abs(y) < 0.01, "Center should project near origin"
    assert abs(z - 0.5) < 0.01, "Depth should match input"

    print(f"  ✓ Center (0.5, 0.5, 0.5) → ({x:.3f}, {y:.3f}, {z:.3f})")


def test_distortion_coefficients():
    """Test distortion coefficient storage."""
    print("\n[Test 5] Distortion coefficients")

    cal = CameraIntrinsics(
        1000, 1000, 500, 500, 1000, 1000,
        k1=-0.05, k2=0.01, p1=-0.001, p2=0.0
    )

    coeffs = cal.get_distortion_coeffs()
    assert coeffs[0] == -0.05, "k1 mismatch"
    assert coeffs[1] == 0.01, "k2 mismatch"
    assert coeffs[2] == -0.001, "p1 mismatch"

    print("  ✓ Distortion coefficients stored correctly")


def test_depth_estimation():
    """Test confidence→depth estimation."""
    print("\n[Test 6] Depth estimation from confidence")

    # High confidence → close to camera
    z_high = DepthEstimator.estimate_from_confidence(0.9, min_z=0.3, max_z=1.0)
    assert z_high > 0.8, "High confidence should give high Z (close)"

    # Low confidence → far from camera
    z_low = DepthEstimator.estimate_from_confidence(0.1, min_z=0.3, max_z=1.0)
    assert z_low < 0.4, "Low confidence should give low Z (far)"

    # Monotonic increase
    z1 = DepthEstimator.estimate_from_confidence(0.3)
    z2 = DepthEstimator.estimate_from_confidence(0.7)
    assert z2 > z1, "Higher confidence should give higher Z"

    print("  ✓ Confidence→depth mapping is correct and monotonic")


def test_calibration_save_load():
    """Test JSON serialization/deserialization."""
    print("\n[Test 7] Calibration save/load")

    import tempfile
    import os

    # Create calibration
    cal_original = CameraCalibrator.estimate_from_image_size(1920, 1080, 65.0)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cal.json")
        CameraCalibrator.save_calibration(cal_original, path)

        # Load back
        cal_loaded = CameraCalibrator.load_calibration(path)

        # Verify
        assert cal_loaded.width == cal_original.width, "Width mismatch after load"
        assert abs(cal_loaded.fx - cal_original.fx) < 0.1, "Focal length mismatch"
        assert abs(cal_loaded.cx - cal_original.cx) < 0.1, "Principal point X mismatch"

    print("  ✓ Save/load preserves calibration data")


def test_landmark_transformation():
    """Test 21-landmark MediaPipe transformation."""
    print("\n[Test 8] 21-landmark transformation")

    cal = CameraCalibrator.estimate_from_image_size(1920, 1080, 60.0)
    transformer = CoordinateTransformer(cal, depth_scale=1.0)

    # Simulate 21 normalized MediaPipe landmarks
    landmarks_norm = [
        (0.5, 0.5, 0.5),    # Wrist (center)
        (0.6, 0.3, 0.6),    # Thumb
        (0.4, 0.2, 0.7),    # Index
        (0.4, 0.3, 0.7),    # Middle
        (0.45, 0.4, 0.6),   # Ring
        (0.5, 0.45, 0.5),   # Pinky
    ] + [(0.5 + 0.05*i%5, 0.5, 0.5 + 0.05*i//5) for i in range(15)]

    assert len(landmarks_norm) == 21, "Should have 21 landmarks"

    # Transform
    landmarks_3d = transformer.transform_landmarks(landmarks_norm)

    assert len(landmarks_3d) == 21, "Should return 21 world coordinates"
    assert all(len(lm) == 3 for lm in landmarks_3d), "Each landmark should be (x, y, z)"

    # Check all are reasonable (not NaN or inf)
    for i, lm in enumerate(landmarks_3d):
        assert all(np.isfinite(coord) for coord in lm), f"Landmark {i} has non-finite coordinates"

    print(f"  ✓ Transformed 21 landmarks to 3D world coordinates")
    print(f"    Sample: Landmark 0 (wrist) → {landmarks_3d[0]}")


def test_depth_smoothing():
    """Test exponential smoothing of depth values."""
    print("\n[Test 9] Depth smoothing")

    z_values = [0.5, 0.52, 0.51, 0.53, 0.50, 0.49]

    # Smooth with high factor (more smoothing)
    z_smooth = DepthEstimator.smooth_depth_sequence(z_values, smoothing_factor=0.9)

    # Smoothed values should have less variance
    var_original = np.var(z_values)
    var_smooth = np.var(z_smooth)
    assert var_smooth < var_original, "Smoothing should reduce variance"

    # First value should match
    assert z_smooth[0] == z_values[0], "First smoothed value should match input"

    print(f"  ✓ Depth smoothing reduced variance from {var_original:.6f} to {var_smooth:.6f}")


def test_combine_depth_sources():
    """Test combining multiple depth estimates."""
    print("\n[Test 10] Combining depth sources")

    z_landmark = 0.6
    z_sensor = 0.4

    # 70% landmark, 30% sensor
    z_combined = DepthEstimator.combine_depth_sources(
        z_landmark, z_sensor, landmark_weight=0.7
    )

    expected = 0.7 * z_landmark + 0.3 * z_sensor
    assert abs(z_combined - expected) < 0.001, "Weight combination incorrect"

    # Handle missing values
    z_only_landmark = DepthEstimator.combine_depth_sources(z_landmark, None, 0.7)
    assert z_only_landmark == z_landmark, "Should use landmark if sensor missing"

    z_only_sensor = DepthEstimator.combine_depth_sources(None, z_sensor, 0.7)
    assert z_only_sensor == z_sensor, "Should use sensor if landmark missing"

    print("  ✓ Depth source combination works correctly")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("CAMERA CALIBRATION TEST SUITE")
    print("="*70)

    tests = [
        test_estimate_from_image_size,
        test_normalized_to_pixel,
        test_pixel_to_3d,
        test_normalized_to_3d,
        test_distortion_coefficients,
        test_depth_estimation,
        test_calibration_save_load,
        test_landmark_transformation,
        test_depth_smoothing,
        test_combine_depth_sources,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
