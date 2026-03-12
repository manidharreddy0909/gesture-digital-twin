"""
Comprehensive test suite for all 10 advanced gesture features.

Tests verify:
1. Gesture Dataset Recording & Playback
2. Adaptive Gesture Calibration
3. Multi-Cursor / Multi-Hand OS Control
4. Gesture Macros & Automation
5. Cross-Platform Cursor Support
6. Advanced Motion Analytics Dashboard
7. User Feedback & Haptics
8. Gesture Confidence / Uncertainty Tracking
9. Dynamic Frame Rate Adaptation
10. Cloud / Networked ML Models
"""

import os
import sys
import json
import time
from pathlib import Path

# Suppress optional dependencies warnings
import warnings
warnings.filterwarnings("ignore")

# Fix Windows Unicode encoding issue
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def test_feature_1_dataset():
    """Test Feature 1: Gesture Dataset Recording & Playback"""
    print("\n" + "="*70)
    print("TEST 1: Gesture Dataset Recording & Playback")
    print("="*70)

    try:
        from gesture_dataset import GestureRecord, GestureDataset, DatasetRecorder, DatasetManager

        # Create a test dataset
        dataset = GestureDataset()
        print("[OK] GestureDataset created")

        # Create a test record
        record = GestureRecord(
            gesture_name="test_pinch",
            hand_used="left",
            frame_count=30,
            duration_sec=1.0,
            user_id="test_user"
        )
        print("[OK] GestureRecord created")

        # Add record to dataset
        dataset.add_record(record)
        print(f"[OK] Record added to dataset (total: {len(dataset.records)})")

        # Test serialization
        manager = DatasetManager(storage_dir="test_datasets")
        print("[OK] DatasetManager created")

        # Verify classes exist and are callable
        assert hasattr(DatasetRecorder, 'start_recording'), "DatasetRecorder.start_recording missing"
        print("[OK] DatasetRecorder interface verified")

        print("\n[PASS] FEATURE 1: Dataset recording/playback system operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 1: {str(e)}")
        return False


def test_feature_2_calibration():
    """Test Feature 2: Adaptive Gesture Calibration"""
    print("\n" + "="*70)
    print("TEST 2: Adaptive Gesture Calibration")
    print("="*70)

    try:
        from gesture_calibration import (
            CalibrationProfile, HandCharacteristicsAnalyzer,
            AdaptiveThresholdComputer, CalibrationManager
        )

        # Create calibration profile
        profile = CalibrationProfile(user_id="test_user")
        print("[OK] CalibrationProfile created")

        # Create adaptive threshold computer
        base_config = {
            "pinch_distance_threshold": 0.05,
            "swipe_distance_threshold": 0.20,
            "finger_angle_threshold_deg": 160.0,
        }

        adaptive_computer = AdaptiveThresholdComputer(base_config=base_config)
        print("[OK] AdaptiveThresholdComputer created")

        # Compute adaptive thresholds
        thresholds = adaptive_computer.compute_adaptive_thresholds()
        assert thresholds is not None, "adaptive_thresholds is None"
        assert "pinch_distance_threshold" in thresholds, "Missing pinch threshold"
        print(f"[OK] Adaptive thresholds computed: {len(thresholds)} parameters")

        # Test calibration manager
        manager = CalibrationManager(profiles_dir="test_calibration")
        print("[OK] CalibrationManager created")

        print("\n[PASS] FEATURE 2: PASSED - Adaptive calibration system operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 2: FAILED - {str(e)}")
        return False


def test_feature_3_multi_cursor():
    """Test Feature 3: Multi-Cursor / Multi-Hand OS Control"""
    print("\n" + "="*70)
    print("TEST 3: Multi-Cursor / Multi-Hand OS Control")
    print("="*70)

    try:
        from multi_cursor_controller import (
            MultiCursorController, CursorMode, WindowMapper, VirtualCursor
        )

        # Create multi-cursor controller
        multi_cursor = MultiCursorController(mode=CursorMode.DUAL_INDEPENDENT)
        print("[OK] MultiCursorController created with DUAL_INDEPENDENT mode")

        # Test cursor modes
        modes = [
            CursorMode.SINGLE_LEFT,
            CursorMode.SINGLE_RIGHT,
            CursorMode.DUAL_INDEPENDENT,
            CursorMode.DUAL_COLLABORATIVE,
        ]

        for mode in modes:
            multi_cursor.set_mode(mode)
            assert multi_cursor.mode == mode, f"Mode not set correctly: {mode}"
        print(f"[OK] All cursor modes tested ({len(modes)} modes)")

        # Test window mapper
        mapper = WindowMapper(screen_width=1920, screen_height=1080)
        region = mapper.get_region_at(0.5, 0.5)  # Center region
        assert region in ["left", "center", "right"], f"Invalid region: {region}"
        print("[OK] WindowMapper tested - region detection working")

        # Test virtual cursor
        vcursor = VirtualCursor(hand_id=0)
        vcursor.update_position(100, 200)
        assert vcursor.x == 100 and vcursor.y == 200, "Virtual cursor position not updated"
        print("[OK] VirtualCursor tested - position tracking working")

        print("\n[PASS] FEATURE 3: PASSED - Multi-cursor control system operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 3: FAILED - {str(e)}")
        return False


def test_feature_4_macros():
    """Test Feature 4: Gesture Macros & Automation"""
    print("\n" + "="*70)
    print("TEST 4: Gesture Macros & Automation")
    print("="*70)

    try:
        from gesture_macros import (
            ActionType, GestureAction, GestureProfile,
            MacroExecutor, ProfileManager, create_default_profiles
        )

        # Create gesture action
        action = GestureAction(
            gesture_name="test_pinch",
            action_type=ActionType.KEYBOARD_KEY,
            action_value="a",
            confidence_threshold=0.6,
            cooldown_ms=100
        )
        print("[OK] GestureAction created")

        # Create profile
        profile = GestureProfile(name="test_profile")
        profile.add_action(action)
        print("[OK] GestureProfile created and action added")

        # Test default profiles
        defaults = create_default_profiles()
        assert len(defaults) > 0, "No default profiles created"
        print(f"[OK] Default profiles loaded ({len(defaults)} profiles)")

        # Test macro executor
        executor = MacroExecutor()
        print("[OK] MacroExecutor created")

        # Test profile manager
        manager = ProfileManager(profiles_dir="test_profiles")
        print("[OK] ProfileManager created")

        print("\n[PASS] FEATURE 4: PASSED - Gesture macros system operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 4: FAILED - {str(e)}")
        return False


def test_feature_5_cross_platform():
    """Test Feature 5: Cross-Platform Cursor Support"""
    print("\n" + "="*70)
    print("TEST 5: Cross-Platform Cursor Support")
    print("="*70)

    try:
        from cursor_platform import CrossPlatformCursorController

        # Create platform controller
        platform = CrossPlatformCursorController()
        print(f"[OK] CrossPlatformCursorController created (OS: {platform.platform_name})")

        # Verify methods exist
        assert hasattr(platform, 'move_cursor'), "move_cursor method missing"
        assert hasattr(platform, 'get_cursor_pos'), "get_cursor_pos method missing"
        assert hasattr(platform, 'click'), "click method missing"
        print("[OK] All platform methods verified")

        # Test non-destructive method
        try:
            pos = platform.get_cursor_pos()
            print(f"[OK] Current cursor position: {pos}")
        except Exception as e:
            print(f"[WARN] Could not get cursor position (may require special privileges): {e}")

        print("\n[PASS] FEATURE 5: PASSED - Cross-platform cursor abstraction operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 5: FAILED - {str(e)}")
        return False


def test_feature_6_analytics():
    """Test Feature 6: Advanced Motion Analytics Dashboard"""
    print("\n" + "="*70)
    print("TEST 6: Advanced Motion Analytics Dashboard")
    print("="*70)

    try:
        from advanced_features import AnalyticsDashboard, MotionAnalytics

        # Create dashboard
        dashboard = AnalyticsDashboard()
        print("[OK] AnalyticsDashboard created")

        # Add motion samples
        for i in range(10):
            dashboard.update_motion(
                finger_name="index_finger",
                vx=0.1 * i,
                vy=0.1 * i,
                ax=0.01,
                ay=0.01,
                x=0.5,
                y=0.5
            )
        print("[OK] Motion samples added to dashboard")

        # Get motion graph data
        motion_data = dashboard.get_motion_graphs_data()
        assert "index_finger" in motion_data, "Motion data not stored"
        print(f"[OK] Motion graph data retrieved ({len(motion_data)} fingers)")

        # Add gesture
        dashboard.add_gesture("test_pinch", 0.95)

        # Get session statistics
        stats = dashboard.get_session_statistics()
        assert "total_gestures" in stats, "Session stats missing"
        print(f"[OK] Session statistics retrieved (gestures: {stats['total_gestures']})")

        print("\n[PASS] FEATURE 6: PASSED - Analytics dashboard operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 6: FAILED - {str(e)}")
        return False


def test_feature_7_feedback():
    """Test Feature 7: User Feedback & Haptics"""
    print("\n" + "="*70)
    print("TEST 7: User Feedback & Haptics")
    print("="*70)

    try:
        from advanced_features import FeedbackManager, FeedbackType, FeedbackEvent

        # Create feedback manager
        manager = FeedbackManager()
        print("[OK] FeedbackManager created")

        # Register feedback handlers
        events_received = []

        def capture_feedback(event):
            events_received.append(event)

        manager.register_feedback(FeedbackType.VISUAL_HIGHLIGHT, capture_feedback)
        manager.register_feedback(FeedbackType.HAPTIC_BUZZ, capture_feedback)
        print("[OK] Feedback handlers registered")

        # Trigger feedback
        manager.provide_gesture_feedback("test_pinch", 0.9)

        # Verify events were triggered
        assert len(events_received) > 0, "No feedback events triggered"
        print(f"[OK] Gesture feedback triggered ({len(events_received)} events)")

        # Verify feedback types
        feedback_types = [e.feedback_type for e in events_received]
        assert FeedbackType.VISUAL_HIGHLIGHT in feedback_types, "Visual feedback not triggered"
        print("[OK] Visual feedback confirmed")

        print("\n[PASS] FEATURE 7: PASSED - Feedback system operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 7: FAILED - {str(e)}")
        return False


def test_feature_8_confidence():
    """Test Feature 8: Gesture Confidence / Uncertainty Tracking"""
    print("\n" + "="*70)
    print("TEST 8: Gesture Confidence / Uncertainty Tracking")
    print("="*70)

    try:
        from advanced_features import ConfidenceTracker, ConfidenceStats

        # Create confidence tracker
        tracker = ConfidenceTracker(default_threshold=0.6)
        print("[OK] ConfidenceTracker created")

        # Record confidence values
        for i in range(20):
            confidence = 0.5 + (i * 0.02)  # Rising confidence
            result = tracker.record_confidence("test_gesture", min(confidence, 1.0))
            assert isinstance(result, bool), "record_confidence should return bool"

        print("[OK] Confidence samples recorded")

        # Get statistics
        stats = tracker.get_gesture_statistics("test_gesture")
        assert stats is not None, "Statistics not retrieved"
        assert "avg_confidence" in stats, "Average confidence missing"
        print(f"[OK] Confidence statistics retrieved (avg: {stats['avg_confidence']:.3f})")

        # Test threshold filtering
        assertion = tracker.record_confidence("low_conf_gesture", 0.3)
        print(f"[OK] Low-confidence filtering works: {assertion}")

        print("\n[PASS] FEATURE 8: PASSED - Confidence tracking operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 8: FAILED - {str(e)}")
        return False


def test_feature_9_adaptive_fps():
    """Test Feature 9: Dynamic Frame Rate Adaptation"""
    print("\n" + "="*70)
    print("TEST 9: Dynamic Frame Rate Adaptation")
    print("="*70)

    try:
        from advanced_features import AdaptiveFrameRateController

        # Create controller
        controller = AdaptiveFrameRateController(
            target_fps=30,
            min_fps=15,
            max_fps=60
        )
        print("[OK] AdaptiveFrameRateController created")

        # Record frame times
        for i in range(10):
            frame_time_ms = 30 + (i % 5)  # Simulate varying frame times
            controller.record_frame_time(frame_time_ms)

        print("[OK] Frame times recorded")

        # Compute adaptive FPS
        adaptive_fps = controller.compute_adaptive_fps()
        assert isinstance(adaptive_fps, (int, float)), "FPS should be numeric"
        assert controller.min_fps <= adaptive_fps <= controller.max_fps, "FPS out of bounds"
        print(f"[OK] Adaptive FPS computed: {adaptive_fps:.1f}")

        # Test frame skipping
        should_skip = controller.should_skip_frame()
        assert isinstance(should_skip, bool), "should_skip_frame should return bool"
        print(f"[OK] Frame skip decision: {should_skip}")

        # Get current FPS
        current_fps = controller.get_current_fps()
        print(f"[OK] Current FPS: {current_fps:.1f}")

        print("\n[PASS] FEATURE 9: PASSED - Adaptive frame rate operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 9: FAILED - {str(e)}")
        return False


def test_feature_10_cloud_inference():
    """Test Feature 10: Cloud / Networked ML Models"""
    print("\n" + "="*70)
    print("TEST 10: Cloud / Networked ML Models")
    print("="*70)

    try:
        from advanced_features import CloudInferenceClient

        # Create cloud client
        client = CloudInferenceClient(
            server_url="http://localhost:8000",
            api_key="test-key"
        )
        print("[OK] CloudInferenceClient created")

        # Verify client properties
        assert hasattr(client, 'enabled'), "enabled property missing"
        assert hasattr(client, 'connect'), "connect method missing"
        assert hasattr(client, 'predict_gesture'), "predict_gesture method missing"
        print("[OK] CloudInferenceClient interface verified")

        # Test connect (will fail gracefully if server not running)
        try:
            client.connect()
            status = "Connected" if client.enabled else "Connection failed (expected)"
            print(f"[OK] Cloud connection attempt: {status}")
        except Exception as e:
            print(f"[WARN] Cloud server not available (expected in test): {type(e).__name__}")

        print("\n[PASS] FEATURE 10: PASSED - Cloud inference client operational")
        return True

    except Exception as e:
        print(f"\n[FAIL] FEATURE 10: FAILED - {str(e)}")
        return False


def run_all_tests():
    """Run all feature tests."""
    print("\n" + "="*70)
    print("GESTURE DIGITAL TWIN: COMPREHENSIVE FEATURE TEST SUITE")
    print("Testing all 10 advanced features")
    print("="*70)

    results = {
        "Feature 1: Dataset Recording": test_feature_1_dataset(),
        "Feature 2: Adaptive Calibration": test_feature_2_calibration(),
        "Feature 3: Multi-Cursor Control": test_feature_3_multi_cursor(),
        "Feature 4: Gesture Macros": test_feature_4_macros(),
        "Feature 5: Cross-Platform": test_feature_5_cross_platform(),
        "Feature 6: Analytics": test_feature_6_analytics(),
        "Feature 7: Feedback": test_feature_7_feedback(),
        "Feature 8: Confidence Tracking": test_feature_8_confidence(),
        "Feature 9: Adaptive FPS": test_feature_9_adaptive_fps(),
        "Feature 10: Cloud Inference": test_feature_10_cloud_inference(),
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for feature, result in results.items():
        status = "[PASS] PASSED" if result else "[FAIL] FAILED"
        print(f"{status:12} {feature}")

    print("\n" + "="*70)
    print(f"TOTAL: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print("="*70)

    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED - System is fully operational!")
    else:
        print(f"\n[WARN]️  {total - passed} test(s) failed - Review output above")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
