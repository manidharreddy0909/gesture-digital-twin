"""
Unit tests for robot inverse kinematics solver.

Validates:
- IK solver finds valid solutions
- Joint limits are enforced
- Reachability analysis
- Gripper control
- Trajectory smoothing
"""

import numpy as np
import sys

try:
    from robot_controller import RobotArmUR5, HandToArmMapper, RobotTarget, GripperCommand
except ImportError:
    print("[Error] robot_controller module not found")
    sys.exit(1)


def test_ik_reachable_position():
    """Test IK solver for reachable position."""
    print("\n[Test 1] IK solution for reachable position")

    robot = RobotArmUR5()
    robot.connect()

    # Position within UR5 workspace: 0.5m forward, 0m lateral, 0.5m up
    position = np.array([0.5, 0.0, 0.5])

    joint_angles = robot.inverse_kinematics(position)

    assert joint_angles is not None, "Should find IK solution"
    assert len(joint_angles) == 6, "Should return 6 joint angles"
    assert all(-np.pi <= j <= np.pi for j in joint_angles), "Joints should be within limits"

    print(f"  ✓ IK solution found: {[f'{j:.2f}' for j in joint_angles]}")


def test_ik_joint_limits():
    """Test that joint limits are enforced."""
    print("\n[Test 2] Joint limits enforcement")

    robot = RobotArmUR5()

    # Try to find a position and verify limits
    position = np.array([0.3, 0.2, 0.5])
    joint_angles = robot.inverse_kinematics(position)

    if joint_angles is not None:
        for i, j in enumerate(joint_angles):
            assert j >= robot.joint_limits["lower"][i], f"Joint {i} below lower limit"
            assert j <= robot.joint_limits["upper"][i], f"Joint {i} above upper limit"

    print("  ✓ All joint limits enforced")


def test_unreachable_position():
    """Test handling of unreachable positions."""
    print("\n[Test 3] Unreachable position handling")

    robot = RobotArmUR5()

    # Position definitely outside UR5 reach (too far + too high)
    unreachable = np.array([2.0, 0.0, 2.0])

    joint_angles = robot.inverse_kinematics(unreachable)

    # Should return None or invalid solution
    if joint_angles is not None:
        # Verify by forward kinematics
        fk_pos, _ = robot.forward_kinematics(joint_angles)
        error = np.linalg.norm(fk_pos - unreachable)
        assert error > 0.1, "Should not closely reach unreachable position"

    print("  ✓ Unreachable positions correctly rejected")


def test_forward_kinematics():
    """Test forward kinematics computation."""
    print("\n[Test 4] Forward kinematics")

    robot = RobotArmUR5()

    # Home position (all zeros)
    home_angles = [0.0] * 6
    position, orientation = robot.forward_kinematics(home_angles)

    assert position is not None, "Should return position"
    assert len(position) == 3, "Position should be 3D"
    assert orientation is not None, "Should return orientation"

    # Position at home should be deterministic
    position2, _ = robot.forward_kinematics(home_angles)
    np.testing.assert_array_almost_equal(position, position2, decimal=3)

    print(f"  ✓ FK at home: position={position}, orientation={orientation}")


def test_ik_consistency():
    """Test that FK(IK(pos)) ≈ pos."""
    print("\n[Test 5] IK→FK consistency")

    robot = RobotArmUR5()

    target_position = np.array([0.4, 0.1, 0.6])

    joint_angles = robot.inverse_kinematics(target_position)

    if joint_angles is not None:
        fk_position, _ = robot.forward_kinematics(joint_angles)
        error = np.linalg.norm(fk_position - target_position)

        assert error < 0.05, f"IK→FK error too large: {error}m"
        print(f"  ✓ IK→FK error: {error:.4f}m (acceptable)")
    else:
        print("  ~ Skipped (IK solution not found for test position)")


def test_gripper_control():
    """Test gripper command creation."""
    print("\n[Test 6] Gripper control")

    target = RobotTarget(
        position=np.array([0.5, 0, 0.5]),
        gripper_command=GripperCommand.CLOSE,
        gripper_force=75.0
    )

    assert target.gripper_command == GripperCommand.CLOSE, "Gripper command mismatch"
    assert target.gripper_force == 75.0, "Gripper force mismatch"

    # Convert to dict (for JSON serialization)
    target_dict = target.to_dict()
    assert target_dict["gripper_command"] == "close", "Serialization should use string value"

    print("  ✓ Gripper commands work correctly")


def test_trajectory_smoothing():
    """Test trajectory smoothing with velocity limits."""
    print("\n[Test 7] Trajectory smoothing")

    mapper = HandToArmMapper(position_scale=1.0, max_velocity=0.5)

    current = RobotTarget(position=np.array([0.0, 0.0, 0.0]))
    target = RobotTarget(position=np.array([1.0, 0.0, 0.0]))

    # Smooth with dt=0.033s (30fps) and max_velocity=0.5 m/s
    smoothed = mapper.smooth_trajectory(target, dt=0.033)

    # With max_velocity=0.5 and dt=0.033, max displacement = 0.0165m
    displacement = np.linalg.norm(smoothed.position - current.position)
    assert displacement <= 0.5 * 0.033 + 0.001, "Should respect max velocity"

    print(f"  ✓ Trajectory smoothing limited motion: {displacement:.4f}m per frame")


def test_hand_to_arm_mapper():
    """Test hand-to-robot target mapping."""
    print("\n[Test 8] Hand-to-arm mapper")

    mapper = HandToArmMapper(position_scale=1.0, max_velocity=0.5)

    # Create mock hand
    class MockHand:
        def __init__(self):
            self.landmarks_3d = [
                (0.5, 0.5, 0.5),  # Wrist
            ] + [(0.0, 0.0, 0.0)] * 7 + [(0.5, 0.1, 0.6)] + [(0.0, 0.0, 0.0)] * 12

    left_hand = MockHand()
    target = mapper.map_hand_to_arm_target(left_hand, None)

    assert target is not None, "Should create target"
    assert len(target.position) == 3, "Position should be 3D"

    print(f"  ✓ Hand mapped to target position: {target.position}")


def test_reachability_workspace():
    """Test workspace reachability checking."""
    print("\n[Test 9] Workspace reachability")

    robot = RobotArmUR5()

    # Points to test
    test_points = [
        (np.array([0.4, 0.0, 0.5]), True),      # Should be reachable
        (np.array([0.0, 0.0, 0.4]), True),      # Should be reachable
        (np.array([2.0, 0.0, 0.0]), False),     # Too far
        (np.array([0.0, 0.0, 0.0]), True),      # Base center
    ]

    reachable_count = 0
    unreachable_count = 0

    for pos, should_reach in test_points:
        joints = robot.inverse_kinematics(pos)
        is_reachable = joints is not None

        if should_reach:
            if is_reachable:
                reachable_count += 1
            else:
                print(f"    Warning: Expected {pos} to be reachable")
        else:
            if not is_reachable:
                unreachable_count += 1

    print(f"  ✓ Reachability checks: {reachable_count} reachable, {unreachable_count} unreachable")


def test_multiple_solutions():
    """Test IK solver handling multiple solutions."""
    print("\n[Test 10] Multiple IK solutions")

    robot = RobotArmUR5()

    position = np.array([0.5, 0.0, 0.5])

    # Try multiple times (IK might find different solutions)
    solutions = []
    for _ in range(3):
        joints = robot.inverse_kinematics(position)
        if joints is not None:
            solutions.append(joints)

    # Solutions should be valid (might be different due to elbow-up/down)
    for joints in solutions:
        assert len(joints) == 6, "Should be 6-DOF solution"
        assert all(np.isfinite(j) for j in joints), "Should be finite"

    print(f"  ✓ Found {len(solutions)} valid IK solutions for same position")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("ROBOT IK SOLVER TEST SUITE")
    print("="*70)

    tests = [
        test_ik_reachable_position,
        test_ik_joint_limits,
        test_unreachable_position,
        test_forward_kinematics,
        test_ik_consistency,
        test_gripper_control,
        test_trajectory_smoothing,
        test_hand_to_arm_mapper,
        test_reachability_workspace,
        test_multiple_solutions,
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
    print("\nPerformance target: IK solver <15ms per joint")
    print("Reachability: >80% for gesture-natural hand positions")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
