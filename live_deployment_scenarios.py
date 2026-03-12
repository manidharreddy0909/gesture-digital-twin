"""
Live deployment scenarios for production readiness checks.

Scenarios:
1) Robot safety envelope and emergency-stop behavior
2) Bidirectional Unreal feedback loop transport smoke test
3) Task-based control flow (intent -> confirmation gesture -> action)

Usage:
  python live_deployment_scenarios.py --scenario all
  python live_deployment_scenarios.py --scenario robot_safety
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from control_intelligence import ControlMode, IntelligentControlLayer
from robot_controller import GripperCommand, RobotArmUR5, RobotSafetyLimits, RobotTarget
from unreal_bridge import UnrealFeedbackEvent, UnrealWebSocketBridge


def scenario_robot_safety() -> bool:
    print("\n[Scenario] robot_safety")
    robot = RobotArmUR5()
    robot.connect()
    robot.set_safety_limits(
        RobotSafetyLimits(
            workspace_min=np.array([0.20, -0.40, 0.20]),
            workspace_max=np.array([0.70, 0.40, 0.80]),
            max_step_m=0.02,
            max_velocity_mps=0.20,
            stop_on_out_of_workspace=True,
        )
    )

    safe_target = RobotTarget(position=np.array([0.40, 0.0, 0.50]), gripper_command=GripperCommand.OPEN)
    ok_safe = robot.move_to_position(safe_target)
    if not ok_safe:
        print("  FAIL: safe target should be accepted")
        return False

    unsafe_target = RobotTarget(position=np.array([1.20, 0.0, 1.50]), gripper_command=GripperCommand.CLOSE)
    ok_unsafe = robot.move_to_position(unsafe_target)
    if ok_unsafe:
        print("  FAIL: out-of-workspace target should be rejected")
        return False

    robot.stop()
    blocked = robot.move_to_position(safe_target)
    if blocked:
        print("  FAIL: commands must be blocked during emergency stop")
        return False
    robot.clear_emergency_stop()
    unblocked = robot.move_to_position(safe_target)
    if not unblocked:
        print("  FAIL: command should resume after e-stop clear")
        return False

    print("  PASS")
    return True


def scenario_unreal_feedback_loop() -> bool:
    print("\n[Scenario] unreal_feedback")
    bridge = UnrealWebSocketBridge("ws://localhost:8765")
    # Offline transport simulation for smoke test.
    # Force send path to succeed without requiring websocket package/network.
    bridge.connected = True
    bridge._send_json_sync = lambda _payload: True  # type: ignore[attr-defined]

    event = UnrealFeedbackEvent(
        event_type="joint_limit_near",
        severity="warn",
        message="Joint 3 near limit",
        data={"joint_index": 3, "ratio": 0.96},
        timestamp=time.time(),
    )
    sent = bridge.send_feedback_event(event)
    if not sent:
        print("  FAIL: feedback event was not sent")
        return False

    events = bridge.poll_feedback_events()
    if len(events) != 1:
        print("  FAIL: expected exactly one feedback event")
        return False
    if events[0].event_type != "joint_limit_near":
        print("  FAIL: incorrect feedback event type")
        return False

    print("  PASS")
    return True


def scenario_task_based_control() -> bool:
    print("\n[Scenario] task_based_control")
    layer = IntelligentControlLayer()
    mode = ControlMode.VIRTUAL_EXECUTION

    # Task decomposition: "Pick cube and place on table"
    # Step1 reach/select, Step2 close gripper, Step3 move, Step4 release.
    steps = [
        {"gesture": "swipe_right", "context": "robot"},
        {"gesture": "pinch", "context": "objects"},
        {"gesture": "push", "context": "robot"},
        {"gesture": "open", "context": "objects"},
    ]

    actions = []
    for s in steps:
        action = layer.resolve_action(
            single_gesture=s["gesture"],
            two_hand_gesture=None,
            mode=mode,
            context=s["context"],
            timestamp=time.time(),
        )
        actions.append(action)
        time.sleep(0.06)  # clear command-rate gate

    expected_non_empty = all(a is not None for a in actions)
    if not expected_non_empty:
        print(f"  FAIL: unresolved action in sequence: {actions}")
        return False

    print(f"  Actions: {actions}")
    print("  PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live deployment scenarios.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", "robot_safety", "unreal_feedback", "task_based_control"],
        help="Scenario to run",
    )
    args = parser.parse_args()

    scenario_map = {
        "robot_safety": scenario_robot_safety,
        "unreal_feedback": scenario_unreal_feedback_loop,
        "task_based_control": scenario_task_based_control,
    }

    if args.scenario == "all":
        results = [fn() for fn in scenario_map.values()]
        ok = all(results)
    else:
        ok = scenario_map[args.scenario]()

    print("\n[Result] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
