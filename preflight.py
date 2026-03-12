"""
One-command preflight verification for Gesture Digital Twin.

Runs:
1) Config sanity checks
2) Core feature tests
3) 3D pipeline tests
4) Live deployment scenarios

Usage:
  python preflight.py
  python preflight.py --quick
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import CursorAIConfig


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _run_command(name: str, cmd: List[str]) -> CheckResult:
    print(f"\n[Run] {name}")
    try:
        proc = subprocess.run(cmd, check=False)
        ok = proc.returncode == 0
        return CheckResult(name=name, ok=ok, detail=f"exit_code={proc.returncode}")
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=str(exc))


def _validate_config(cfg: CursorAIConfig) -> CheckResult:
    errors: List[str] = []

    if cfg.control_mode not in {"real_interface", "virtual_execution"}:
        errors.append(f"control_mode invalid: {cfg.control_mode}")

    if cfg.control_context not in {"default", "robot", "objects", "ui"}:
        errors.append(f"control_context invalid: {cfg.control_context}")

    if not (0.0 < cfg.stability_smoothing_alpha <= 1.0):
        errors.append("stability_smoothing_alpha must be in (0, 1]")

    if cfg.robot_max_velocity <= 0.0:
        errors.append("robot_max_velocity must be > 0")

    if cfg.robot_max_step_m <= 0.0:
        errors.append("robot_max_step_m must be > 0")

    wmin = cfg.robot_workspace_min_xyz
    wmax = cfg.robot_workspace_max_xyz
    if len(wmin) != 3 or len(wmax) != 3:
        errors.append("robot workspace bounds must be xyz tuples")
    else:
        if any(wmin[i] >= wmax[i] for i in range(3)):
            errors.append(f"robot_workspace_min_xyz must be < robot_workspace_max_xyz, got {wmin} / {wmax}")

    if cfg.base_pipeline_latency_ms < 0.0:
        errors.append("base_pipeline_latency_ms must be >= 0")

    if cfg.max_command_rate_hz <= 0.0:
        errors.append("max_command_rate_hz must be > 0")

    if errors:
        return CheckResult(name="config_sanity", ok=False, detail="; ".join(errors))
    return CheckResult(name="config_sanity", ok=True, detail="ok")


def _validate_required_files() -> CheckResult:
    required = [
        "main_3d_vr.py",
        "control_intelligence.py",
        "live_deployment_scenarios.py",
        "test_all_features.py",
        "test_3d_pipeline.py",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        return CheckResult(name="required_files", ok=False, detail=f"missing: {missing}")
    return CheckResult(name="required_files", ok=True, detail="ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run preflight checks.")
    parser.add_argument("--quick", action="store_true", help="Skip long test suites and run only sanity + scenarios.")
    args = parser.parse_args()

    results: List[CheckResult] = []

    print("\n=== PRE-FLIGHT CHECK ===")
    results.append(_validate_required_files())
    results.append(_validate_config(CursorAIConfig()))

    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")

    if not all(r.ok for r in results):
        print("\n[Result] FAIL (sanity checks)")
        return 1

    if not args.quick:
        results.append(_run_command("test_all_features", [sys.executable, "test_all_features.py"]))
        results.append(_run_command("test_3d_pipeline", [sys.executable, "test_3d_pipeline.py"]))
    else:
        print("\n[Info] Quick mode enabled: skipping long tests")

    results.append(
        _run_command("live_deployment_scenarios", [sys.executable, "live_deployment_scenarios.py", "--scenario", "all"])
    )

    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed

    print("\n=== PRE-FLIGHT SUMMARY ===")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name} ({r.detail})")

    print(f"\nTotals: {passed} passed, {failed} failed")
    print("[Result] " + ("PASS" if failed == 0 else "FAIL"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
