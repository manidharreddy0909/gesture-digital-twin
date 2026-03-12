"""
Unified production entrypoint for Gesture Digital Twin.

Single command to:
1) (Optional) run preflight checks
2) launch integrated 3D/AR/VR runtime

Usage examples:
  python main_unified.py
  python main_unified.py --skip-preflight
  python main_unified.py --mode virtual_execution --context robot
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from config import CursorAIConfig
from main_3d_vr import main_3d_vr


def run_preflight(quick: bool) -> bool:
    """Run preflight checks before launch."""
    cmd = [sys.executable, "preflight.py"]
    if quick:
        cmd.append("--quick")

    print("[Unified] Running preflight...")
    proc = subprocess.run(cmd, check=False)
    ok = proc.returncode == 0
    print(f"[Unified] Preflight status: {'PASS' if ok else 'FAIL'}")
    return ok


def build_runtime_config(args: argparse.Namespace) -> CursorAIConfig:
    """Build runtime config from CLI overrides."""
    cfg = CursorAIConfig()

    if args.mode is not None:
        cfg.control_mode = args.mode
    if args.context is not None:
        cfg.control_context = args.context

    if args.enable_robot:
        cfg.enable_robot_control = True
    if args.enable_unreal:
        cfg.enable_ue5_integration = True
    if args.enable_objects:
        cfg.enable_3d_world = True
    if args.profile:
        cfg.enable_3d_profiling = True

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified launcher for Gesture Digital Twin.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks before launch.")
    parser.add_argument("--quick-preflight", action="store_true", help="Run quick preflight mode.")
    parser.add_argument(
        "--mode",
        choices=["real_interface", "virtual_execution"],
        default=None,
        help="Runtime control mode override.",
    )
    parser.add_argument(
        "--context",
        choices=["default", "robot", "objects", "ui"],
        default=None,
        help="Runtime control context override.",
    )
    parser.add_argument("--enable-robot", action="store_true", help="Enable robot control.")
    parser.add_argument("--enable-unreal", action="store_true", help="Enable Unreal integration.")
    parser.add_argument("--enable-objects", action="store_true", help="Enable 3D object manipulation.")
    parser.add_argument("--profile", action="store_true", help="Enable 3D profiling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_preflight:
        if not run_preflight(quick=args.quick_preflight):
            print("[Unified] Launch blocked: preflight failed.")
            return 1

    cfg = build_runtime_config(args)
    print("[Unified] Starting main_3d_vr runtime...")
    main_3d_vr(cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
