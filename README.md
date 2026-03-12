# Gesture Digital Twin

[![CI](https://github.com/manidharreddy0909/gesture-digital-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/manidharreddy0909/gesture-digital-twin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MediaPipe-based hand tracking and gesture control system for cursor interaction, 3D manipulation, robot control, and Unreal Engine integration.

This repository combines real-time hand landmark tracking, gesture recognition, motion smoothing, and downstream control pipelines for cursor control, robot targeting, and Unreal Engine skeletal interaction.

## Highlights

- Dual-hand tracking with 21 landmarks per hand
- Motion-driven cursor control with smoothing and Kalman filtering
- Static, dynamic, and two-hand gesture recognition
- 3D transformation pipeline for AR/VR and robotics workflows
- Modular integrations for Unreal Engine, sockets, HTTP APIs, and robot targets
- Optional PyQt6 dashboard and profiling utilities

## Status

- Local integration validation: `10/10` tests passed via `python test_3d_pipeline.py`
- CI workflow: runs on pushes and pull requests through GitHub Actions
- Repository license: MIT

## Repository Structure

```text
.
|-- main.py
|-- main_enhanced.py
|-- main_3d_vr.py
|-- hand_tracker.py
|-- gesture_detector.py
|-- cursor_controller.py
|-- cursor_controller_3d.py
|-- robot_controller.py
|-- unreal_bridge.py
|-- gui_pyqt6.py
|-- models/
|   `-- hand_landmarker.task
`-- test_3d_pipeline.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional GUI dependencies are already listed in `requirements.txt` and include `PyQt6` and `PyQt6-Charts`.

## Run

Basic pipeline:

```bash
python main.py
```

Enhanced pipeline:

```bash
python main_enhanced.py
```

3D / VR pipeline:

```bash
python main_3d_vr.py
```

## Validation

Run the integration test suite:

```bash
python test_3d_pipeline.py
```

Current local result: `10/10` tests passed.

## Notes

- The MediaPipe model is stored at `models/hand_landmarker.task`.
- Local virtual environments, caches, and scratch files are excluded from Git.
- Extended project notes remain available in `README_ENHANCEMENTS.md`.
- The test and runtime paths now consistently reference `models/hand_landmarker.task`.
