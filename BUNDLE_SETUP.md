# One-Click Bundle Guide

## What you get

- `setup.bat` : Creates `.venv`, installs Python dependencies, runs quick preflight.
- `run.bat` : Runs the app (default profile or custom args).
- `install_ue_python_deps.bat` : Installs `websockets` into UE5 bundled Python.
- `requirements.txt` : Full dependency list for this project.

## New laptop flow

1. Copy this folder as-is to new laptop:
   - `C:\Gesture_Digital_Twin`
2. Double-click `setup.bat`.
3. Open Unreal project and run `Content/python/init_unreal_ws.py` once.
4. Double-click `run.bat`.

## Default run profile

If you run `run.bat` with no arguments, it launches:

```bat
python main_unified.py --skip-preflight --mode virtual_execution --context objects --enable-unreal --enable-objects
```

## Custom run examples

```bat
run.bat --skip-preflight --mode real_interface --context robot --enable-unreal --enable-objects
run.bat --skip-preflight --mode virtual_execution --context ui
```

## Notes

- If UE websocket script errors with missing `websockets`, run `install_ue_python_deps.bat` as Administrator.
- Ensure actor label in Unreal is `ManipulatedObject` (or change `unreal_object_actor_name` in `config.py`).