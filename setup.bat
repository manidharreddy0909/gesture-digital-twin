@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo =====================================================
echo Gesture Digital Twin - One-Click Setup
echo =====================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  echo Install Python 3.11+ and check "Add Python to PATH" during install.
  pause
  exit /b 1q
)

for /f "tokens=2 delims= " %%V in ('python -V 2^>^&1') do set PYVER=%%V
echo [INFO] Detected Python %PYVER%

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  pause
  exit /b 1
)

echo [INFO] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] pip bootstrap failed.
  pause
  exit /b 1
)

if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found in project root.
  pause
  exit /b 1
)

echo [INFO] Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Dependency install failed.
  echo Try rerunning this file as Administrator if needed.
  pause
  exit /b 1
)

if exist "install_ue_python_deps.bat" (
  echo [INFO] Installing UE Python dependency (websockets) if UE Python is found...
  call install_ue_python_deps.bat --quiet
)

echo [INFO] Running quick preflight...
python preflight.py --quick
if errorlevel 1 (
  echo [WARN] Quick preflight reported failures. Review output above.
) else (
  echo [INFO] Quick preflight passed.
)

echo.
echo [DONE] Setup finished.
echo Next: double-click run.bat
pause
exit /b 0