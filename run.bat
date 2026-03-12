@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Virtual environment missing. Running setup first...
  call setup.bat
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Could not activate virtual environment.
  pause
  exit /b 1
)

if "%~1"=="" (
  echo [INFO] Starting default runtime profile...
  python main_unified.py --skip-preflight --mode virtual_execution --context objects --enable-unreal --enable-objects
) else (
  echo [INFO] Starting with custom args: %*
  python main_unified.py %*
)

set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [WARN] Runtime exited with code %EXIT_CODE%
  pause
)
exit /b %EXIT_CODE%