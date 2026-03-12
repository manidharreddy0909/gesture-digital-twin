@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set QUIET=0
if /I "%~1"=="--quiet" set QUIET=1

set UE_PY=C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\ThirdParty\Python3\Win64\python.exe

if not exist "%UE_PY%" (
  if %QUIET%==0 echo [INFO] UE Python not found at: %UE_PY%
  if %QUIET%==0 echo [INFO] Skipping UE Python dependency install.
  exit /b 0
)

if %QUIET%==0 echo [INFO] Found UE Python: %UE_PY%
if %QUIET%==0 echo [INFO] Installing websockets into UE Python...

"%UE_PY%" -m pip install --upgrade pip >nul 2>&1
"%UE_PY%" -m pip install websockets
if errorlevel 1 (
  if %QUIET%==0 echo [WARN] UE Python pip install failed.
  if %QUIET%==0 echo [WARN] Try running this script as Administrator.
  exit /b 1
)

if %QUIET%==0 echo [INFO] UE Python dependency install complete.
exit /b 0