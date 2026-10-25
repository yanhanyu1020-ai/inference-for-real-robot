@echo off
setlocal
set "APP_DIR=%~dp0JointCapture"
set "PYTHON_FILE=%APP_DIR%\python_path.txt"
set "MANIFEST_FILE=%APP_DIR%\manifest_root.txt"
if not exist "%PYTHON_FILE%" (
  echo JointCapture is not installed: "%PYTHON_FILE%" 1>&2
  exit /b 2
)
if not exist "%MANIFEST_FILE%" (
  echo JointCapture manifest path is missing: "%MANIFEST_FILE%" 1>&2
  exit /b 2
)
set /p "PYTHON_EXE="<"%PYTHON_FILE%"
set /p "MANIFEST_ROOT="<"%MANIFEST_FILE%"
if not defined MANIFEST_ROOT exit /b 2
if not exist "%PYTHON_EXE%" (
  echo Configured Python was not found: "%PYTHON_EXE%" 1>&2
  exit /b 2
)
set "PYTHONPATH=%APP_DIR%\src;%PYTHONPATH%"
"%PYTHON_EXE%" -m jointctl --config "%APP_DIR%\config\default.json" --manifest-root "%MANIFEST_ROOT%" start
set "JOINTCTL_EXIT=%ERRORLEVEL%"
if not defined JOINTCTL_NO_PAUSE "%PYTHON_EXE%" -c "import sys; sys.stdin.isatty() and input('Press Enter to close...')"
exit /b %JOINTCTL_EXIT%
