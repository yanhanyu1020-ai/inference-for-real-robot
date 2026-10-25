@echo off
setlocal
set "APP_DIR=%~dp0JointCapture"
set /p "PYTHON_EXE="<"%APP_DIR%\python_path.txt"
if not exist "%APP_DIR%\manifest_root.txt" exit /b 2
set /p "MANIFEST_ROOT="<"%APP_DIR%\manifest_root.txt"
if not defined MANIFEST_ROOT exit /b 2
if not exist "%PYTHON_EXE%" exit /b 2
set "PYTHONPATH=%APP_DIR%\src;%PYTHONPATH%"
"%PYTHON_EXE%" -m jointctl.console --config "%APP_DIR%\config\default.json" --manifest-root "%MANIFEST_ROOT%"
set "JOINTCTL_EXIT=%ERRORLEVEL%"
if not "%JOINTCTL_EXIT%"=="0" if not defined JOINTCTL_NO_PAUSE pause
exit /b %JOINTCTL_EXIT%
