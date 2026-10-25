#!/bin/bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")" && pwd)"
APP_ROOT="$REPOSITORY_ROOT/controller"

# First run only creates local files; no packages are downloaded or robot
# commands issued by the installer.
if [ ! -x "$APP_ROOT/.venv-macos/bin/python" ] || [ ! -f "$APP_ROOT/macos_paths.json" ]; then
    JOINTCTL_NO_PAUSE=1 /bin/bash "$APP_ROOT/scripts/Install-JointCapture.command"
fi

exec /bin/bash "$APP_ROOT/scripts/JointConsole.command" "$@"
