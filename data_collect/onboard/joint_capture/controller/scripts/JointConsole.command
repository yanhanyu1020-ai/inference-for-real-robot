#!/bin/bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"
APP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

finish() {
    status=$?
    trap - EXIT
    if [ "$status" -ne 0 ] && [ -t 0 ] && [ "${JOINTCTL_NO_PAUSE:-}" != "1" ]; then
        printf '\nPress Return to close this window... '
        read -r _ || true
    fi
    exit "$status"
}
trap finish EXIT

if [ ! -x "$APP_ROOT/.venv-macos/bin/python" ] || [ ! -f "$APP_ROOT/macos_paths.json" ]; then
    printf 'Run controller/scripts/Install-JointCapture.command first, or use JointConsole-macOS.command at the repository root.\n' >&2
    exit 2
fi

export PYTHONPATH="$APP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$APP_ROOT/.venv-macos/bin/python" -m jointctl.macos launch --app-root "$APP_ROOT" "$@"
