#!/bin/bash
set -euo pipefail

# Finder does not inherit an interactive shell's Homebrew PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"
APP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

finish() {
    status=$?
    trap - EXIT
    if [ -t 0 ] && [ "${JOINTCTL_NO_PAUSE:-}" != "1" ]; then
        printf '\nPress Return to close this window... '
        read -r _ || true
    fi
    exit "$status"
}
trap finish EXIT

if ! command -v ssh >/dev/null 2>&1; then
    printf 'OpenSSH was not found. Enable or install the macOS OpenSSH client, then run this installer again.\n' >&2
    exit 2
fi

PYTHON_EXE="${JOINTCAPTURE_PYTHON:-}"
if [ -n "$PYTHON_EXE" ]; then
    if ! "$PYTHON_EXE" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        printf 'JOINTCAPTURE_PYTHON must point to a working Python 3.10 or newer executable: %s\n' "$PYTHON_EXE" >&2
        exit 2
    fi
else
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        candidate_path="$(command -v "$candidate" 2>/dev/null || true)"
        if [ -n "$candidate_path" ] && "$candidate_path" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            PYTHON_EXE="$candidate_path"
            break
        fi
    done
    if [ -z "$PYTHON_EXE" ]; then
        printf 'Python 3.10 or newer was not found. Install Python for macOS from https://www.python.org/downloads/macos/, then run this installer again.\n' >&2
        exit 2
    fi
fi

printf 'Preparing the Mac desktop console with %s\n' "$PYTHON_EXE"
"$PYTHON_EXE" -m venv --without-pip "$APP_ROOT/.venv-macos"
export PYTHONPATH="$APP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$APP_ROOT/.venv-macos/bin/python" -m jointctl.macos setup --app-root "$APP_ROOT" "$@"
printf '\nReady. Double-click JointConsole-macOS.command in the repository to open the console.\n'
