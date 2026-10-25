#!/usr/bin/env bash
set -euo pipefail

# Resolve compatibility entry-point symlinks before finding the module root.
SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_PATH" ]; do
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
  SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
  [[ "$SCRIPT_PATH" = /* ]] || SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH"
done
ONBOARD_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
ROOT="$(cd "$ONBOARD_DIR/../../.." && pwd)"
CONFIG="${COLLECTION_CONFIG:-${ROOT}/config/collection.json}"
PYTHON="${COLLECTION_PYTHON:-python3}"

config_value() {
  "$PYTHON" - "$CONFIG" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for part in sys.argv[2].split('.'):
    value = value[part]
print(value)
PY
}

usage() {
  cat <<EOF
Usage:
  $0 TASK
  $0 TASK EPISODE_ID

Transfer stopped collection episodes from Unitree to A800.

Examples:
  $0 go2_remote_test
  $0 go2_remote_test episode_001

Each episode must already have manifest.json (run collection_ctl.sh stop
or validate_episode.py --write first).
EOF
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || { usage; exit 2; }
task="$1"
episode_filter="${2:-}"
[[ "$task" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "invalid task name: $task" >&2
  exit 2
}
if [ -n "$episode_filter" ]; then
  [[ "$episode_filter" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "invalid episode id: $episode_filter" >&2
    exit 2
  }
fi

data_root="$(config_value data_root)"
task_dir="$data_root/$task"
transfer_one="$ONBOARD_DIR/transfer_episode.sh"

if [ ! -d "$task_dir" ]; then
  echo "task directory not found: $task_dir" >&2
  exit 1
fi

episodes=()
if [ -f "$task_dir/manifest.json" ]; then
  episodes=("$(basename "$task_dir")")
elif [ -n "$episode_filter" ]; then
  [ -d "$task_dir/$episode_filter" ] || {
    echo "episode not found: $task_dir/$episode_filter" >&2
    exit 1
  }
  episodes=("$episode_filter")
else
  shopt -s nullglob
  for candidate in "$task_dir"/*/; do
    [ -d "$candidate" ] || continue
    episodes+=("$(basename "$candidate")")
  done
  shopt -u nullglob
fi

if [ "${#episodes[@]}" -eq 0 ]; then
  echo "no episodes found under $task_dir" >&2
  exit 1
fi

failed=0
for episode_id in "${episodes[@]}"; do
  echo "transferring $task/$episode_id"
  if [ -f "$task_dir/manifest.json" ]; then
    if ! "$transfer_one" "$episode_id" "$task" "$task_dir"; then
      failed=1
    fi
  elif ! "$transfer_one" "$episode_id" "$task"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "one or more episodes failed to transfer" >&2
  exit 1
fi

echo "task transfer complete: $task (${#episodes[@]} episode(s))"
