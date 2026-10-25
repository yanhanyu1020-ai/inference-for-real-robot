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
RUN_DIR="${ROOT}/run"
BUILD_DIR="${ROOT}/build"
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

data_root="$(config_value data_root)"
network_interface="$(config_value network_interface)"
can_interface="$(config_value can_interface)"
gamepad_python="$(config_value piper_gamepad.python)"
gamepad_runtime="$(config_value piper_gamepad.runtime)"

usage() {
  cat <<EOF
Usage:
  $0 start EPISODE_ID INSTRUCTION [TASK]
  $0 status
  $0 stop
EOF
}

active_episode() {
  [ -f "${RUN_DIR}/active_episode" ] && perl -0777 -pe 's/\s+$//' "${RUN_DIR}/active_episode"
}

pid_running() {
  local pid_file="$1"
  [ -s "$pid_file" ] || return 1
  local pid
  pid="$(perl -0777 -pe 's/\s+$//' "$pid_file")"
  kill -0 "$pid" 2>/dev/null
}

wait_stopped() {
  local pid="$1"
  local timeout="$2"
  local deadline=$((SECONDS + timeout))
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      return 1
    fi
    sleep 0.2
  done
}

start_collection() {
  [ "$#" -ge 2 ] && [ "$#" -le 3 ] || { usage; exit 2; }
  local episode_id="$1"
  local instruction="$2"
  local task="${3:-default}"
  [[ "$episode_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "invalid episode id: $episode_id" >&2
    exit 2
  }
  [[ "$task" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "invalid task name: $task" >&2
    exit 2
  }
  mkdir -p "$RUN_DIR" "$data_root/$task"
  if [ -n "$(active_episode || true)" ]; then
    echo "a collection is already active: $(active_episode)" >&2
    exit 1
  fi
  local episode_dir="$data_root/$task/$episode_id"
  if [ -e "$episode_dir" ]; then
    echo "refusing to overwrite existing episode: $episode_dir" >&2
    exit 1
  fi
  if pgrep -af 'piper_(chunk|stream)_loop|hetero_(teleop|teach)_loop|piper_gamepad_teleop.py|Gamepad_PiPER.*main' >/dev/null; then
    echo "conflicting Piper writer is running; stop it explicitly before collection" >&2
    pgrep -af 'piper_(chunk|stream)_loop|hetero_(teleop|teach)_loop|piper_gamepad_teleop.py|Gamepad_PiPER.*main' >&2 || true
    exit 1
  fi
  "$PYTHON" "${ONBOARD_DIR}/hetero_pkl_recorder.py" \
    --config "$CONFIG" --check-cameras
  SDL_VIDEODRIVER=dummy PYTHONPATH="$gamepad_runtime" \
    "$gamepad_python" "${ONBOARD_DIR}/piper_gamepad_teleop.py" \
    --config "$CONFIG" --check
  if ! ip -brief link show "$can_interface" | perl -ne '$ok = 1 if /\bUP\b/; END { exit($ok ? 0 : 1) }'; then
    echo "bringing up Piper CAN ${can_interface} at 1 Mbps"
    sudo ip link set "$can_interface" up type can bitrate 1000000
  fi
  [ -x "${BUILD_DIR}/go2_state_bridge" ] || {
    echo "missing ${BUILD_DIR}/go2_state_bridge; deploy/build the collection package first" >&2
    exit 1
  }
  mkdir -p "$episode_dir/frames" "$episode_dir/raw" "$episode_dir/logs"
  "$PYTHON" - "$episode_dir/meta.json" "$episode_id" "$task" "$instruction" "$CONFIG" <<'PY'
import json, pathlib, platform, sys, time
path, episode, task, instruction, config = sys.argv[1:]
payload = {
    "episode_id": episode,
    "task": task,
    "instruction": instruction,
    "start_wall_time_ns": time.time_ns(),
    "host": platform.node(),
    "collection_config": json.load(open(config)),
}
pathlib.Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
  printf '%s\n' "$episode_dir" > "${RUN_DIR}/active_episode"
  nohup env -i \
    HOME="$HOME" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "${BUILD_DIR}/go2_state_bridge" "$network_interface" "$episode_dir/raw" \
    > "$episode_dir/logs/go2_state_bridge.log" 2>&1 &
  printf '%s\n' "$!" > "${RUN_DIR}/go2.pid"
  sleep 1
  if ! pid_running "${RUN_DIR}/go2.pid"; then
    echo "Go2 bridge failed to start; see $episode_dir/logs/go2_state_bridge.log" >&2
    rm -f "${RUN_DIR}/active_episode" "${RUN_DIR}/go2.pid"
    exit 1
  fi
  nohup env SDL_VIDEODRIVER=dummy PYTHONPATH="$gamepad_runtime" \
    "$gamepad_python" "${ONBOARD_DIR}/piper_gamepad_teleop.py" \
    --config "$CONFIG" --episode-dir "$episode_dir" \
    > "$episode_dir/logs/piper_gamepad.log" 2>&1 &
  printf '%s\n' "$!" > "${RUN_DIR}/gamepad.pid"
  local gamepad_deadline=$((SECONDS + 60))
  while [ ! -f "$episode_dir/raw/piper_gamepad_snapshot.json" ]; do
    if ! pid_running "${RUN_DIR}/gamepad.pid"; then
      echo "Piper gamepad failed to start; see $episode_dir/logs/piper_gamepad.log" >&2
      kill -TERM "$(perl -0777 -pe 's/\s+$//' "${RUN_DIR}/go2.pid")" 2>/dev/null || true
      rm -f "${RUN_DIR}/active_episode" "${RUN_DIR}/go2.pid" "${RUN_DIR}/gamepad.pid"
      exit 1
    fi
    if [ "$SECONDS" -ge "$gamepad_deadline" ]; then
      echo "Piper gamepad did not become ready within 60 seconds" >&2
      kill -TERM "$(perl -0777 -pe 's/\s+$//' "${RUN_DIR}/gamepad.pid")" 2>/dev/null || true
      kill -TERM "$(perl -0777 -pe 's/\s+$//' "${RUN_DIR}/go2.pid")" 2>/dev/null || true
      rm -f "${RUN_DIR}/active_episode" "${RUN_DIR}/go2.pid" "${RUN_DIR}/gamepad.pid"
      exit 1
    fi
    sleep 0.2
  done
  nohup "$PYTHON" "${ONBOARD_DIR}/hetero_pkl_recorder.py" \
    --config "$CONFIG" --episode-dir "$episode_dir" \
    > "$episode_dir/logs/recorder.log" 2>&1 &
  printf '%s\n' "$!" > "${RUN_DIR}/recorder.pid"
  sleep 2
  if ! pid_running "${RUN_DIR}/recorder.pid"; then
    echo "PKL recorder failed to start; see $episode_dir/logs/recorder.log" >&2
    kill -TERM "$(perl -0777 -pe 's/\s+$//' "${RUN_DIR}/go2.pid")" 2>/dev/null || true
    kill -TERM "$(perl -0777 -pe 's/\s+$//' "${RUN_DIR}/gamepad.pid")" 2>/dev/null || true
    rm -f "${RUN_DIR}/active_episode" "${RUN_DIR}/go2.pid" "${RUN_DIR}/gamepad.pid" "${RUN_DIR}/recorder.pid"
    exit 1
  fi
  echo "collection started: $episode_dir"
  echo "Go2 remains operator-controlled by its native remote"
  echo "Piper gamepad is ready; press HOME to enable and move to the zero-joint pose"
}

status_collection() {
  local episode_dir
  episode_dir="$(active_episode || true)"
  if [ -z "$episode_dir" ]; then
    echo "stopped"
    return
  fi
  echo "episode=$episode_dir"
  echo "go2_pid=$([ -s "${RUN_DIR}/go2.pid" ] && perl -0777 -pe 's/\s+$//' "${RUN_DIR}/go2.pid" || echo none)"
  echo "recorder_pid=$([ -s "${RUN_DIR}/recorder.pid" ] && perl -0777 -pe 's/\s+$//' "${RUN_DIR}/recorder.pid" || echo none)"
  echo "gamepad_pid=$([ -s "${RUN_DIR}/gamepad.pid" ] && perl -0777 -pe 's/\s+$//' "${RUN_DIR}/gamepad.pid" || echo none)"
  if [ -f "$episode_dir/raw/piper_gamepad_snapshot.json" ]; then
    "$PYTHON" -m json.tool "$episode_dir/raw/piper_gamepad_snapshot.json"
  fi
  if [ -f "$episode_dir/status.json" ]; then
    "$PYTHON" -m json.tool "$episode_dir/status.json"
  else
    echo "status.json not available yet"
  fi
  df -h "$data_root"
}

stop_collection() {
  local episode_dir
  episode_dir="$(active_episode || true)"
  if [ -z "$episode_dir" ]; then
    echo "no collection is active"
    return
  fi
  local recorder_pid=""
  local gamepad_pid=""
  if pid_running "${RUN_DIR}/recorder.pid"; then
    recorder_pid="$(perl -0777 -pe 's/\s+$//' "${RUN_DIR}/recorder.pid")"
    kill -TERM "$recorder_pid"
  fi
  if pid_running "${RUN_DIR}/gamepad.pid"; then
    gamepad_pid="$(perl -0777 -pe 's/\s+$//' "${RUN_DIR}/gamepad.pid")"
    kill -TERM "$gamepad_pid"
  fi
  if [ -n "$gamepad_pid" ]; then
    wait_stopped "$gamepad_pid" 10 || {
      echo "Piper gamepad did not stop within 10 seconds; data remains local" >&2
      exit 1
    }
  fi
  if [ -n "$recorder_pid" ]; then
    wait_stopped "$recorder_pid" 60 || {
      echo "recorder did not stop within 60 seconds; data remains local" >&2
      exit 1
    }
  fi
  if pid_running "${RUN_DIR}/go2.pid"; then
    local go2_pid
    go2_pid="$(perl -0777 -pe 's/\s+$//' "${RUN_DIR}/go2.pid")"
    kill -TERM "$go2_pid"
    wait_stopped "$go2_pid" 10 || {
      echo "Go2 bridge did not stop within 10 seconds" >&2
      exit 1
    }
  fi
  "$PYTHON" - "$episode_dir/meta.json" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
value["stop_wall_time_ns"] = time.time_ns()
path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
PY
  local validation=0
  "$PYTHON" "${ONBOARD_DIR}/validate_episode.py" \
    --episode "$episode_dir" --config "$CONFIG" --write || validation=$?
  rm -f "${RUN_DIR}/active_episode" "${RUN_DIR}/go2.pid" "${RUN_DIR}/gamepad.pid" "${RUN_DIR}/recorder.pid"
  if [ "$validation" -ne 0 ]; then
    echo "collection stopped but validation failed: $episode_dir" >&2
    exit "$validation"
  fi
  echo "collection stopped and validated locally: $episode_dir"
  echo "no data was transferred"
}

command="${1:-}"
case "$command" in
  start) shift; start_collection "$@" ;;
  status) status_collection ;;
  stop) stop_collection ;;
  *) usage; exit 2 ;;
esac
