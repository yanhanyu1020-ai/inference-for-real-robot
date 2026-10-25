#!/usr/bin/env bash
set -euo pipefail

install_dir="${GO2_RECORDER_HOME:-/home/unitree/heterovla-recorder}"
data_root="${GO2_DATA_ROOT:-/home/unitree/heterovla-data/raw}"
pid_file="${install_dir}/recorder.pid"
active_file="${install_dir}/active_episode"
binary="${install_dir}/bin/go2_sdk_recorder"

usage() {
  echo "Usage:"
  echo "  $0 start EPISODE_ID [INSTRUCTION]"
  echo "  $0 stop"
  echo "  $0 status"
}

is_running() {
  [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null
}

case "${1:-}" in
  start)
    episode_id="${2:-}"
    instruction="${3:-}"
    if [[ -z "${episode_id}" ]]; then
      usage
      exit 2
    fi
    if is_running; then
      echo "Recorder already running: PID $(cat "${pid_file}")"
      exit 1
    fi
    episode_dir="${data_root}/${episode_id}"
    if [[ -e "${episode_dir}" ]]; then
      echo "Refusing to overwrite existing episode: ${episode_dir}"
      exit 1
    fi
    mkdir -p "${episode_dir}"
    printf '%s\n' "${instruction}" > "${episode_dir}/instruction.txt"
    date -Ins > "${episode_dir}/start_time.txt"
    if [[ -f "${install_dir}/source_git_commit.txt" ]]; then
      cp "${install_dir}/source_git_commit.txt" "${episode_dir}/source_git_commit.txt"
    fi
    printf '%s\n' "${episode_dir}" > "${active_file}"
    nohup "${binary}" eth0 "${episode_dir}" \
      > "${episode_dir}/recorder.log" 2>&1 &
    recorder_pid=$!
    printf '%s\n' "${recorder_pid}" > "${pid_file}"
    sleep 2
    if ! kill -0 "${recorder_pid}" 2>/dev/null; then
      echo "Recorder failed to start."
      tail -50 "${episode_dir}/recorder.log"
      rm -f "${pid_file}" "${active_file}"
      exit 1
    fi
    echo "Started ${episode_id}, PID ${recorder_pid}"
    tail -5 "${episode_dir}/recorder.log"
    ;;

  stop)
    if ! is_running; then
      echo "Recorder is not running."
      exit 1
    fi
    recorder_pid="$(cat "${pid_file}")"
    episode_dir="$(cat "${active_file}")"
    kill -INT "${recorder_pid}"
    for _ in $(seq 1 50); do
      if ! kill -0 "${recorder_pid}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "${recorder_pid}" 2>/dev/null; then
      echo "Recorder did not stop cleanly within 5 seconds."
      exit 1
    fi
    date -Ins > "${episode_dir}/stop_time.txt"
    rm -f "${pid_file}" "${active_file}"
    echo "Stopped recording: ${episode_dir}"
    tail -8 "${episode_dir}/recorder.log"
    ;;

  status)
    if is_running; then
      episode_dir="$(cat "${active_file}")"
      echo "running PID=$(cat "${pid_file}") episode=${episode_dir}"
      tail -5 "${episode_dir}/recorder.log"
    else
      echo "stopped"
    fi
    ;;

  *)
    usage
    exit 2
    ;;
esac
