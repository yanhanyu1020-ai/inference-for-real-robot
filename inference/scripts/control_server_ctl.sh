#!/usr/bin/env bash
set -euo pipefail

# Resolve physical source paths, including the repository-root compatibility links.
script_source="${BASH_SOURCE[0]}"
while [[ -L "${script_source}" ]]; do
  script_dir="$(cd -P "$(dirname "${script_source}")" && pwd)"
  script_source="$(readlink "${script_source}")"
  [[ "${script_source}" = /* ]] || script_source="${script_dir}/${script_source}"
done
script_dir="$(cd -P "$(dirname "${script_source}")" && pwd)"
inference_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${inference_root}/.." && pwd)"
state_dir="${HOME}/.local/state/heterovla"
token_file="${HOME}/.config/heterovla/control-token"
pid_file="${state_dir}/control-server.pid"
log_file="${state_dir}/control-server.log"
python_bin="${CONTROL_PYTHON:-/usr/bin/python3}"

is_running() {
  [[ -s "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null
}

start() {
  if [[ ! -s "${token_file}" ]]; then
    echo "Missing control token: ${token_file}" >&2
    exit 1
  fi
  if is_running; then
    echo "Control server is already running (pid $(<"${pid_file}"))."
    return
  fi
  mkdir -p "${state_dir}"
  chmod 700 "${state_dir}"
  cd "${repo_root}"
  nohup env \
    HETEROVLA_CONTROL_TOKEN="$(<"${token_file}")" \
    PYTHONPATH="${inference_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -m heterovla.control_server \
      --host 127.0.0.1 --port 8766 \
      >"${log_file}" 2>&1 </dev/null &
  echo "$!" >"${pid_file}"
  sleep 1
  if ! is_running; then
    echo "Control server failed to start. Log:" >&2
    tail -n 30 "${log_file}" >&2
    exit 1
  fi
  echo "Control server started (pid $(<"${pid_file}"))."
}

stop() {
  if ! is_running; then
    echo "Control server is not running."
    return
  fi
  pid="$(<"${pid_file}")"
  kill "${pid}"
  for _ in 1 2 3 4 5; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Control server did not stop cleanly (pid ${pid})." >&2
    exit 1
  fi
  echo "Control server stopped."
}

status() {
  if is_running; then
    echo "Control server is running (pid $(<"${pid_file}"))."
    ss -ltn | grep '127.0.0.1:8766' || true
  else
    echo "Control server is not running."
    exit 1
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop || true; start ;;
  status) status ;;
  logs) tail -n "${2:-50}" "${log_file}" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs [LINES]}" >&2; exit 2 ;;
esac
