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
token_file="${HOME}/.config/heterovla/control-token"

if [[ ! -s "${token_file}" ]]; then
  echo "Missing control token: ${token_file}" >&2
  exit 1
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 COMMAND [DATA_JSON|-|--data-file PATH]" >&2
  echo "example: $0 system.status" >&2
  echo "example: $0 piper.action_chunk '{\"actions\": [[0,5,-10,0,15,0,2]]}'" >&2
  echo "example: $0 piper.action_chunk --data-file arm_grasp.json" >&2
  echo "example: $0 go2.action_chunk '{\"actions\": [[0.3,0,0],[0,0,0]]}'" >&2
  echo "example: $0 system.record_start '{\"episode_id\":\"demo_001\"}'" >&2
  echo "example: $0 system.teleop_start" >&2
  echo "example: $0 system.teach_start" >&2
  exit 2
fi

command="$1"
shift

tmp=""
cleanup() {
  if [[ -n "${tmp}" && -f "${tmp}" ]]; then
    rm -f "${tmp}"
  fi
}
trap cleanup EXIT

data_arg=()
if [[ $# -ge 1 ]]; then
  if [[ "$1" == "--data-file" ]]; then
    data_arg=(--data-file "${2:?missing --data-file path}")
  elif [[ "$1" == "-" ]]; then
    data_arg=(--data-file /dev/stdin)
  else
    # Keep large JSON off Python's argv (Linux MAX_ARG_STRLEN is 128KiB).
    tmp="$(mktemp)"
    printf '%s' "$1" > "${tmp}"
    data_arg=(--data-file "${tmp}")
  fi
fi

cd "${repo_root}"
env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  HETEROVLA_CONTROL_TOKEN="$(<"${token_file}")" \
  PYTHONPATH="${inference_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  /usr/bin/python3 "${inference_root}/scripts/send_robot_command.py" "${command}" "${data_arg[@]}"
