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
source_dir="${inference_root}/onboard/unitree_go2/control"
gateway_dir="${inference_root}/onboard/robot_gateway"
robot_host="${GO2_HOST:-unitree}"
remote_dir="${GO2_CONTROL_DIR:-/home/unitree/heterovla-control}"
sdk_root="${UNITREE_SDK2_ROOT:-/home/unitree/unitree_sdk2-main}"
ssh_config_source="${GO2_SSH_CONFIG:-}"

for required in \
  robot_command_agent.py \
  piper_command_executor.py \
  go2_command_executor.cpp \
  go2_velocity_bridge.cpp \
  go2_send.py \
  go2_chunk_loop.py \
  go2_stream_loop.py \
  hetero_teleop_loop.py \
  hetero_teach_loop.py \
  episode_log.py \
  CMakeLists.txt
do
  if [[ ! -f "${source_dir}/${required}" ]]; then
    echo "Missing control source: ${source_dir}/${required}" >&2
    exit 1
  fi
done
for required in piper_chunk_loop.py piper_stream_loop.py; do
  if [[ ! -f "${gateway_dir}/${required}" ]]; then
    echo "Missing gateway source: ${gateway_dir}/${required}" >&2
    exit 1
  fi
done

ssh "${robot_host}" "mkdir -p '${remote_dir}/build'"
scp \
  "${source_dir}/robot_command_agent.py" \
  "${source_dir}/piper_command_executor.py" \
  "${source_dir}/go2_command_executor.cpp" \
  "${source_dir}/go2_velocity_bridge.cpp" \
  "${source_dir}/go2_send.py" \
  "${source_dir}/go2_chunk_loop.py" \
  "${source_dir}/go2_stream_loop.py" \
  "${source_dir}/hetero_teleop_loop.py" \
  "${source_dir}/hetero_teach_loop.py" \
  "${source_dir}/episode_log.py" \
  "${source_dir}/CMakeLists.txt" \
  "${gateway_dir}/piper_chunk_loop.py" \
  "${gateway_dir}/piper_stream_loop.py" \
  "${robot_host}:${remote_dir}/"

if [[ -n "${ssh_config_source}" ]]; then
  if [[ ! -f "${ssh_config_source}" ]]; then
    echo "GO2_SSH_CONFIG does not exist: ${ssh_config_source}" >&2
    exit 1
  fi
  scp "${ssh_config_source}" "${robot_host}:${remote_dir}/ssh_config"
fi

ssh "${robot_host}" \
  "cmake -S '${remote_dir}' -B '${remote_dir}/build' \
     -DUNITREE_SDK2_ROOT='${sdk_root}' && \
   cmake --build '${remote_dir}/build' -j2 && \
   cp '${remote_dir}/build/go2_command_executor' \
      '${remote_dir}/go2_command_executor' && \
   cp '${remote_dir}/build/go2_velocity_bridge' \
      '${remote_dir}/go2_velocity_bridge' && \
   chmod +x '${remote_dir}/robot_command_agent.py' \
             '${remote_dir}/piper_command_executor.py' \
             '${remote_dir}/go2_send.py' \
             '${remote_dir}/go2_chunk_loop.py' \
             '${remote_dir}/go2_stream_loop.py' \
             '${remote_dir}/hetero_teleop_loop.py' \
             '${remote_dir}/hetero_teach_loop.py' \
             '${remote_dir}/episode_log.py' \
             '${remote_dir}/piper_chunk_loop.py' \
             '${remote_dir}/piper_stream_loop.py' \
             '${remote_dir}/go2_command_executor' \
             '${remote_dir}/go2_velocity_bridge' && \
   if [[ -f '${remote_dir}/ssh_config' ]]; then \
     chmod 600 '${remote_dir}/ssh_config'; \
   fi"

echo "Control agent deployed to ${robot_host}:${remote_dir}"
echo "It has not been started and cannot move hardware yet."
