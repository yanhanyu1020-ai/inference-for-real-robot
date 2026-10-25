#!/usr/bin/env bash
# Run this ON the Go2 Jetson.  A800 cannot SSH to hostname "unitree".
# Pulls control files from the GPU server over the existing tunnel SSH config.
set -euo pipefail

ssh_config="${HOME}/heterovla-control/ssh_config"
remote_dir="${HOME}/heterovla-control"
a800_repo="${A800_HETEROVLA:-~/heterovla}"

if [[ ! -f "${ssh_config}" ]]; then
  echo "missing ${ssh_config}" >&2
  exit 1
fi
mkdir -p "${remote_dir}"

scp -F "${ssh_config}" \
  "heterovla-a800:${a800_repo}/inference/onboard/unitree_go2/control/go2_chunk_loop.py" \
  "heterovla-a800:${a800_repo}/inference/onboard/unitree_go2/control/go2_send.py" \
  "heterovla-a800:${a800_repo}/inference/onboard/unitree_go2/control/episode_log.py" \
  "heterovla-a800:${a800_repo}/inference/onboard/unitree_go2/control/hetero_teleop_loop.py" \
  "heterovla-a800:${a800_repo}/inference/onboard/unitree_go2/control/hetero_teach_loop.py" \
  "heterovla-a800:${a800_repo}/inference/onboard/unitree_go2/control/robot_command_agent.py" \
  "heterovla-a800:${a800_repo}/inference/onboard/robot_gateway/piper_chunk_loop.py" \
  "heterovla-a800:${a800_repo}/inference/onboard/robot_gateway/piper_stream_loop.py" \
  "${remote_dir}/"

chmod +x \
  "${remote_dir}/go2_chunk_loop.py" \
  "${remote_dir}/go2_send.py" \
  "${remote_dir}/episode_log.py" \
  "${remote_dir}/hetero_teleop_loop.py" \
  "${remote_dir}/hetero_teach_loop.py" \
  "${remote_dir}/robot_command_agent.py" \
  "${remote_dir}/piper_chunk_loop.py" \
  "${remote_dir}/piper_stream_loop.py"

echo "updated files in ${remote_dir}"
echo "on A800 run:"
echo "  bash inference/scripts/robotctl.sh system.go2_control_stop"
echo "  bash inference/scripts/robotctl.sh system.go2_control_start"
echo "then restart the agent if you also pulled robot_command_agent.py:"
echo "  ~/heterovla-control/control_up.sh"
