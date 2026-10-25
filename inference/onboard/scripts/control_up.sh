#!/usr/bin/env bash
# Restore the HeteroVLA control chain on the Go2 onboard computer:
#   1) bring up Piper CAN
#   2) open the SSH tunnel to the GPU server control broker
#   3) start robot_command_agent with a clean environment (no ROS/CycloneDDS LD_LIBRARY_PATH)
set -euo pipefail

# Resolve physical source paths, including the repository-root compatibility links.
script_source="${BASH_SOURCE[0]}"
while [[ -L "${script_source}" ]]; do
  script_dir="$(cd -P "$(dirname "${script_source}")" && pwd)"
  script_source="$(readlink "${script_source}")"
  [[ "${script_source}" = /* ]] || script_source="${script_dir}/${script_source}"
done
script_dir="$(cd -P "$(dirname "${script_source}")" && pwd)"
here="${script_dir}"
agent_script="${here}/robot_command_agent.py"
if [[ ! -f "${agent_script}" ]]; then
  agent_script="${here}/../unitree_go2/control/robot_command_agent.py"
fi
python="${HOME}/heterovla-client-env/bin/python"
token_file="${HOME}/.config/heterovla/control-token"
local_port=18766
log_file="${HOME}/heterovla-control/agent.log"

echo "[1/3] bring up Piper CAN"
if ! ip -brief link show can0 | grep -q UP; then
  sudo ip link set can0 up type can bitrate 1000000
fi
ip -brief link show can0 | grep -q UP || { echo "can0 is not UP"; exit 1; }

echo "[2/3] open SSH tunnels (unitree -> h100 -> a800)"
# 18766 -> 8766: control broker; 18765 -> 8765: policy server
# A while-loop watchdog keeps the tunnel alive: SSH -fNT tunnels are passive
# and exit when the connection drops (no auto-reconnect).
watchdog="${HOME}/heterovla-control/tunnel_watchdog.log"
start_tunnel() {
  while true; do
    ssh -F "${HOME}/heterovla-control/ssh_config" -fNT \
      -o ExitOnForwardFailure=yes \
      -L "127.0.0.1:${local_port}:127.0.0.1:8766" \
      -L "127.0.0.1:18765:127.0.0.1:8765" heterovla-a800
    echo "$(date '+%F %T') tunnel exited, restarting in 5s" >> "${watchdog}"
    sleep 5
  done
}
if ss -tln | grep -q "127.0.0.1:${local_port}"; then
  echo "tunnel already listening on ${local_port}, skipping"
else
  export local_port watchdog
  nohup bash -c "$(declare -f start_tunnel); start_tunnel" \
    > /dev/null 2>&1 &
  sleep 2
fi
ss -tln | grep -q "127.0.0.1:${local_port}" || { echo "tunnel failed"; exit 1; }

echo "[3/3] restart robot_command_agent with clean env"
pkill -f robot_command_agent || true
sleep 1
env -i \
  HOME="${HOME}" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  USER="${USER:-unitree}" \
  HETEROVLA_CONTROL_TOKEN="$(<"${token_file}")" \
  nohup "${python}" "${agent_script}" \
    --uri "ws://127.0.0.1:${local_port}" --robot-id unitree-go2-01 --execute \
    > "${log_file}" 2>&1 &
sleep 2

echo "--- agent.log ---"
tail -n 3 "${log_file}"
if grep -q "registered as" "${log_file}"; then
  echo "OK: agent registered (execute=True)"
else
  echo "WARN: agent not registered yet, check ${log_file}"
fi
