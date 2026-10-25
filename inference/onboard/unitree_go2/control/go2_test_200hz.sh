#!/usr/bin/env bash
# Go2 200 Hz velocity test: bridge(lease) -> stand -> stream 0.5 m/s 2 s -> down.
# REQUIREMENT: the handheld remote must be OFF (or unlocked) - the dog's
# locomotion gate ignores SDK Move while locked to the remote.
set -euo pipefail

cd ~/heterovla-control
clean_env=(env -i HOME="$HOME" PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")

BRIDGE_PID=""
cleanup() {
  echo ">> cleanup"
  if [[ -n "$BRIDGE_PID" ]]; then
    kill -TERM "$BRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[1/4] starting bridge (lease holder)"
rm -f go2.sock go2.log
"${clean_env[@]}" ./go2_velocity_bridge eth0 go2.sock > go2.log 2>&1 &
BRIDGE_PID=$!
sleep 2
grep -q "lease applied" go2.log || { echo "bridge failed:"; cat go2.log; exit 1; }

echo "[2/4] standing up (through the bridge)"
python3 go2_send.py --stand-up
sleep 3

echo "[3/4] streaming 0.5 m/s at 200 Hz for 2 s"
python3 - <<'PYEOF'
import socket, time
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/home/unitree/heterovla-control/go2.sock")
t0 = time.time()
while time.time() - t0 < 2:
    s.sendall(b"V 0.5 0 0\n")
    time.sleep(0.005)
s.sendall(b"V 0 0 0\n")
s.close()
print("200 Hz stream done")
PYEOF
sleep 3

echo "[4/4] lying down, stopping bridge"
python3 go2_send.py --stand-down
sleep 2
kill -TERM "$BRIDGE_PID" 2>/dev/null || true
BRIDGE_PID=""

echo "done. bridge log:"
cat go2.log
