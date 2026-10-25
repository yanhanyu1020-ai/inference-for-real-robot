#!/usr/bin/env bash
# Run in an interactive 16.7 terminal. The remote sudo password stays in that TTY.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
clock_epoch="$(date +%s)"
exec ssh -tt -F "$project_root/.cache/real_robot/ssh_config" v3-p450 \
  "bash -lc 'source /opt/ros/noetic/setup.bash && source /home/amov/p450_experiment/devel/setup.bash && cd /home/amov/heterovla-live && /usr/bin/python3 -m real_robot.clock_repair --unix ${clock_epoch}'"
