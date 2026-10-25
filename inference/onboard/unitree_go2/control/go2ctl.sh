#!/usr/bin/env bash
# Run Go2 SDK commands with a clean environment.
#
# The interactive shell sources cyclonedds_ws/ROS, whose DDS libraries clash
# with the SDK's bundled libddsc -> "free(): invalid pointer" crashes.
set -euo pipefail

exec env -i \
  HOME="$HOME" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "$HOME/heterovla-control/go2_command_executor" eth0 "$@"
