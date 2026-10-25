#!/usr/bin/env bash
set -euo pipefail
v2_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$v2_directory/env.sh"
v2_python="${HETEROVLA_V2_PYTHON:-$HETEROVLA_ROOT/.venvs/openpi-pi05/bin/python}"
if [[ ! -x "$v2_python" ]]; then
  echo "Missing existing openpi Python: $v2_python" >&2
  exit 1
fi
exec "$v2_python" "$v2_directory/scripts/launch_full.py" "$@"

