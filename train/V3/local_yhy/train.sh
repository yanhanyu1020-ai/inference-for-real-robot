#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$code_dir/../../../env.sh"
exec "$HETEROVLA_ROOT/.venvs/openpi-pi05/bin/python" -B "$code_dir/launch_training.py" "$@"
