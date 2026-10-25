#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec /meta_eon_cfs/home/hph/heterovla/.venvs/openpi-pi05/bin/python -B "$code_dir/launch_training.py" "$@"
