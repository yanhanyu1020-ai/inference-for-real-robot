#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec /meta_eon_cfs/home/hph/heterovla/.venvs/openpi-pi05/bin/python -B /meta_eon_cfs/home/hph/heterovla/train/V3/official_vitlora_r8_a16_h15_data20261021_resume/resume_training.py "$@"
