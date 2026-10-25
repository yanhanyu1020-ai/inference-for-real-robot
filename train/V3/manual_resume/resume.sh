#!/usr/bin/env bash
set -euo pipefail
exec /meta_eon_cfs/home/hph/heterovla/.venvs/openpi-pi05/bin/python -B \
  /meta_eon_cfs/home/hph/heterovla/train/V3/manual_resume/resume_training.py "$@"
