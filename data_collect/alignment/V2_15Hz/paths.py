"""Paths for the independently rebuilt, video-timed V2_15Hz release."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('HETEROVLA_ROOT') or '/meta_eon_cfs/home/hph/heterovla').expanduser().resolve()
BENCHMARK_ROOT = Path(os.environ.get('HETEROVLA_BENCHMARK_ROOT') or PROJECT_ROOT.parent/'heterovla-data/raw/benchmark').expanduser().resolve()
REFERENCE_ROOT = Path(os.environ.get('HETEROVLA_REFERENCE_ROOT') or PROJECT_ROOT.parent/'heterovla-data/aligned/V2').expanduser().resolve()
DATA_ROOT = Path(os.environ.get('HETEROVLA_DATA_ROOT') or PROJECT_ROOT.parent/'heterovla-data/aligned/V2_15Hz').expanduser().resolve()
TRAIN_SCRIPTS = PROJECT_ROOT/'train/V2_15Hz/scripts'
