"""Canonical V2 paths; environment overrides support isolated builds."""
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get('HETEROVLA_ROOT') or Path.home()/'heterovla').expanduser().resolve()
DATA_ROOT = Path(os.environ.get('HETEROVLA_DATA_ROOT') or Path.home()/'heterovla-data/aligned/V2').expanduser().resolve()
RAW_ROOT = Path(os.environ.get('HETEROVLA_RAW_ROOT') or Path.home()/'heterovla-data/raw').expanduser().resolve()
BENCHMARK_ROOT = Path(os.environ.get('HETEROVLA_BENCHMARK_ROOT') or RAW_ROOT/'benchmark').expanduser().resolve()
V1_ROOT = Path(os.environ.get('HETEROVLA_V1_ROOT') or Path.home()/'heterovla-data/aligned/V1').expanduser().resolve()
TRAIN_SCRIPTS = PROJECT_ROOT/'train/V2/scripts'
