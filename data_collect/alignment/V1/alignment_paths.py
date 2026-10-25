"""Default V1 conversion paths, independent of the working directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('HETEROVLA_ROOT') or Path(__file__).resolve().parents[3]).expanduser().resolve()
DATA_ROOT = Path(os.environ.get('HETEROVLA_DATA_ROOT') or Path.home() / 'heterovla-data/aligned/V1').expanduser().resolve()
RAW_ROOT = Path(os.environ.get('HETEROVLA_RAW_ROOT') or Path.home() / 'heterovla-data/raw').expanduser().resolve()
BENCHMARK_ROOT = Path(os.environ.get('HETEROVLA_BENCHMARK_ROOT') or RAW_ROOT / 'benchmark').expanduser().resolve()
TRAIN_SCRIPTS = PROJECT_ROOT / 'train/V1/scripts'
