"""V1 source, data and output paths; importing this module starts no work."""
import os
from pathlib import Path


def configured_path(name, default):
    return Path(os.environ.get(name) or default).expanduser().resolve()


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = configured_path('HETEROVLA_ROOT', SCRIPT_ROOT.parents[2])
DATA_ROOT = configured_path('HETEROVLA_DATA_ROOT', PROJECT_ROOT.parent / 'heterovla-data/aligned/V1')
WEIGHTS = configured_path('PI05_PARAMS', PROJECT_ROOT.parent / 'heterovla-model/pretrained/pi05_base/params')
RUN_ROOT = configured_path('HETEROVLA_RUN_ROOT', PROJECT_ROOT.parent / 'heterovla-model/V1/runs/pi05_v1')
OPENPI_ROOT = configured_path('OPENPI_ROOT', PROJECT_ROOT / 'third_party/openpi')
