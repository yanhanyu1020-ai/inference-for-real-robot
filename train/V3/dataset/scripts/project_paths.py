"""Locate project sources from either repository code or a release snapshot."""
import os
from pathlib import Path


def project_root():
    override = os.environ.get('HETEROVLA_ROOT')
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent, parent / 'heterovla'):
            if (candidate / 'third_party/openpi').is_dir():
                return candidate
    raise FileNotFoundError('Set HETEROVLA_ROOT to the project containing third_party/openpi')
