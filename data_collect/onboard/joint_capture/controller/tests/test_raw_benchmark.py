import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest


def test_benchmark_durable_counts_and_no_overwrite(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/benchmark_unitree_raw.py'
    spec=importlib.util.spec_from_file_location('raw_benchmark',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result=module.run_workload(tmp_path/'new',.05,'lz4',[np.zeros((3,4,3),np.uint8)])
    assert result['raw_bytes_per_s']>0
    assert result['fsync']['count']>=2
    assert all(s['written']==s['durable'] and s['written']>0 for s in result['streams'].values())
    assert not list((tmp_path/'new').glob('*.active'))
    with pytest.raises(FileExistsError):
        module.run_workload(tmp_path/'new',.05,'lz4',[np.zeros((3,4,3),np.uint8)])
