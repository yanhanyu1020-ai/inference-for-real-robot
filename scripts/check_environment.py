"""Check local dependencies and paths without training or robot connections."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    import os
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.require_gpu else 'cpu'
    os.environ.setdefault('OPENPI_DATA_HOME', str(ROOT / '.cache/openpi'))
    versions = {}
    locations = {}
    for name, package in [('jax', 'jax'), ('jaxlib', 'jaxlib'), ('flax', 'flax'),
                          ('torch', 'torch'), ('numpy', 'numpy'), ('lerobot', 'lerobot'),
                          ('orbax.checkpoint', 'orbax-checkpoint'), ('websockets', 'websockets'),
                          ('av', 'av'), ('pyarrow', 'pyarrow'), ('openpi', 'openpi'),
                          ('openpi_client', 'openpi-client')]:
        module = importlib.import_module(name)
        versions[package] = importlib.metadata.version(package)
        locations[name] = str(Path(module.__file__).resolve())
    assert sys.version_info[:2] == (3, 11), sys.version
    assert Path(sys.prefix).resolve() == ROOT / '.venvs/openpi-pi05', sys.prefix
    assert not any('/home/hph/' in p for p in [*sys.path, *locations.values()])
    for name, expected in {'jax': '0.5.3', 'jaxlib': '0.5.3', 'flax': '0.10.2',
                           'torch': '2.7.1', 'numpy': '1.26.4'}.items():
        assert versions[name] == expected, (name, versions[name], expected)
    data = Path(os.environ.get('HETEROVLA_DATA_ROOT') or ROOT.parent / 'heterovla-data/aligned/V3').resolve()
    models = Path(os.environ.get('HETEROVLA_MODEL_ROOT') or ROOT.parent / 'heterovla-model').resolve()
    paths = [data / 'STATUS.json', data / 'assets/heterovla_v3/norm_stats.json',
             models / 'pretrained/pi05_base/params/_METADATA',
             Path(os.environ['OPENPI_DATA_HOME']) / 'big_vision/paligemma_tokenizer.model']
    assert all(path.is_file() for path in paths), [str(p) for p in paths if not p.is_file()]
    import jax
    import torch
    devices = [str(d) for d in jax.devices()]
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                         text=True, capture_output=True)
    result = dict(cpu_checks_passed=True, python=sys.version, prefix=sys.prefix,
                  versions=versions, module_locations=locations, data_root=str(data),
                  model_root=str(models), jax_devices=devices,
                  cache_dirs={key: os.environ.get(key) for key in
                              ('OPENPI_DATA_HOME', 'HF_HOME', 'HF_DATASETS_CACHE', 'HF_HUB_CACHE', 'WANDB_CACHE_DIR')},
                  torch_cuda_available=torch.cuda.is_available(),
                  nvidia_smi_ok=gpu.returncode == 0,
                  nvidia_smi_output=(gpu.stdout + gpu.stderr).strip(),
                  gpu_required=args.require_gpu, training_started=False)
    if args.require_gpu:
        assert gpu.returncode == 0 and torch.cuda.is_available()
        assert all(d.platform == 'gpu' for d in jax.devices())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
