"""CPU-only full numeric audit and deterministic evaluation cache, without data writes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np
import pyarrow.parquet as pq
from openpi.training import data_loader
from experiment_configs import DATA_ROOT, RUN_ROOT, make_config, install_dataset_adapter
from scan_common import atomic_json, early_stop_update, overfit


def main():
    started = time.time()
    root = RUN_ROOT / 'lr_scan_20261005'
    root.mkdir(exist_ok=True)
    if (root / 'preflight.json').exists():
        raise FileExistsError('Use the existing audited cache; do not overwrite it')
    manifest = json.loads((DATA_ROOT / 'reports/episode_manifest.json').read_text())
    assert [x['source_episode'][:7] for x in manifest if x['split'] == 'train'] == ['task_01', 'task_02', 'task_04', 'task_05']
    assert [x['source_episode'][:7] for x in manifest if x['split'] == 'val'] == ['task_06']
    stats_path = DATA_ROOT / 'assets/heterovla_v1/norm_stats.json'
    stats = json.loads(stats_path.read_text())['norm_stats']
    audit = {'splits': {}, 'train_only_norm': {}, 'norm_sha256': hashlib.sha256(stats_path.read_bytes()).hexdigest()}
    install_dataset_adapter()
    for split in ('train', 'val'):
        arrays = {'state': [], 'actions': []}
        base = DATA_ROOT / f'lerobot/local/heterovla_v1_{split}'
        info = json.loads((base / 'meta/info.json').read_text())
        names = info['features']['action']['names']
        assert all(x.startswith('go2.') for x in names[:3])
        assert all(x.startswith('piper.') for x in names[3:10])
        assert all(x.startswith('uav.') for x in names[10:14])
        for path in sorted((base / 'data').rglob('*.parquet')):
            data = pq.read_table(path).to_pydict()
            assert np.asarray(data['action_valid']).all()
            for name, key in [('state', 'observation.state'), ('actions', 'action')]:
                values = np.asarray(data[key], dtype=np.float64)
                assert np.isfinite(values).all(), str(path)
                arrays[name].extend(values)
            times = np.asarray(data['timestamp']).reshape(-1)
            assert np.allclose(np.diff(times), .1, atol=1e-4), str(path)
        if split == 'train':
            for name, values in arrays.items():
                a = np.asarray(values)
                for key, expected in [('mean', a.mean(0)), ('std', a.std(0)),
                                      ('q01', np.quantile(a, .01, axis=0)), ('q99', np.quantile(a, .99, axis=0))]:
                    assert np.allclose(stats[name][key], expected, atol=1e-7, rtol=1e-5), (name, key)
                audit['train_only_norm'][name] = {'frames': len(a), 'all_four_statistics_match': True}
        config = make_config('dual_lora', batch_size=16, split=split,
                             remat_policy='dots_with_no_batch_dims_saveable')
        dc = config.data.create(config.assets_dirs, config.model)
        raw = data_loader.create_torch_dataset(dc, 10, config.model)
        transformed = data_loader.transform_dataset(raw, dc)
        assert len(raw) == (2945 if split == 'train' else 710)
        indices = np.arange(len(raw)) if split == 'val' else np.linspace(0, len(raw)-1, 128, dtype=int)
        directory = root / ('val_cache' if split == 'val' else 'train_probe_cache')
        directory.mkdir(exist_ok=True)
        hashes = {}
        for offset in range(0, len(indices), 16):
            samples = [transformed[int(i)] for i in indices[offset:offset+16]]
            count = len(samples)
            for sample in samples:
                assert sample['actions'].shape == (10, 32)
                assert np.isfinite(sample['actions']).all() and np.isfinite(sample['state']).all()
                assert (sample['actions'][:, 14:] == 0).all()
            samples += [samples[-1]] * (16-count)
            batch = data_loader._collate_fn(samples)
            flat = {'count': count, 'offset': offset}
            for key, value in batch.items():
                if isinstance(value, dict):
                    flat.update({key + '/' + k: v for k, v in value.items()})
                else:
                    flat[key] = value
            path = directory / f'batch_{offset:04d}.npz'
            np.savez(path, **flat)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if offset % 160 == 0:
                print(f'CACHE {split} {offset}/{len(indices)}', flush=True)
        audit['splits'][split] = {'windows': len(raw), 'cached_windows': len(indices),
            'indices': indices.tolist(), 'cache_sha256': hashes, 'action_names': names,
            'frames': len(arrays['actions']), 'quantile_norm': dc.use_quantile_norm}
    # Verify boundary semantics for the stopping rule without model allocation.
    history = []
    for step in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        history.append(early_stop_update(history, step, 1.0))
    assert history[-2]['streak'] == 4 and history[-1]['streak'] == 5
    assert early_stop_update(history, 1100, .99)['streak'] == 0
    assert overfit([{'step': s, 'val': 1+i*.1, 'train': 2-i*.1} for i,s in enumerate([200,300,400,500])])
    audit.update(status='passed', seed=42, eval_batch_size=16, complete_val=True,
                 val_augmentation=False, elapsed_seconds=time.time()-started,
                 dataset_writes=False, devices=[str(d) for d in jax.devices()])
    atomic_json(root / 'preflight.json', audit)
    print('SCAN_PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    main()
