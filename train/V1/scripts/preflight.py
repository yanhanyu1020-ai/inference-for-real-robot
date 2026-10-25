"""CPU-only data, tokenizer, checkpoint metadata and freezing checks; no training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
import importlib.metadata
import json
import math
from pathlib import Path
from path_config import OPENPI_ROOT
import subprocess
import time

import flax.nnx as nnx
import jax
import numpy as np
from openpi.training import data_loader
from experiment_configs import (DATA_ROOT, PROJECT_ROOT, RUN_ROOT, WEIGHTS, VARIANTS,
                                make_config, summary, install_dataset_adapter)


def main():
    started = time.time()
    reports = RUN_ROOT / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / 'configs').mkdir(parents=True, exist_ok=True)
    result = {'training_started': False, 'gradient_updates': 0,
              'jax_devices': [str(d) for d in jax.devices()], 'variants': {}, 'datasets': {}}
    assert all(d.platform == 'cpu' for d in jax.devices())
    result['versions'] = {k: importlib.metadata.version(k) for k in
                         ['jax', 'jaxlib', 'flax', 'torch', 'lerobot', 'wandb', 'orbax-checkpoint', 'numpy']}
    result['openpi_commit'] = subprocess.check_output(
        ['git', '-c', f'safe.directory={OPENPI_ROOT}', '-C', str(OPENPI_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert result['openpi_commit'] == '15a9616a00943ada6c20a0f158e3adb39df2ccac'
    assert (WEIGHTS / '_METADATA').is_file()
    result['checkpoint_metadata_present'] = True
    metadata = json.loads((WEIGHTS / '_METADATA').read_text())['tree_metadata']
    checkpoint_shapes = {
        tuple(part['key'] for part in record['key_metadata'])[1:]:
        tuple(record['value_metadata']['write_shape'])
        for record in metadata.values()
        if 'write_shape' in record['value_metadata']
    }
    install_dataset_adapter()

    for variant in VARIANTS:
        config = make_config(variant)
        # NNX eval_shape constructs abstract arrays only, not the multi-GB weights.
        abstract_model = nnx.eval_shape(lambda: config.model.create(jax.random.key(42)))
        params = nnx.state(abstract_model, nnx.Param)
        all_params = params.flat_state()
        checked_weights = 0
        for path, leaf in all_params.items():
            name = '/'.join(map(str, path))
            if 'lora' in name:
                continue
            key = tuple(map(str, path))
            assert key in checkpoint_shapes, f'Missing pretrained parameter: {name}'
            assert tuple(leaf.value.shape) == checkpoint_shapes[key], f'Weight shape mismatch: {name}'
            checked_weights += 1
        selected = params.filter(config.trainable_filter).flat_state()
        groups = {}
        for path, leaf in all_params.items():
            name = '/'.join(map(str, path))
            group = ('vision' if 'img' in name else
                     'action_expert' if 'llm' in name and '_1' in name else
                     'language' if 'llm' in name else 'projections_and_time_mlp')
            count = math.prod(leaf.value.shape)
            g = groups.setdefault(group, {'total': 0, 'trainable': 0, 'trainable_non_lora': 0})
            g['total'] += count
            if path in selected:
                g['trainable'] += count
                if 'lora' not in name:
                    g['trainable_non_lora'] += count
        if variant != 'full':
            assert groups['vision']['trainable'] == 0
        if variant == 'action_lora':
            assert groups['language']['trainable'] == 0
        if variant in ('action_lora', 'dual_lora'):
            assert groups['action_expert']['trainable'] > 0
            assert groups['action_expert']['trainable_non_lora'] == 0
        if variant in ('dual_lora', 'expert_full'):
            assert groups['language']['trainable'] > 0
            assert groups['language']['trainable_non_lora'] == 0
        if variant in ('expert_full', 'full'):
            assert groups['action_expert']['trainable'] == groups['action_expert']['total']
        if variant == 'full':
            assert all(g['total'] == g['trainable'] for g in groups.values())
        assert groups['projections_and_time_mlp']['trainable'] == groups['projections_and_time_mlp']['total']
        record = {**summary(config), 'parameter_groups': groups,
                  'pretrained_parameter_shapes_checked': checked_weights,
                  'total_parameters': sum(g['total'] for g in groups.values()),
                  'trainable_parameters': sum(g['trainable'] for g in groups.values())}
        result['variants'][variant] = record
        (RUN_ROOT / 'configs' / f'{variant}.json').write_text(json.dumps(record, indent=2))
        print('FREEZE_CHECK_OK', variant, record['trainable_parameters'], flush=True)

    for split in ('train', 'val'):
        config = make_config('dual_lora', split=split)
        data_config = config.data.create(config.assets_dirs, config.model)
        raw = data_loader.create_torch_dataset(data_config, 10, config.model)
        transformed = data_loader.transform_dataset(raw, data_config)
        indices = sorted(set([0, len(raw) // 2, len(raw) - 1]))
        for index in indices:
            source = raw[index]
            assert source['prompt'] == '抓取桌子上的纸箱内的方块'
            assert source['state'].shape == (30,) and source['actions'].shape == (10, 14)
            sample = transformed[index]
            assert sample['state'].shape == (32,) and sample['actions'].shape == (10, 32)
            assert np.isfinite(sample['state']).all() and np.isfinite(sample['actions']).all()
            assert np.all(sample['actions'][:, 14:] == 0)
            assert sample['tokenized_prompt_mask'].sum() < config.model.max_token_len
            assert all(a.shape == (224, 224, 3) for a in sample['image'].values())
        assert len(raw) == (2945 if split == 'train' else 710)
        result['datasets'][split] = {'windows': len(raw), 'sample_indices_checked': indices,
                                    'normalized_state_shape': [32], 'action_shape': [10, 32],
                                    'images': 3, 'image_shape': [224, 224, 3]}
        print('DATA_CHECK_OK', split, len(raw), flush=True)
    result['elapsed_seconds'] = round(time.time() - started, 1)
    result['not_yet_tested'] = ['GPU model allocation', 'forward/backward pass', 'max batch size',
                              'training throughput', 'checkpoint save/restore after training',
                              'offline action evaluation and gradient accumulation']
    (reports / 'preflight.json').write_text(json.dumps(result, indent=2))
    print('PREFLIGHT_PASSED_NO_TRAINING', flush=True)


if __name__ == '__main__':
    main()
