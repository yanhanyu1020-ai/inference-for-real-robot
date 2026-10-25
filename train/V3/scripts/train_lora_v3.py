#!/usr/bin/env python3
"""V3 frozen-ViT LoRA finetuning: CPU preflight/cache preparation or explicit single-GPU execution."""
import argparse
import ast
import dataclasses
import fcntl
import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import traceback

from v3_config import install_dataset_adapter, make_config, paths, phase_steps, source_step, PARENT_RUN, MODEL_NAME
from lora_state_v3 import check_runtime_contract, init_lora_train_state, parameter_report, frozen_digest, cpu_shape_preflight


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='CPU-only config, local weight metadata and real V2 loader checks')
    mode.add_argument('--prepare-cache', action='store_true', help='CPU preflight plus complete deterministic val/train-probe cache')
    mode.add_argument('--execute', action='store_true', help='Actually allocate the assigned GPU and train; never implicit')
    parser.add_argument('--gpu', '--gpu-uuid', dest='gpu', help='One physical GPU index or UUID, required only with --execute')
    parser.add_argument('--name', default='preflight', help='Unique independent V3 run name; required for execution')
    parser.add_argument('--resume', action='store_true', help='Resume only this V3 LoRA regular checkpoint and optimizer')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--base-params', type=Path)
    parser.add_argument('--warm-start-params', type=Path, help='Explicit parent best params; weights only, always a fresh optimizer on a new run')
    parser.add_argument('--source-step', type=int, default=1000, help='Parent global step; added to the new optimizer phase step')
    parser.add_argument('--parent-run', default=PARENT_RUN, help='Existing V2 parent run name associated with warm-start params')
    parser.add_argument('--reuse-eval-cache', type=Path, help='Reuse an audited pre-warm-start cache after exact contract/SHA checks; no decoding')
    parser.add_argument('--preallocate-for-handoff', action='store_true', help='Explicitly reserve a 90% JAX allocator pool after the old trainer exits')
    parser.add_argument('--handoff-release-file', type=Path, help='CPU-audit first, publish PATH.ready.json, then wait for the external supervisor release file')
    parser.add_argument('--handoff-timeout', type=float, default=1800)
    parser.add_argument('--_handoff-audit-file', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--parent-run-root', type=Path, help='Immutable parent V2 runs directory; separate from all V3 output')
    parser.add_argument('--run-root', type=Path, help='Independent V3 run metadata and evaluation cache root')
    parser.add_argument('--report', type=Path, help='Optional path for the CPU preflight JSON report')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--peak-lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=20000, help='Target global step, including source-step for a warm start')
    parser.add_argument('--eval-interval', type=int, default=200)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--keep-period', type=int, default=5000)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    if not args.warm_start_params:
        args.warm_start_params = paths(args)['weights']
    if args.source_step != 1000 or args.parent_run != PARENT_RUN:
        parser.error('V3 source must be the audited low-LR V2 best1000 parent')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name):
        parser.error('--name must be one safe directory name')
    if args.execute and (not args.gpu or args.name == 'preflight'):
        parser.error('--execute requires --gpu and a unique --name')
    if args.gpu and not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)', args.gpu):
        parser.error('--gpu must select exactly one GPU index or UUID')
    if args.resume and not args.execute:
        parser.error('--resume requires --execute')
    if (args.preallocate_for_handoff or args.handoff_release_file) and not args.execute:
        parser.error('GPU handoff flags require --execute')
    if args.handoff_timeout <= 0 or not math.isfinite(args.handoff_timeout):
        parser.error('--handoff-timeout must be finite and positive')
    if args._handoff_audit_file and not (args.execute and args.handoff_release_file):
        parser.error('The internal audited phase requires an explicit handoff release file')
    if args.warm_start_params:
        if args.base_params or args.source_step <= 0 or not args.parent_run or args.parent_run == args.name:
            parser.error('Warm start requires a distinct --parent-run, positive --source-step and no --base-params')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.parent_run):
            parser.error('--parent-run must be one safe existing run name')
    elif args.source_step or args.parent_run:
        parser.error('--source-step and --parent-run require --warm-start-params')
    for key in ('batch_size', 'eval_interval', 'eval_batch_size', 'save_interval', 'keep_period', 'log_interval'):
        if getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if phase_steps(args) <= 100 or args.peak_lr <= 0 or not math.isfinite(args.peak_lr) or args.num_workers < 0:
        parser.error('steps must exceed warmup=100, LR must be positive and workers nonnegative')
    if args.save_interval > 1000 or args.keep_period % args.save_interval:
        parser.error('Recovery saves must be at least every 1000 steps; keep-period must be a save-interval multiple')
    return args


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configure_environment(args):
    location = paths(args)
    cpu_handoff = args.handoff_release_file and not args._handoff_audit_file
    gpu_phase = args.execute and not cpu_handoff
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu if gpu_phase else '',
        JAX_PLATFORMS='cuda' if gpu_phase else 'cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='true' if args.preallocate_for_handoff and gpu_phase else 'false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
        WANDB_MODE='offline', WANDB_DISABLE_CODE='true', WANDB_SILENT='true',
        WANDB_GROUP='pi05-v3-lora-frozen-vit-20000', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_LEROBOT_HOME=str(location['data'] / 'lerobot'))
    return location


def plan(args, location):
    record = {
        'name': args.name, 'mode': 'execute' if args.execute else ('prepare_cache' if args.prepare_cache else 'preflight'),
        'variant': 'lora_frozen_vit', 'batch_size': args.batch_size, 'peak_lr': args.peak_lr,
        'warmup_steps': 100, 'decay_lr': args.peak_lr / 10, 'steps': args.steps, 'seed': args.seed,
        'optimizer': {'name': 'AdamW', 'b1': 0.9, 'b2': 0.95, 'eps': 1e-8, 'weight_decay': 1e-10, 'clip_gradient_norm': 1.0},
        'remat_policy': 'nothing_saveable', 'parameter_dtype': 'float32', 'compute_dtype': 'bfloat16',
        'ema_decay': None, 'full_unfreeze': False, 'lora': True, 'vit_frozen': True, 'gemma_base_frozen': True, 'early_stopping': False,
        'initialization': 'official_pi05_base_new_optimizer_step0' if not args.resume else 'same_V3_regular_checkpoint',
        'wandb_mode': 'offline', 'eval_interval': args.eval_interval, 'eval_batch_size': args.eval_batch_size,
        'save_interval': args.save_interval, 'keep_period': args.keep_period,
        'log_interval': args.log_interval, 'num_workers': args.num_workers,
        'data_root': str(location['data']), 'base_params': str(location['weights']),
        'openpi_root': str(location['openpi']), 'run_dir': str(location['run']),
        'parent_run_root': str(location['parent_runs']),
        'checkpoint_dir': str(location['checkpoint_base'] / MODEL_NAME / args.name),
        'best_dir': str(location['run'] / 'best'), 'resume': args.resume,
        'objective': 'official mean flow matching loss over all 32 dimensions; real14 metrics diagnostic',
    }
    if args.warm_start_params:
        record.update(initialization='same_V3_regular_checkpoint' if args.resume else 'V2_best1000_dense_zero_lora_fresh_optimizer_phase0',
                      parent_run=args.parent_run, source_step=source_step(args),
                      phase_steps=phase_steps(args), target_global_step=args.steps,
                      phase_optimizer_start=0, sampler_restarts_with_seed=args.seed,
                      warm_start_params=str(location['weights']))
    if args.preallocate_for_handoff:
        record['preallocate_for_handoff'] = True
    return record


def warm_start_provenance(args, location, record):
    """CPU streaming checksums only; no checkpoint tensors are instantiated."""
    if not args.warm_start_params:
        return None
    parent = location['parent_runs'] / args.parent_run
    parent_config = read_json(parent / 'config.json')
    best = read_json(parent / 'best.json')
    if args.resume:
        saved = read_json(location['run'] / 'config.json')['warm_start_provenance']
        assert saved['source_step'] == source_step(args) and saved['parent_run'] == args.parent_run
        assert saved['params_path'] == str(location['weights'])
        best = {'best_step': saved['source_step'], 'best_checkpoint': str(location['weights'].parent), 'best': saved['expected_initial_val_real14_loss'], 'best_metrics': saved['expected_initial_val_metrics']}
    assert int(best['best_step']) == source_step(args), 'Parent best step does not match the requested source step'
    assert (Path(best['best_checkpoint']) / 'params').resolve() == location['weights'], 'Warm start must reference the parent best params exactly'
    for key in ('norm_sha256', 'manifest_sha256', 'source_inventory_sha256'):
        assert parent_config[key] == record[key], f'Parent V2 data differs: {key}'
    parent_norm = location['weights'].parent / 'assets/heterovla_v2/norm_stats.json'
    assert parent_norm.is_file() and read_json(parent_norm) == read_json(location['data'] / 'assets/heterovla_v2/norm_stats.json'), 'Parent checkpoint normalization differs from aligned/V2'
    entries = []
    for path in sorted(location['weights'].rglob('*')):
        if path.is_file():
            assert not path.is_symlink(), 'Warm-start parameter files must be immutable regular checkpoint files'
            stat = path.stat()
            entries.append({'path': str(path.relative_to(location['weights'])), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    assert entries and any(entry['path'] == '_METADATA' for entry in entries)
    memo_key = hashlib.sha256(str(location['weights']).encode()).hexdigest()[:20]
    memo_path = location['runs'] / '_warm_start_audits' / f'{memo_key}.json'
    memo = read_json(memo_path) if memo_path.exists() else None
    if memo is None or memo.get('file_stats') != entries:
        content = [{'path': entry['path'], 'size': entry['size'], 'sha256': sha256(location['weights'] / entry['path'])} for entry in entries]
        memo = {'params_path': str(location['weights']), 'file_stats': entries, 'file_sha256': content,
                'params_sha256': hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
                'files': len(content), 'bytes': sum(entry['size'] for entry in entries)}
        atomic_json(memo_path, memo)
    result = {key: memo[key] for key in ('params_path', 'params_sha256', 'files', 'bytes')}
    result.update(parent_run=args.parent_run, source_step=source_step(args),
                  parent_config_sha256=sha256(parent / 'config.json'),
                  expected_initial_val_real14_loss=float(best['best']),
                  expected_initial_val_metrics=best['best_metrics'],
                  fresh_optimizer=True, optimizer_phase_step=0)
    if args.resume:
        assert result == saved, 'Warm-start source checkpoint changed after this V2 run was created'
    return result


def reuse_eval_cache(args, audit):
    """Only config weight/schedule/clock changes are compatible with this old cache."""
    if not args.reuse_eval_cache:
        return
    original = args.reuse_eval_cache.expanduser().resolve()
    target = Path(audit['cache_dir']).resolve()
    if original == target:
        return
    original_manifest = read_json(original / 'manifest.json')
    assert original_manifest['complete'], 'The source evaluation cache is incomplete'
    old_contract = original_manifest['cache_contract']
    new_contract = audit['cache_contract']
    old_without_transform = {key: value for key, value in old_contract.items() if key != 'transform_sha256'}
    new_without_transform = {key: value for key, value in new_contract.items() if key != 'transform_sha256'}
    assert old_without_transform == new_without_transform, 'Evaluation cache data, seed, batching or builder changed'
    old_transform, new_transform = dict(old_contract['transform_sha256']), dict(new_contract['transform_sha256'])
    legacy_config = old_transform.pop('v2_config.py')
    new_transform.pop('v3_config.py')
    assert legacy_config == '468f1383c8518ed67677b810650e0af2a28710d8493002a7ab4485816d3d33c8', 'Only the audited low-LR V2 config may supply V3 evaluation data'
    assert old_transform.pop('dataset_v2.py') == new_transform.pop('dataset_v3.py') == '2adc6c25c616d2271e08ec0821ef3e8e390a5a8b9971fbf0dfc7477065d07984'
    config_tree = ast.parse(Path(__file__).with_name('v3_config.py').read_text())
    transform_classes = [node for node in config_tree.body if isinstance(node, ast.ClassDef) and node.name in ('Output14', 'JointTransforms')]
    def canonical_ast(node):
        # ast.dump formatting/empty fields changed between Python versions.
        if node is Ellipsis:
            return {'_literal': 'Ellipsis'}
        if isinstance(node, ast.AST):
            return {'_type': type(node).__name__, **{key: canonical_ast(value) for key, value in ast.iter_fields(node)
                    if key != 'type_params' and value is not None and value != []}}
        if isinstance(node, list):
            return [canonical_ast(value) for value in node]
        return node
    transform_ast = json.dumps(canonical_ast(ast.Module(body=transform_classes, type_ignores=[])), sort_keys=True, separators=(',', ':'))
    assert hashlib.sha256(transform_ast.encode()).hexdigest() == 'b1c140d8285c586e64eb89f3b795ecbac4754b8ae476fe951d039e8cba32f423', 'V3 data transforms differ from the audited V2 transforms'
    assert old_transform == new_transform, 'Image, loader, tokenizer or normalization transforms changed'
    target.mkdir(parents=True, exist_ok=True)
    locks = []
    try:
        # The V2 source is immutable. Only the independent V3 destination gets
        # a lock file; never create or modify anything under the parent cache.
        source_manifest_sha = sha256(original / 'manifest.json')
        for directory in (target,):
            lock = (directory / '.prepare.lock').open('a')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            locks.append(lock)
        for split in original_manifest['splits'].values():
            for name, expected in split['files'].items():
                relative = Path(name)
                assert not relative.is_absolute() and '..' not in relative.parts
                source, destination = original / relative, target / relative
                assert source.is_file() and sha256(source) == expected, source
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    assert sha256(destination) == expected
                    continue
                try:
                    os.link(source, destination)
                except OSError:
                    temporary = destination.with_suffix('.reuse.tmp')
                    shutil.copy2(source, temporary)
                    temporary.replace(destination)
                assert sha256(destination) == expected
        assert sha256(original / 'manifest.json') == source_manifest_sha, 'Immutable parent cache changed during reuse'
        migrated = {**original_manifest, 'cache_contract': new_contract, 'reused_from': str(original), 'source_manifest_sha256': source_manifest_sha}
        atomic_json(target / 'manifest.json', migrated)
    finally:
        for lock in reversed(locks):
            lock.close()


def check_transformed(sample):
    import numpy as np
    assert np.asarray(sample['state']).shape == (32,)
    assert np.asarray(sample['actions']).shape == (10, 32)
    assert np.isfinite(sample['state']).all() and np.isfinite(sample['actions']).all()
    assert (np.asarray(sample['actions'])[:, 14:] == 0).all()
    assert set(sample['image']) == {'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'}
    assert all(np.asarray(value).shape == (224, 224, 3) for value in sample['image'].values())
    assert np.asarray(sample['tokenized_prompt_mask']).any()


def cpu_audit(args, location, *, write_report=True):
    """CPU shape tracing and checkpoint metadata only; no GPU or tensor restore."""
    import numpy as np
    from openpi.training import data_loader
    from dataset_v3 import JointV2Dataset
    assert os.environ['JAX_PLATFORMS'] in ('cpu', 'cuda')
    validation_path = location['data'] / 'reports/validation.json'
    validation = read_json(validation_path)
    assert validation['format_validation_passed'] and validation['normalization_train_only']
    norm_path = location['data'] / 'assets/heterovla_v2/norm_stats.json'
    stats = read_json(norm_path)['norm_stats']
    for key, dimension in [('state', 30), ('actions', 14)]:
        for field in ('mean', 'std', 'q01', 'q99'):
            values = np.asarray(stats[key][field])
            assert values.shape == (dimension,) and np.isfinite(values).all(), (key, field)
        assert (np.asarray(stats[key]['std']) >= 0).all()
        assert (np.asarray(stats[key]['q99']) >= np.asarray(stats[key]['q01'])).all()
    assert (location['weights'] / '_METADATA').is_file(), 'Local initialization params metadata missing'
    assert (location['openpi'] / 'scripts/train.py').is_file()
    # Inspect local checkpoint metadata only, never load multi-GB tensor arrays.
    metadata = read_json(location['weights'] / '_METADATA')
    assert metadata.get('tree_metadata'), 'Invalid local initialization parameter metadata'
    install_dataset_adapter(location['data'])
    record = plan(args, location)
    record.update(status='passed', training_started=False, gradient_updates=0,
                  gpu_enumerated=False, model_instantiated=False, pretrained_arrays_loaded=False,
                  norm_sha256=sha256(norm_path), validation_sha256=sha256(validation_path),
                  manifest_sha256=sha256(location['data'] / 'reports/episode_manifest.json'),
                  source_inventory_sha256=sha256(location['data'] / 'configs/source_inventory.json'),
                  base_metadata_sha256=sha256(location['weights'] / '_METADATA'), datasets={})
    provenance = warm_start_provenance(args, location, record)
    if provenance is not None:
        record['warm_start_provenance'] = provenance
    record['openpi_commit'] = subprocess.check_output(['git', '-C', str(location['openpi']), 'rev-parse', 'HEAD'], text=True).strip()
    record['runtime_source_contract'] = check_runtime_contract(location['openpi'])
    if os.environ['JAX_PLATFORMS'] == 'cpu':
        record['abstract_parameter_report'] = cpu_shape_preflight(make_config(args), location['weights'])
        record['cpu_parameter_report'] = record['abstract_parameter_report']
        record['model_abstractly_traced'] = True
        assert record['abstract_parameter_report']['trainable_parameters'] == 52153376
    download_report = location['weights'].parent.parent / 'pi05_base.download_verified.json'
    if download_report.is_file():
        verification = read_json(download_report)
        assert verification['source'] == 'gs://openpi-assets/checkpoints/pi05_base'
        assert verification.get('all_official_md5_verified') is True or verification.get('all_official_crc32c_verified') is True, 'Official base download was not verified'
        assert not verification.get('errors'), 'Official base download reported errors'
        record['base_download_report_sha256'] = sha256(download_report)
    for split in ('train', 'val'):
        config = make_config(args, split)
        assert config.model.remat_policy == 'nothing_saveable'
        assert config.model.paligemma_variant == 'gemma_2b_lora' and config.model.action_expert_variant == 'gemma_300m_lora'
        assert config.ema_decay is None and config.optimizer.weight_decay == 1e-10
        dc = config.data.create(config.assets_dirs, config.model)
        assert dc.norm_stats is not None and dc.asset_id == 'heterovla_v2' and dc.use_quantile_norm
        raw = data_loader.create_torch_dataset(dc, 10, config.model)
        transformed = data_loader.transform_dataset(raw, dc)
        strict = JointV2Dataset(location['data'], split, 10)
        expected = int(validation['datasets'][split]['h10_windows'])
        assert len(raw) == len(strict) == expected and expected > 0
        indices = sorted({0, expected // 2, expected - 1})
        for index in indices:
            item = raw[index]
            assert np.asarray(item['state']).shape == (30,) and np.asarray(item['actions']).shape == (10, 14)
            check_transformed(transformed[index])
        record['datasets'][split] = {'windows': expected, 'checked_indices': indices, 'state_shape': [32], 'action_shape': [10, 32], 'quantile_norm': True}
    transform_files = [Path(__file__).with_name('dataset_v3.py'), Path(__file__).with_name('v3_config.py')]
    for relative in ('src/openpi/transforms.py', 'src/openpi/models/tokenizer.py', 'src/openpi/training/config.py'):
        candidate = location['openpi'] / relative
        if candidate.is_file():
            transform_files.append(candidate)
    record['transform_sha256'] = {str(path.relative_to(location['openpi'])) if path.is_relative_to(location['openpi']) else path.name: sha256(path) for path in transform_files}
    training_files = [Path(__file__), Path(__file__).with_name('v3_config.py'), Path(__file__).with_name('dataset_v3.py'), Path(__file__).with_name('lora_state_v3.py')]
    training_files.extend(location['openpi'] / relative for relative in (
        'scripts/train.py', 'src/openpi/training/optimizer.py', 'src/openpi/models/pi0.py',
        'src/openpi/models/pi0_config.py', 'src/openpi/models/gemma.py', 'src/openpi/models/lora.py',
        'src/openpi/models/model.py', 'src/openpi/shared/nnx_utils.py', 'src/openpi/training/weight_loaders.py'))
    record['training_code_sha256'] = {str(path.relative_to(location['openpi'])) if path.is_relative_to(location['openpi']) else path.name: sha256(path) for path in training_files}
    cache_contract = {key: record[key] for key in ('norm_sha256', 'validation_sha256', 'manifest_sha256', 'source_inventory_sha256', 'transform_sha256', 'openpi_commit')}
    builder_source = '\n'.join(inspect.getsource(function) for function in (prepare_cache, _prepare_cache_locked, check_transformed, cache_batches))
    cache_contract['cache_builder_code_sha256'] = hashlib.sha256(builder_source.encode()).hexdigest()
    cache_contract.update(eval_batch_size=args.eval_batch_size, seed=args.seed, train_probe_windows=min(128, record['datasets']['train']['windows']), validation_windows=record['datasets']['val']['windows'], augmentation=False)
    record['cache_contract'] = cache_contract
    record['cache_fingerprint'] = hashlib.sha256(json.dumps(cache_contract, sort_keys=True).encode()).hexdigest()
    assert record['datasets']['val']['windows'] == 3532 and cache_contract['train_probe_windows'] == 128
    record['cache_dir'] = str(location['runs'] / '_eval_cache' / record['cache_fingerprint'][:20])
    if write_report:
        report = args.report or location['runs'] / '_preflight' / f'{args.name}.json'
        atomic_json(report, record)
    return record


def prepare_cache(args, location, audit):
    directory = Path(audit['cache_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    # A shared fingerprint can be requested by CPU preparation and multiple
    # later GPU jobs. Serialize validation/resumption of the entire build.
    with (directory / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _prepare_cache_locked(args, location, audit)


def _prepare_cache_locked(args, location, audit):
    import numpy as np
    from openpi.training import data_loader
    directory = Path(audit['cache_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {'cache_contract': audit['cache_contract'], 'splits': {}, 'complete': False}
    assert manifest['cache_contract'] == audit['cache_contract']
    for split in ('val', 'train'):
        config = make_config(args, split)
        dc = config.data.create(config.assets_dirs, config.model)
        raw = data_loader.create_torch_dataset(dc, 10, config.model)
        transformed = data_loader.transform_dataset(raw, dc)
        indices = np.arange(len(raw)) if split == 'val' else np.unique(np.linspace(0, len(raw) - 1, min(128, len(raw)), dtype=int))
        folder = 'val_cache' if split == 'val' else 'train_probe_cache'
        (directory / folder).mkdir(exist_ok=True)
        entry = manifest['splits'].setdefault(split, {'windows': len(raw), 'cached_windows': len(indices), 'indices': indices.tolist(), 'files': {}})
        assert entry['indices'] == indices.tolist() and entry['windows'] == len(raw)
        for offset in range(0, len(indices), args.eval_batch_size):
            name = f'{folder}/batch_{offset:06d}.npz'
            path = directory / name
            if name in entry['files']:
                assert path.is_file() and sha256(path) == entry['files'][name], path
                continue
            selected = [transformed[int(index)] for index in indices[offset:offset + args.eval_batch_size]]
            count = len(selected)
            for sample in selected:
                check_transformed(sample)
            selected += [selected[-1]] * (args.eval_batch_size - count)
            batch = data_loader._collate_fn(selected)
            flat = {'count': np.int64(count), 'offset': np.int64(offset)}
            for key, value in batch.items():
                if isinstance(value, dict):
                    flat.update({key + '/' + child: array for child, array in value.items()})
                else:
                    flat[key] = value
            if 'estimated_bytes' not in manifest:
                one_batch = sum(np.asarray(value).nbytes for value in flat.values())
                batches = math.ceil(audit['datasets']['val']['windows'] / args.eval_batch_size) + math.ceil(min(128, audit['datasets']['train']['windows']) / args.eval_batch_size)
                manifest['estimated_bytes'] = int(one_batch * batches)
                manifest['free_bytes_before_cache'] = shutil.disk_usage(directory).free
                if manifest['estimated_bytes'] * 1.1 + 2**30 > manifest['free_bytes_before_cache']:
                    raise RuntimeError('Insufficient disk for complete deterministic evaluation cache plus 1 GiB headroom')
            temporary = path.with_suffix('.tmp.npz')
            np.savez(temporary, **flat)
            temporary.replace(path)
            entry['files'][name] = sha256(path)
            atomic_json(manifest_path, manifest)
            if offset % (args.eval_batch_size * 10) == 0:
                print(f'CACHE {split} {offset + count}/{len(indices)}', flush=True)
    manifest.update(complete=True, actual_bytes=sum((directory / filename).stat().st_size for entry in manifest['splits'].values() for filename in entry['files']), validation_augmentation=False)
    atomic_json(manifest_path, manifest)
    return manifest


def cache_batches(directory):
    import numpy as np
    from openpi.models.model import Observation
    for path in sorted(Path(directory).glob('batch_*.npz')):
        with np.load(path, allow_pickle=False) as file:
            values = {key: file[key] for key in file.files}
        count, offset = int(values.pop('count')), int(values.pop('offset'))
        data = {'image': {}, 'image_mask': {}}
        for key, value in values.items():
            if '/' in key:
                parent, child = key.split('/', 1)
                data[parent][child] = value
            else:
                data[key] = value
        actions = data.pop('actions')
        yield offset, count, (Observation.from_dict(data), actions)


def derive_loss_elements():
    from openpi.models import pi0
    tree = ast.parse(textwrap.dedent(inspect.getsource(pi0.Pi0.compute_loss)))
    function = tree.body[0]
    expected = ast.parse('jnp.mean(jnp.square(v_t - u_t), axis=-1)', mode='eval').body
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert len(returns) == 1 and ast.dump(returns[0].value) == ast.dump(expected), 'Official loss changed; review before training'
    returns[0].value = ast.parse('jnp.square(v_t - u_t)', mode='eval').body
    function.name, function.decorator_list, function.returns = 'loss_elements', [], None
    ast.fix_missing_locations(tree)
    namespace = dict(vars(pi0))
    exec(compile(tree, '<official_pi0_unreduced_loss>', 'exec'), namespace)
    return namespace['loss_elements']


def scalar_metrics(values):
    import numpy as np
    result = {'loss': float(np.mean(values)), 'real14_loss': float(np.mean(values[:14]))}
    result.update({name + '_loss': float(np.mean(values[left:right])) for name, left, right in [('go2', 0, 3), ('piper', 3, 10), ('uav', 10, 14)]})
    return result


def execute(args, location, audit):
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(int(args.gpu)) if args.gpu.isdigit() else nv.nvmlDeviceGetHandleByUUID(args.gpu)
    total_memory = int(nv.nvmlDeviceGetMemoryInfo(handle).total)
    minimum_free = max(60 * 2**30, int(0.90 * total_memory) + 2**30) if args.preallocate_for_handoff else 60 * 2**30
    if nv.nvmlDeviceGetMemoryInfo(handle).free < minimum_free:
        raise RuntimeError('Batch32 full finetuning requires at least 60 GiB currently free; no automatic batch reduction')
    for process in nv.nvmlDeviceGetComputeRunningProcesses(handle):
        if args.preallocate_for_handoff and process.pid != os.getpid():
            raise RuntimeError('Handoff reservation refuses any competing GPU compute process')
        try:
            if process.pid != os.getpid() and Path(f'/proc/{process.pid}').stat().st_uid == os.getuid():
                raise RuntimeError('Another own GPU process is active on the assigned GPU')
        except FileNotFoundError:
            pass
    reservation = None
    if args.preallocate_for_handoff:
        # Reserve before importing the remaining training stack. This is only
        # reached after the external supervisor released the old GPU process.
        import jax
        import numpy as np
        reservation = jax.device_put(np.zeros((1,), dtype=np.float32))
        jax.block_until_ready(reservation)
        own_reserved = 0
        for _ in range(25):
            own_reserved = sum(int(process.usedGpuMemory) for process in nv.nvmlDeviceGetComputeRunningProcesses(handle)
                               if process.pid == os.getpid() and 0 <= int(process.usedGpuMemory) <= total_memory)
            if own_reserved >= int(0.80 * total_memory):
                break
            time.sleep(0.2)
        if own_reserved < int(0.80 * total_memory):
            raise RuntimeError('Explicit handoff preallocation did not establish the expected allocator reservation')
        print(json.dumps({'status': 'gpu_allocator_reserved', 'pid': os.getpid(), 'gpu': args.gpu,
                          'own_nvml_bytes': own_reserved, 'allocator_fraction': 0.90}), flush=True)
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import wandb
    from openpi.shared import array_typing as at
    from openpi.training import checkpoints as ckpt, data_loader, sharding
    # The handoff exec starts a clean interpreter after the CPU audit.
    install_dataset_adapter(location['data'])
    config = make_config(args)
    directory = location['run']
    if directory.exists() and not args.resume:
        raise FileExistsError(f'Use a new run name; refusing to overwrite {directory}')
    if args.resume and not (directory / 'config.json').is_file():
        raise FileNotFoundError('This V2 run has no recorded configuration to resume')
    directory.mkdir(parents=True, exist_ok=True)
    contract = {**plan(args, location), **{key: audit[key] for key in ('norm_sha256', 'manifest_sha256', 'source_inventory_sha256', 'base_metadata_sha256', 'openpi_commit', 'transform_sha256', 'training_code_sha256', 'cache_fingerprint')}}
    if args.warm_start_params:
        contract['warm_start_provenance'] = audit['warm_start_provenance']
    mutable = {'resume', 'initialization'}
    if args.resume:
        original = read_json(directory / 'config.json')
        assert {key: value for key, value in original.items() if key not in mutable} == {key: value for key, value in contract.items() if key not in mutable}, 'Resume configuration or V2 data changed'
    else:
        atomic_json(directory / 'config.json', contract)
    atomic_json(directory / 'effective_config.json', contract)
    # Official initializer is source-pinned for comparison, never called: its
    # frozen-weight bf16 cast would change the dense V2 starting function.
    loss_elements = derive_loss_elements()
    assert jax.device_count() == 1 and jax.devices()[0].platform == 'gpu'
    started = time.monotonic()
    prior_seconds = read_json(directory / 'status.json').get('elapsed_seconds', 0.0) if args.resume and (directory / 'status.json').exists() else 0.0
    offset, phase_target = source_step(args), phase_steps(args)
    status = {'name': args.name, 'gpu': args.gpu, 'pid': os.getpid(), 'status': 'initializing', 'step': offset}

    def publish(**values):
        if 'step' in values:
            if offset:
                values['phase_step'] = int(values['step'])
                values['global_step'] = offset + int(values['step'])
            values['step'] = offset + int(values['step'])
        status.update(values, elapsed_seconds=prior_seconds + time.monotonic() - started)
        atomic_json(directory / 'status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    def interrupted(signum, frame):
        raise RuntimeError(f'Run interrupted by signal {signum}; resume from the last regular checkpoint')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    managers, wb = [], None
    try:
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        manager, resuming = ckpt.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=args.keep_period, overwrite=False, resume=args.resume)
        managers.append(manager)
        if args.resume and not resuming:
            raise RuntimeError('No completed regular V2 checkpoint exists for --resume')
        best_manager, _ = ckpt.initialize_checkpoint_dir(directory / 'best', keep_period=None, overwrite=False, resume=args.resume)
        managers.append(best_manager)
        (directory / 'wandb').mkdir(exist_ok=True)
        run_id_path = directory / 'wandb_id.txt'
        wb = wandb.init(project=config.project_name, name=args.name, mode='offline', dir=str(directory / 'wandb'),
                        id=run_id_path.read_text().strip() if args.resume and run_id_path.exists() else None, config=contract)
        run_id_path.write_text(wb.id)
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        iterator = iter(loader)
        train_rng, init_rng = jax.random.split(jax.random.key(args.seed))
        publish(status='checking_parameter_shapes')
        state, state_sharding = init_lora_train_state(config, init_rng, mesh, resume=True)
        atomic_json(directory / 'parameter_report.json', parameter_report(state.params.flat_state(), state.params.filter(config.trainable_filter).flat_state()))
        batch = next(iterator)
        jax.block_until_ready(batch)

        def train_step(rng, state, batch):
            model = nnx.merge(state.model_def, state.params)
            model.train()

            def objective(model, key, observation, actions):
                errors = loss_elements(model, key, observation, actions, train=True)
                return errors.mean(axis=-1).mean(), errors.mean(axis=(0, 1))

            (loss, dimensions), grads = nnx.value_and_grad(objective, argnums=nnx.DiffState(0, config.trainable_filter), has_aux=True)(model, jax.random.fold_in(rng, state.step), *batch)
            params = state.params.filter(config.trainable_filter)
            updates, opt_state = state.tx.update(grads, state.opt_state, params)
            params = optax.apply_updates(params, updates)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves((params, opt_state))]))
            nnx.update(model, params)
            return dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=opt_state), {'loss': loss, 'dimensions': dimensions, 'grad_norm': optax.global_norm(grads), 'finite_update': finite}

        train_fn = jax.jit(train_step, in_shardings=(replicated, state_sharding, data_sharding), out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        compiled_structure, compiled_tx = jax.tree.structure(state), state.tx
        publish(status='compiling_train', step=0)
        with sharding.set_mesh(mesh), at.disable_typechecking():
            executable = train_fn.lower(train_rng, state, batch).compile()
        memory = executable.memory_analysis()
        demand = memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes
        required = demand + int(1.5 * 2**30)
        budget = int(0.90 * nv.nvmlDeviceGetMemoryInfo(handle).total - 2 * 2**30)
        atomic_json(directory / 'compiled_memory.json', {'live_estimate_bytes': demand, 'required_free_before_weights_bytes': required, 'safe_budget_bytes': budget, 'weight_allocation_started': False})
        free_memory = int(nv.nvmlDeviceGetMemoryInfo(handle).free)
        own_memory = sum(int(process.usedGpuMemory) for process in nv.nvmlDeviceGetComputeRunningProcesses(handle)
                         if process.pid == os.getpid() and 0 <= int(process.usedGpuMemory) <= total_memory)
        usable_memory = free_memory + own_memory if args.preallocate_for_handoff else free_memory
        if args.preallocate_for_handoff and any(process.pid != os.getpid() for process in nv.nvmlDeviceGetComputeRunningProcesses(handle)):
            raise RuntimeError('A competing compute process appeared after handoff reservation')
        if demand > budget or usable_memory < required:
            raise RuntimeError('Compiled demand plus headroom exceeds currently available GPU memory; weights remain unloaded')
        atomic_json(directory / 'compiled_memory.json', {'live_estimate_bytes': demand, 'required_free_before_weights_bytes': required, 'safe_budget_bytes': budget,
                    'external_free_bytes': free_memory, 'own_reserved_bytes': own_memory,
                    'usable_including_own_reservation_bytes': usable_memory, 'weight_allocation_started': True})
        publish(status='restoring_v2_checkpoint' if resuming else ('loading_parent_best_params' if offset else 'loading_official_pi05_base'))
        state, state_sharding = init_lora_train_state(config, init_rng, mesh, resume=resuming, weights_path=location['weights'], proof_path=directory / 'dense_load_proof.json')
        if resuming:
            state = ckpt.restore_state(manager, state, loader)
        # The optimizer closure is static pytree metadata: equality of settings
        # is insufficient. Reuse the exact object captured by abstract compile.
        state = dataclasses.replace(state, tx=compiled_tx)
        assert jax.tree.structure(state) == compiled_structure, 'Compiled and actual TrainState pytree mismatch'
        jax.block_until_ready(state)
        frozen_initial_sha = frozen_digest(state.params)
        atomic_json(directory / 'frozen_parameters.json', {'initial_sha256': frozen_initial_sha, 'all_frozen_fp32': True, 'first_update_verified': False})
        start = int(state.step)
        if not resuming:
            assert start == 0, 'Fresh V2 must start from base weights and a new step-zero optimizer'
        else:
            assert offset + start == int(manager.latest_step()) and start <= phase_target
        state_record = {'pytree_structure_equal': True, 'optimizer_static_object_reused': True, 'optimizer_hyperparameters_changed': False, 'from_step': start, 'fresh_optimizer_step0': not resuming}
        if offset:
            state_record.update(source_step=offset, phase_step=start, global_step=offset + start, target_global_step=args.steps, phase_target_steps=phase_target)
        atomic_json(directory / 'compiled_state_contract.json', state_record)

        def evaluation_elements(params, rng, observation, actions):
            model = nnx.merge(state.model_def, params)
            model.eval()
            return loss_elements(model, rng, observation, actions, train=False).mean(axis=1)

        eval_fn = jax.jit(evaluation_elements)
        cache = Path(audit['cache_dir'])

        def evaluate(folder):
            total, count = np.zeros(32, dtype=np.float64), 0
            for offset, n, values in cache_batches(cache / folder):
                dimensions = np.asarray(eval_fn(state.params, jax.random.fold_in(jax.random.key(args.seed), offset), *values))
                if not np.isfinite(dimensions).all():
                    raise FloatingPointError('Non-finite deterministic evaluation loss')
                total += dimensions[:n].sum(axis=0, dtype=np.float64)
                count += n
            assert count == (audit['datasets']['val']['windows'] if folder == 'val_cache' else min(128, audit['datasets']['train']['windows']))
            return {**scalar_metrics(total / count), 'windows': count}

        control = {'best': None, 'best_step': None, 'best_checkpoint': None, 'evaluations': []}
        if resuming:
            control = read_json(directory / f'control_{offset + start}.json')
            if (directory / 'best.json').exists():
                global_best = read_json(directory / 'best.json')
                if control['best'] is None or global_best['best'] < control['best']:
                    for field in ('best', 'best_step', 'best_phase_step', 'best_checkpoint', 'best_metrics', 'best_train_probe', 'best_checkpoint_tag'):
                        if field in global_best:
                            control[field] = global_best[field]
            publish(status='restoring_data_position', step=start)
            # The first shuffled batch was already consumed for compilation.
            for _ in range(max(start - 1, 0)):
                next(iterator)
            batch = next(iterator)

        def save(which, checkpoint_tag):
            ckpt.save_state(which, state, loader, checkpoint_tag)
            which.wait_until_finished()
            which.check_for_errors()

        def validate(step):
            beginning = time.monotonic()
            validation, probe = evaluate('val_cache'), evaluate('train_probe_cache')
            value = validation['real14_loss']
            global_step = offset + step
            evaluation_record = {'step': global_step, 'val_real14_loss': value, 'train_probe_real14_loss': probe['real14_loss']}
            if offset:
                evaluation_record['phase_step'] = step
            control['evaluations'].append(evaluation_record)
            if control['best'] is None or value < control['best']:
                # On resume, a valid best checkpoint can be newer than the last
                # recovery save. A monotonically increasing save tag avoids
                # collisions while best_step records the actual optimizer step.
                last_best_tag = best_manager.latest_step()
                tag = max(global_step, int(last_best_tag) + 1 if last_best_tag is not None else global_step)
                save(best_manager, tag)
                control.update(best=value, best_step=global_step, best_metrics=validation, best_train_probe=probe, best_checkpoint_tag=tag, best_checkpoint=str(directory / 'best' / str(tag)))
                if offset:
                    control['best_phase_step'] = step
                atomic_json(directory / 'best.json', control)
            return {**{'val/' + key: value for key, value in validation.items()}, **{'train_probe/' + key: value for key, value in probe.items()}, 'eval_seconds': time.monotonic() - beginning}

        def emit(step, metrics):
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError('Non-finite logged metric')
            entry = {'step': offset + step, **metrics, 'elapsed_seconds': prior_seconds + time.monotonic() - started}
            if offset:
                entry.update(phase_step=step, global_step=offset + step)
            with (directory / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
            wb.log(entry, step=offset + step)

        if not resuming:
            _, _, check_batch = next(cache_batches(cache / 'val_cache'))
            key = jax.random.key(args.seed)

            def official_loss(params, key, observation, actions):
                model = nnx.merge(state.model_def, params)
                model.eval()
                return model.compute_loss(key, observation, actions, train=False).mean()

            official_value = float(jax.jit(official_loss)(state.params, key, *check_batch))
            instrumented_value = float(np.asarray(eval_fn(state.params, key, *check_batch)).mean())
            np.testing.assert_allclose(official_value, instrumented_value, rtol=1e-5, atol=1e-6)
            atomic_json(directory / 'loss_equivalence.json', {'official': official_value, 'instrumented': instrumented_value})
            publish(status='validating_step0', step=0)
            initial_metrics = validate(0)
            if offset:
                expected = audit['warm_start_provenance']['expected_initial_val_real14_loss']
                np.testing.assert_allclose(initial_metrics['val/real14_loss'], expected, rtol=1e-4, atol=1e-6,
                                           err_msg='Warm-start full validation differs from the parent best checkpoint')
                expected_metrics = audit['warm_start_provenance']['expected_initial_val_metrics']
                comparisons = {}
                for metric in ('loss', 'real14_loss', 'go2_loss', 'piper_loss', 'uav_loss'):
                    actual_value = initial_metrics['val/' + metric]
                    np.testing.assert_allclose(actual_value, expected_metrics[metric], rtol=1e-4, atol=1e-6,
                                               err_msg=f'Zero-LoRA parent validation differs: {metric}')
                    comparisons[metric] = {'parent': expected_metrics[metric], 'zero_lora': actual_value}
                atomic_json(directory / 'warm_start_equivalence.json', {'parent_run': args.parent_run, 'source_step': offset,
                            'phase_step': 0, 'global_step': offset, 'parent_val_real14_loss': expected,
                            'new_initial_val_real14_loss': initial_metrics['val/real14_loss'], 'full_validation_matched': True,
                            'all_metric_comparisons': comparisons, 'windows': 3532, 'rtol': 1e-4, 'atol': 1e-6,
                            'params_sha256': audit['warm_start_provenance']['params_sha256'], 'optimizer_was_reset': True})
            emit(0, initial_metrics)
        accum = []
        publish(status='training', step=start)
        for step in range(start + 1, phase_target + 1):
            tick = time.monotonic()
            if step != start + 1:
                batch = next(iterator)
            jax.block_until_ready(batch)
            data_wait = time.monotonic() - tick
            if not bool(jnp.isfinite(batch[1]).all()) or not bool(jnp.isfinite(batch[0].state).all()):
                raise FloatingPointError('Non-finite training data')
            state, info = executable(train_rng, state, batch)
            jax.block_until_ready((state, info))
            if step == start + 1:
                current_frozen_sha = frozen_digest(state.params)
                assert current_frozen_sha == frozen_initial_sha, 'Frozen ViT/Gemma changed after first update'
                atomic_json(directory / 'frozen_parameters.json', {'initial_sha256': frozen_initial_sha, 'after_first_update_sha256': current_frozen_sha, 'first_update_verified': True, 'all_frozen_fp32': True})
            host = jax.device_get(info)
            if not bool(host['finite_update']) or not all(np.isfinite(value).all() for value in host.values()):
                raise FloatingPointError('Non-finite loss, gradient, model parameters or optimizer state')
            accum.append({'dimensions': np.asarray(host['dimensions']), 'loss': float(host['loss']), 'grad_norm': float(host['grad_norm']), 'step_seconds': time.monotonic() - tick, 'data_wait_seconds': data_wait})
            metrics = {}
            if (offset + step) % args.log_interval == 0 or step == phase_target:
                metrics.update({'train/' + key: value for key, value in scalar_metrics(np.mean([entry['dimensions'] for entry in accum], axis=0)).items()})
                metrics.update({'train/' + key: float(np.mean([entry[key] for entry in accum])) for key in ('loss', 'grad_norm', 'step_seconds', 'data_wait_seconds')})
                metrics['lr'] = float(config.lr_schedule.create()(step - 1))
                accum = []
            if (offset + step) % args.eval_interval == 0 or step == phase_target:
                metrics.update(validate(step))
            if metrics:
                emit(step, metrics)
                publish(status='training', step=step, best_val=control['best'], best_step=control['best_step'], best_checkpoint=control['best_checkpoint'])
            if (offset + step) % args.save_interval == 0 or step == phase_target:
                atomic_json(directory / f'control_{offset + step}.json', control)
                save(manager, offset + step)
        assert frozen_digest(state.params) == frozen_initial_sha, 'Frozen ViT/Gemma changed during training'
        atomic_json(directory / 'frozen_parameters.json', {'initial_sha256': frozen_initial_sha, 'first_update_verified': True, 'final_all_frozen_exactly_unchanged': True, 'all_frozen_fp32': True})
        assert int(state.step) == phase_target
        publish(status='completed', step=int(state.step), stop_reason=f'completed_requested_{args.steps}_steps', best_val=control['best'], best_step=control['best_step'], best_checkpoint=control['best_checkpoint'], final_checkpoint=str(config.checkpoint_dir / str(args.steps)))
        wb.summary.update(status)
        wb.finish()
    except BaseException as error:
        publish(status='failed', stop_reason=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if wb is not None:
            wb.summary.update(status)
            wb.finish(exit_code=1)
        raise
    finally:
        for manager in reversed(managers):
            manager.close()
        nv.nvmlShutdown()


def handoff_ready_path(args):
    return Path(str(args.handoff_release_file.expanduser().resolve()) + '.ready.json')


def wait_for_handoff(args, audit):
    release = args.handoff_release_file.expanduser().resolve()
    ready_path = handoff_ready_path(args)
    atomic_json(ready_path, {'pid': os.getpid(), 'name': args.name, 'release_file': str(release),
                            'audit': audit, 'ready_time': time.time(), 'gpu_initialized': False})
    print(json.dumps({'status': 'cpu_ready_waiting_for_handoff', 'pid': os.getpid(), 'ready_file': str(ready_path)}), flush=True)
    deadline = time.monotonic() + args.handoff_timeout
    while not release.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('External supervisor did not release GPU handoff before the timeout')
        time.sleep(0.2)
    # CPU imports may already have initialized JAX's CPU backend. A same-PID
    # exec resets runtime selection cleanly and preserves inherited lock FDs.
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--_handoff-audit-file', str(ready_path)]
    os.execve(sys.executable, command, dict(os.environ))


def main():
    args = arguments()
    location = configure_environment(args)
    logging.basicConfig(level=logging.INFO, force=True)
    if not (args.execute or args.preflight or args.prepare_cache):
        print(json.dumps(plan(args, location), ensure_ascii=False, indent=2))
        print('CONFIGURATION ONLY. Use --preflight or --prepare-cache on CPU; training requires --execute --gpu --name.')
        return
    if args._handoff_audit_file:
        assert args._handoff_audit_file.resolve() == handoff_ready_path(args)
        ready = read_json(args._handoff_audit_file)
        assert ready['pid'] == os.getpid() and ready['name'] == args.name
        assert args.handoff_release_file.expanduser().resolve().is_file()
        audit = ready['audit']
        for key, value in plan(args, location).items():
            assert audit[key] == value, f'Handoff argument changed after CPU audit: {key}'
        assert audit['training_code_sha256']['train_lora_v3.py'] == sha256(__file__)
        assert audit['training_code_sha256']['v3_config.py'] == sha256(Path(__file__).with_name('v3_config.py'))
        assert audit['training_code_sha256']['lora_state_v3.py'] == sha256(Path(__file__).with_name('lora_state_v3.py'))
        assert audit['training_code_sha256']['dataset_v3.py'] == sha256(Path(__file__).with_name('dataset_v3.py'))
        assert check_runtime_contract(location['openpi']) == audit['runtime_source_contract']
        released = read_json(args.handoff_release_file)
        assert released['pid'] == os.getpid() and released['gpu'] == args.gpu, 'Handoff release targets another trainer/device'
    else:
        audit = cpu_audit(args, location, write_report=not args.execute)
        if args.prepare_cache or args.execute:
            if args.warm_start_params and not args.reuse_eval_cache:
                parent = read_json(location['parent_runs'] / args.parent_run / 'config.json')
                args.reuse_eval_cache = location['parent_runs'] / '_eval_cache' / parent['cache_fingerprint'][:20]
            reuse_eval_cache(args, audit)
            cache = prepare_cache(args, location, audit)
            audit.update(cache_prepared=True, cache_actual_bytes=cache['actual_bytes'], cache_estimated_bytes=cache['estimated_bytes'])
    if not args.execute:
        report = args.report or location['runs'] / '_preflight' / f'{args.name}.json'
        atomic_json(report, audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    if args.handoff_release_file and not args._handoff_audit_file:
        wait_for_handoff(args, audit)
    execute(args, location, audit)


if __name__ == '__main__':
    main()
