#!/usr/bin/env python3
"""V2 full finetuning: CPU preflight/cache preparation or explicit single-GPU execution."""
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
import textwrap
import time
import traceback

from v2_config import install_dataset_adapter, make_config, paths


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='CPU-only config, local weight metadata and real V2 loader checks')
    mode.add_argument('--prepare-cache', action='store_true', help='CPU preflight plus complete deterministic val/train-probe cache')
    mode.add_argument('--execute', action='store_true', help='Actually allocate the assigned GPU and train; never implicit')
    parser.add_argument('--gpu', '--gpu-uuid', dest='gpu', help='One physical GPU index or UUID, required only with --execute')
    parser.add_argument('--name', default='preflight', help='Unique V2 run name; required for execution')
    parser.add_argument('--resume', action='store_true', help='Resume this V2 run regular checkpoint, never a V1 checkpoint')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--base-params', type=Path)
    parser.add_argument('--run-root', type=Path, help='Parent directory for V2 run metadata and shared evaluation caches')
    parser.add_argument('--report', type=Path, help='Optional path for the CPU preflight JSON report')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--peak-lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--eval-interval', type=int, default=200)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--keep-period', type=int, default=5000)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name):
        parser.error('--name must be one safe directory name')
    if args.execute and (not args.gpu or args.name == 'preflight'):
        parser.error('--execute requires --gpu and a unique --name')
    if args.gpu and not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)', args.gpu):
        parser.error('--gpu must select exactly one GPU index or UUID')
    if args.resume and not args.execute:
        parser.error('--resume requires --execute')
    for key in ('batch_size', 'eval_interval', 'eval_batch_size', 'save_interval', 'keep_period', 'log_interval'):
        if getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.steps <= 100 or args.peak_lr <= 0 or not math.isfinite(args.peak_lr) or args.num_workers < 0:
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
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu if args.execute else '',
        JAX_PLATFORMS='cuda' if args.execute else 'cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
        WANDB_MODE='offline', WANDB_DISABLE_CODE='true', WANDB_SILENT='true',
        WANDB_GROUP='pi05-v2-full-20000', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_LEROBOT_HOME=str(location['data'] / 'lerobot'))
    return location


def plan(args, location):
    return {
        'name': args.name, 'mode': 'execute' if args.execute else ('prepare_cache' if args.prepare_cache else 'preflight'),
        'variant': 'full', 'batch_size': args.batch_size, 'peak_lr': args.peak_lr,
        'warmup_steps': 100, 'decay_lr': args.peak_lr / 10, 'steps': args.steps, 'seed': args.seed,
        'optimizer': {'name': 'AdamW', 'b1': 0.9, 'b2': 0.95, 'eps': 1e-8, 'weight_decay': 1e-10, 'clip_gradient_norm': 1.0},
        'remat_policy': 'nothing_saveable', 'parameter_dtype': 'float32', 'compute_dtype': 'bfloat16',
        'ema_decay': None, 'full_unfreeze': True, 'lora': False, 'early_stopping': False,
        'initialization': 'official_pi05_base_new_optimizer_step0' if not args.resume else 'same_V2_regular_checkpoint',
        'wandb_mode': 'offline', 'eval_interval': args.eval_interval, 'eval_batch_size': args.eval_batch_size,
        'save_interval': args.save_interval, 'keep_period': args.keep_period,
        'log_interval': args.log_interval, 'num_workers': args.num_workers,
        'data_root': str(location['data']), 'base_params': str(location['weights']),
        'openpi_root': str(location['openpi']), 'run_dir': str(location['run']),
        'checkpoint_dir': str(location['checkpoint_base'] / 'pi05_heterovla_v2_full' / args.name),
        'best_dir': str(location['run'] / 'best'), 'resume': args.resume,
        'objective': 'official mean flow matching loss over all 32 dimensions; real14 metrics diagnostic',
    }


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
    """No model.create/eval_shape/init_state, gradient, NVML or device enumeration."""
    import numpy as np
    from openpi.training import data_loader
    from dataset_v2 import JointV2Dataset
    assert not args.execute or os.environ['JAX_PLATFORMS'] == 'cuda'
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
    assert (location['weights'] / '_METADATA').is_file(), 'Local official pi05_base params metadata missing'
    assert (location['openpi'] / 'scripts/train.py').is_file()
    # Inspect local checkpoint metadata only, never load multi-GB tensor arrays.
    metadata = read_json(location['weights'] / '_METADATA')
    assert metadata.get('tree_metadata'), 'Invalid local base parameter metadata'
    install_dataset_adapter(location['data'])
    record = plan(args, location)
    record.update(status='passed', training_started=False, gradient_updates=0,
                  gpu_enumerated=False, model_instantiated=False, pretrained_arrays_loaded=False,
                  norm_sha256=sha256(norm_path), validation_sha256=sha256(validation_path),
                  manifest_sha256=sha256(location['data'] / 'reports/episode_manifest.json'),
                  source_inventory_sha256=sha256(location['data'] / 'configs/source_inventory.json'),
                  base_metadata_sha256=sha256(location['weights'] / '_METADATA'), datasets={})
    record['openpi_commit'] = subprocess.check_output(['git', '-C', str(location['openpi']), 'rev-parse', 'HEAD'], text=True).strip()
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
        assert config.model.paligemma_variant == 'gemma_2b' and config.model.action_expert_variant == 'gemma_300m'
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
    transform_files = [Path(__file__).with_name('dataset_v2.py'), Path(__file__).with_name('v2_config.py')]
    for relative in ('src/openpi/transforms.py', 'src/openpi/models/tokenizer.py', 'src/openpi/training/config.py'):
        candidate = location['openpi'] / relative
        if candidate.is_file():
            transform_files.append(candidate)
    record['transform_sha256'] = {str(path.relative_to(location['openpi'])) if path.is_relative_to(location['openpi']) else path.name: sha256(path) for path in transform_files}
    training_files = [Path(__file__), Path(__file__).with_name('v2_config.py'), Path(__file__).with_name('dataset_v2.py')]
    training_files.extend(location['openpi'] / relative for relative in (
        'scripts/train.py', 'src/openpi/training/optimizer.py', 'src/openpi/models/pi0.py',
        'src/openpi/models/pi0_config.py', 'src/openpi/models/gemma.py'))
    record['training_code_sha256'] = {str(path.relative_to(location['openpi'])) if path.is_relative_to(location['openpi']) else path.name: sha256(path) for path in training_files}
    cache_contract = {key: record[key] for key in ('norm_sha256', 'validation_sha256', 'manifest_sha256', 'source_inventory_sha256', 'transform_sha256', 'openpi_commit')}
    builder_source = '\n'.join(inspect.getsource(function) for function in (prepare_cache, _prepare_cache_locked, check_transformed, cache_batches))
    cache_contract['cache_builder_code_sha256'] = hashlib.sha256(builder_source.encode()).hexdigest()
    cache_contract.update(eval_batch_size=args.eval_batch_size, seed=args.seed, train_probe_windows=min(128, record['datasets']['train']['windows']), validation_windows=record['datasets']['val']['windows'], augmentation=False)
    record['cache_contract'] = cache_contract
    record['cache_fingerprint'] = hashlib.sha256(json.dumps(cache_contract, sort_keys=True).encode()).hexdigest()
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
    if nv.nvmlDeviceGetMemoryInfo(handle).free < 60 * 2**30:
        raise RuntimeError('Batch32 full finetuning requires at least 60 GiB currently free; no automatic batch reduction')
    for process in nv.nvmlDeviceGetComputeRunningProcesses(handle):
        try:
            if process.pid != os.getpid() and Path(f'/proc/{process.pid}').stat().st_uid == os.getuid():
                raise RuntimeError('Another own GPU process is active on the assigned GPU')
        except FileNotFoundError:
            pass
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import wandb
    from openpi.shared import array_typing as at
    from openpi.training import checkpoints as ckpt, data_loader, sharding
    config = make_config(args)
    directory = location['run']
    if directory.exists() and not args.resume:
        raise FileExistsError(f'Use a new run name; refusing to overwrite {directory}')
    if args.resume and not (directory / 'config.json').is_file():
        raise FileNotFoundError('This V2 run has no recorded configuration to resume')
    directory.mkdir(parents=True, exist_ok=True)
    contract = {**plan(args, location), **{key: audit[key] for key in ('norm_sha256', 'manifest_sha256', 'source_inventory_sha256', 'base_metadata_sha256', 'openpi_commit', 'transform_sha256', 'training_code_sha256', 'cache_fingerprint')}}
    mutable = {'resume', 'initialization'}
    if args.resume:
        original = read_json(directory / 'config.json')
        assert {key: value for key, value in original.items() if key not in mutable} == {key: value for key, value in contract.items() if key not in mutable}, 'Resume configuration or V2 data changed'
    else:
        atomic_json(directory / 'config.json', contract)
    atomic_json(directory / 'effective_config.json', contract)
    spec = importlib.util.spec_from_file_location('official_v2_train', location['openpi'] / 'scripts/train.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    loss_elements = derive_loss_elements()
    assert jax.device_count() == 1 and jax.devices()[0].platform == 'gpu'
    started = time.monotonic()
    prior_seconds = read_json(directory / 'status.json').get('elapsed_seconds', 0.0) if args.resume and (directory / 'status.json').exists() else 0.0
    status = {'name': args.name, 'gpu': args.gpu, 'pid': os.getpid(), 'status': 'initializing', 'step': 0}

    def publish(**values):
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
        state, state_sharding = official.init_train_state(config, init_rng, mesh, resume=True)
        flat = state.params.flat_state()
        trainable = state.params.filter(config.trainable_filter).flat_state()
        assert set(flat) == set(trainable) and not any('lora' in '/'.join(map(str, key)).lower() for key in flat)
        assert {str(leaf.value.dtype) for leaf in flat.values()} == {'float32'}
        total = sum(math.prod(leaf.value.shape) for leaf in flat.values())
        atomic_json(directory / 'parameter_report.json', {'total_parameters': total, 'trainable_parameters': total, 'parameter_dtypes': ['float32'], 'full_unfreeze': True, 'lora_parameters': 0})
        del flat, trainable
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
        if demand > budget or nv.nvmlDeviceGetMemoryInfo(handle).free < required:
            raise RuntimeError('Compiled demand plus headroom exceeds currently available GPU memory; weights remain unloaded')
        atomic_json(directory / 'compiled_memory.json', {'live_estimate_bytes': demand, 'required_free_before_weights_bytes': required, 'safe_budget_bytes': budget, 'weight_allocation_started': True})
        publish(status='restoring_v2_checkpoint' if resuming else 'loading_official_pi05_base')
        state, state_sharding = official.init_train_state(config, init_rng, mesh, resume=resuming)
        if resuming:
            state = ckpt.restore_state(manager, state, loader)
        # The optimizer closure is static pytree metadata: equality of settings
        # is insufficient. Reuse the exact object captured by abstract compile.
        state = dataclasses.replace(state, tx=compiled_tx)
        assert jax.tree.structure(state) == compiled_structure, 'Compiled and actual TrainState pytree mismatch'
        jax.block_until_ready(state)
        start = int(state.step)
        if not resuming:
            assert start == 0, 'Fresh V2 must start from base weights and a new step-zero optimizer'
        else:
            assert start == int(manager.latest_step()) and start <= args.steps
        atomic_json(directory / 'compiled_state_contract.json', {'pytree_structure_equal': True, 'optimizer_static_object_reused': True, 'optimizer_hyperparameters_changed': False, 'from_step': start, 'fresh_optimizer_step0': not resuming})

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
            control = read_json(directory / f'control_{start}.json')
            if (directory / 'best.json').exists():
                global_best = read_json(directory / 'best.json')
                if control['best'] is None or global_best['best'] < control['best']:
                    for field in ('best', 'best_step', 'best_checkpoint', 'best_metrics', 'best_train_probe', 'best_checkpoint_tag'):
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
            control['evaluations'].append({'step': step, 'val_real14_loss': value, 'train_probe_real14_loss': probe['real14_loss']})
            if control['best'] is None or value < control['best']:
                # On resume, a valid best checkpoint can be newer than the last
                # recovery save. A monotonically increasing save tag avoids
                # collisions while best_step records the actual optimizer step.
                last_best_tag = best_manager.latest_step()
                tag = max(step, int(last_best_tag) + 1 if last_best_tag is not None else step)
                save(best_manager, tag)
                control.update(best=value, best_step=step, best_metrics=validation, best_train_probe=probe, best_checkpoint_tag=tag, best_checkpoint=str(directory / 'best' / str(tag)))
                atomic_json(directory / 'best.json', control)
            return {**{'val/' + key: value for key, value in validation.items()}, **{'train_probe/' + key: value for key, value in probe.items()}, 'eval_seconds': time.monotonic() - beginning}

        def emit(step, metrics):
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError('Non-finite logged metric')
            entry = {'step': step, **metrics, 'elapsed_seconds': prior_seconds + time.monotonic() - started}
            with (directory / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
            wb.log(entry, step=step)

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
            emit(0, validate(0))
        accum = []
        publish(status='training', step=start)
        for step in range(start + 1, args.steps + 1):
            tick = time.monotonic()
            if step != start + 1:
                batch = next(iterator)
            jax.block_until_ready(batch)
            data_wait = time.monotonic() - tick
            if not bool(jnp.isfinite(batch[1]).all()) or not bool(jnp.isfinite(batch[0].state).all()):
                raise FloatingPointError('Non-finite training data')
            state, info = executable(train_rng, state, batch)
            jax.block_until_ready((state, info))
            host = jax.device_get(info)
            if not bool(host['finite_update']) or not all(np.isfinite(value).all() for value in host.values()):
                raise FloatingPointError('Non-finite loss, gradient, model parameters or optimizer state')
            accum.append({'dimensions': np.asarray(host['dimensions']), 'loss': float(host['loss']), 'grad_norm': float(host['grad_norm']), 'step_seconds': time.monotonic() - tick, 'data_wait_seconds': data_wait})
            metrics = {}
            if step % args.log_interval == 0 or step == args.steps:
                metrics.update({'train/' + key: value for key, value in scalar_metrics(np.mean([entry['dimensions'] for entry in accum], axis=0)).items()})
                metrics.update({'train/' + key: float(np.mean([entry[key] for entry in accum])) for key in ('loss', 'grad_norm', 'step_seconds', 'data_wait_seconds')})
                metrics['lr'] = float(config.lr_schedule.create()(step - 1))
                accum = []
            if step % args.eval_interval == 0 or step == args.steps:
                metrics.update(validate(step))
            if metrics:
                emit(step, metrics)
                publish(status='training', step=step, best_val=control['best'], best_step=control['best_step'], best_checkpoint=control['best_checkpoint'])
            if step % args.save_interval == 0 or step == args.steps:
                atomic_json(directory / f'control_{step}.json', control)
                save(manager, step)
        assert int(state.step) == args.steps
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


def main():
    args = arguments()
    location = configure_environment(args)
    logging.basicConfig(level=logging.INFO, force=True)
    if not (args.execute or args.preflight or args.prepare_cache):
        print(json.dumps(plan(args, location), ensure_ascii=False, indent=2))
        print('CONFIGURATION ONLY. Use --preflight or --prepare-cache on CPU; training requires --execute --gpu --name.')
        return
    audit = cpu_audit(args, location, write_report=not args.execute)
    if args.prepare_cache or args.execute:
        cache = prepare_cache(args, location, audit)
        audit.update(cache_prepared=True, cache_actual_bytes=cache['actual_bytes'], cache_estimated_bytes=cache['estimated_bytes'])
    if not args.execute:
        report = args.report or location['runs'] / '_preflight' / f'{args.name}.json'
        atomic_json(report, audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    execute(args, location, audit)


if __name__ == '__main__':
    main()
