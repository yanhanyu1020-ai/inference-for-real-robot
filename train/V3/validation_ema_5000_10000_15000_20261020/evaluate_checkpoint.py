#!/usr/bin/env python3
"""Independent fixed-protocol EMA validation. No training/optimizer entry point."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import traceback

GPU_UUID = 'GPU-e66640c0-8e1e-3f09-8f3f-2d10f877ec04'
WINDOWS = 2426
BATCH = 16
HORIZON = 15


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def unreduced_loss(pi0):
    """Preserve the official forward/RNG; only retain its final action dimension."""
    source = textwrap.dedent(inspect.getsource(pi0.Pi0.compute_loss))
    tree = ast.parse(source)
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    expected = ast.parse('jnp.mean(jnp.square(v_t - u_t), axis=-1)', mode='eval').body
    if len(returns) != 1 or ast.dump(returns[0].value) != ast.dump(expected):
        raise ValueError('Pinned official loss return expression changed')
    returns[0].value = ast.parse('jnp.square(v_t - u_t)', mode='eval').body
    function = tree.body[0]
    function.name, function.decorator_list, function.returns = 'validation_errors', [], None
    ast.fix_missing_locations(tree)
    namespace = dict(vars(pi0))
    exec(compile(tree, '<official_loss_keep_dimensions>', 'exec'), namespace)
    return namespace['validation_errors'], hashlib.sha256(source.encode()).hexdigest()


def load_batch(cache, record, np, Observation, jax=None):
    path = cache / record['file']
    if sha256(path) != record['sha256']:
        raise ValueError(f'Validation cache SHA mismatch: {path}')
    observation = {}
    with np.load(path, allow_pickle=False) as arrays:
        actions = arrays['actions']
        indices = arrays['window_indices']
        mask = arrays['valid_mask']
        for key in arrays.files:
            if not key.startswith('observation/'):
                continue
            parts = key.split('/')[1:]
            branch = observation
            for part in parts[:-1]:
                branch = branch.setdefault(part, {})
            branch[parts[-1]] = arrays[key]
    if actions.shape != (BATCH, HORIZON, 32) or actions.dtype != np.float32:
        raise ValueError('Cached actions do not match B16/H15/D32')
    count = int(record['count'])
    if not np.array_equal(indices[mask], np.arange(record['window_start'], record['window_end_exclusive'])):
        raise ValueError('Cached validation indices changed')
    if count != int(mask.sum()) or not np.array_equal(mask, np.arange(BATCH) < count):
        raise ValueError('Unexpected final-batch validity mask')
    obs = Observation.from_dict(observation)
    if jax is not None:
        import jax.numpy as jnp
        obs = jax.tree.map(jnp.asarray, obs)
        actions = jnp.asarray(actions)
    return obs, actions, indices, mask


def expected_model(cfg, nnx, jax, jnp):
    from openpi.shared import nnx_utils
    model_shape = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    model_shape.eval()
    graphdef = nnx.graphdef(model_shape)

    def state_fn(key):
        model = cfg.model.create(key)
        model.eval()
        return nnx_utils.state_map(nnx.state(model), cfg.freeze_filter,
                                   lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    state_shape = jax.eval_shape(state_fn, jax.random.key(0))
    return graphdef, state_shape


def validate_checkpoint_schema(checkpoint, state_shape, np):
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict
    expected = flatten_dict(state_shape.to_pure_dict())
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(str(checkpoint / 'params'))
    actual = flatten_dict(metadata['params'])
    if all(path[-1] == 'value' for path in actual):
        actual = {path[:-1]: value for path, value in actual.items()}
    if set(actual) != set(expected):
        raise ValueError('Checkpoint EMA parameter keys differ from the ViT-LoRA model')
    total_bytes, lora_b = 0, []
    for path, value in actual.items():
        spec = expected[path]
        if tuple(value.shape) != tuple(spec.shape) or str(value.dtype) != str(spec.dtype):
            raise ValueError(f'Checkpoint EMA shape/dtype mismatch: {path}')
        total_bytes += math.prod(value.shape) * np.dtype(value.dtype).itemsize
        if 'lora_b' in path:
            if path[:2] != ('PaliGemma', 'img'):
                raise ValueError('Unexpected non-ViT LoRA')
            lora_b.append('/'.join(path))
    if not lora_b or total_bytes != 12601518688:
        raise ValueError('Unexpected EMA parameter architecture/size')
    return dict(passed=True, leaves=len(actual), parameter_bytes=total_bytes,
                vit_lora_b_tensors=len(lora_b), original_dtypes_preserved=True,
                metadata_sha256=sha256(checkpoint / 'params/_METADATA'))


def gpu_state():
    result = subprocess.check_output([
        'nvidia-smi', '-i', GPU_UUID,
        '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    index, uuid, total, free, utilization = [x.strip() for x in result.strip().split(',')]
    if int(index) != 5 or uuid != GPU_UUID:
        raise ValueError('Only the user-authorized GPU5 may be used')
    return dict(index=5, uuid=uuid, total_mib=int(total), free_mib=int(free), utilization=int(utilization))


def run(args):
    start = time.monotonic()
    output = args.output_dir.resolve()
    checkpoint, cache, code = args.checkpoint.resolve(), args.cache_dir.resolve(), args.code_dir.resolve()
    if output.is_relative_to(checkpoint) or output.is_relative_to(code):
        raise ValueError('Validation cannot write inside training sources/checkpoints')
    output.mkdir(parents=True, exist_ok=True)
    status = dict(status='checking_contract', pid=os.getpid(), checkpoint=str(checkpoint),
                  checkpoint_step=int(checkpoint.name), actual_optimizer_step=int(checkpoint.name) + 1,
                  gpu=None if args.cpu_preflight else GPU_UUID, started_utc=now())

    def phase(name, **values):
        status.update(status=name, updated_utc=now(), elapsed_seconds=time.monotonic() - start, **values)
        atomic_json(output / 'status.json', status)
        print(json.dumps({k: status[k] for k in ('status', 'checkpoint_step', 'elapsed_seconds')},
                         ensure_ascii=False), flush=True)

    phase('checking_contract')
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('complete') is not True or manifest['windows'] != WINDOWS
            or manifest['batch_size'] != BATCH or manifest['fps'] != 15 or manifest['action_horizon'] != HORIZON):
        raise ValueError('Require complete original V3 2426-window B16/H15/15Hz cache')
    if sha256(code / 'manifest.json') != manifest['config_manifest_sha256']:
        raise ValueError('Original training manifest changed')
    for filename, digest in manifest.get('sidecar_files', {}).items():
        if sha256(cache / filename) != digest:
            raise ValueError(f'Validation cache provenance changed: {filename}')
    locked = json.loads((code / 'manifest.json').read_text())
    for relative, digest in locked['files'].items():
        if sha256(code / relative) != digest:
            raise ValueError(f'Original code changed: {relative}')
    norm = checkpoint / 'assets/heterovla_v3/norm_stats.json'
    if sha256(norm) != manifest['norm_sha256']:
        raise ValueError('Checkpoint normalization differs from the fixed validation cache')
    batches = manifest['batches']
    if (sum(x['count'] for x in batches) != WINDOWS or
            [i for x in batches for i in range(x['window_start'], x['window_end_exclusive'])] != list(range(WINDOWS))):
        raise ValueError('Validation windows are not covered exactly once')
    sys.path.insert(0, str(code))
    import configuration as c
    if Path(c.__file__).resolve() != code / 'configuration.py':
        raise ValueError('Unexpected original configuration module')
    c.bootstrap()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from openpi.models import model as model_api, pi0
    from openpi.shared import array_typing as at
    from compilation_compat import lower_and_compile
    cfg = c.make_config()
    graphdef, shape = expected_model(cfg, nnx, jax, jnp)
    schema = validate_checkpoint_schema(checkpoint, shape, np)
    errors_fn, official_loss_sha = unreduced_loss(pi0)
    observation, actions, _, _ = load_batch(cache, batches[0], np, model_api.Observation)
    tail = load_batch(cache, batches[-1], np, model_api.Observation)
    if int(tail[-1].sum()) != 10:
        raise ValueError('Expected final 10 real validation windows')
    proof = dict(passed=True, checkpoint_step=int(checkpoint.name), windows=WINDOWS,
                 seed=args.seed, batch_size=BATCH, train=False, weights='EMA',
                 cache_manifest_sha256=sha256(manifest_path), norm_sha256=manifest['norm_sha256'],
                 schema=schema, official_loss_source_sha256=official_loss_sha,
                 optimizer_loaded=False, base_weights_loaded=False, model_arrays_loaded=False,
                 cpu_only=args.cpu_preflight)
    atomic_json(output / 'contract_proof.json', proof)
    if args.cpu_preflight:
        if jax.default_backend() != 'cpu':
            raise ValueError('CPU preflight must not initialize a GPU')
        phase('cpu_preflight_passed')
        return
    if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise ValueError('Only one GPU must be visible to the evaluator')

    def forward(state, key, obs, act):
        model = nnx.merge(graphdef, state)
        return jnp.mean(errors_fn(model, key, obs, act, train=False), axis=1)

    def official_forward(state, key, obs, act):
        model = nnx.merge(graphdef, state)
        return model.compute_loss(key, obs, act, train=False)

    abstract_observation, abstract_actions = cfg.model.inputs_spec(batch_size=BATCH)
    key = jax.random.fold_in(jax.random.key(args.seed), 0)
    phase('compiling_validation_forward')
    old_cache_flag = jax.config.jax_enable_compilation_cache
    try:
        # JAX0.5.3 cache hits can lose memory-analysis alias metadata.
        jax.config.update('jax_enable_compilation_cache', False)
        compiled = lower_and_compile(jax.jit(forward), shape, key, abstract_observation, abstract_actions)
        reference = lower_and_compile(jax.jit(official_forward), shape, key, abstract_observation, abstract_actions)
    finally:
        jax.config.update('jax_enable_compilation_cache', old_cache_flag)
    memory = []
    pool = int(gpu_state()['total_mib'] * 2**20 * .30)
    for name, function in [('per_dimension', compiled), ('official_reference', reference)]:
        stats = function.memory_analysis()
        if stats is None:
            raise ValueError('No GPU compiled-memory analysis')
        report = {name: int(getattr(stats, name)) for name in (
            'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
        report.update(program=name,
                      estimated_peak_bytes=report['argument_size_in_bytes'] + report['output_size_in_bytes']
                      + report['temp_size_in_bytes'] - report['alias_size_in_bytes'])
        if report['estimated_peak_bytes'] + 512 * 2**20 > pool:
            raise MemoryError('Validation exceeds the fixed 24GiB pool; no model weights loaded')
        memory.append(report)
    atomic_json(output / 'compiled_memory.json', dict(passed=True, programs=memory, pool_bytes=pool,
                                                      reserved_fraction=.30, no_optimizer=True))
    phase('loading_ema_weights')
    params = model_api.restore_params(checkpoint / 'params', dtype=None)
    state = shape
    state.replace_by_pure_dict(params)
    del params
    model = nnx.merge(graphdef, state)
    model.eval()
    state = nnx.state(model)
    del model
    b_nonzero = 0
    for path, variable in state.flat_state().items():
        if 'lora_b' in path:
            b_nonzero += int(np.count_nonzero(np.asarray(variable.value)))
    if b_nonzero == 0:
        raise ValueError('Trained checkpoint LoRA B is entirely zero')
    phase('checking_official_loss_equivalence')
    obs_gpu, act_gpu, _, _ = load_batch(cache, batches[0], np, model_api.Observation, jax)
    first_dimensions = np.asarray(compiled(state, key, obs_gpu, act_gpu))
    first_official = np.asarray(reference(state, key, obs_gpu, act_gpu))
    np.testing.assert_allclose(first_dimensions.mean(axis=1), first_official.mean(axis=1), rtol=2e-5, atol=1e-7)
    atomic_json(output / 'loss_equivalence.json', dict(
        passed=True, official32=float(first_official.mean()), from_dimensions=float(first_dimensions.mean()),
        maximum_absolute_difference=float(np.max(np.abs(first_dimensions.mean(axis=1) - first_official.mean(axis=1)))),
        rtol=2e-5, atol=1e-7, official_loss_source_sha256=official_loss_sha))
    del reference, obs_gpu, act_gpu, first_official, first_dimensions
    provenance = [json.loads(line) for line in (cache / 'windows.jsonl').read_text().splitlines() if line]
    if len(provenance) != WINDOWS:
        raise ValueError('Incomplete validation provenance')
    records = []
    phase('evaluating', windows_completed=0, total_windows=WINDOWS)
    with (output / 'per_window.jsonl').open('x') as stream:
        for batch in batches:
            obs_gpu, act_gpu, indices, mask = load_batch(cache, batch, np, model_api.Observation, jax)
            key = jax.random.fold_in(jax.random.key(args.seed), int(batch['window_start']))
            dimensions = np.asarray(compiled(state, key, obs_gpu, act_gpu))
            if dimensions.shape != (BATCH, 32) or not np.isfinite(dimensions).all():
                raise FloatingPointError('Nonfinite or invalid validation loss; no samples discarded')
            for row in np.flatnonzero(mask):
                index = int(indices[row])
                values = dimensions[row].astype(np.float64)
                record = dict(provenance[index], official32_loss=float(values.mean()),
                              real14_loss=float(values[:14].mean()), padding18_loss=float(values[14:].mean()),
                              go2_loss=float(values[:3].mean()), piper_loss=float(values[3:10].mean()),
                              uav_loss=float(values[10:14].mean()), dimension14_loss=values[:14].tolist())
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            phase('evaluating', windows_completed=len(records),
                  official32_loss_so_far=float(np.mean([r['official32_loss'] for r in records])))
    if [r['window_index'] for r in records] != list(range(WINDOWS)):
        raise ValueError('Final validation coverage is not exact')
    metrics = {}
    for name in ('official32_loss', 'real14_loss', 'padding18_loss', 'go2_loss', 'piper_loss', 'uav_loss'):
        values = np.asarray([r[name] for r in records], dtype=np.float64)
        metrics[name] = dict(mean=float(values.mean()), median=float(np.median(values)),
                             p95=float(np.quantile(values, .95)), maximum=float(values.max()))
    if sha256(checkpoint / 'params/_METADATA') != schema['metadata_sha256']:
        raise ValueError('Checkpoint metadata changed during validation')
    report = dict(complete=True, checkpoint_step=int(checkpoint.name), actual_optimizer_step=int(checkpoint.name)+1,
                  checkpoint=str(checkpoint), windows=WINDOWS, batch_size=BATCH, seed=args.seed,
                  fps=15, action_horizon=HORIZON, weights='EMA', train=False, metrics=metrics,
                  official32_loss=metrics['official32_loss']['mean'], real14_loss=metrics['real14_loss']['mean'],
                  cache_manifest_sha256=sha256(manifest_path), checkpoint_schema=schema,
                  per_window_sha256=sha256(output/'per_window.jsonl'), completed_utc=now(),
                  elapsed_seconds=time.monotonic()-start, gpu_uuid=GPU_UUID,
                  gpu_memory_stats={k: int(v) for k, v in jax.devices()[0].memory_stats().items()
                                    if v is not None and isinstance(v, (int, np.integer))},
                  optimizer_loaded=False, training_modified=False,
                  key_policy='fold_in(key(seed=42), fixed batch window_start); identical across checkpoints')
    atomic_json(output / 'summary.json', report)
    phase('completed', windows_completed=WINDOWS, official32_loss=report['official32_loss'],
          real14_loss=report['real14_loss'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', default=GPU_UUID)
    parser.add_argument('--batch-size', type=int, default=BATCH)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    if args.gpu != GPU_UUID or args.batch_size != BATCH or args.seed != 42:
        parser.error('This protocol fixes GPU5, batch16 and seed42')
    if args.checkpoint.name not in ('5000', '10000', '15000'):
        parser.error('Only requested checkpoints5000/10000/15000 are authorized')
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.cpu_preflight else GPU_UUID,
                      JAX_PLATFORMS='cpu' if args.cpu_preflight else 'cuda',
                      XLA_PYTHON_CLIENT_MEM_FRACTION='0.30',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false' if args.cpu_preflight else 'true',
                      PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
    sys.dont_write_bytecode = True
    if not args.cpu_preflight and gpu_state()['free_mib'] < 28672:
        raise MemoryError('GPU5 has less than28GiB free; refusing CUDA initialization')
    try:
        run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / 'error.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                                       traceback=traceback.format_exc(), updated_utc=now()))
        raise


if __name__ == '__main__':
    main()
