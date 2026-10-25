#!/usr/bin/env python3
"""Short single-GPU LoRA benchmark using official steps. No checkpoint is saved."""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
from path_config import OPENPI_ROOT
import statistics
import sys
import threading
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--timed-steps', type=int, default=15)
    parser.add_argument('--pipeline-steps', type=int, default=8)
    parser.add_argument('--remat-policy', choices=['nothing_saveable', 'dots_saveable', 'dots_with_no_batch_dims_saveable', 'none'], default='nothing_saveable')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing benchmark report')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['OMP_NUM_THREADS'] = '4'
    logging.basicConfig(level=logging.WARNING, force=True)
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(args.gpu_uuid)
    before = nv.nvmlDeviceGetMemoryInfo(handle)
    other_pids = [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid != os.getpid()]
    if other_pids or before.free < 70 * 2**30:
        raise RuntimeError('Selected GPU is no longer empty; benchmark not started')
    report = {'status': 'running', 'batch_size': args.batch_size, 'variant': 'dual_lora',
              'gpu_uuid': args.gpu_uuid, 'gpu_name': nv.nvmlDeviceGetName(handle),
              'host': os.uname().nodename, 'pid': os.getpid(), 'gpu_total_bytes': before.total,
              'gpu_initial_free_bytes': before.free, 'preallocation': False,
              'allocator_fraction': 0.90, 'checkpoint_saved': False, 'phases': {},
              'remat_policy': args.remat_policy}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    phase = ['imports']
    samples = []

    def monitor():
        while not stop.wait(0.25):
            try:
                procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                own = sum(p.usedGpuMemory for p in procs if p.pid == os.getpid())
                util = nv.nvmlDeviceGetUtilizationRates(handle).gpu
                samples.append({'phase': phase[0], 'process_bytes': own, 'utilization': util,
                                'other_processes': len([p for p in procs if p.pid != os.getpid()])})
            except nv.NVMLError:
                pass

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()

    def announce(name):
        phase[0] = name
        print(json.dumps({'phase': name, 'batch_size': args.batch_size,
                          'elapsed_seconds': round(time.perf_counter() - started, 2)}), flush=True)

    try:
        import jax
        import numpy as np
        from openpi.shared import array_typing as at
        from openpi.training import data_loader, sharding
        from experiment_configs import make_config, install_dataset_adapter, PROJECT_ROOT
        assert jax.device_count() == 1 and jax.devices()[0].platform == 'gpu'
        jax.config.update('jax_compilation_cache_dir', str(args.output.parent / 'jax_cache'))
        report['jax_version'] = jax.__version__
        config = dataclasses.replace(make_config('dual_lora', batch_size=args.batch_size,
                                                remat_policy=args.remat_policy), wandb_enabled=False)
        install_dataset_adapter()
        spec = importlib.util.spec_from_file_location('openpi_official_benchmark_trainer',
                    OPENPI_ROOT / 'scripts/train.py')
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        rng, init_rng = jax.random.split(jax.random.key(42))

        announce('data_loading')
        t0 = time.perf_counter()
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        iterator = iter(loader)
        cached_batches = [next(iterator) for _ in range(2)]
        jax.block_until_ready(cached_batches)
        report['cached_batch_sha256'] = hashlib.sha256(b''.join(
            np.asarray(x).tobytes() for x in jax.tree.leaves(jax.device_get(cached_batches)))).hexdigest()
        report['initial_two_batches_seconds'] = time.perf_counter() - t0

        announce('weight_and_optimizer_initialization')
        t0 = time.perf_counter()
        state, state_sharding = trainer.init_train_state(config, init_rng, mesh, resume=False)
        jax.block_until_ready(state)
        report['initialization_seconds'] = time.perf_counter() - t0
        def parameter_sketch():
            leaves = jax.tree.leaves(state.params.filter(config.trainable_filter))
            return np.concatenate([np.asarray(x).reshape(-1)[
                np.linspace(0, x.size - 1, min(256, x.size), dtype=int)].astype(np.float32)
                for x in jax.device_get(leaves)])
        initial_sketch = parameter_sketch()
        step = jax.jit(functools.partial(trainer.train_step, config),
                       in_shardings=(replicated, state_sharding, data_sharding),
                       out_shardings=(state_sharding, replicated), donate_argnums=(1,))

        announce('compile')
        t0 = time.perf_counter()
        # JAX AOT lowering rebuilds dataclasses with ArgInfo placeholders, which
        # are not arrays and cannot pass TrainState's runtime type checks.
        with sharding.set_mesh(mesh), at.disable_typechecking():
            executable = step.lower(rng, state, cached_batches[0]).compile()
        report['compile_seconds'] = time.perf_counter() - t0
        mem = executable.memory_analysis()
        fields = ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes',
                  'alias_size_in_bytes', 'generated_code_size_in_bytes']
        report['compiled_memory'] = {k: int(getattr(mem, k, 0)) for k in fields}
        m = report['compiled_memory']
        estimate = m['argument_size_in_bytes'] + m['output_size_in_bytes'] + m['temp_size_in_bytes'] - m['alias_size_in_bytes']
        report['compiled_live_memory_estimate_bytes'] = estimate
        if estimate > 0.90 * before.total - 2 * 2**30:
            report['status'] = 'skipped_memory_estimate'
            print('Compiled memory estimate exceeds safe allocation budget', flush=True)
            return

        losses = []
        def one(batch):
            nonlocal state
            if any(p.pid != os.getpid() for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)):
                raise RuntimeError('Another process appeared on the GPU; stopping benchmark')
            state, info = executable(rng, state, batch)
            jax.block_until_ready((state, info))
            scalar = {k: float(v) for k, v in jax.device_get(info).items()}
            if not all(math.isfinite(x) for x in scalar.values()):
                raise RuntimeError('Non-finite training metric')
            losses.append(scalar['loss'])
            if len(losses) == 1:
                report['first_step_metrics'] = scalar

        announce('warmup')
        for i in range(args.warmup):
            one(cached_batches[i % 2])
        np.savez(args.output.with_suffix('.npz'), initial=initial_sketch,
                 after_warmup=parameter_sketch())
        announce('compute_timing')
        durations = []
        for i in range(args.timed_steps):
            t0 = time.perf_counter()
            one(cached_batches[i % 2])
            durations.append(time.perf_counter() - t0)
        report['compute'] = {'steps': len(durations), 'step_seconds_mean': statistics.mean(durations),
                             'step_seconds_median': statistics.median(durations),
                             'step_seconds_p95': float(np.percentile(durations, 95)),
                             'samples_per_second': args.batch_size * len(durations) / sum(durations),
                             'step_seconds': durations}

        announce('pipeline_timing')
        durations, waits = [], []
        for _ in range(args.pipeline_steps):
            t0 = time.perf_counter()
            batch = next(iterator)
            waits.append(time.perf_counter() - t0)
            one(batch)
            durations.append(time.perf_counter() - t0)
        report['pipeline'] = {'steps': len(durations), 'step_seconds_mean': statistics.mean(durations),
                              'samples_per_second': args.batch_size * len(durations) / sum(durations),
                              'data_wait_seconds_mean': statistics.mean(waits),
                              'data_wait_fraction': sum(waits) / sum(durations),
                              'num_workers': config.num_workers}
        report['benchmark_optimizer_steps'] = int(state.step)
        report['first_loss'] = losses[0]
        report['last_loss'] = losses[-1]
        report['loss_sequence'] = losses
        report['jax_memory_stats'] = jax.devices()[0].memory_stats()
        report['status'] = 'passed'
        announce('finished')
    except Exception as exc:
        report['status'] = 'failed'
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        traceback.print_exc()
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        report['elapsed_seconds'] = time.perf_counter() - started
        report['peak_process_memory_bytes'] = max((s['process_bytes'] for s in samples), default=0)
        for name in sorted({s['phase'] for s in samples}):
            rows = [s for s in samples if s['phase'] == name]
            report['phases'][name] = {
                'peak_process_memory_bytes': max(s['process_bytes'] for s in rows),
                'gpu_utilization_mean': statistics.mean(s['utilization'] for s in rows),
                'other_gpu_process_seen': any(s['other_processes'] for s in rows)}
        args.output.write_text(json.dumps(report, indent=2))
        print('RESULT', json.dumps({k: report[k] for k in ['status', 'batch_size', 'elapsed_seconds',
                                                          'peak_process_memory_bytes']}), flush=True)


if __name__ == '__main__':
    main()
