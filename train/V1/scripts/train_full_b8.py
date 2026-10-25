"""Single-GPU offline full-finetuning run with deterministic validation and resumable state."""
import argparse
import dataclasses
import functools
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
from path_config import OPENPI_ROOT
import signal
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--peak-lr', type=float, choices=[1e-4], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    allowed = ['GPU-87fdd428-76b9-b8ca-7898-fe5ed5907150']
    if args.gpu_uuid not in allowed or '/' in args.name:
        p.error('Full-finetuning comparison uses user-authorized 16.7 GPU0')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, JAX_PLATFORMS='cuda',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.68',
                      WANDB_GROUP='pi05-v1-full-vs-lora-20261006')
    logging.basicConfig(level=logging.INFO, force=True)
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(args.gpu_uuid)
    # Abstract compilation does not allocate the multi-GB model/Adam arrays.
    # Check measured compiled demand plus headroom immediately before weight allocation.
    if nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2).free < 2*2**30:
        raise RuntimeError('Insufficient GPU context headroom for abstract compilation')
    for proc in nv.nvmlDeviceGetComputeRunningProcesses(handle):
        try:
            if Path(f'/proc/{proc.pid}').stat().st_uid == os.getuid():
                raise RuntimeError('Another own GPU process is active; one experiment per GPU')
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
    from experiment_configs import make_config, summary, install_dataset_adapter, PROJECT_ROOT, RUN_ROOT
    from scan_common import atomic_json, cache_batches, loss_elements, eval_elements, evaluate, early_stop_update, overfit
    scan = RUN_ROOT / 'full_ft_20261006'
    cache = RUN_ROOT / 'lr_scan_20261005'
    audit = json.loads((cache / 'preflight.json').read_text())
    assert audit['status'] == 'passed' and audit['complete_val']
    config = dataclasses.replace(make_config('full', batch_size=8, peak_lr=args.peak_lr,
        exp_name=args.name, remat_policy='nothing_saveable'),
        save_interval=250, keep_period=250, resume=args.resume)
    directory = scan / args.name
    if directory.exists() and not args.resume:
        raise FileExistsError(directory)
    directory.mkdir(parents=True, exist_ok=True)
    status = {'name': args.name, 'peak_lr': args.peak_lr, 'gpu_uuid': args.gpu_uuid,
              'pid': os.getpid(), 'status': 'initializing', 'step': 0,
              'remat_policy': 'nothing_saveable', 'allocator_fraction': 0.68}
    started = time.monotonic()
    prior_seconds = 0.0
    if args.resume:
        prior_seconds = json.loads((directory / 'status.json').read_text()).get('elapsed_seconds', 0)
    def publish(**values):
        status.update(values, elapsed_seconds=prior_seconds+time.monotonic()-started)
        atomic_json(directory / 'status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    def fail_signal(signum, frame):
        raise RuntimeError(f'Own run received signal {signum}')
    signal.signal(signal.SIGTERM, fail_signal)
    signal.signal(signal.SIGINT, fail_signal)
    managers = []
    try:
        assert jax.device_count() == 1 and jax.devices()[0].platform == 'gpu'
        install_dataset_adapter()
        spec = importlib.util.spec_from_file_location('official_scan_trainer', OPENPI_ROOT / 'scripts/train.py')
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        cfg_record = {**summary(config), 'warmup_steps': 100, 'decay_lr': args.peak_lr/10,
            'save_interval': 250, 'eval_interval': 100, 'log_interval': 25,
            'validation_windows': 710, 'validation_seed': 42, 'validation_augmentation': False,
            'training_objective': 'official mean over all 32 dimensions',
            'early_stop': 'after step 500: five consecutive evals without >=1% improvement over preceding best',
            'overfit_rule': '3 consecutive fixed-val increases and fixed-train-probe decreases, each endpoint >1%',
            'train_probe': '128 fixed evenly spaced training windows; no augmentation; diagnostic only',
            'gpu_uuid': args.gpu_uuid, 'preflight_norm_sha256': audit['norm_sha256'],
            'lora_reference': 'dual_lora_lr1e-4_s42_20261005',
            'precision': 'official full training: all parameters float32; compute dtype unchanged',
            'batch_change': 'User requested batch8 after batch16 compile estimate 54.15 GiB; validation remains batch16'}
        if args.resume:
            original_config = json.loads((directory/'config.json').read_text())
            assert original_config['remat_policy'] in ['nothing_saveable', 'dots_with_no_batch_dims_saveable']
            # The user authorized only this memory policy and placement change.
            assert {k:v for k,v in original_config.items() if k not in ['remat_policy','gpu_uuid']} == {
                k:v for k,v in cfg_record.items() if k not in ['remat_policy','gpu_uuid']}
        else:
            atomic_json(directory/'config.json', cfg_record)
        atomic_json(directory/'effective_config.json', {**cfg_record, 'allocator_fraction': .68})
        mesh = sharding.make_mesh(1)
        ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        manager, resuming = ckpt.initialize_checkpoint_dir(config.checkpoint_dir,
            keep_period=250, overwrite=False, resume=args.resume)
        managers.append(manager)
        best_manager, _ = ckpt.initialize_checkpoint_dir(directory/'best', keep_period=None,
            overwrite=False, resume=args.resume)
        managers.append(best_manager)
        if args.resume and not resuming:
            raise RuntimeError('Requested resume but no regular checkpoint exists')
        run_id_path = directory/'wandb_id.txt'
        wb = wandb.init(project=config.project_name, name=args.name, config=cfg_record,
            id=run_id_path.read_text().strip() if args.resume else None,
            resume='must' if args.resume else None)
        run_id_path.write_text(wb.id)
        wandb.config.update(cfg_record, allow_val_change=True)
        wandb.config.update({'allocator_fraction': .68, 'execution': 'full_finetuning_shared_gpu0'}, allow_val_change=True)
        status['wandb_url'] = wb.url
        loader = data_loader.create_data_loader(config, sharding=ds, shuffle=True)
        iterator = iter(loader)
        train_rng, init_rng = jax.random.split(jax.random.key(42))
        publish(status='checking_parameter_shapes')
        state, state_sharding = official.init_train_state(config, init_rng, mesh, resume=True)
        flat = state.params.flat_state()
        selected = state.params.filter(config.trainable_filter).flat_state()
        assert set(flat) == set(selected) and not any('lora' in '/'.join(map(str,k)) for k in flat)
        total = sum(math.prod(v.value.shape) for v in flat.values())
        groups = {}
        for path, leaf in flat.items():
            name = '/'.join(map(str,path))
            group = ('vision' if 'img' in name else 'action_expert' if 'llm' in name and '_1' in name
                     else 'language' if 'llm' in name else 'projections_and_time_mlp')
            groups[group] = groups.get(group, 0) + math.prod(leaf.value.shape)
        atomic_json(directory/'parameter_report.json', {'total_parameters': total,
            'trainable_parameters': total, 'trainable_groups': groups,
            'parameter_dtypes': sorted({str(v.value.dtype) for v in flat.values()})})
        del flat, selected
        _, _, batch = next(cache_batches(cache/'val_cache'))
        batch = jax.tree.map(lambda x: x[:config.batch_size], batch)
        start = 0
        def train_step(rng, state, batch):
            model = nnx.merge(state.model_def, state.params)
            model.train()
            def objective(model, key, observation, actions):
                errors = loss_elements(model, key, observation, actions, train=True)
                # Same reduction order as the installed official objective.
                return errors.mean(axis=-1).mean(), errors.mean(axis=(0, 1))
            (loss, dimensions), grads = nnx.value_and_grad(objective,
                argnums=nnx.DiffState(0, config.trainable_filter), has_aux=True)(
                model, jax.random.fold_in(rng, state.step), *batch)
            params = state.params.filter(config.trainable_filter)
            updates, opt_state = state.tx.update(grads, state.opt_state, params)
            params = optax.apply_updates(params, updates)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((params, opt_state))]))
            nnx.update(model, params)
            new_state = dataclasses.replace(state, step=state.step+1, params=nnx.state(model), opt_state=opt_state)
            return new_state, {'loss': loss, 'dimensions': dimensions,
                               'grad_norm': optax.global_norm(grads), 'finite_update': finite}
        train_fn = jax.jit(train_step, in_shardings=(replicated, state_sharding, ds),
            out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        eval_fn = jax.jit(functools.partial(eval_elements, state.model_def))
        def emit(step, metrics):
            if not all(math.isfinite(float(x)) for x in metrics.values()):
                raise FloatingPointError('Non-finite logged metric')
            record = {'step': step, **metrics, 'elapsed_seconds': prior_seconds+time.monotonic()-started}
            with (directory/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False)+'\n')
            wandb.log(record, step=step)
        def save(manager, step):
            ckpt.save_state(manager, state, loader, step)
            manager.wait_until_finished()  # No buffer donation until snapshot completes.
            manager.check_for_errors()
        def validate(step):
            t0 = time.monotonic()
            val = evaluate(eval_fn, state.params, cache/'val_cache')
            probe = evaluate(eval_fn, state.params, cache/'train_probe_cache')
            assert val['windows'] == 710 and probe['windows'] == 128
            value = val['real14_loss']
            control['early_history'].append(early_stop_update(control['early_history'], step, value))
            control['comparison_history'].append({'step': step, 'val': value, 'train': probe['real14_loss']})
            if control['best'] is None or value < control['best']:
                control.update(best=value, best_step=step, best_metrics=val,
                               best_train_probe=probe, best_checkpoint=str(directory/'best'/str(step)))
                save(best_manager, step)
                atomic_json(directory/'best.json', control)
            return {**{'val/'+k:v for k,v in val.items()},
                    **{'train_probe/'+k:v for k,v in probe.items()},
                    'eval_seconds': time.monotonic()-t0}
        publish(status='compiling_train', step=start)
        with sharding.set_mesh(mesh), at.disable_typechecking():
            executable = train_fn.lower(train_rng, state, batch).compile()
        memory = executable.memory_analysis()
        estimate = memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes
        budget = .68*nv.nvmlDeviceGetMemoryInfo(handle).total - 2*2**30
        atomic_json(directory/'compiled_memory.json', {'live_estimate_bytes': estimate,
            'safe_budget_bytes': int(budget), 'argument_bytes': memory.argument_size_in_bytes,
            'output_bytes': memory.output_size_in_bytes, 'temporary_bytes': memory.temp_size_in_bytes,
            'alias_bytes': memory.alias_size_in_bytes, 'weight_allocation_started': False})
        if estimate > budget:
            raise RuntimeError(f'Insufficient shared-GPU memory: compiled live estimate {estimate/2**30:.2f} GiB exceeds safe budget {budget/2**30:.2f} GiB; no weights loaded or updates performed')
        required_free = estimate + int(1.5*2**30)
        if nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2).free < required_free:
            raise RuntimeError(f'GPU0 lacks compiled demand plus 1.5 GiB headroom before weight allocation: need {required_free/2**30:.2f} GiB free')
        record = json.loads((directory/'compiled_memory.json').read_text())
        record.update(required_free_before_weights_bytes=required_free, weight_allocation_started=True)
        atomic_json(directory/'compiled_memory.json', record)
        publish(status='loading_weights')
        state, state_sharding = official.init_train_state(config, init_rng, mesh, resume=resuming)
        if resuming:
            state = ckpt.restore_state(manager, state, loader)
        jax.block_until_ready(state)
        start = int(state.step)
        atomic_json(directory/f'runtime_profile_from_step_{start}.json', {
            'from_step': start, 'remat_policy': config.model.remat_policy,
            'gpu_uuid': args.gpu_uuid, 'allocator_fraction': .68,
            'reason': 'User authorized reducing full-finetuning batch16 to batch8 on 2026-10-06'})
        control = {'best': None, 'best_step': None, 'early_history': [], 'comparison_history': []}
        if resuming:
            control = json.loads((directory/f'control_{start}.json').read_text())
            # Official loader uses a seeded generator. Replaying its batches also
            # restores the exact shuffled data position; training RNG is step-keyed.
            publish(status='restoring_data_position', step=start)
            for _ in range(start):
                next(iterator)
        batch = next(iterator)
        if not resuming:
            publish(status='verifying_official_loss', step=0)
            _, _, check_batch = next(cache_batches(cache/'val_cache'))
            key = jax.random.key(42)
            def official_eval(params, key, observation, actions):
                model = nnx.merge(state.model_def, params)
                model.eval()
                return model.compute_loss(key, observation, actions, train=False).mean()
            check_fn = jax.jit(official_eval)
            official_value = float(check_fn(state.params, key, *check_batch))
            observed = float(np.asarray(eval_fn(state.params, key, *check_batch)).mean())
            np.testing.assert_allclose(observed, official_value, rtol=1e-5, atol=1e-6)
            atomic_json(directory/'loss_equivalence.json', {'official': official_value, 'instrumented': observed})
            del check_fn
            publish(status='validating_step0')
            emit(0, validate(0))

        accum = []
        reason = 'max_steps_2000'
        publish(status='training', step=start)
        for step in range(start+1, 2001):
            tick = time.monotonic()
            # Step 1 uses the already-loaded compile batch; subsequent waits are measured.
            if step != start+1:
                batch = next(iterator)
            jax.block_until_ready(batch)
            wait = time.monotonic()-tick
            if not bool(jnp.all(jnp.isfinite(batch[1]))) or not bool(jnp.all(jnp.isfinite(batch[0].state))):
                raise ValueError('Non-finite training data')
            state, info = executable(train_rng, state, batch)
            jax.block_until_ready((state, info))
            host = jax.device_get(info)
            if not host['finite_update'] or not all(np.isfinite(v).all() for v in host.values()):
                raise FloatingPointError('NaN/Inf in loss, gradients, parameters or optimizer')
            own = sum(p.usedGpuMemory for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid == os.getpid())
            accum.append({'loss': float(host['loss']), 'grad_norm': float(host['grad_norm']),
                'dimensions': np.asarray(host['dimensions']), 'step_seconds': time.monotonic()-tick,
                'data_wait_seconds': wait, 'gpu_bytes': own})
            metrics = {}
            if step % 25 == 0:
                from scan_common import scalar_metrics
                metrics = {'train/'+k: v for k,v in scalar_metrics(np.mean([x['dimensions'] for x in accum],axis=0)).items()}
                metrics.update({'train/'+k:float(np.mean([x[k] for x in accum])) for k in
                                ['loss','grad_norm','step_seconds','data_wait_seconds']})
                metrics.update(lr=float(config.lr_schedule.create()(step-1)),
                    gpu_memory_bytes=max(x['gpu_bytes'] for x in accum),
                    jax_live_bytes=jax.devices()[0].memory_stats().get('bytes_in_use',0))
                accum = []
            if step % 100 == 0:
                metrics.update(validate(step))
            if metrics:
                emit(step, metrics)
                publish(status='training', step=step, best_val=control['best'], best_step=control['best_step'],
                        train_loss=metrics.get('train/loss'), best_checkpoint=control.get('best_checkpoint'))
            stop = None
            if step % 100 == 0:
                if overfit(control['comparison_history']):
                    stop = 'overfitting_fixed_train_down_val_up'
                elif control['early_history'][-1]['streak'] >= 5:
                    stop = 'early_stop_5_evals_without_1pct_improvement'
                elif step == 500:
                    data = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
                    training = [row for row in data if 'train/loss' in row]
                    first = np.mean([row['train/loss'] for row in training[:4]])
                    last = np.mean([row['train/loss'] for row in training[-4:]])
                    clear = last <= .95*first and control['comparison_history'][-1]['train'] < control['comparison_history'][0]['train']
                    atomic_json(directory/'stage500_review.json', {
                        'step': 500, 'first100_train_mean': float(first), 'last100_train_mean': float(last),
                        'clear_training_decrease': bool(clear), 'decision': 'continue' if clear else 'pause',
                        'gradient_norm_min': min(row['train/grad_norm'] for row in training),
                        'gradient_norm_max': max(row['train/grad_norm'] for row in training),
                        'validation': {k:v for k,v in metrics.items() if k.startswith('val/')},
                        'fixed_train_probe': {k:v for k,v in metrics.items() if k.startswith('train_probe/')}})
                    if not clear:
                        stop = 'paused_stage500_no_clear_training_decrease'
            if step % 250 == 0 or stop:
                atomic_json(directory/f'control_{step}.json', control)
                save(manager, step)
            if stop:
                reason = stop
                break
        publish(status='paused' if reason.startswith('paused_') else 'completed',
                step=int(state.step), stop_reason=reason, best_val=control['best'],
                best_step=control['best_step'], best_checkpoint=control['best_checkpoint'])
        wandb.run.summary.update(status)
        wandb.finish()
    except BaseException as exc:
        publish(status='failed', stop_reason=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        if wandb.run:
            wandb.run.summary.update(status)
            wandb.finish(exit_code=1)
        raise
    finally:
        for manager in managers:
            manager.close()


if __name__ == '__main__':
    main()
