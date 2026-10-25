"""Resume the locked official V3 experiment after the server reboot.

Uses upstream main, restore_state, optimizer, loss and EMA; never loads the
base checkpoint or resets Adam/LoRA. The upstream loader restarts its shuffle.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/meta_eon_cfs/home/hph/heterovla/train/V3/official_vitlora_r8_a16_h15_data20261021')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    # JAX0.5.3 memoizes is_cache_used() at the first compilation. Disable
    # disk-cache use before any JAX/model/data import, rather than only lowering.
    os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
    # The queue is a separate one-shot claim; original attempt remains intact.
    from queue_training import verify_preflight, query_gpus, available
    proof = verify_preflight(HERE)
    plan = json.loads((HERE / 'resume_plan.json').read_text())
    assert args.run_name == plan['run_name']
    device = next(g for g in query_gpus() if g['uuid'] == args.gpu)
    if not available(device):
        raise RuntimeError('Selected GPU is no longer idle; no model was loaded')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
        WANDB_MODE='offline', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(ORIGINAL))
    import configuration as c
    from runtime_reporting import atomic_json, append_metrics, dumps_record
    original_contract = verify_preflight(ORIGINAL)
    assert original_contract['manifest_sha256'] == plan['original_manifest_sha256']
    run = c.MODELS / 'V3/runs' / args.run_name
    report = run / 'resume_reports' / plan['resume_id']
    report.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_DIR'] = str(run)
    prior = json.loads((run / 'runtime_status.json').read_text())
    atomic_json(report / 'runtime_before_resume.json', prior)
    started = time.monotonic()
    prior_elapsed = float(prior.get('elapsed_seconds', 0))
    expected = plan['source_actual_step']
    status = {'status':'initializing_resume', 'step':expected, 'pid':os.getpid(),
        'gpu':args.gpu, 'run_name':args.run_name, 'resume_id':plan['resume_id'],
        'resumed_from_loop_step':plan['source_loop_step'],
        'source_actual_step':expected, 'target_steps':20000,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(), **proof}

    def publish(**values):
        status.update(values, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            resume_elapsed_seconds=time.monotonic()-started,
            elapsed_seconds=prior_elapsed+time.monotonic()-started)
        atomic_json(run / 'runtime_status.json', status)
        atomic_json(report / 'runtime_status.json', status)
        print(dumps_record(status), flush=True)

    try:
        publish()
        c.bootstrap(); c.install_data_adapter()
        config = dataclasses.replace(c.make_config(), resume=True, overwrite=False)
        assert config.num_train_steps == 20000 and config.ema_decay == .99
        assert str(Path(config.checkpoint_dir) / str(plan['source_loop_step'])) == plan['source_checkpoint']
        official = c.official_trainer()
        import jax
        assert jax.config.jax_enable_compilation_cache is False
        from jax._src import compiler as _jax_compiler
        cache_reads = []
        original_cache_read = _jax_compiler._cache_read
        def observe_cache_read(*a, **kw):
            cache_reads.append(1)
            return original_cache_read(*a, **kw)
        _jax_compiler._cache_read = observe_cache_read
        import numpy as np
        from openpi.training import sharding
        from compilation_compat import lower_and_compile
        official.train_step = c.corrected_train_step(official.train_step)
        old_init = official.init_train_state
        old_restore = official._checkpoints.restore_state
        old_save = official._checkpoints.save_state
        old_wandb_init = official.init_wandb
        holder = {}
        initial = json.loads((run / 'initialization_proof.json').read_text())

        def digest(params):
            h = hashlib.sha256()
            for path, variable in sorted(params.filter(config.freeze_filter).flat_state().items()):
                h.update('/'.join(map(str,path)).encode())
                h.update(np.asarray(jax.device_get(variable.value)).tobytes())
            return h.hexdigest()

        def no_base_load(*a, **kw):
            raise RuntimeError('Resume must not load base weights or reset trained LoRA')
        official._load_weights_and_validate = no_base_load

        def init(cfg, rng, mesh, *, resume):
            assert resume, 'Checkpoint was not selected for resume'
            publish(status='compiling_resume_memory_check')
            shape, placement = old_init(cfg, rng, mesh, resume=True)
            groups = c.parameter_report(shape, cfg)
            atomic_json(report / 'parameter_report.json', groups)
            rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
            fn = jax.jit(functools.partial(official.train_step, cfg),
                in_shardings=(rep,placement,data), out_shardings=(placement,rep), donate_argnums=(1,))
            cache_was_enabled = jax.config.jax_enable_compilation_cache
            try:
                jax.config.update('jax_enable_compilation_cache', False)
                with sharding.set_mesh(mesh):
                    executable = lower_and_compile(fn, jax.random.split(jax.random.key(42))[0],
                        shape, cfg.model.inputs_spec(batch_size=32))
            finally:
                jax.config.update('jax_enable_compilation_cache', cache_was_enabled)
            if jax.config.jax_enable_compilation_cache or cache_reads:
                raise RuntimeError('Global compilation-cache guard failed; no checkpoint weights loaded')
            m = executable.memory_analysis()
            hlo = executable.as_text()
            peak = m.argument_size_in_bytes+m.temp_size_in_bytes+m.output_size_in_bytes-m.alias_size_in_bytes
            memory = {'arguments':m.argument_size_in_bytes,'temporary':m.temp_size_in_bytes,
                'outputs':m.output_size_in_bytes,'aliases':m.alias_size_in_bytes,
                'estimated_peak_bytes':peak,'pool_bytes':int(device['total_mib']*2**20*.9),
                'verified_on_selected_gpu':True,'resume':True,'memory_stats_source':'fresh_compile_cache_disabled','cache_setting_restored':jax.config.jax_enable_compilation_cache==cache_was_enabled,
                'cache_globally_disabled':True, 'cache_read_calls':len(cache_reads),
                'hlo_sha256':hashlib.sha256(hlo.encode()).hexdigest(),
                'hlo_declares_input_output_alias': 'input_output_alias=' in hlo.splitlines()[0]}
            atomic_json(report / 'compiled_memory.json', memory)
            if m.alias_size_in_bytes == 0:
                raise RuntimeError('Compiled donation statistics missing; memory safety unverified, not a proven OOM')
            if peak > memory['pool_bytes']-2**30:
                raise MemoryError('Resume B32+EMA exceeds checked memory; parameters remain unchanged')
            del executable
            publish(status='restoring_checkpoint', step=expected)
            return shape, placement

        def restore(manager, shape, loader, step=None):
            assert manager.latest_step() == plan['source_loop_step'], 'Checkpoint selection changed'
            state = old_restore(manager, shape, loader, step=plan['source_loop_step'])
            jax.block_until_ready(state)
            assert int(state.step) == expected
            c.parameter_report(state, config)
            counts = []
            for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
                if path and getattr(path[-1], 'name', None) == 'count':
                    counts.append(int(jax.device_get(value)))
            assert len(counts) == 2 and all(v == expected for v in counts), counts
            raw_sha, ema_sha = digest(state.params), digest(state.ema_params)
            assert raw_sha == ema_sha == initial['initial_frozen_sha256']
            holder['restored'] = True
            record = {'passed':True,'from_loop_step':plan['source_loop_step'],
                'from_optimizer_step':int(state.step),'adam_counts':counts,
                'fresh_optimizer':False,'base_weights_reloaded':False,'lora_b_reset':False,
                'raw_and_ema_restored':True,'frozen_sha256':raw_sha,
                'raw_and_ema_frozen_exactly_unchanged':True,
                'learning_rate_next_update':float(config.lr_schedule.create()(expected)),
                'seed':42,'rng_policy':'official seed split, fold_in(restored state.step)',
                'data_order_policy':'official resume: shuffle restarts; no historical video replay',
                'target_updates':20000, 'original_contract':original_contract, **proof}
            atomic_json(report / 'restore_proof.json', record)
            atomic_json(run / 'resume_proof.json', record)
            publish(status='training', step=expected, optimizer_restored=True, ema_restored=True)
            return state

        def save(manager, state, loader, loop_step):
            old_save(manager, state, loader, loop_step)
            publish(status='training', step=int(state.step), last_checkpoint_loop_step=int(loop_step))
            if int(state.step) == 20000:
                holder.update(final_state=state, final_loop_step=int(loop_step))

        def wandb_init(cfg, **kwargs):
            assert kwargs.get('resuming') is True
            old_wandb_init(cfg, **kwargs)
            # Bind after wandb.init, which replaces the global wandb.log.
            base_log = official.wandb.log
            def log(data, *a, **kw):
                if 'loss' in data:
                    row={'step':int(kw['step'])+1,'official_loop_step':int(kw['step']),
                         **{k:float(v) for k,v in data.items()}, 'optimizer_step':int(kw['step'])+1, 'source':'official_wandb_log', 'precision':'full_float'}
                    append_metrics(run / 'metrics.jsonl', row)
                    publish(status='training',step=row['step'],train_loss=row['loss'])
                return base_log(data,*a,**kw)
            official.wandb.log = log

        official.init_train_state=init
        official._checkpoints.restore_state=restore
        official._checkpoints.save_state=save
        official.init_wandb=wandb_init
        atomic_json(report / 'effective_config.json', {'resume':True,'source_checkpoint':plan['source_checkpoint'],
            'batch_size':32,'warmup_steps':500,'peak_lr':1e-5,'decay_steps':20000,'decay_lr':1e-6,
            'ema_decay':.99,'seed':42,'fps':15,'action_horizon':15,'target_updates':20000,
            'vit_lora_rank':8,'vit_lora_alpha':16,'other_layers':'dense_full',
            'official_main_unmodified':True,'official_loss_and_adam_unmodified':True,
            'data_order_policy':'official resume shuffle restart', **proof})
        logging.basicConfig(level=logging.INFO)
        official.main(config)
        assert holder.get('restored')
        state=holder.pop('final_state')
        assert int(state.step)==20000 and holder['final_loop_step']==19999
        final=Path(config.checkpoint_dir)/'19999'
        assert json.loads((final/'_CHECKPOINT_METADATA').read_text())['commit_timestamp_nsecs'] is not None
        assert (final/'params/_METADATA').is_file() and (final/'train_state/_METADATA').is_file()
        raw_sha,ema_sha=digest(state.params),digest(state.ema_params)
        assert raw_sha==ema_sha==initial['initial_frozen_sha256']
        atomic_json(run/'final_proof.json',{'step':20000,'checkpoint_loop_step':19999,
            'raw_and_ema_frozen_exactly_unchanged':True,'frozen_sha256':raw_sha,
            'final_checkpoint':str(final),'resumed_from_step':expected,
            'official_train_sha256':c.sha256(c.HERE/'official_source/scripts/train.py')})
        publish(status='completed',step=20000,final_checkpoint=str(final),frozen_parameters_verified=True)
    except BaseException as error:
        publish(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
