"""Run the pinned official V3 strategy from pi05_base on merged V3 data.

The upstream main loop, loss, Adam, LR, and trainable EMA are unchanged.
Runtime wrappers verify initialization, compilation memory, and saved state;
the existing frozen-ViT EMA correction is retained.
"""
from __future__ import annotations

import os

# JAX 0.5.3 memoizes persistent-cache eligibility at its first compilation.
# A cached executable can lose donation statistics. Disable it before ANY
# JAX/model/data import, including imports made by configuration.make_config().
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'

import argparse
import datetime
import functools
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback

from runtime_reporting import atomic_json, append_metrics, dumps_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    import configuration as c
    from queue_training import query_gpus, available, verify_preflight

    assert args.run_name == c.RUN_NAME
    contract = verify_preflight(c.HERE)
    device = next(g for g in query_gpus() if g['uuid'] == args.gpu)
    if not available(device):
        raise RuntimeError('GPU ceased to be completely idle; refusing this training attempt')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
        WANDB_MODE='offline', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false')
    run = c.MODELS / 'V3/runs' / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_DIR'] = str(run)
    started = time.monotonic()
    status = {'status':'initializing', 'pid':os.getpid(), 'gpu':args.gpu,
        'run_name':args.run_name, 'step':0, 'target_steps':20000, **contract}
    original_cache_read = None
    compiler = None

    def publish(**values):
        status.update(values, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started)
        atomic_json(run / 'runtime_status.json', status)
        print(dumps_record(status), flush=True)

    try:
        publish()
        c.bootstrap()
        import jax
        assert jax.config.jax_enable_compilation_cache is False
        from jax._src import compiler
        cache_reads = []
        original_cache_read = compiler._cache_read

        def observe_cache_read(*a, **kw):
            cache_reads.append(1)
            return original_cache_read(*a, **kw)

        compiler._cache_read = observe_cache_read
        cfg = c.make_config()
        c.install_data_adapter()
        official = c.official_trainer()
        import numpy as np
        from openpi.training import sharding
        from compilation_compat import lower_and_compile

        manifest = json.loads((c.HERE / 'manifest.json').read_text())
        for name, digest in manifest['runtime_files'].items():
            assert c.sha256(name) == digest, name
        dataset = manifest['dataset']
        expected_data = {'train_source_count':55, 'val_source_count':7,
                         'train_windows':34013, 'val_windows':743}
        for name, expected in expected_data.items():
            assert dataset[name] == expected, (name, dataset[name], expected)
        norm_path = Path(dataset['norm_stats_path'])
        assert norm_path.resolve() == (c.DATA / 'assets/heterovla_v3/norm_stats.json').resolve()
        assert c.sha256(norm_path) == dataset['norm_stats_sha256'], 'Normalization changed'
        base_params = c.MODELS / 'pretrained/pi05_base/params'
        assert Path(cfg.weight_loader.params_path).resolve() == base_params.resolve()
        assert cfg.resume is False and cfg.overwrite is False
        assert cfg.num_train_steps == 20000 and cfg.batch_size == 32 and cfg.ema_decay == .99
        assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'

        atomic_json(run / 'effective_config.json', {
            'official_commit':c.OFFICIAL_COMMIT, 'official_train_sha256':c.OFFICIAL_TRAIN_SHA,
            'batch_size':32, 'warmup_steps':500, 'peak_lr':1e-5, 'decay_steps':20000,
            'decay_lr':1e-6, 'ema_decay':.99, 'fps':15, 'action_horizon':15,
            'steps':20000, 'seed':42, 'resume':False, 'source_base_params':str(base_params),
            'initialization':'official_pi05_base_new_optimizer_step0',
            'vit_lora_rank':8, 'vit_lora_alpha':16, 'other_layers':'dense_full',
            'data_root':str(c.DATA.resolve()), 'dataset':{**dataset, **expected_data},
            'norm_stats_path':str(norm_path), 'norm_stats_sha256':dataset['norm_stats_sha256'],
            'official_main_unmodified':True, 'official_loss_and_adam_unmodified':True,
            'ema_frozen_dense_policy':'copy_raw_frozen_parameters_exactly',
            'validation_in_official_loop':False, 'cache_globally_disabled':True, **contract})

        official.train_step = c.corrected_train_step(official.train_step)
        original_init = official.init_train_state
        original_save = official._checkpoints.save_state
        original_wandb_init = official.init_wandb
        holder = {'metric_records':0}

        def digest(params):
            h = hashlib.sha256()
            for path, variable in sorted(params.filter(cfg.freeze_filter).flat_state().items()):
                h.update('/'.join(map(str, path)).encode())
                h.update(np.asarray(jax.device_get(variable.value)).tobytes())
            return h.hexdigest()

        def adam_counts(state):
            counts = []
            for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
                if path and getattr(path[-1], 'name', None) == 'count':
                    counts.append(int(jax.device_get(value)))
            return counts

        def assert_no_cache_reads():
            if jax.config.jax_enable_compilation_cache or cache_reads:
                raise RuntimeError('Global compilation-cache guard failed; compilation statistics untrusted')

        def init(config, rng, mesh, *, resume):
            assert resume is False, 'This run must initialize from pi05_base, never a trained checkpoint'
            assert config.resume is False
            assert Path(config.weight_loader.params_path).resolve() == base_params.resolve()
            publish(status='compiling_memory_check')
            # Upstream resume=True produces abstract state only: no weights and
            # no optimizer values are restored. Real initialization below is fresh.
            shape, placement = original_init(config, rng, mesh, resume=True)
            groups = c.parameter_report(shape, config)
            atomic_json(run / 'parameter_report.json', groups)
            rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
            pstep = jax.jit(functools.partial(official.train_step, config),
                in_shardings=(rep, placement, data), out_shardings=(placement, rep),
                donate_argnums=(1,))
            with sharding.set_mesh(mesh):
                compiled = lower_and_compile(pstep, jax.random.key(42), shape,
                    config.model.inputs_spec(batch_size=32))
            assert_no_cache_reads()
            memory = compiled.memory_analysis()
            hlo = compiled.as_text()
            estimate = (memory.argument_size_in_bytes + memory.temp_size_in_bytes
                        + memory.output_size_in_bytes - memory.alias_size_in_bytes)
            report = {'arguments':memory.argument_size_in_bytes,
                'temporary':memory.temp_size_in_bytes, 'outputs':memory.output_size_in_bytes,
                'aliases':memory.alias_size_in_bytes, 'estimated_peak_bytes':estimate,
                'pool_bytes':int(device['total_mib'] * 2**20 * .9),
                'verified_on_selected_gpu':True,
                'memory_stats_source':'fresh_compile_global_cache_disabled',
                'cache_globally_disabled':True, 'cache_read_calls':len(cache_reads),
                'hlo_sha256':hashlib.sha256(hlo.encode()).hexdigest(),
                'hlo_declares_input_output_alias':'input_output_alias=' in hlo.splitlines()[0]}
            atomic_json(run / 'compiled_memory.json', report)
            if memory.alias_size_in_bytes == 0:
                raise RuntimeError('Compiled donation statistics missing; memory safety unverified, not a proven OOM')
            if estimate > report['pool_bytes'] - 2**30:
                raise MemoryError('B32+EMA requested configuration exceeds checked GPU memory; no automatic hyperparameter changes')
            del compiled
            publish(status='loading_base_weights')
            state, placement = original_init(config, rng, mesh, resume=False)
            jax.block_until_ready(state)
            c.parameter_report(state, config)
            assert int(state.step) == 0, 'Fresh official TrainState must start at optimizer step 0'
            counts = adam_counts(state)
            assert len(counts) == 2 and all(value == 0 for value in counts), counts
            b_values = [np.asarray(jax.device_get(variable.value))
                for path, variable in state.params.flat_state().items() if path[-1] == 'lora_b']
            assert len(b_values) == 6 and all(not np.any(value) for value in b_values), \
                'All six scanned ViT LoRA B tensors must initialize to zero'
            holder['initial_frozen_sha256'] = digest(state.params)
            holder['initial_ema_frozen_sha256'] = digest(state.ema_params)
            assert holder['initial_frozen_sha256'] == holder['initial_ema_frozen_sha256']
            assert_no_cache_reads()
            atomic_json(run / 'initialization_proof.json', {
                'all_lora_b_zero':True, 'dense_shapes_verified_in_cpu_preflight':True,
                'fresh_optimizer_step0':True, 'initial_optimizer_step':0,
                'adam_counts':counts, 'resume':False, 'source_base_params':str(base_params),
                'cache_globally_disabled':True, 'cache_read_calls':len(cache_reads),
                'initial_frozen_sha256':holder['initial_frozen_sha256'],
                'initial_ema_frozen_sha256':holder['initial_ema_frozen_sha256']})
            publish(status='training', step=0)
            return state, placement

        def save(manager, state, loader, loop_step):
            updates = int(state.step)
            publish(status='saving_checkpoint', step=updates, checkpoint_loop_step=int(loop_step))
            original_save(manager, state, loader, loop_step)
            publish(status='training', step=updates, last_checkpoint_loop_step=int(loop_step))
            if updates == 20000:
                holder.update(final_state=state, final_loop_step=int(loop_step))

        def wandb_init(config, **kwargs):
            assert kwargs.get('resuming') is False
            result = original_wandb_init(config, **kwargs)
            # wandb.init rebinds the module's log function. Install the observer
            # afterwards so runtime_status follows official loss records.
            original_log = official.wandb.log

            def log(data, *a, **kw):
                if 'loss' in data:
                    loop_step = int(kw['step'])
                    row = {**{name:float(value) for name, value in data.items()},
                        'step':loop_step+1, 'optimizer_step':loop_step+1,
                        'official_loop_step':loop_step, 'source':'official_wandb_log',
                        'precision':'full_float'}
                    append_metrics(run / 'metrics.jsonl', row)
                    holder['metric_records'] += 1
                    publish(status='training', step=row['step'], train_loss=row['loss'])
                return original_log(data, *a, **kw)

            official.wandb.log = log
            return result

        official.init_train_state = init
        official._checkpoints.save_state = save
        official.init_wandb = wandb_init
        logging.basicConfig(level=logging.INFO)
        official.main(cfg)
        assert holder['metric_records'] > 0, 'No official loss records reached the observer'
        state = holder.pop('final_state')
        assert int(state.step) == 20000 and holder['final_loop_step'] == 19999
        final = Path(cfg.checkpoint_dir) / '19999'
        metadata = json.loads((final / '_CHECKPOINT_METADATA').read_text())
        assert int(metadata['commit_timestamp_nsecs']) > 0, 'Final checkpoint not committed'
        assert (final / 'params/_METADATA').is_file() and (final / 'train_state/_METADATA').is_file()
        raw_sha, ema_sha = digest(state.params), digest(state.ema_params)
        assert raw_sha == ema_sha == holder['initial_frozen_sha256']
        counts = adam_counts(state)
        assert len(counts) == 2 and all(value == 20000 for value in counts), counts
        assert_no_cache_reads()
        atomic_json(run / 'final_proof.json', {
            'step':20000, 'checkpoint_loop_step':19999, 'adam_counts':counts,
            'raw_and_ema_frozen_exactly_unchanged':True, 'frozen_sha256':raw_sha,
            'final_checkpoint':str(final), 'checkpoint_commit_timestamp_nsecs':metadata['commit_timestamp_nsecs'],
            'official_train_sha256':c.sha256(c.HERE / 'official_source/scripts/train.py'),
            'cache_globally_disabled':True, 'cache_read_calls':len(cache_reads),
            'metric_records':holder['metric_records']})
        publish(status='completed', step=20000, final_checkpoint=str(final), frozen_parameters_verified=True)
    except BaseException as error:
        values = {'status':'failed', 'error':f'{type(error).__name__}: {error}',
                  'traceback':traceback.format_exc()}
        if hasattr(error, 'nonfinite_metrics'):
            values.update(nonfinite_metrics=error.nonfinite_metrics, failed_metric_record=error.record,
                          step=error.record.get('step', status['step']))
        publish(**values)
        raise
    finally:
        if compiler is not None and original_cache_read is not None:
            compiler._cache_read = original_cache_read


if __name__ == '__main__':
    main()
