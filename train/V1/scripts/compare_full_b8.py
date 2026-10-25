"""Offline action sampling: true pretrained base versus the full-tuning run and the existing best LoRA reference."""
import argparse
import dataclasses
import gc
import importlib.util
import json
import os
from pathlib import Path
from path_config import OPENPI_ROOT
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--run-name', required=True)
    args = p.parse_args()
    assert args.gpu_uuid in ['GPU-87fdd428-76b9-b8ca-7898-fe5ed5907150']
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, JAX_PLATFORMS='cuda',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.68')
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(args.gpu_uuid)
    assert nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2).free >= 42*2**30, 'Insufficient free memory'
    assert not any(Path(f'/proc/{p.pid}').stat().st_uid == os.getuid()
                   for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)), 'Another own GPU process is active'
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import wandb
    from openpi import transforms
    from openpi.training import checkpoints, data_loader, sharding
    from experiment_configs import RUN_ROOT, PROJECT_ROOT, make_config, install_dataset_adapter
    from scan_common import atomic_json, cache_batches, GROUPS
    scan = RUN_ROOT/'full_ft_20261006'
    cache = RUN_ROOT/'lr_scan_20261005'
    run = scan/args.run_name
    best = json.loads((run/'best.json').read_text())
    record = json.loads((run/'config.json').read_text())
    out = scan/'action_comparison'/args.run_name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError('Comparison already completed')
    config = make_config('full', batch_size=8, peak_lr=record['peak_lr'], exp_name=args.run_name,
                         remat_policy='nothing_saveable')
    spec = importlib.util.spec_from_file_location('official_compare_trainer', OPENPI_ROOT / 'scripts/train.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    install_dataset_adapter()
    dc = config.data.create(config.assets_dirs, config.model)
    inverse = transforms.Unnormalize({'actions': dc.norm_stats['actions']}, use_quantiles=dc.use_quantile_norm)
    mesh = sharding.make_mesh(1)
    _, init_rng = jax.random.split(jax.random.key(42))
    report = {'best_run': args.run_name, 'best_step': best['best_step'],
              'checkpoint': best['best_checkpoint'], 'seed': 42, 'integration_steps': 10,
              'windows': 710, 'horizon': 10,
              'group_errors': 'normalized action space; physical per-dimension errors carry distinct units',
              'base_definition': 'original pretrained full architecture, no LoRA adapters; full float32 parameter precision',
              'results': {}}
    predictions = {}
    target = None
    for label in ['base', 'best']:
        t0 = time.monotonic()
        state, _ = official.init_train_state(config, init_rng, mesh, resume=(label=='best'))
        if label == 'best':
            manager, _ = checkpoints.initialize_checkpoint_dir(run/'best', keep_period=None,
                                                               overwrite=False, resume=True)
            state = checkpoints.restore_state(manager, state, None, step=best['best_step'])
            manager.close()
        jax.block_until_ready(state)
        state = dataclasses.replace(state, opt_state=())  # Inference does not need Adam moments.
        model_def = state.model_def
        def predict(params, rng, obs):
            model = nnx.merge(model_def, params)
            model.eval()
            return model.sample_actions(rng, obs, num_steps=10)[..., :14]
        sample = jax.jit(predict)
        preds, targets = [], []
        for offset, n, (obs, actions) in cache_batches(cache/'val_cache'):
            pred = np.asarray(sample(state.params, jax.random.fold_in(jax.random.key(42), offset), obs))[:n]
            assert np.isfinite(pred).all(), 'Non-finite sampled action'
            preds.append(pred)
            targets.append(actions[:n, :, :14])
            if offset % 160 == 0:
                print(f'PREDICT {label} {offset}/710', flush=True)
        norm = np.concatenate(preds)
        target_norm = np.concatenate(targets)
        physical = inverse({'actions': norm})['actions']
        truth = inverse({'actions': target_norm})['actions']
        if target is not None:
            np.testing.assert_array_equal(target, truth)
        target = truth
        predictions[label] = physical
        np.savez(out/f'{label}_predictions.npz', normalized=norm, physical=physical,
                 target_normalized=target_norm, target_physical=truth)
        error = norm-target_norm
        physical_error = physical-truth
        report['results'][label] = {
            'normalized_rmse': float(np.sqrt(np.mean(error**2))),
            'normalized_mae': float(np.mean(np.abs(error))),
            'groups': {k: {'normalized_rmse': float(np.sqrt(np.mean(error[...,a:b]**2))),
                           'normalized_mae': float(np.mean(np.abs(error[...,a:b])))} for k,(a,b) in GROUPS.items()},
            'physical_rmse_per_dim': np.sqrt(np.mean(physical_error**2,axis=(0,1))).tolist(),
            'physical_mae_per_dim': np.mean(np.abs(physical_error),axis=(0,1)).tolist(),
            'seconds': time.monotonic()-t0}
        del sample, state
        gc.collect()
        jax.clear_caches()
    audit = json.loads((cache/'preflight.json').read_text())
    names = audit['splits']['val']['action_names']
    units = ['m/s','m/s','rad/s'] + ['rad']*6 + ['m'] + ['m/s','m/s','m/s','rad/s']
    report.update(action_names=names, units=units)
    reference = cache/'action_comparison/dual_lora_lr1e-4_s42_20261005'
    with np.load(reference/'best_predictions.npz') as file:
        np.testing.assert_array_equal(target, file['target_physical'])
        predictions['lora'] = file['physical'].copy()
    report['results']['lora'] = json.loads((reference/'metrics.json').read_text())['results']['best']
    report['lora_reference'] = {'run': 'dual_lora_lr1e-4_s42_20261005', 'step': 700,
        'precision_note': 'LoRA freezes backbone parameters in bf16; full tuning uses official float32 parameters'}
    times = 3.0 + np.arange(710)/10
    paths = []
    for group,(a,b) in GROUPS.items():
        fig, axes = plt.subplots(b-a, 1, figsize=(13, 2.1*(b-a)), sharex=True, squeeze=False)
        for axis, dim in zip(axes[:,0], range(a,b)):
            axis.plot(times, target[:,0,dim], color='black', lw=1.3, label='Target')
            axis.plot(times, predictions['base'][:,0,dim], alpha=.65, lw=.8, label='Pretrained base')
            axis.plot(times, predictions['best'][:,0,dim], alpha=.85, lw=.8, label='Best full tuning')
            axis.plot(times, predictions['lora'][:,0,dim], color='green', alpha=.65, lw=.8, label='Best LoRA (700)')
            axis.set_ylabel(f'{names[dim]}\n({units[dim]})', fontsize=8)
            axis.grid(alpha=.2)
        axes[0,0].legend(ncol=4, fontsize=9)
        axes[-1,0].set_xlabel('Source task_06 time (s); first action of every validation window')
        fig.suptitle(f'{group}: identical observations and sampling seeds; best step {best["best_step"]}')
        fig.tight_layout()
        path = out/f'{group}_trajectory.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    wandb.init(project='heterovla-pi05', name='action_compare_'+args.run_name,
               group='pi05-v1-full-vs-lora-20261006', config=report)
    wandb.log({p.stem: wandb.Image(str(p)) for p in paths})
    rows = [[label,group,vals['normalized_rmse'],vals['normalized_mae']]
            for label,result in report['results'].items() for group,vals in result['groups'].items()]
    wandb.log({'group_errors': wandb.Table(columns=['model','group','normalized_rmse','normalized_mae'],data=rows)})
    report['wandb_url'] = wandb.run.url
    atomic_json(out/'metrics.json', report)
    wandb.finish()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
