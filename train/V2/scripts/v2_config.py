"""Lazy V2 full-finetuning configuration; importing this module allocates no model."""
import dataclasses
import os
from pathlib import Path


def paths(args):
    project = Path(os.environ.get('HETEROVLA_ROOT') or Path(__file__).resolve().parents[3]).expanduser().resolve()
    models = Path(os.environ.get('HETEROVLA_MODEL_ROOT') or Path.home() / 'heterovla-model').expanduser().resolve()
    runs = Path(args.run_root or os.environ.get('HETEROVLA_RUN_ROOT') or models / 'V2/runs').expanduser().resolve()
    return {
        'project': project,
        'openpi': Path(os.environ.get('OPENPI_ROOT') or project / 'third_party/openpi').expanduser().resolve(),
        'data': Path(args.data_root or os.environ.get('HETEROVLA_DATA_ROOT') or Path.home() / 'heterovla-data/aligned/V2').expanduser().resolve(),
        'weights': Path(args.base_params or os.environ.get('PI05_PARAMS') or models / 'pretrained/pi05_base/params').expanduser().resolve(),
        'models': models,
        'runs': runs,
        'run': runs / args.name,
        'checkpoint_base': models / 'V2/checkpoints',
    }


@dataclasses.dataclass(frozen=True)
class Output14:
    def __call__(self, data):
        return {**data, 'actions': data['actions'][..., :14]}


@dataclasses.dataclass(frozen=True)
class JointTransforms:
    def __call__(self, model_config):
        from openpi import transforms
        return transforms.Group(outputs=[Output14()])


def make_config(args, split='train'):
    import flax.nnx as nnx
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as cfg, optimizer, weight_loaders
    if split not in ('train', 'val'):
        raise ValueError(split)
    location = paths(args)
    return cfg.TrainConfig(
        name='pi05_heterovla_v2_full', project_name='heterovla-pi05', exp_name=args.name,
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=10,
                        paligemma_variant='gemma_2b', action_expert_variant='gemma_300m',
                        remat_policy='nothing_saveable'),
        freeze_filter=nnx.Nothing,
        data=cfg.SimpleDataConfig(
            repo_id=f'local/heterovla_v2_{split}',
            assets=cfg.AssetsConfig(assets_dir=str(location['data'] / 'assets'), asset_id='heterovla_v2'),
            data_transforms=JointTransforms(),
            base_config=cfg.DataConfig(action_sequence_keys=('action',), prompt_from_task=False)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(location['weights'])),
        optimizer=optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=args.peak_lr,
                                                 decay_steps=args.steps, decay_lr=args.peak_lr / 10),
        ema_decay=None, seed=args.seed, fsdp_devices=1,
        batch_size=args.batch_size, num_workers=args.num_workers, num_train_steps=args.steps,
        save_interval=args.save_interval, keep_period=args.keep_period, log_interval=args.log_interval,
        resume=args.resume, overwrite=False, wandb_enabled=True,
        assets_base_dir=str(location['run'] / 'assets'), checkpoint_base_dir=str(location['checkpoint_base']),
        policy_metadata={'fps': 10, 'physical_action_dim': 14, 'action_horizon': 10,
                         'variant': 'full', 'dataset_version': 'V2', 'dataset_root': str(location['data']),
                         'initialization': 'official_pi05_base_new_optimizer_step0',
                         'target_type': 'kinematic_imitation_not_operator_command_bc'})


def install_dataset_adapter(data_root):
    from openpi.training import data_loader
    from dataset_v2 import OpenpiV2Dataset
    original = data_loader.create_torch_dataset
    if getattr(original, '_heterovla_v2_adapter', False):
        if original._heterovla_v2_root != Path(data_root):
            raise RuntimeError('V2 adapter was already installed for a different data root')
        return

    def create(data_config, action_horizon, model_config):
        for split in ('train', 'val'):
            if data_config.repo_id == f'local/heterovla_v2_{split}':
                return OpenpiV2Dataset(data_root, split, action_horizon)
        return original(data_config, action_horizon, model_config)

    create._heterovla_v2_adapter = True
    create._heterovla_v2_root = Path(data_root)
    data_loader.create_torch_dataset = create
