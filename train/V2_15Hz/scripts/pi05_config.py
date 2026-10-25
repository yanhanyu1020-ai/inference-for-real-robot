"""Pi0.5 configuration for 15 Hz data with independently configurable action chunks."""
import dataclasses
import os
import json
from chunk_settings import resolve_action_horizon
from pathlib import Path


def paths(args):
    candidate = Path(__file__).resolve().parents[3]
    if not (candidate/'third_party/openpi').is_dir():
        candidate = Path(__file__).resolve().parents[4]/'heterovla'
    project = Path(os.environ.get('HETEROVLA_ROOT') or candidate).expanduser().resolve()
    models = Path(os.environ.get('HETEROVLA_MODEL_ROOT') or project.parent / 'heterovla-model').expanduser().resolve()
    runs = Path(args.run_root or os.environ.get('HETEROVLA_RUN_ROOT') or models / 'V2_15Hz/runs').expanduser().resolve()
    return {
        'project': project,
        'openpi': Path(os.environ.get('OPENPI_ROOT') or project / 'third_party/openpi').expanduser().resolve(),
        'data': Path(args.data_root or os.environ.get('HETEROVLA_DATA_ROOT') or project.parent / 'heterovla-data/V2_15Hz_train35').expanduser().resolve(),
        'weights': Path(args.base_params or os.environ.get('PI05_PARAMS') or models / 'pretrained/pi05_base/params').expanduser().resolve(),
        'models': models,
        'runs': runs,
        'run': runs / args.name,
        'checkpoint_base': models / 'V2_15Hz/checkpoints',
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
    horizon = resolve_action_horizon(location['data'], getattr(args, 'action_horizon', None))
    info = json.loads((location['data'] / 'lerobot/local' / f'heterovla_v2_15hz_{split}' / 'meta/info.json').read_text())
    schema = json.loads((location['data'] / 'configs/schema.json').read_text())
    return cfg.TrainConfig(
        name='pi05_heterovla_v2_15hz_full', project_name='heterovla-pi05', exp_name=args.name,
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=horizon,
                        paligemma_variant='gemma_2b', action_expert_variant='gemma_300m',
                        remat_policy='nothing_saveable'),
        freeze_filter=nnx.Nothing,
        data=cfg.SimpleDataConfig(
            repo_id=f'local/heterovla_v2_15hz_{split}',
            assets=cfg.AssetsConfig(assets_dir=str(location['data'] / 'assets'), asset_id='heterovla_v2_15hz'),
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
        policy_metadata={'fps': info['fps'], 'physical_action_dim': 14, 'action_horizon': horizon,
                         'variant': 'full', 'dataset_version': location['data'].name, 'dataset_root': str(location['data']),
                         'validation_imputation': schema.get('validation_imputation'),
                         'no_source_session_leakage': schema.get('no_source_session_leakage', True),
                         'initialization': 'official_pi05_base_new_optimizer_step0',
                         'target_type': 'kinematic_imitation_not_operator_command_bc'})


def install_dataset_adapter(data_root):
    from openpi.training import data_loader
    from dataset_15hz import OpenpiV215HzDataset
    data_root = Path(data_root).expanduser().resolve()
    original = data_loader.create_torch_dataset
    if getattr(original, '_heterovla_15hz_adapter', False):
        if original._heterovla_15hz_root != Path(data_root):
            raise RuntimeError('V2 adapter was already installed for a different data root')
        return

    def create(data_config, action_horizon, model_config):
        for split in ('train', 'val'):
            if data_config.repo_id == f'local/heterovla_v2_15hz_{split}':
                if action_horizon != model_config.action_horizon:
                    raise ValueError('Dataset chunk length must match model.action_horizon')
                return OpenpiV215HzDataset(data_root, split, action_horizon)
        return original(data_config, action_horizon, model_config)

    create._heterovla_15hz_adapter = True
    create._heterovla_15hz_root = Path(data_root)
    data_loader.create_torch_dataset = create
