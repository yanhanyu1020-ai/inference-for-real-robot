"""Independent V3 LoRA configuration; the immutable dataset remains aligned/V2."""
import dataclasses
import os
from pathlib import Path

HEADS = ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out')
MODEL_NAME = 'pi05_heterovla_v3_lora_frozen_vit'
PARENT_RUN = 'full_b32_lr1e-5_s42_from600_20261016_170448'


def path_parts(path):
    return tuple(map(str, path)) if not isinstance(path, str) else tuple(path.split('/'))


def parameter_group(path):
    parts = path_parts(path)
    if parts[:2] == ('PaliGemma', 'img'):
        return 'vit'
    if parts[:2] == ('PaliGemma', 'llm'):
        if any('lora' in part.lower() for part in parts):
            return 'action_expert_lora' if any(part.endswith('_1') for part in parts) else 'paligemma_lora'
        return 'gemma_base'
    if parts and parts[0] in HEADS:
        return parts[0]
    return 'other'


def trainable_path(path):
    return parameter_group(path) in ('paligemma_lora', 'action_expert_lora', *HEADS)


def frozen_filter(path, value):
    del value
    return not trainable_path(path)


def source_step(args):
    return int(args.source_step)


def phase_steps(args):
    return args.steps - source_step(args)


def paths(args):
    project = Path(os.environ.get('HETEROVLA_ROOT') or Path.home() / 'heterovla').expanduser().resolve()
    models = Path(os.environ.get('HETEROVLA_MODEL_ROOT') or Path.home() / 'heterovla-model').expanduser().resolve()
    runs = Path(args.run_root or os.environ.get('HETEROVLA_V3_RUN_ROOT') or models / 'V3/runs').expanduser().resolve()
    parent_runs = Path(args.parent_run_root or models / 'V2/runs').expanduser().resolve()
    if runs == parent_runs or runs.is_relative_to(parent_runs):
        raise ValueError('V3 output cannot be inside the immutable V2 parent run root')
    return dict(project=project, openpi=Path(os.environ.get('OPENPI_ROOT') or project / 'third_party/openpi').expanduser().resolve(),
                data=Path(args.data_root or os.environ.get('HETEROVLA_V3_DATA_ROOT') or Path.home() / 'heterovla-data/aligned/V2').expanduser().resolve(),
                weights=Path(args.warm_start_params or models / 'V2/runs' / PARENT_RUN / 'best/1000/params').expanduser().resolve(),
                models=models, runs=runs, parent_runs=parent_runs, run=runs / args.name,
                checkpoint_base=models / 'V3/checkpoints')


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
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as cfg, optimizer, weight_loaders
    if split not in ('train', 'val'):
        raise ValueError(split)
    location = paths(args)
    return cfg.TrainConfig(
        name=MODEL_NAME, project_name='heterovla-pi05', exp_name=args.name,
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=10,
                        paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora',
                        remat_policy='nothing_saveable'), freeze_filter=frozen_filter,
        data=cfg.SimpleDataConfig(repo_id=f'local/heterovla_v2_{split}',
            assets=cfg.AssetsConfig(assets_dir=str(location['data'] / 'assets'), asset_id='heterovla_v2'),
            data_transforms=JointTransforms(),
            base_config=cfg.DataConfig(action_sequence_keys=('action',), prompt_from_task=False)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(location['weights'])),
        optimizer=optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=args.peak_lr,
                                                 decay_steps=phase_steps(args), decay_lr=args.peak_lr / 10),
        ema_decay=None, seed=args.seed, fsdp_devices=1,
        batch_size=args.batch_size, num_workers=args.num_workers, num_train_steps=phase_steps(args),
        save_interval=args.save_interval, keep_period=args.keep_period, log_interval=args.log_interval,
        resume=args.resume, overwrite=False, wandb_enabled=True,
        assets_base_dir=str(location['run'] / 'assets'), checkpoint_base_dir=str(location['checkpoint_base']),
        policy_metadata=dict(fps=10, physical_action_dim=14, action_horizon=10, variant='lora_frozen_vit',
                             dataset_version='V2', dataset_root=str(location['data']), source_step=source_step(args),
                             initialization='parent_dense_best1000_zero_lora_new_optimizer',
                             target_type='kinematic_imitation_not_operator_command_bc'))


def install_dataset_adapter(data_root):
    from openpi.training import data_loader
    from dataset_v3 import OpenpiV2Dataset
    original = data_loader.create_torch_dataset
    if getattr(original, '_heterovla_v3_adapter', False):
        if original._heterovla_v3_root != Path(data_root):
            raise RuntimeError('V3 adapter already installed for different data')
        return

    def create(data_config, action_horizon, model_config):
        for split in ('train', 'val'):
            if data_config.repo_id == f'local/heterovla_v2_{split}':
                return OpenpiV2Dataset(data_root, split, action_horizon)
        return original(data_config, action_horizon, model_config)

    create._heterovla_v3_adapter = True
    create._heterovla_v3_root = Path(data_root)
    data_loader.create_torch_dataset = create
