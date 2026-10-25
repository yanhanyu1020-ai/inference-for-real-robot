"""Four single-GPU pi0.5 configurations; registering them never starts training."""
import dataclasses

import flax.nnx as nnx
from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils
from openpi.training import config as cfg, data_loader, optimizer, weight_loaders

from path_config import DATA_ROOT, PROJECT_ROOT, RUN_ROOT, WEIGHTS
from dataset_v1 import OpenpiV1Dataset

VARIANTS = ('action_lora', 'dual_lora', 'expert_full', 'full')


@dataclasses.dataclass(frozen=True)
class Output14:
    def __call__(self, data):
        return {**data, 'actions': data['actions'][..., :14]}


@dataclasses.dataclass(frozen=True)
class JointTransforms:
    def __call__(self, model_config):
        return transforms.Group(outputs=[Output14()])


def make_config(variant, *, batch_size=8, peak_lr=2.5e-5, steps=2000, exp_name='prepared', split='train', remat_policy='nothing_saveable'):
    if variant not in VARIANTS or split not in ('train', 'val'):
        raise ValueError((variant, split))
    language_lora = variant in ('dual_lora', 'expert_full')
    expert_lora = variant in ('action_lora', 'dual_lora')
    model = Pi0Config(pi05=True, action_dim=32, action_horizon=10, remat_policy=remat_policy,
                     paligemma_variant='gemma_2b_lora' if language_lora else 'gemma_2b',
                     action_expert_variant='gemma_300m_lora' if expert_lora else 'gemma_300m')
    if variant == 'full':
        freeze = nnx.Nothing
    else:
        filters = [model.get_freeze_filter(), nnx_utils.PathRegex('.*img.*')]
        if variant == 'action_lora':
            filters.append(nnx.All(nnx_utils.PathRegex('.*llm.*'),
                                   nnx.Not(nnx_utils.PathRegex('.*llm.*_1.*'))))
        freeze = nnx.Any(*filters)
    return cfg.TrainConfig(
        name=f'pi05_heterovla_{variant}', project_name='heterovla-pi05', exp_name=exp_name,
        model=model, freeze_filter=freeze,
        data=cfg.SimpleDataConfig(
            repo_id=f'local/heterovla_v1_{split}',
            assets=cfg.AssetsConfig(assets_dir=str(DATA_ROOT / 'assets'), asset_id='heterovla_v1'),
            data_transforms=JointTransforms(),
            base_config=cfg.DataConfig(action_sequence_keys=('action',), prompt_from_task=False)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(WEIGHTS)),
        batch_size=batch_size, num_workers=2, num_train_steps=steps,
        save_interval=500, keep_period=500, log_interval=25,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=peak_lr,
                                                 decay_steps=steps, decay_lr=peak_lr / 10),
        ema_decay=None, seed=42, fsdp_devices=1, wandb_enabled=True,
        assets_base_dir=str(RUN_ROOT / 'assets'), checkpoint_base_dir=str(RUN_ROOT / 'checkpoints'),
        policy_metadata={'fps': 10, 'physical_action_dim': 14, 'action_horizon': 10,
                         'variant': variant, 'dataset_root': str(DATA_ROOT),
                         'target_type': 'kinematic_imitation_not_operator_command_bc'})


def install_dataset_adapter():
    if getattr(data_loader.create_torch_dataset, '_heterovla_adapter', False):
        return
    original = data_loader.create_torch_dataset

    def create(data_config, action_horizon, model_config):
        for split in ('train', 'val'):
            if data_config.repo_id == f'local/heterovla_v1_{split}':
                return OpenpiV1Dataset(DATA_ROOT, split, action_horizon)
        return original(data_config, action_horizon, model_config)

    create._heterovla_adapter = True
    data_loader.create_torch_dataset = create


def summary(config):
    return {'name': config.name, 'project': config.project_name,
            'variant': config.policy_metadata['variant'], 'seed': config.seed,
            'batch_size': config.batch_size, 'num_train_steps': config.num_train_steps,
            'remat_policy': config.model.remat_policy,
            'peak_lr': config.lr_schedule.peak_lr, 'action_horizon': config.model.action_horizon,
            'physical_action_dim': 14, 'model_action_dim': config.model.action_dim,
            'language_variant': config.model.paligemma_variant,
            'expert_variant': config.model.action_expert_variant,
            'weights': config.weight_loader.params_path, 'data_root': str(DATA_ROOT),
            'checkpoint_dir': str(config.checkpoint_dir), 'ema_decay': config.ema_decay,
            'wandb_enabled': config.wandb_enabled,
            'batch_size_status': 'see_reports_for_measured_variant_batch_and_remat_combinations',
            'gradient_accumulation': 'not_implemented_in_this_official_trainer'}
