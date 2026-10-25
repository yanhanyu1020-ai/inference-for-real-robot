"""Locked user-requested V3 experiment using the upstream TrainConfig."""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = Path('/meta_eon_cfs/home/hph/heterovla')
DATA = PROJECT.parent / 'heterovla-data/aligned/V3'
MODELS = PROJECT.parent / 'heterovla-model'
RUN_NAME = 'vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_data20261021'
CONFIG_NAME = 'pi05_heterovla_v3_vitlora'
OFFICIAL_COMMIT = '15a9616a00943ada6c20a0f158e3adb39df2ccac'
OFFICIAL_TRAIN_SHA = '22f98a06aaa6e68e0f725d727ab95398238d1cae8a46bb94e829f330758a9baf'

def bootstrap():
    sys.path.insert(0, str(HERE / 'official_source/src'))
    sys.path.insert(0, str(HERE / 'data_adapter'))

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def official_trainer():
    path = HERE / 'official_source/scripts/train.py'
    assert sha256(path) == OFFICIAL_TRAIN_SHA, 'Official train.py changed'
    spec = importlib.util.spec_from_file_location('v3_upstream_trainer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@dataclasses.dataclass(frozen=True)
class FreezeViTDense:
    def __call__(self, path, value):
        return tuple(path[:2]) == ('PaliGemma', 'img') and not any(str(p).startswith('lora_') for p in path)

@dataclasses.dataclass(frozen=True)
class PhysicalOutput:
    def __call__(self, data):
        return {**data, 'actions': data['actions'][..., :14]}

@dataclasses.dataclass(frozen=True)
class JointTransforms:
    def __call__(self, model_config):
        from openpi import transforms
        return transforms.Group(outputs=[PhysicalOutput()])

def make_config(run_name=RUN_NAME):
    bootstrap()
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as cfg, optimizer, weight_loaders
    import vit_lora
    vit_lora.install(rank=8, alpha=16)
    return cfg.TrainConfig(
        name=CONFIG_NAME, project_name='heterovla-pi05', exp_name=run_name,
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=15,
                        paligemma_variant='gemma_2b', action_expert_variant='gemma_300m'),
        freeze_filter=FreezeViTDense(),
        data=cfg.SimpleDataConfig(repo_id='local/heterovla_v3_train',
            assets=cfg.AssetsConfig(assets_dir=str(DATA/'assets'), asset_id='heterovla_v3'),
            data_transforms=JointTransforms(),
            base_config=cfg.DataConfig(action_sequence_keys=('action',), prompt_from_task=False)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(MODELS/'pretrained/pi05_base/params')),
        optimizer=optimizer.AdamW(b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=1e-5,
                                                decay_steps=20000, decay_lr=1e-6),
        ema_decay=.99, seed=42, batch_size=32, num_workers=2, num_train_steps=20000,
        save_interval=1000, keep_period=5000, log_interval=100, fsdp_devices=1,
        checkpoint_base_dir=str(MODELS/'V3/checkpoints'),
        assets_base_dir=str(MODELS/'V3/runs'/run_name/'assets'),
        resume=False, overwrite=False, wandb_enabled=True,
        policy_metadata={'fps':15,'action_horizon':15,'physical_action_dim':14,
            'vit_lora_rank':8,'vit_lora_alpha':16,'other_layers':'dense_full',
            'dataset_root':str(DATA.resolve()),'asset_id':'heterovla_v3',
            'official_commit':OFFICIAL_COMMIT,'ema_decay':.99,
            'ema_frozen_dense_policy':'copy_raw_frozen_parameters_exactly',
            'architecture_adapter':str(HERE/'vit_lora.py')})

def install_data_adapter():
    from openpi.training import data_loader
    from dataset_v3 import OpenpiV3Dataset
    original = data_loader.create_torch_dataset
    def create(data_config, action_horizon, model_config):
        if data_config.repo_id in ('local/heterovla_v3_train','local/heterovla_v3_val'):
            assert action_horizon == model_config.action_horizon == 15
            return OpenpiV3Dataset(DATA, data_config.repo_id.rsplit('_',1)[1], action_horizon)
        return original(data_config, action_horizon, model_config)
    data_loader.create_torch_dataset = create

def corrected_train_step(original):
    """Delegate loss/gradient/Adam/trainable EMA unchanged; freeze the EMA base too.

    Upstream bf16 old*.99+raw*.01 can drift even if old==raw. Only the
    frozen ViT leaves of its returned EMA are replaced; raw state is untouched.
    """
    import dataclasses
    from flax import nnx
    def step(config, rng, state, batch):
        new, info = original(config, rng, state, batch)
        if new.ema_params is not None:
            ema = nnx.State.merge(new.ema_params.filter(config.trainable_filter),
                                  new.params.filter(config.freeze_filter))
            new = dataclasses.replace(new, ema_params=ema)
        return new, info
    return step

def parameter_report(state, config):
    import math
    flat = state.params.flat_state()
    trained = set(state.params.filter(config.trainable_filter).flat_state())
    groups = {k:{'parameters':0,'trainable':0,'dtypes':set()} for k in ('vit_dense','vit_lora','other_dense')}
    for path, variable in flat.items():
        image = tuple(path[:2]) == ('PaliGemma','img')
        lora = any(str(p).startswith('lora_') for p in path)
        assert image or not lora, 'Unexpected non-ViT LoRA'
        group = groups['vit_lora' if image and lora else 'vit_dense' if image else 'other_dense']
        n = math.prod(variable.value.shape)
        group['parameters'] += n
        group['trainable'] += n if path in trained else 0
        group['dtypes'].add(str(variable.value.dtype))
    for group in groups.values(): group['dtypes'] = sorted(group['dtypes'])
    assert groups['vit_dense']['trainable'] == 0
    assert groups['vit_lora']['parameters'] == groups['vit_lora']['trainable'] == 4347648
    assert groups['other_dense']['parameters'] == groups['other_dense']['trainable'] == 2938630176
    return groups
