#!/usr/bin/env python3
"""Register the V1 configuration in an existing openpi environment, then train.

Run with openpi's own Python/JAX environment, not the CPU conversion venv.
No benchmark files or installed openpi source files are modified.
"""
import os
import sys
import runpy
from pathlib import Path
from path_config import DATA_ROOT, OPENPI_ROOT, RUN_ROOT, WEIGHTS


def register(root):
    import dataclasses
    import flax.nnx as nnx
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as cfg, data_loader, optimizer, weight_loaders
    from openpi.shared import nnx_utils
    from openpi import transforms
    from dataset_v1 import OpenpiV1Dataset

    @dataclasses.dataclass(frozen=True)
    class Output14:
        def __call__(self,data):return {'actions':data['actions'][...,:14]}

    model=Pi0Config(pi05=True,action_dim=32,action_horizon=10,
        paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    config=cfg.TrainConfig(name='pi05_heterovla_v1_lora',model=model,
        data=cfg.SimpleDataConfig(repo_id='local/heterovla_v1_train',
            assets=cfg.AssetsConfig(assets_dir=str(root/'assets'),asset_id='heterovla_v1'),
            data_transforms=lambda _:transforms.Group(outputs=[Output14()]),
            base_config=cfg.DataConfig(action_sequence_keys=('action',),prompt_from_task=False)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(WEIGHTS)),
        freeze_filter=nnx.Any(model.get_freeze_filter(),nnx_utils.PathRegex('.*img.*')),
        batch_size=8,num_workers=2,num_train_steps=2000,save_interval=500,log_interval=25,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,peak_lr=2.5e-5,decay_steps=2000,decay_lr=2.5e-6),
        ema_decay=None,wandb_enabled=False,assets_base_dir=str(RUN_ROOT/'assets'),
        checkpoint_base_dir=str(RUN_ROOT/'checkpoints'),
        policy_metadata={'fps':10,'physical_action_dim':14,'action_horizon':10,
            'target_type':'kinematic_imitation_not_operator_command_bc','dataset_root':str(root)})
    cfg._CONFIGS_DICT[config.name]=config
    if not any(c.name==config.name for c in cfg._CONFIGS):cfg._CONFIGS.append(config)
    original=data_loader.create_torch_dataset
    def create(data_config,action_horizon,model_config):
        if data_config.repo_id=='local/heterovla_v1_train':
            return OpenpiV1Dataset(root,'train',action_horizon)
        return original(data_config,action_horizon,model_config)
    data_loader.create_torch_dataset=create
    return config


def main():
    root=DATA_ROOT
    os.environ.setdefault('HF_LEROBOT_HOME',str(root/'lerobot'))
    script=OPENPI_ROOT/'scripts/train.py'
    if not script.is_file():raise SystemExit(f'Missing openpi trainer: {script}')
    register(root)
    if '--config-only' in sys.argv:
        print('V1 configuration registered successfully');return
    sys.argv=[str(script),'pi05_heterovla_v1_lora',*sys.argv[1:]]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
