"""Reuse the pinned V3 model/trainer, with separate data semantics and assets."""
import dataclasses
import os
from pathlib import Path
import sys

from inference.real_robot.relative14 import contract_metadata

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'local_yhy'
PROJECT = HERE.parents[2]
DATA = Path(os.environ.get('HETEROVLA_DATA_ROOT') or PROJECT.parent/'heterovla-data/aligned/V3').resolve()
MODELS = Path(os.environ.get('HETEROVLA_MODEL_ROOT') or PROJECT.parent/'heterovla-model').resolve()
ASSET_ID = 'heterovla_v3_relative14'
ASSETS = MODELS/'V3/relative14/assets'
ASSET_DIR = ASSETS/ASSET_ID
CONFIG_NAME = 'pi05_heterovla_v3_relative14'
RUN_NAME = 'vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_relative14'


def bootstrap():
    sys.path.insert(0, str(BASE))
    from train.V3.local_yhy import configuration as base
    base.bootstrap()
    return base


def make_config(run_name=RUN_NAME):
    base = bootstrap()
    from openpi.training import config as cfg
    from .transforms import RelativeTransforms
    original = base.make_config(run_name)
    metadata = {**original.policy_metadata, **contract_metadata(), 'asset_id': ASSET_ID}
    return dataclasses.replace(original, name=CONFIG_NAME, project_name='heterovla-relative14',
        model=dataclasses.replace(original.model, discrete_state_input=True),
        data=cfg.SimpleDataConfig(repo_id='local/heterovla_v3_train',
            assets=cfg.AssetsConfig(assets_dir=str(ASSETS),asset_id=ASSET_ID),
            data_transforms=RelativeTransforms(),
            base_config=cfg.DataConfig(action_sequence_keys=('action',),prompt_from_task=False)),
        assets_base_dir=str(MODELS/'V3/relative14/runs'/run_name/'assets'),
        policy_metadata=metadata, resume=False, overwrite=False)


class Dataset14:
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        from inference.real_robot.relative14 import state14_from_state30
        sample = self.dataset[index]
        return {**sample, 'state': state14_from_state30(sample['state'])}


def install_data_adapter():
    bootstrap()
    from openpi.training import data_loader
    from dataset_v3 import OpenpiV3Dataset
    original = data_loader.create_torch_dataset
    def create(data_config, action_horizon, model_config):
        if data_config.asset_id == ASSET_ID:
            if data_config.repo_id not in ('local/heterovla_v3_train','local/heterovla_v3_val'):
                raise ValueError('Unknown relative14 dataset')
            if action_horizon != 15 or model_config.action_horizon != 15:
                raise ValueError('This experiment requires H15')
            return Dataset14(OpenpiV3Dataset(DATA,data_config.repo_id.rsplit('_',1)[1],15))
        return original(data_config,action_horizon,model_config)
    data_loader.create_torch_dataset = create
    return original
