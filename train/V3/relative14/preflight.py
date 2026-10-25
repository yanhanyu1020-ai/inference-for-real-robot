"""CPU-only validation of real data, state tokens and relative/physical round trips."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',JAX_ENABLE_COMPILATION_CACHE='false',
                  HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='disabled')

import dataclasses
import json
from pathlib import Path
import numpy as np

from inference.real_robot.relative14 import DELTA_MASK, chunk_delta, state14_from_state30
from . import configuration as c
from .prepare import atomic_json, digest, verify_assets

REPORT = c.MODELS/'V3/relative14/preflight.json'


def source_hashes():
    paths = list(c.HERE.glob('*.py'))
    paths += [c.PROJECT/'inference/real_robot/relative14.py']
    paths += [c.BASE/name for name in ('configuration.py','vit_lora.py','compilation_compat.py')]
    paths += [c.BASE/'queue_training.py',c.HERE/'train.sh',c.PROJECT/'env.sh']
    paths += list((c.BASE/'data_adapter').glob('*.py'))
    paths += [c.BASE/'official_source/scripts/train.py']
    paths += [c.BASE/'official_source/src/openpi'/name for name in
              ('transforms.py','models/pi0_config.py','models/pi0.py','training/data_loader.py',
               'policies/policy.py','policies/policy_config.py','training/config.py')]
    return {str(p.relative_to(c.PROJECT)):digest(p) for p in sorted(paths)}


def run():
    provenance = verify_assets()
    for item in provenance['sources']:
        if digest(c.DATA/item['path'])!=item['sha256']:
            raise ValueError('Training data changed: '+item['path'])
    base = c.bootstrap()
    base.official_trainer()  # checks the pinned upstream trainer hash
    cfg = c.make_config()
    c.install_data_adapter()
    from openpi import transforms
    from openpi.models import model
    from openpi.training import data_loader
    import jax
    if any(device.platform!='cpu' for device in jax.devices()):
        raise RuntimeError('Preflight must run on CPU')
    dc = cfg.data.create(cfg.assets_dirs,cfg.model)
    assert cfg.model.discrete_state_input is True and cfg.model.action_dim==32
    assert dc.use_quantile_norm and len(dc.norm_stats['state'].mean)==14
    normalize = transforms.Normalize(dc.norm_stats,use_quantiles=True)
    unnormalize = transforms.Unnormalize(dc.norm_stats,use_quantiles=True)
    before = transforms.compose(dc.data_transforms.inputs)
    after = transforms.compose(dc.data_transforms.outputs)
    splits = {}
    max_error = 0.
    for split, expected in (('train',34013),('val',743)):
        dataset = data_loader.create_torch_dataset(dataclasses.replace(dc,repo_id='local/heterovla_v3_'+split),15,cfg.model)
        assert len(dataset)==expected
        transformed = data_loader.transform_dataset(dataset,dc)
        for index in (0,len(dataset)//2,len(dataset)-1):
            sample = dataset[index]
            original = dataset.dataset[index]
            np.testing.assert_array_equal(sample['state'],state14_from_state30(original['state']))
            expected_delta = chunk_delta(sample['state'],sample['actions'])
            relative = before(sample)
            np.testing.assert_array_equal(relative['actions'],expected_delta)
            np.testing.assert_array_equal(relative['actions'][:,9],sample['actions'][:,9])
            normalized = normalize({'state':relative['state'],'actions':relative['actions']})
            padded = transforms.PadStatesAndActions(32)(normalized)
            physical = after(unnormalize(padded))['actions']
            error = float(np.max(np.abs(physical-sample['actions'])))
            max_error = max(max_error,error)
            np.testing.assert_allclose(physical,sample['actions'],atol=2e-6,rtol=1e-5)
            batch_item = transformed[index]
            assert batch_item['state'].shape==(32,) and batch_item['actions'].shape==(15,32)
            np.testing.assert_array_equal(batch_item['state'][14:],0)
            np.testing.assert_array_equal(batch_item['actions'][:,14:],0)
            assert batch_item['tokenized_prompt_mask'].sum()>0
            # Changing only state must change the PI05 prefix: state input is enabled.
            alternate = {**sample,'state':sample['state']+1}
            pipeline = transforms.compose([*dc.data_transforms.inputs,normalize,*dc.model_transforms.inputs])
            assert not np.array_equal(pipeline(sample)['tokenized_prompt'],pipeline(alternate)['tokenized_prompt'])
            batched = jax.tree.map(lambda x:np.asarray(x)[None,...],batch_item)
            obs = model.Observation.from_dict(batched)
            assert obs.state.shape==(1,32) and set(obs.images)=={'base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'}
        splits[split] = dict(windows=len(dataset),tested_indices=[0,len(dataset)//2,len(dataset)-1])
    report = dict(passed=True,gpu_used=False,training_started=False,state_physical_dim=14,
                  state_padded_dim=32,action_physical_dim=14,delta_mask=list(DELTA_MASK),
                  max_roundtrip_absolute_error=max_error,splits=splits,
                  normalization_sha256=provenance['normalization_sha256'],
                  source_hashes=source_hashes(),train_source_hashes_verified=True)
    atomic_json(REPORT,report)
    return report


def main():
    result=run()
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
    print('Preflight report:',REPORT)


if __name__=='__main__':main()
