"""Behavioral checks for anchoring, frame channels, real samples and serving transforms."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',JAX_ENABLE_COMPILATION_CACHE='false')

import dataclasses
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from inference.real_robot.relative14 import (DELTA_MASK, chunk_delta, contract_metadata,
    policy_observation, state14_from_state30)
from . import configuration as c
from .prepare import episode_windows, verify_assets

c.bootstrap()
from openpi import transforms
from .transforms import RelativeTransforms, ValidateInputs


def test_state14_selects_measured_counterparts_not_first_fourteen_values():
    state=np.arange(30,dtype=np.float32)
    expected=np.array([7,8,9,10,11,12,13,14,15,16,24,25,26,27],dtype=np.float32)
    np.testing.assert_array_equal(state14_from_state30(state),expected)
    stacked=np.stack([state,state+100])
    np.testing.assert_array_equal(state14_from_state30(stacked),[expected,expected+100])


def test_all_steps_use_one_anchor_and_gripper_is_absolute():
    state=np.arange(14,dtype=np.float32)/10
    targets=np.arange(15*14,dtype=np.float32).reshape(15,14)/100
    source=targets.copy()
    group=RelativeTransforms()(None)
    encoded=transforms.compose(group.inputs)({'state':state,'actions':targets})
    for k in (0,1,14):
        expected=targets[k]-state
        expected[9]=targets[k,9]
        np.testing.assert_allclose(encoded['actions'][k],expected)
    np.testing.assert_array_equal(targets,source)
    decoded=transforms.compose(group.outputs)(encoded)['actions']
    np.testing.assert_allclose(decoded,targets,atol=2e-7)


def test_relative_velocities_and_yaw_rates_are_restored_not_integrated():
    state=np.zeros(14,dtype=np.float32)
    channels=[0,1,2,10,11,12,13]
    state[channels]=[.4,-.2,.3,-.1,.2,.05,-.4]
    physical=np.tile(state,(15,1))
    physical[:,channels]+=np.arange(15,dtype=np.float32)[:,None]*.01
    delta=chunk_delta(state,physical)
    np.testing.assert_allclose(delta[14,channels],.14,atol=1e-7)
    decoded=transforms.AbsoluteActions(DELTA_MASK)({'state':state,'actions':delta.copy()})
    np.testing.assert_allclose(decoded['actions'],physical,atol=1e-7)
    # Adding a new measured state per step, or cumulative integration, is not this contract.
    assert not np.allclose(np.cumsum(delta[:,0])+state[0],physical[:,0])


def test_window_norm_population_uses_each_window_start():
    state=np.zeros((17,30),dtype=np.float32)
    state[:,7]=np.arange(17)
    action=np.zeros((17,14),dtype=np.float32)
    action[:,0]=10+np.arange(17)
    anchors,targets=episode_windows(state,action)
    assert anchors.shape==(3,14) and targets.shape==(3,15,14)
    # Same source action at row 2 has different deltas in windows 0 and 2.
    assert targets[0,2,0]==12 and targets[2,0,0]==10
    np.testing.assert_array_equal(targets[:,:,0],np.tile(10+np.arange(15),(3,1)))


def test_reject_wrong_shapes_and_nonfinite_values():
    with pytest.raises(ValueError):ValidateInputs()({'state':np.zeros(30)})
    with pytest.raises(ValueError):chunk_delta(np.zeros(14),np.zeros((14,14)))
    with pytest.raises(ValueError):state14_from_state30(np.full(30,np.nan))
    with pytest.raises(ValueError):chunk_delta(np.zeros(14),np.full((15,14),np.inf))


def test_client_projects_only_matching_semantics_and_preserves_old_server():
    obs={'state':np.arange(30,dtype=np.float32),'prompt':'task'}
    assert policy_observation(obs,{}) is obs
    metadata=contract_metadata()
    assert policy_observation(obs,metadata)['state'].shape==(14,)
    assert obs['state'].shape==(30,)
    with pytest.raises(ValueError):policy_observation(obs,{**metadata,'returned_action_encoding':'raw_deltas'})


def test_real_norm_and_data_pipeline_roundtrip():
    verify_assets()
    cfg=c.make_config()
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    from dataset_v3 import OpenpiV3Dataset
    raw=OpenpiV3Dataset(c.DATA,'train',15)[0]
    sample={**raw,'state':state14_from_state30(raw['state'])}
    pipeline=transforms.compose([*dc.data_transforms.inputs,
        transforms.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])
    transformed=pipeline(sample)
    assert transformed['state'].shape==(32,) and transformed['actions'].shape==(15,32)
    np.testing.assert_array_equal(transformed['state'][14:],0)
    np.testing.assert_array_equal(transformed['actions'][:,14:],0)
    output=transforms.compose([transforms.Unnormalize(dc.norm_stats,use_quantiles=True),*dc.data_transforms.outputs])
    np.testing.assert_allclose(output(transformed)['actions'],raw['actions'],atol=2e-6,rtol=1e-5)
    assert cfg.model.discrete_state_input is True
    assert cfg.name!='pi05_heterovla_v3_vitlora'


def test_policy_serving_adds_anchor_exactly_once_and_sends_state14_tokens():
    """Use the real OpenPI Policy inference wrapper with a deterministic test sampler."""
    from openpi.policies.policy import Policy
    from openpi.shared import nnx_utils
    import jax.numpy as jnp
    cfg=c.make_config();dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    state=np.linspace(-.2,.3,14,dtype=np.float32)
    target=np.tile(state,(15,1));target[:,:]+=np.arange(15,dtype=np.float32)[:,None]*.001
    target[:,9]=.03
    physical={'state':state,'actions':target}
    relative=transforms.compose(dc.data_transforms.inputs)(physical)
    normalized=transforms.Normalize(dc.norm_stats,use_quantiles=True)(relative)
    output_actions=transforms.PadStatesAndActions(32)(normalized)['actions']
    def sample(rng,obs,**kwargs):
        assert obs.state.shape==(1,32)
        np.testing.assert_array_equal(np.asarray(obs.state)[0,14:],0)
        assert obs.tokenized_prompt.shape==(1,200)
        return jnp.asarray(output_actions[None])
    class Model:
        sample_actions=staticmethod(sample)
    with patch.object(nnx_utils,'module_jit',lambda f:f):
        policy=Policy(Model(),transforms=[*dc.data_transforms.inputs,
            transforms.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs],
            output_transforms=[transforms.Unnormalize(dc.norm_stats,use_quantiles=True),*dc.data_transforms.outputs],
            metadata=cfg.policy_metadata)
        obs={'state':state,'prompt':'pick up the green block',
             'image':{name:np.zeros((480,640,3),np.uint8) for name in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
             'image_mask':{name:True for name in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}}
        result=policy.infer(obs)
    np.testing.assert_allclose(result['actions'],target,atol=2e-6)


def test_old_checkpoint_rejected_before_loading_weights():
    from .serve_policy import verify_checkpoint
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError,match='semantic marker'):
            verify_checkpoint(Path(directory))
