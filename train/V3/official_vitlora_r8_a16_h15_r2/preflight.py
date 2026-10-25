"""CPU-only contracts; never reserve a GPU or load base model arrays."""
import os
os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled',
                  HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import configuration as c
c.bootstrap()
import jax
from openpi.training import sharding

def run():
    (c.HERE/'preflight_report.json').unlink(missing_ok=True)
    tests={}
    for name,command in {
        'vit_lora':[sys.executable,'test_vit_lora.py'],
        'official_contract':[sys.executable,'test_official_contract.py'],
        'official_trainstate_compilation':[sys.executable,'test_compilation_compat.py'],
        'finite_runtime_reporting':[sys.executable,'-m','unittest','-v','test_runtime_reporting'],
        'one_shot_queue':[sys.executable,'-m','unittest','-v','test_queue_training'],
    }.items():
        result=subprocess.run(command,cwd=c.HERE,text=True,capture_output=True,timeout=180)
        tests[name]={'passed':result.returncode==0,'stdout':result.stdout,'stderr':result.stderr}
        (c.HERE/'cpu_tests.json').write_text(json.dumps(tests,indent=2)+'\n')
        assert result.returncode==0, f'{name}: {result.stdout}\n{result.stderr}'
    cfg=c.make_config()
    official=c.official_trainer()
    mesh=sharding.make_mesh(1)
    state,_=official.init_train_state(cfg,jax.random.key(42),mesh,resume=True)
    groups=c.parameter_report(state,cfg)
    metadata=json.loads((Path(cfg.weight_loader.params_path)/'_METADATA').read_text())['tree_metadata']
    dense={}
    for value in metadata.values():
        path=tuple(k['key'] for k in value['key_metadata'])
        if path and path[0]=='params':path=path[1:]
        if path and path[-1]=='value':path=path[:-1]
        dense[path]=tuple(value['value_metadata']['write_shape'])
    expected={p:tuple(v.value.shape) for p,v in state.params.flat_state().items()
              if not any(str(k).startswith('lora_') for k in p)}
    assert dense==expected, 'Dense base checkpoint key/shape mismatch'
    manifest=json.loads((c.HERE/'manifest.json').read_text())
    for name,digest in manifest['files'].items():assert c.sha256(c.HERE/name)==digest,name
    for name,digest in manifest['runtime_files'].items():assert c.sha256(name)==digest,name
    norm=json.loads((c.DATA/'assets/heterovla_v3/norm_stats.json').read_text())
    assert set(norm['norm_stats'])=={'state','actions'}
    c.install_data_adapter()
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    from openpi.training import data_loader
    datasets={}
    for split in ('train','val'):
        import dataclasses
        ds=data_loader.create_torch_dataset(dataclasses.replace(dc,repo_id='local/heterovla_v3_'+split),15,cfg.model)
        assert ds.base.fps==15 and ds.base.horizon==15
        for i in (0,len(ds)//2,len(ds)-1):
            sample=ds[i]; assert sample['actions'].shape==(15,14) and sample['state'].shape==(30,)
        transformed=data_loader.transform_dataset(ds,dc)
        sample=transformed[0]
        assert sample['actions'].shape==(15,32) and sample['state'].shape==(32,)
        datasets[split]={'windows':len(ds),'fps':15,'raw_action_shape':[15,14],
                         'transformed_action_shape':list(sample['actions'].shape),'prompt':ds.base.prompt}
    loader=data_loader.create_data_loader(cfg,shuffle=True,num_batches=1)
    iterator=iter(loader)
    batch=next(iterator)
    spec=cfg.model.inputs_spec(batch_size=32)
    assert jax.tree.structure(batch)==jax.tree.structure(spec), 'Actual batch pytree differs from model spec'
    batch_shapes=[]
    for actual,expected_spec in zip(jax.tree.leaves(batch),jax.tree.leaves(spec)):
        assert actual.shape==expected_spec.shape and actual.dtype==expected_spec.dtype, (actual.shape,actual.dtype,expected_spec)
        assert np.isfinite(np.asarray(actual)).all()
        batch_shapes.append({'shape':list(actual.shape),'dtype':str(actual.dtype)})
    del iterator,loader,batch
    schedule=cfg.lr_schedule.create()
    rates={str(s):float(schedule(s)) for s in (0,499,500,19999,20000)}
    assert np.isclose(rates['500'],1e-5) and np.isclose(rates['20000'],1e-6)
    def size(tree):return sum(math.prod(x.shape)*np.dtype(x.dtype).itemsize for x in jax.tree.leaves(tree) if hasattr(x,'dtype'))
    result={'passed':True,'manifest_sha256':c.sha256(c.HERE/'manifest.json'),
        'official_train_sha256':c.OFFICIAL_TRAIN_SHA,'official_commit':c.OFFICIAL_COMMIT,
        'groups':groups,'datasets':datasets,'learning_rates':rates,'batch_size':32,
        'warmup_steps':500,'decay_steps':20000,'ema_decay':.99,'vit_rank':8,'vit_alpha':16,
        'base_dense_shapes_exact':True,'train_state_bytes':size(state),
        'gpu_used':False,'base_arrays_loaded':False,'gpu_memory_verified':False,
        'cpu_tests':{name:result['passed'] for name,result in tests.items()},
        'official_trainstate_lower_compile_checked':True,
        'actual_batch_matches_model_spec':True,'actual_batch_leaves':batch_shapes,
        'ema_frozen_dense_policy':'copy_raw_frozen_parameters_exactly'}
    (c.HERE/'preflight_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':run()
