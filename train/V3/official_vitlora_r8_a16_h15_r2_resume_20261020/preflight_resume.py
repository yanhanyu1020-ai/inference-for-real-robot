"""CPU and checkpoint metadata gates for the one authorized reboot resume."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',WANDB_MODE='disabled')
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ORIGINAL=Path('/meta_eon_cfs/home/hph/heterovla/train/V3/official_vitlora_r8_a16_h15_r2')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    sys.path.insert(0,str(ORIGINAL))
    import configuration as c
    from queue_training import verify_preflight
    old=verify_preflight(ORIGINAL)
    plan=json.loads((HERE/'resume_plan.json').read_text())
    assert old['manifest_sha256']==plan['original_manifest_sha256']
    cpu=json.loads((HERE/'cpu_resume_contract.json').read_text())
    assert cpu['passed'] is True
    assert cpu['test_sha256']==sha(HERE/'test_resume_contract.py')
    cp=Path(plan['source_checkpoint'])
    meta=json.loads((cp/'_CHECKPOINT_METADATA').read_text())
    assert meta['commit_timestamp_nsecs'] is not None
    complete=[int(p.name) for p in cp.parent.iterdir() if p.is_dir() and p.name.isdigit()
              and json.loads((p/'_CHECKPOINT_METADATA').read_text()).get('commit_timestamp_nsecs')]
    assert max(complete)==plan['source_loop_step']
    import tensorstore as ts
    base={'driver':'file','path':str(cp/'train_state')+'/'}
    scalars={}
    for key in ['step','opt_state.1.0.count','opt_state.1.2.count']:
        a=ts.open({'driver':'zarr','kvstore':{'driver':'ocdbt','base':base,'path':key+'/'}}).result()
        scalars[key]=int(a.read().result())
    assert set(scalars.values())=={plan['source_actual_step']}
    c.bootstrap()
    import jax
    from openpi.training import sharding
    cfg=dataclasses.replace(c.make_config(),resume=True,overwrite=False)
    trainer=c.official_trainer()
    shape,_=trainer.init_train_state(cfg,jax.random.key(42),sharding.make_mesh(1),resume=True)
    c.parameter_report(shape,cfg)
    manifest=json.loads((HERE/'manifest.json').read_text())
    for path,digest in manifest['files'].items():assert sha(HERE/path)==digest,path
    for path,digest in manifest['runtime_files'].items():assert sha(path)==digest,path
    out={'passed':True,'manifest_sha256':sha(HERE/'manifest.json'),
        'original_contract':old,'source_loop_step':plan['source_loop_step'],
        'source_actual_step':plan['source_actual_step'],'checkpoint_scalars':scalars,
        'cpu_resume_roundtrip_passed':True,'gpu_used':False,'large_checkpoint_arrays_loaded':False,
        'data_order_policy':'official resume shuffle restart','target_updates':20000}
    (HERE/'preflight_report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__': main()
