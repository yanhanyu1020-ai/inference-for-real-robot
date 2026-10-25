"""Fresh relative14 training with the pinned official loop and V3 memory guards."""
import argparse
import functools
import json
import os
from pathlib import Path
import re

os.environ['JAX_ENABLE_COMPILATION_CACHE']='false'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Run CPU-only preflight; no GPU allocation')
    parser.add_argument('--gpu',help='GPU index or UUID; required to actually train')
    parser.add_argument('--run-name')
    args=parser.parse_args()
    if args.check:
        from .preflight import main as check
        check();return
    if args.gpu is None:
        parser.error('Use --check, or explicitly select --gpu to start fresh training')
    # Select the device before importing JAX/OpenPI.
    from . import configuration as c
    from .prepare import atomic_json, digest, verify_assets
    base=c.bootstrap()
    from queue_training import query_gpus
    from launch_training import query_indices
    indices=query_indices('nvidia-smi')
    gpu_uuid=indices.get(args.gpu,args.gpu)
    device=next((g for g in query_gpus() if g['uuid']==gpu_uuid),None)
    if device is None:
        raise RuntimeError('Selected GPU was not found: '+args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid,JAX_PLATFORMS='cuda',
                      XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
                      WANDB_MODE='offline',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    # Do not import preflight here: that module deliberately forces the CPU backend.
    proof=json.loads((c.MODELS/'V3/relative14/preflight.json').read_text())
    if not proof.get('passed') or proof.get('gpu_used'):
        raise RuntimeError('Run CPU preflight first')
    for path, expected in proof['source_hashes'].items():
        if digest(c.PROJECT/path)!=expected:
            raise RuntimeError('Source changed since preflight: '+path)
    provenance=verify_assets()
    if proof['normalization_sha256']!=provenance['normalization_sha256']:
        raise RuntimeError('Normalization changed since preflight')
    for item in provenance['sources']:
        if digest(c.DATA/item['path'])!=item['sha256']:
            raise RuntimeError('Dataset changed since normalization')
    run_name=args.run_name or c.RUN_NAME
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',run_name):
        raise ValueError('Invalid run name')
    cfg=c.make_config(run_name)
    if cfg.checkpoint_dir.exists():
        raise FileExistsError('Fresh run refuses existing checkpoint destination: '+str(cfg.checkpoint_dir))
    run=c.MODELS/'V3/relative14/runs'/run_name
    run.mkdir(parents=True,exist_ok=False)
    os.environ['WANDB_DIR']=str(run)
    import fcntl
    lock_path=c.MODELS/'V3/relative14'/('gpu_'+gpu_uuid+'.lock')
    lock=lock_path.open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import jax
    from openpi.training import sharding
    from compilation_compat import lower_and_compile
    from inference.real_robot.relative14 import contract_metadata
    assert not jax.config.jax_enable_compilation_cache
    assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
    c.install_data_adapter()
    official=base.official_trainer()
    official.train_step=base.corrected_train_step(official.train_step)
    original_init=official.init_train_state
    original_save=official._checkpoints.save_state
    status=dict(status='initializing',gpu=gpu_uuid,run_name=run_name,
                state_dim=14,action_horizon=15,resume=False,overwrite=False)
    def publish(**values):
        status.update(values)
        atomic_json(run/'runtime_status.json',status)
        print(json.dumps(status),flush=True)
    def initialize(config,rng,mesh,*,resume):
        if resume:raise RuntimeError('This entry only supports fresh training')
        publish(status='compiling_memory_check')
        shape,placement=original_init(config,rng,mesh,resume=True)
        atomic_json(run/'parameter_report.json',base.parameter_report(shape,config))
        rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        data=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        step=jax.jit(functools.partial(official.train_step,config),
                     in_shardings=(rep,placement,data),out_shardings=(placement,rep),donate_argnums=(1,))
        with sharding.set_mesh(mesh):
            compiled=lower_and_compile(step,jax.random.key(42),shape,
                                       config.model.inputs_spec(batch_size=config.batch_size))
        mem=compiled.memory_analysis()
        peak=mem.argument_size_in_bytes+mem.temp_size_in_bytes+mem.output_size_in_bytes-mem.alias_size_in_bytes
        pool=int(device['total_mib']*2**20*.90)
        atomic_json(run/'compiled_memory.json',dict(estimated_peak_bytes=peak,pool_bytes=pool,
                    aliases=mem.alias_size_in_bytes,cache_globally_disabled=True))
        if not mem.alias_size_in_bytes or peak>pool-2**30:
            raise RuntimeError('Compiled memory/donation check failed; hyperparameters remain unchanged')
        del compiled
        publish(status='loading_base_weights')
        state,placement=original_init(config,rng,mesh,resume=False)
        jax.block_until_ready(state)
        if int(state.step)!=0:raise RuntimeError('Expected fresh optimizer state')
        publish(status='training',optimizer_step=0)
        return state,placement
    def save(manager,state,loader,step):
        original_save(manager,state,loader,step)
        manager.wait_until_finished()
        asset=cfg.checkpoint_dir/str(step)/'assets'/c.ASSET_ID
        # Semantic marker prevents loading an old absolute-action checkpoint as relative.
        atomic_json(asset/'contract.json',contract_metadata())
        publish(status='training',optimizer_step=int(state.step),checkpoint=str(cfg.checkpoint_dir/str(step)))
    official.init_train_state=initialize
    official._checkpoints.save_state=save
    atomic_json(run/'effective_config.json',dict(**contract_metadata(),config_name=cfg.name,
                run_name=run_name,batch_size=cfg.batch_size,num_train_steps=cfg.num_train_steps,
                base_params=str(cfg.weight_loader.params_path),asset_dir=str(c.ASSET_DIR),
                normalization_sha256=provenance['normalization_sha256'],source_hashes=proof['source_hashes']))
    try:
        publish()
        official.main(cfg)
        publish(status='complete')
    except BaseException as exc:
        publish(status='failed',error=str(exc))
        raise
    finally:
        official._checkpoints.save_state=original_save
        lock.close()


if __name__=='__main__':main()
