"""Invoke the pinned official main loop with V3 data and scoped ViT LoRA.

    No copied optimizer or training loop. Observers record official logging and
    checkpoint events. The sole step correction fixes frozen bf16 EMA drift.
"""
from __future__ import annotations
import argparse
import dataclasses
import datetime
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from runtime_reporting import atomic_json, append_metrics, dumps_record

def atomic(path, value):
    return atomic_json(path,value)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',required=True);ap.add_argument('--run-name',required=True)
    args=ap.parse_args()
    import configuration as c
    assert args.run_name==c.RUN_NAME
    # This must run before importing JAX, Flax, or model code.
    from queue_training import query_gpus,available,verify_preflight
    contract=verify_preflight(c.HERE)
    device=next(g for g in query_gpus() if g['uuid']==args.gpu)
    if not available(device):raise RuntimeError('GPU ceased to be completely idle; refusing this training attempt')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
        WANDB_MODE='offline',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false')
    run=c.MODELS/'V3/runs'/args.run_name
    run.mkdir(parents=True,exist_ok=True)
    os.environ['WANDB_DIR']=str(run)
    started=time.monotonic();status={'status':'initializing','pid':os.getpid(),'gpu':args.gpu,
        'run_name':args.run_name,'step':0,**contract}
    def publish(**values):
        status.update(values,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        atomic(run/'runtime_status.json',status);print(dumps_record(status),flush=True)
    try:
        c.bootstrap();cfg=c.make_config();c.install_data_adapter();official=c.official_trainer()
        import jax
        import numpy as np
        from openpi.training import sharding
        from compilation_compat import lower_and_compile
        manifest=json.loads((c.HERE/'manifest.json').read_text())
        for name,digest in manifest['runtime_files'].items():assert c.sha256(name)==digest,name
        assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
        atomic(run/'effective_config.json',{'official_commit':c.OFFICIAL_COMMIT,
            'official_train_sha256':c.OFFICIAL_TRAIN_SHA,'batch_size':32,'warmup_steps':500,
            'peak_lr':1e-5,'decay_steps':20000,'decay_lr':1e-6,'ema_decay':.99,
            'fps':15,'action_horizon':15,'steps':20000,'seed':42,
            'initialization':'official_pi05_base_new_optimizer_step0','vit_lora_rank':8,
            'vit_lora_alpha':16,'other_layers':'dense_full','data_root':str(c.DATA.resolve()),
            'official_main_unmodified':True,'official_loss_and_adam_unmodified':True,
            'ema_frozen_dense_policy':'copy_raw_frozen_parameters_exactly',
            'validation_in_official_loop':False,**contract})
        official.train_step=c.corrected_train_step(official.train_step)
        original_init=official.init_train_state;original_save=official._checkpoints.save_state
        original_log=official.wandb.log
        holder={}
        def digest(params):
            h=hashlib.sha256()
            for path,var in sorted(params.filter(cfg.freeze_filter).flat_state().items()):
                h.update('/'.join(map(str,path)).encode());h.update(np.asarray(jax.device_get(var.value)).tobytes())
            return h.hexdigest()
        def init(config,rng,mesh,*,resume):
            assert not resume
            publish(status='compiling_memory_check')
            shape,placement=original_init(config,rng,mesh,resume=True)
            groups=c.parameter_report(shape,config);atomic(run/'parameter_report.json',groups)
            rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
            data=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
            pstep=jax.jit(functools.partial(official.train_step,config),
                in_shardings=(rep,placement,data),out_shardings=(placement,rep),donate_argnums=(1,))
            with sharding.set_mesh(mesh):
                compiled=lower_and_compile(pstep,jax.random.key(42),shape,config.model.inputs_spec(batch_size=32))
            m=compiled.memory_analysis()
            estimate=m.argument_size_in_bytes+m.temp_size_in_bytes+m.output_size_in_bytes-m.alias_size_in_bytes
            atomic(run/'compiled_memory.json',{'arguments':m.argument_size_in_bytes,'temporary':m.temp_size_in_bytes,
                'outputs':m.output_size_in_bytes,'aliases':m.alias_size_in_bytes,'estimated_peak_bytes':estimate,
                'pool_bytes':int(device['total_mib']*2**20*.9),'verified_on_selected_gpu':True})
            if estimate>int(device['total_mib']*2**20*.9)-2**30:
                raise MemoryError('B32+EMA requested configuration exceeds checked GPU memory; no automatic hyperparameter changes')
            del compiled
            publish(status='loading_base_weights')
            state,placement=original_init(config,rng,mesh,resume=False)
            jax.block_until_ready(state)
            c.parameter_report(state,config)
            b_values=[np.asarray(jax.device_get(v.value)) for p,v in state.params.flat_state().items() if p[-1]=='lora_b']
            assert len(b_values)==6 and all(not np.any(v) for v in b_values), 'All six scanned ViT LoRA B tensors must initialize to zero'
            holder['initial_frozen_sha256']=digest(state.params)
            holder['initial_ema_frozen_sha256']=digest(state.ema_params)
            assert holder['initial_frozen_sha256']==holder['initial_ema_frozen_sha256']
            atomic(run/'initialization_proof.json',{'all_lora_b_zero':True,'dense_shapes_verified_in_cpu_preflight':True,
                'fresh_optimizer_step0':int(state.step)==0,**holder})
            publish(status='training');return state,placement
        def save(manager,state,loader,step):
            original_save(manager,state,loader,step)
            updates=int(state.step)
            publish(status='training',step=updates,last_checkpoint_loop_step=int(step))
            if updates==20000:
                holder['final_state']=state;holder['final_loop_step']=int(step)
        def log(data,*a,**kw):
            if 'loss' in data:
                updates=int(kw.get('step',0))+1
                row={'step':updates,'official_loop_step':updates-1,
                     **{k:float(v) for k,v in data.items()}}
                append_metrics(run/'metrics.jsonl',row)
                result=original_log(data,*a,**kw)
                publish(status='training',step=updates,train_loss=row['loss'])
                return result
            return original_log(data,*a,**kw)
        official.init_train_state=init;official._checkpoints.save_state=save;official.wandb.log=log
        logging.basicConfig(level=logging.INFO)
        official.main(cfg)
        state=holder.pop('final_state');assert int(state.step)==20000
        final=Path(cfg.checkpoint_dir)/str(holder['final_loop_step'])
        assert holder['final_loop_step']==19999 and (final/'_CHECKPOINT_METADATA').is_file()
        assert (final/'params/_METADATA').is_file() and (final/'train_state/_METADATA').is_file()
        raw_sha=digest(state.params);ema_sha=digest(state.ema_params)
        assert raw_sha==ema_sha==holder['initial_frozen_sha256']
        atomic(run/'final_proof.json',{'step':20000,'checkpoint_loop_step':19999,
            'raw_and_ema_frozen_exactly_unchanged':True,'frozen_sha256':raw_sha,
            'final_checkpoint':str(final),'official_train_sha256':c.sha256(c.HERE/'official_source/scripts/train.py')})
        publish(status='completed',step=20000,final_checkpoint=str(final),frozen_parameters_verified=True)
    except BaseException as exc:
        values={'status':'failed','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()}
        if hasattr(exc,'nonfinite_metrics'):
            values.update(nonfinite_metrics=exc.nonfinite_metrics,failed_metric_record=exc.record,
                          step=exc.record.get('step',status['step']))
        publish(**values)
        raise

if __name__=='__main__':main()
