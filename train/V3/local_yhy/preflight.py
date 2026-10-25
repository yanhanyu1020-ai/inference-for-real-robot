"""CPU-only contracts; never reserve a GPU or load base model arrays."""
import os
os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled',
                  HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  JAX_ENABLE_COMPILATION_CACHE='false')
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

EXPECTED_DATA = {
    'train': {'sources': 55, 'episodes': 856, 'frames': 45997, 'windows': 34013},
    'val': {'sources': 7, 'episodes': 251, 'frames': 4257, 'windows': 743},
}
OLD_NORM_SHA = '5f66271f42de5dd7c73626d48420da18591bddd2df9cc97e8a1b315dbd3bfe90'
NEW_NORM_SHA = '4d2329417c7ddcd93d75776f04373e21d823574b28af527f00c3afb5d037d33c'


def check_strict_data():
    """Inspect numerical columns once on CPU; never decode all source videos."""
    import pyarrow.parquet as pq
    status_path = c.DATA/'STATUS.json'
    status = json.loads(status_path.read_text())
    assert status['fps'] == 15 and status['action_horizon'] == 15
    assert status['imputation_applied'] is False
    assert status['format_validation_passed'] is True
    source_path = c.DATA/'reports/episode_manifest.json'
    validation_path = c.DATA/'reports/validation.json'
    validation = json.loads(validation_path.read_text())
    assert validation['passed'] is True and validation['format_validation_passed'] is True
    assert len(validation['sources']) == 62
    for source, proof in validation['sources'].items():
        assert all(proof[key] is True for key in (
            'all_original_state_action_values_verified', 'all_validity_masks_true', 'no_imputation'
        )), source
    raw = json.loads(source_path.read_text())
    records = raw.get('episodes', raw.get('records', [])) if isinstance(raw, dict) else raw
    assert isinstance(records, list) and records, 'Episode provenance manifest is empty'
    reports = {}
    required = ('observation.state', 'action', 'action_valid', 'source_task_time')
    for split, expected in EXPECTED_DATA.items():
        declared = status['splits'][split]
        for key, value in expected.items():
            actual = declared.get(key, declared.get('action_windows') if key == 'windows' else None)
            assert actual == value, (split, key, actual, value)
        provenance = [record for record in records if record.get('split') == split]
        assert len(provenance) == expected['episodes'], (split, 'provenance episode count')
        assert all('source_episode' in record for record in provenance), split
        source_count = len({json.dumps(record['source_episode'], sort_keys=True) for record in provenance})
        assert source_count == expected['sources'], (split, source_count)
        root = c.DATA/'lerobot/local'/('heterovla_v3_'+split)
        info = json.loads((root/'meta/info.json').read_text())
        assert info['fps'] == 15 and info['total_episodes'] == expected['episodes']
        assert info['total_frames'] == expected['frames']
        files = sorted((root/'data').rglob('*.parquet'))
        assert len(files) == expected['episodes'], (split, 'parquet episode count')
        rows = windows = 0
        optional_present = set()
        for path in files:
            names = set(pq.read_schema(path).names)
            assert set(required).issubset(names), (path, names)
            optional = [key for key in ('state_valid', 'state_imputed', 'action_imputed') if key in names]
            optional_present.update(key for key in optional if key.endswith('_imputed'))
            table = pq.read_table(path, columns=[*required, *optional])
            n = len(table)
            arrays = {key: np.asarray(table[key].to_pylist()) for key in (*required, *optional)}
            for key, width in (('observation.state', 30), ('action', 14)):
                array = arrays[key]
                assert array.shape == (n, width) and np.isfinite(array).all(), (path, key)
            for key, width in (('state_valid', 30), ('action_valid', 14)):
                if key not in arrays:
                    # Strict export stores no state mask; its locked source audit
                    # proves every exported state/action value and validity bit.
                    assert key == 'state_valid'
                    continue
                array = arrays[key]
                assert array.shape == (n, width) and np.all(array == 1), (path, key)
            for key in optional:
                if key == 'state_valid':
                    continue
                width = 30 if key == 'state_imputed' else 14
                assert arrays[key].shape == (n, width) and np.all(arrays[key] == 0), (path, key)
            times = np.asarray(arrays['source_task_time'], dtype=np.float64).reshape(-1)
            assert times.shape == (n,) and n > 0 and np.isfinite(times).all(), path
            tolerance = max(1e-5, np.finfo(np.float32).eps * max(1., float(np.abs(times).max())) * 2)
            assert tolerance < 0.25/15 and np.allclose(np.diff(times), 1/15, atol=tolerance, rtol=0), path
            rows += n
            windows += max(0, n-15+1)
        assert rows == expected['frames'] and windows == expected['windows'], (split, rows, windows)
        reports[split] = {**expected, 'parquet_files_scanned': len(files),
                         'all_numeric_rows_finite_and_valid': True,
                         'imputed_columns_present': sorted(optional_present),
                         'all_imputation_masks_zero': True, 'episode_times_continuous_15hz': True}
    return {'status_sha256': c.sha256(status_path),
            'source_validity_report_sha256': c.sha256(validation_path),
            'all_62_sources_original_values_and_masks_verified': True,
            'episode_manifest_sha256': c.sha256(source_path), 'splits': reports,
            'videos_fully_decoded': False}


def check_normalization(dc, norm, path):
    assert dc.use_quantile_norm is True, 'PI05 must use the unchanged quantile normalization'
    digest = c.sha256(path)
    assert digest != OLD_NORM_SHA and digest == NEW_NORM_SHA, 'Expected the new training-set normalization asset'
    assert set(norm['norm_stats']) == {'state', 'actions'}
    assert dc.norm_stats is not None and set(dc.norm_stats) == {'state', 'actions'}
    for group, width in (('state', 30), ('actions', 14)):
        stats = norm['norm_stats'][group]
        for field in ('mean', 'std', 'q01', 'q99'):
            stored = np.asarray(stats[field])
            loaded = np.asarray(getattr(dc.norm_stats[group], field))
            assert stored.shape == loaded.shape == (width,), (group, field)
            assert np.isfinite(stored).all() and np.isfinite(loaded).all(), (group, field)
            np.testing.assert_array_equal(loaded, stored, err_msg=f'{group}.{field} loaded normalization mismatch')
        assert np.all(np.asarray(stats['std']) >= 0), group
        assert np.all(np.asarray(stats['q99']) >= np.asarray(stats['q01'])), group
    return {'sha256': digest, 'old_norm_rejected': True, 'use_quantile_norm': True,
            'actual_loaded_fields_exactly_match_new_asset': True, 'state_width': 30, 'action_width': 14}


def run():
    (c.HERE/'preflight_report.json').unlink(missing_ok=True)
    tests={}
    for name,command in {
        'vit_lora':[sys.executable,'test_vit_lora.py'],
        'official_contract':[sys.executable,'test_official_contract.py'],
        'official_trainstate_compilation':[sys.executable,'test_compilation_compat.py'],
        'finite_runtime_reporting':[sys.executable,'-m','unittest','-v','test_runtime_reporting'],
        'one_shot_queue':[sys.executable,'-m','unittest','-v','test_queue_training'],
        'manual_fresh_launcher':[sys.executable,'-m','unittest','-v','test_launch_training'],
        'global_cache_before_import':[sys.executable,'-B','test_global_cache.py','--original-code',str(c.HERE)],
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
    strict_data=check_strict_data()
    norm_path=c.DATA/'assets/heterovla_v3/norm_stats.json'
    norm=json.loads(norm_path.read_text())
    c.install_data_adapter()
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    normalization=check_normalization(dc,norm,norm_path)
    cache_proof=json.loads((c.HERE/'global_cache_cpu_result.json').read_text())
    assert cache_proof['passed'] is True and cache_proof['gpu_used'] is False
    assert cache_proof['global_cache_disabled_before_jax_import'] is True
    assert cache_proof['populated_cache_bypassed_without_reader_calls'] is True
    from openpi.training import data_loader
    datasets={}
    for split in ('train','val'):
        import dataclasses
        ds=data_loader.create_torch_dataset(dataclasses.replace(dc,repo_id='local/heterovla_v3_'+split),15,cfg.model)
        assert ds.base.fps==15 and ds.base.horizon==15
        assert len(ds)==EXPECTED_DATA[split]['windows'], (split,len(ds))
        assert ds.base.allow_imputed is False, 'The new data must be strict for both splits'
        for i in (0,len(ds)//2,len(ds)-1):
            sample=ds[i]; assert sample['actions'].shape==(15,14) and sample['state'].shape==(30,)
        transformed=data_loader.transform_dataset(ds,dc)
        sample=transformed[0]
        assert sample['actions'].shape==(15,32) and sample['state'].shape==(32,)
        datasets[split]={'windows':len(ds),'sources':EXPECTED_DATA[split]['sources'],
                         'allow_imputed':False,'fps':15,'raw_action_shape':[15,14],
                         'transformed_action_shape':list(sample['actions'].shape),'prompt':ds.base.prompt}
    # CPU acceptance can run in containers without multiprocessing shared memory.
    # Keep the production config's worker count and validate the same batch here.
    loader=data_loader.create_data_loader(dataclasses.replace(cfg,num_workers=0),shuffle=True,num_batches=1)
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
        'cpu_data_loader_workers':0,'configured_training_workers':cfg.num_workers,
        'official_trainstate_lower_compile_checked':True,
        'actual_batch_matches_model_spec':True,'actual_batch_leaves':batch_shapes,
        'strict_data_cpu_scan':strict_data,'normalization':normalization,
        'global_cache_cpu_result_sha256':c.sha256(c.HERE/'global_cache_cpu_result.json'),
        'ema_frozen_dense_policy':'copy_raw_frozen_parameters_exactly'}
    (c.HERE/'preflight_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':run()
