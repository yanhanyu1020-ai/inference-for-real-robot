"""Full verification of a V3 release with no imputation in either split."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow.parquet as pq

from alignment_core import dump, segments
from raw_checks import episode_file
from verify_batch import video


def read(path):
    return json.loads(path.read_text())


def verify(root,raw,train_scripts):
    sys.path.insert(0,str(train_scripts))
    from dataset_v3 import JointV3Dataset,as_openpi_sample
    from check_action_chunks import check
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    imputation=read(root/'configs/imputation.json')
    assert imputation['method']=='none' and not imputation['splits']
    horizon=read(root/'configs/action_chunk.json')['action_horizon']
    manifest=read(root/'reports/episode_manifest.json')
    inventory=read(root/'configs/source_inventory.json')
    expected={s:{e['source_episode'] for e in manifest if e['split']==s} for s in ('train','val')}
    assert not expected['train'] & expected['val']
    entries={e['source_episode']:e for e in inventory['sources']}
    assert len(entries)==len(inventory['sources'])
    for split in ('train','val'):
        for name in expected[split]:assert entries[name]['split']==split and entries[name]['include']
    source_checks={};splits={};train_arrays={'state':[],'actions':[]}
    for split in ('train','val'):
        directory=root/'lerobot/local'/f'heterovla_v3_{split}'
        info=read(directory/'meta/info.json');episodes=[e for e in manifest if e['split']==split]
        assert info['fps']==15 and info['codebase_version']=='v2.1'
        assert info['total_frames']==sum(e['frames'] for e in episodes)
        assert info['total_episodes']==len(episodes)
        assert len(list((directory/'data').rglob('*.parquet')))==len(episodes)
        assert len(list((directory/'videos').rglob('*.mp4')))==3*len(episodes)
        assert not any('imputed' in key for key in info['features'])
        assert [e['episode_index'] for e in episodes]==list(range(len(episodes)))
        offset=0
        # Source order is the order in the native episode manifest.
        for name in dict.fromkeys(e['source_episode'] for e in episodes):
            entry=entries[name]
            audit=pq.read_table(root/'aligned'/f'{name}.parquet').to_pylist()
            runs=segments(audit);selected=[e for e in episodes if e['source_episode']==name]
            assert len(runs)==len(selected)
            compare={c:{} for c in ('front','wrist','uav')}
            raw_compare={c:{} for c in compare}
            for run,e in zip(runs,selected):
                index=e['episode_index'];n=len(run)
                assert n==e['frames'] and n>=15
                provenance=pq.read_table(root/'provenance'/f'{split}_{index:06d}.parquet').to_pylist()
                assert provenance==run,(name,index,'provenance')
                table=pq.read_table(episode_file(directory,info,index)).to_pydict()
                for key,field in [('state','observation.state'),('action','action')]:
                    x=np.asarray(table[field])
                    assert x.shape==(n,30 if key=='state' else 14) and np.isfinite(x).all()
                    np.testing.assert_array_equal(x,np.asarray([r[key] for r in run]))
                    if split=='train':train_arrays['state' if key=='state' else 'actions'].extend(x)
                    assert all(all(r[f'{key}_valid']) for r in run)
                    assert all(f'{key}_imputed' not in r for r in run)
                assert np.asarray(table['action_valid']).all()
                np.testing.assert_array_equal(table['index'],np.arange(offset,offset+n))
                np.testing.assert_array_equal(table['episode_index'],np.full(n,index))
                np.testing.assert_array_equal(table['frame_index'],np.arange(n))
                np.testing.assert_allclose(table['timestamp'],np.arange(n)/15,atol=1e-5,rtol=0)
                np.testing.assert_array_equal(np.asarray(table['source_task_time']).reshape(-1),
                    np.asarray([r['source_task_time_s'] for r in run],np.float32))
                np.testing.assert_allclose(np.diff([r['source_task_time_s'] for r in run]),1/15,atol=2e-9,rtol=0)
                offset+=n
                picks=sorted({0,n//2,n-1}|{i for i,r in enumerate(run) if any(x>=0 for x in r.get('camera_raw_frames',[-1]*3))})
                for ci,camera in enumerate(compare):
                    decoded=video(episode_file(directory,info,index,f'observation.images.{camera}'),n,picks)
                    for j,pixels in decoded.items():
                        raw_index=run[j].get('camera_raw_frames',[-1]*3)[ci]
                        if raw_index>=0:raw_compare[camera].setdefault(raw_index,[]).append(pixels)
                        else:compare[camera].setdefault(run[j]['camera_video_frames'][ci],[]).append(pixels)
            errors=[]
            for raw_camera,camera in zip(('front','wrist','p450'),compare):
                remaining=set(compare[camera])
                with av.open(str(raw/entry['source_path']/'videos'/f'{raw_camera}.mp4')) as container:
                    container.streams.video[0].thread_count=2
                    for i,frame in enumerate(container.decode(video=0)):
                        if i not in remaining:continue
                        pixels=frame.to_ndarray(format='rgb24')
                        for exported in compare[camera][i]:
                            error=float(np.abs(exported.astype(np.int16)-pixels.astype(np.int16)).mean())
                            assert error<6,(name,camera,i,error)
                            errors.append(error)
                        remaining.remove(i)
                assert not remaining
                if raw_compare[camera]:
                    from raw_camera import selected_raw_frames
                    originals=selected_raw_frames(raw/entry['source_path'],raw_camera,raw_compare[camera])
                    for i,images in raw_compare[camera].items():
                        for exported in images:
                            error=float(np.abs(exported.astype(np.int16)-originals[i].astype(np.int16)).mean())
                            assert error<6,(name,camera,i,error)
                            errors.append(error)
            source_checks[name]=dict(split=split,episodes=len(selected),frames=sum(e['frames'] for e in selected),
                all_original_state_action_values_verified=True,all_validity_masks_true=True,no_imputation=True,
                all_videos_decoded=True,source_image_comparisons=len(errors),max_rgb_mean_absolute_error=max(errors,default=0))
            print('STRICT_SOURCE_VERIFIED',split,name,source_checks[name]['frames'],flush=True)
        splits[split]=dict(sources=len(expected[split]),episodes=len(episodes),frames=offset,
            action_windows=sum(max(0,e['frames']-horizon+1) for e in episodes))
    norm=read(root/'assets/heterovla_v3/norm_stats.json')['norm_stats']
    metadata=LeRobotDatasetMetadata('local/heterovla_v3_train',root=root/'lerobot/local/heterovla_v3_train')
    for key,field in [('state','observation.state'),('actions','action')]:
        x=np.asarray(train_arrays[key],np.float64)
        for stat,value in [('mean',x.mean(0)),('std',x.std(0)),('q01',np.quantile(x,.01,axis=0)),('q99',np.quantile(x,.99,axis=0))]:
            np.testing.assert_allclose(norm[key][stat],value,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(metadata.stats[field]['mean'],norm[key]['mean'],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(metadata.stats[field]['std'],norm[key]['std'],atol=1e-6,rtol=1e-6)
    chunks=check(root,horizon)
    prompt=read(root/'configs/task.json')['english_translation'].strip();loader={}
    for split in ('train','val'):
        ds=JointV3Dataset(root,split);assert not ds.allow_imputed and ds.horizon==horizon
        lookup={start:i for i,start in enumerate(ds.starts)}
        boundaries={}
        for e,start,end in zip([e for e in manifest if e['split']==split],ds.dataset.episode_data_index['from'],ds.dataset.episode_data_index['to']):
            start,end=int(start),int(end)
            if end-start<horizon:continue
            pair=boundaries.setdefault(e['source_episode'],[lookup[start],lookup[end-horizon]])
            pair[1]=lookup[end-horizon]
        samples={i for pair in boundaries.values() for i in pair}
        for i in sorted(samples):
            sample=as_openpi_sample(ds[i])
            assert sample['prompt']==prompt and sample['actions'].shape==(horizon,14) and sample['state'].shape==(30,)
        loader[split]=dict(checked_source_boundary_windows=len(samples),all_episode_window_boundaries_verified=True,
            actual_batch_shape=chunks[split]['batch_action_shape'])
        splits[split].update(chunks[split])
    groups={s:sorted({e['source_session'] for e in manifest if e['split']==s}) for s in ('train','val')}
    overlap=sorted(set(groups['train']) & set(groups['val']))
    assert overlap==read(root/'configs/schema.json')['overlapping_source_sessions']
    for name,entry in entries.items():
        if name not in source_checks:
            assert not segments(pq.read_table(root/'aligned'/f'{name}.parquet').to_pylist())
    result=dict(passed=True,format_validation_passed=True,fps=15,default_action_horizon=horizon,
        datasets=splits,sources=source_checks,normalization_train_only=True,imputation_applied=False,
        all_exported_frames_original_valid=True,decoded_exported_videos=sum(s['episodes'] for s in splits.values())*3,
        no_source_episode_leakage=True,no_source_session_leakage=not overlap,overlapping_source_sessions=overlap,
        loader=loader,task_prompt=prompt,training_started=False)
    dump(root/'reports/validation.json',result)
    dump(root/'reports/action_chunk_validation.json',dict(default_action_horizon=horizon,configured_default_datasets=chunks,
        checks={str(horizon):chunks},fps_independent_of_action_horizon=True))
    status=read(root/'STATUS.json');status.update(stage='verified',validation_pending=False,format_validation_passed=True,
        splits=splits,imputation_applied=False,no_source_episode_leakage=True)
    dump(root/'STATUS.json',status)
    print('STRICT_RELEASE_VERIFIED',json.dumps(splits),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','raw','train-scripts'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();verify(args.root.resolve(),args.raw.resolve(),args.train_scripts.resolve())
