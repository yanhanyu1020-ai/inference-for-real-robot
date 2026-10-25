"""Verify every exported batch episode, video and training label, then exercise the V3 loader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow.parquet as pq

from alignment_core import segments, dump
from raw_checks import episode_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def video(path, count, picks):
    selected={}
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        stream.thread_count=2
        assert float(stream.average_rate)==15 and (stream.width,stream.height)==(640,480)
        n=0
        for i,frame in enumerate(container.decode(video=0)):
            assert abs(float(frame.pts*frame.time_base)-i/15)<1e-4
            if i in picks:selected[i]=frame.to_ndarray(format='rgb24')
            n+=1
    assert n==count,(path,n,count)
    assert set(selected)==set(picks)
    return selected


def verify(source,root,train_scripts):
    sys.path.insert(0,str(train_scripts))
    from dataset_v3 import JointV3Dataset,as_openpi_sample
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from torch.utils.data import DataLoader
    upload=read(root/'reports/upload_integrity.json')
    assert upload['all_manifest_files_passed'] and upload['all_task_artifacts_passed']
    manifest=read(root/'reports/episode_manifest.json')
    summary=read(root/'reports/alignment_summary.json')
    directory=root/'lerobot/local/heterovla_v3_train'
    info=read(directory/'meta/info.json')
    assert info['fps']==15 and info['codebase_version']=='v2.1'
    assert info['total_frames']==summary['exported_frames'] and info['total_episodes']==summary['exported_segments']
    assert len(manifest)==info['total_episodes']
    assert len(list((directory/'data').rglob('*.parquet')))==len(manifest)
    assert len(list((directory/'videos').rglob('*.mp4')))==len(manifest)*3
    arrays={'state':[],'actions':[]}
    verified_sources={}
    offset=0
    for report in summary['sources']:
        name=report['source_episode']
        audit=pq.read_table(root/'aligned'/f'{name}.parquet').to_pylist()
        expected=segments(audit)
        episodes=[e for e in manifest if e['source_episode']==name]
        assert len(expected)==len(episodes)
        compare={camera:{} for camera in ('front','wrist','uav')}
        raw_compare={camera:{} for camera in ('front','wrist','uav')}
        for run,e in zip(expected,episodes):
            index=e['episode_index'];n=len(run)
            assert n==e['frames']
            provenance=pq.read_table(root/'provenance'/f'train_{index:06d}.parquet').to_pylist()
            assert provenance==run
            table=pq.read_table(episode_file(directory,info,index)).to_pydict()
            for field,key in [('observation.state','state'),('action','action')]:
                x=np.asarray(table[field])
                assert x.shape==(n,30 if key=='state' else 14) and np.isfinite(x).all()
                np.testing.assert_array_equal(x,[r[key] for r in run])
                arrays['state' if key=='state' else 'actions'].extend(x)
            assert np.asarray(table['action_valid']).all()
            np.testing.assert_array_equal(np.asarray(table['source_task_time']).reshape(-1),np.asarray([r['source_task_time_s'] for r in run],np.float32))
            np.testing.assert_array_equal(table['index'],np.arange(offset,offset+n))
            np.testing.assert_array_equal(table['episode_index'],np.full(n,index))
            np.testing.assert_array_equal(table['frame_index'],np.arange(n))
            np.testing.assert_allclose(table['timestamp'],np.arange(n)/15,atol=1e-5,rtol=0)
            offset+=n
            picks=sorted({0,n//2,n-1} | {i for i,r in enumerate(run) if any(x>=0 for x in r.get('camera_raw_frames',[-1]*3))})
            for ci,camera in enumerate(('front','wrist','uav')):
                decoded=video(episode_file(directory,info,index,f'observation.images.{camera}'),n,picks)
                for row_index,pixels in decoded.items():
                    original_index=run[row_index].get('camera_raw_frames',[-1]*3)[ci]
                    if original_index>=0:
                        raw_compare[camera].setdefault(original_index,[]).append(pixels)
                        continue
                    raw_index=run[row_index]['camera_video_frames'][ci]
                    compare[camera].setdefault(raw_index,[]).append(pixels)
        errors=[]
        for raw_camera,camera in zip(('front','wrist','p450'),('front','wrist','uav')):
            with av.open(str(source.parent/report['source_path']/'videos'/f'{raw_camera}.mp4')) as container:
                container.streams.video[0].thread_count=2
                remaining=set(compare[camera])
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
                originals=selected_raw_frames(source.parent/report['source_path'],raw_camera,raw_compare[camera])
                for index,images in raw_compare[camera].items():
                    for exported in images:
                        error=float(np.abs(exported.astype(np.int16)-originals[index].astype(np.int16)).mean())
                        assert error<6,(name,camera,index,error)
                        errors.append(error)
        verified_sources[name]=dict(episodes=len(episodes),frames=sum(e['frames'] for e in episodes),
            all_tables_match_aligned_and_provenance=True,all_videos_decoded=True,source_image_comparisons=len(errors),
            max_rgb_mean_absolute_error=max(errors,default=0),h20_windows=report['h20_windows'])
        print('BATCH_VERIFIED',report['task_number'],len(episodes),report['exported_frames'],flush=True)
    norm=read(root/'assets/heterovla_v3/norm_stats.json')['norm_stats']
    metadata=LeRobotDatasetMetadata('local/heterovla_v3_train',root=directory)
    for key,field in [('state','observation.state'),('actions','action')]:
        x=np.asarray(arrays[key],np.float64)
        for name,value in [('mean',x.mean(0)),('std',x.std(0)),('q01',np.quantile(x,.01,axis=0)),('q99',np.quantile(x,.99,axis=0))]:
            np.testing.assert_allclose(norm[key][name],value,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(metadata.stats[field]['mean'],norm[key]['mean'],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(metadata.stats[field]['std'],norm[key]['std'],atol=1e-6,rtol=1e-6)
    dataset=JointV3Dataset(root,'train')
    assert dataset.horizon==20 and len(dataset)==summary['h20_windows']
    expected=sum(max(0,e['frames']-19) for e in manifest)
    assert len(dataset)==expected
    lookup={start:i for i,start in enumerate(dataset.starts)}
    samples=set()
    for start,end in zip(dataset.dataset.episode_data_index['from'],dataset.dataset.episode_data_index['to']):
        start,end=int(start),int(end)
        if end-start>=20:samples.update([lookup[start],lookup[end-20]])
    for i in samples:
        sample=as_openpi_sample(dataset[i])
        assert sample['actions'].shape==(20,14) and sample['state'].shape==(30,)
        assert sample['prompt']==read(root/'configs/task.json')['english_translation'].strip()
    batch=next(iter(DataLoader(dataset,batch_size=4,num_workers=0)))
    assert list(batch['action'].shape)==[4,20,14]
    result=dict(passed=True,format_validation_passed=True,fps=15,action_horizon=20,source_count=len(summary['sources']),
        h20_usable_sources=sum(r['h20_windows']>0 for r in summary['sources']),
        h20_unusable_sources=[r['source_episode'] for r in summary['sources'] if not r['h20_windows']],
        episodes=len(manifest),frames=offset,windows=len(dataset),decoded_exported_videos=len(manifest)*3,
        train_only_normalization_verified=True,loader_boundary_samples=len(samples),batch_action_shape=[4,20,14],
        task_prompt=read(root/'configs/task.json')['english_translation'],sources=verified_sources,
        active_v3_modified=False,training_started=False,validation_split_created=False,imputation_applied=False)
    dump(root/'reports/validation.json',result)
    print('STANDALONE_BATCH_VERIFIED',len(manifest),offset,len(dataset),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','root','train-scripts'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    verify(args.source.resolve(),args.root.resolve(),args.train_scripts.resolve())
