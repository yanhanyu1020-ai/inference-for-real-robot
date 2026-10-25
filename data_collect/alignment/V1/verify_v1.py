#!/usr/bin/env python3
"""Validate numerical data, source boundaries, every video, and real loader batches."""
import argparse
import json
import sys
from pathlib import Path
from alignment_paths import BENCHMARK_ROOT, DATA_ROOT, TRAIN_SCRIPTS
import numpy as np
import pyarrow.parquet as pq
import av
from prepare_v1 import TASK,dump,rotate_xy,wrap,Series,causal_video


def test_invariants():
    import pandas as pd
    assert np.allclose(rotate_xy([0,1],np.pi/2),[1,0],atol=1e-12)
    assert abs(wrap(-np.pi+.01-(np.pi-.01))-.02)<1e-10
    s=Series(pd.DataFrame({'a':[0.,10.,20.]}),[0.,.1,.5],['a'])
    assert s.past(.099)[0][0]==0
    assert s.past(-.01) is None
    assert s.interpolate_target(.3) is None
    assert not s.interval_ok(.1,.5)
    m={'times':np.array([.05,.15]),'frames':np.array([0,1]),'common_rows':np.array([0,1]),'warnings':np.array([False,False])}
    assert causal_video(m,.1)[0]==0
    assert causal_video(m,.04) is None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DATA_ROOT)
    ap.add_argument('--source',type=Path,default=BENCHMARK_ROOT);args=ap.parse_args()
    root=args.root.expanduser().resolve();source=args.source.expanduser().resolve()
    test_invariants()
    manifest=json.loads((root/'reports/episode_manifest.json').read_text())
    train_sessions={e['source_session'] for e in manifest if e['split']=='train'}
    val_sessions={e['source_session'] for e in manifest if e['split']=='val'}
    assert not train_sessions&val_sessions
    results={'invariant_tests_passed':True,'no_source_session_leakage':True,'datasets':{},'videos':[]}
    for split in ['train','val']:
        p=root/'lerobot/local'/f'heterovla_v1_{split}'
        info=json.loads((p/'meta/info.json').read_text());frames=0
        for ep in [e for e in manifest if e['split']==split]:
            idx=ep['episode_index'];f=p/'data/chunk-000'/f'episode_{idx:06d}.parquet'
            d=pq.read_table(f).to_pydict();n=len(d['index']);frames+=n
            assert np.asarray(d['observation.state']).shape==(n,30)
            assert np.asarray(d['action']).shape==(n,14)
            assert np.isfinite(d['observation.state']).all() and np.isfinite(d['action']).all()
            assert np.asarray(d['action_valid']).all()
            assert np.allclose(np.diff(d['timestamp']),.1,atol=1e-5)
            assert np.allclose(np.diff(np.asarray(d['source_task_time']).reshape(-1)),.1,atol=1e-5)
            prov=pq.read_table(root/'provenance'/f'{split}_{idx:06d}.parquet').to_pylist()
            assert len(prov)==n and all(r['eligible'] for r in prov)
            for r in prov:
                t=r['source_task_time_s']
                assert all(-1e-9<=t-v<=.15+1e-9 for v in r['camera_source_times_s'])
                assert all(v<=t+1e-9 for v in r['state_source_times_s'])
                assert r['source_episode']!='task_03_1144-1300'
            for cam in ['front','wrist','uav']:
                video=p/'videos/chunk-000'/f'observation.images.{cam}'/f'episode_{idx:06d}.mp4'
                with av.open(str(video)) as c:
                    stream=c.streams.video[0];count=0;last=-1.
                    assert stream.width==640 and stream.height==480
                    for frame in c.decode(video=0):
                        ts=float(frame.pts*frame.time_base);assert ts>last;last=ts;count+=1
                    assert count==n,(video,count,n)
                results['videos'].append({'path':str(video.relative_to(root)),'frames':count})
        assert frames==info['total_frames']
        tasks=[json.loads(l) for l in (p/'meta/tasks.jsonl').read_text().splitlines()]
        assert len(tasks)==1 and tasks[0]['task']==TASK
        results['datasets'][split]={'frames':frames,'episodes':info['total_episodes']}
    # Actual pinned LeRobot loader with decoded images and unpadded chunks.
    sys.path.insert(0, str(TRAIN_SCRIPTS))
    from dataset_v1 import JointV1Dataset,as_openpi_sample
    from torch.utils.data import DataLoader
    import cv2
    for split in ['train','val']:
        ds=JointV1Dataset(root,split,10)
        choices=np.unique(np.linspace(0,len(ds)-1,12,dtype=int))
        for i in choices:
            sample=ds[int(i)];x=as_openpi_sample(sample)
            assert x['state'].shape==(30,) and x['actions'].shape==(10,14)
            assert x['prompt']==TASK
            assert all(a.shape==(480,640,3) and a.dtype==np.uint8 for a in x['image'].values())
        # Check source frame identity and RGB channel order through the actual reader.
        image_errors=[]
        for ep in [e for e in manifest if e['split']==split]:
            epi=ep['episode_index'];n=ep['frames'];j=min(n//2,n-10)
            global_index=int(ds.dataset.episode_data_index['from'][epi])+j
            loaded=ds.dataset[global_index]
            row=pq.read_table(root/'provenance'/f'{split}_{epi:06d}.parquet').to_pylist()[j]
            for ci,(source_cam,key_cam) in enumerate([('front','front'),('wrist','wrist'),('p450','uav')]):
                path=source/ep['source_episode']/'videos'/f'{source_cam}.mp4'
                cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,row['camera_video_frames'][ci])
                ok,original=cap.read();cap.release();assert ok
                expected=cv2.cvtColor(original,cv2.COLOR_BGR2RGB).astype(float)/255
                actual=loaded[f'observation.images.{key_cam}'].numpy().transpose(1,2,0)
                error=float(np.mean(np.abs(expected-actual))*255)
                assert error<8,(str(path),error,'Wrong frame or channel order')
                image_errors.append(error)
        batch=next(iter(DataLoader(ds,batch_size=4,num_workers=0,shuffle=False)))
        assert tuple(batch['action'].shape)==(4,10,14)
        results['datasets'][split].update(h10_windows=len(ds),real_lerobot_loader_passed=True,
            dataloader_batch_action_shape=list(batch['action'].shape),
            source_frame_identity_rgb_check=True,max_reencode_mean_absolute_pixel_error=max(image_errors))
    stats=json.loads((root/'assets/heterovla_v1/norm_stats.json').read_text())['norm_stats']
    train_arrays={'state':[],'actions':[]}
    for f in sorted((root/'lerobot/local/heterovla_v1_train/data').rglob('*.parquet')):
        d=pq.read_table(f).to_pydict();train_arrays['state'].extend(d['observation.state']);train_arrays['actions'].extend(d['action'])
    for k,a in train_arrays.items():assert np.allclose(stats[k]['mean'],np.mean(a,axis=0),atol=1e-8)
    results['normalization_train_only']=True
    results['format_validation_passed']=True
    results['full_vla_training_run']=False
    results['physical_alignment_independently_validated']=False
    results['action_semantics']='kinematic proxy for Go2/UAV; logged target for Piper'
    dump(root/'reports/validation.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
