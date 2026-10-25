#!/usr/bin/env python3
"""Validate all 15 Hz alignment labels against raw timestamps and actual LeRobot data."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import av

from paths import DATA_ROOT, BENCHMARK_ROOT, REFERENCE_ROOT, TRAIN_SCRIPTS
from prepare import FPS, DT, HORIZON, TASK, JOINTS, unitree, uav, pose_series, dump

CAMERAS = (("front", "front"), ("wrist", "wrist"), ("p450", "uav"))


def read(path):
    return json.loads(path.read_text())


def episode_file(directory, info, index, video_key=None):
    chunk = int(index) // int(info.get("chunks_size", 1000))
    if video_key is None:
        template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    else:
        template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return directory / template.format(episode_chunk=chunk, episode_index=int(index), video_key=video_key)


def check_video(path, expected_frames):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert abs(float(stream.average_rate) - FPS) < 1e-8, path
        assert (stream.width, stream.height) == (640, 480), path
        count, last = 0, -1.0
        for frame in container.decode(video=0):
            timestamp = float(frame.pts * frame.time_base)
            assert timestamp > last, (path, count, timestamp)
            assert abs(timestamp - count / 15) <= 1e-4, (path, count, timestamp)
            last = timestamp
            count += 1
        assert count == expected_frames, (path, count, expected_frames)
    return {"frames": count, "width": 640, "height": 480}


def past_index(series, t, age):
    i = int(np.searchsorted(series.times, t+1e-10, side='right')-1)
    assert i >= 0 and -1e-9 <= t-series.times[i] <= age+1e-9
    assert not series.warnings[i] and np.isfinite(series.values[i]).all()
    return i


def heading(xy, yaw):
    # Explicit rotation matrix, independent of the producer's rotate_xy helper.
    return np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]]) @ xy


def angle_delta(a,b):
    return (b-a+np.pi) % (2*np.pi) - np.pi


def motion(series, t, end, yaw_index, include_z, indices):
    endpoints=[];expected=[]
    for stamp in [t,end]:
        j=int(np.searchsorted(series.times,stamp,side='left'));i=j-1
        assert i>=0 and j<len(series.times)
        assert 0<series.times[j]-series.times[i]<=.12
        assert not series.warnings[i:j+1].any()
        expected.extend([i,j])
        endpoints.append(np.array([np.interp(stamp,series.times[i:j+1],series.values[i:j+1,c]) for c in range(series.values.shape[1])]))
    assert indices==expected
    i=max(0,int(np.searchsorted(series.times,t,side='right'))-1)
    j=min(len(series.times),int(np.searchsorted(series.times,end,side='left'))+1)
    assert not series.warnings[i:j].any() and np.max(np.diff(series.times[i:j]))<=.12
    a,b=endpoints;dt=end-t
    xy=heading((b[:2]-a[:2])/dt,a[yaw_index])
    yaw=angle_delta(a[yaw_index],b[yaw_index])/dt
    return [*xy,(b[2]-a[2])/dt,yaw] if include_z else [*xy,yaw]


def past_velocity(series,t,yaw_index,include_z):
    i=past_index(series,t-DT,.075);j=past_index(series,t,.075)
    assert series.times[j]>series.times[i]
    a,b=series.values[i],series.values[j];dt=series.times[j]-series.times[i]
    xy=heading((b[:2]-a[:2])/dt,b[yaw_index])
    yaw=angle_delta(a[yaw_index],b[yaw_index])/dt
    return [*xy,(b[2]-a[2])/dt,yaw] if include_z else [*xy,yaw]


def check_raw_alignment(raw, rows, entry):
    info=read(raw/'episode.json');mapping=pd.read_csv(raw/'timing/frame_mapping.csv')
    grid=mapping[mapping.task_time_s<float(info['duration_s'])]
    assert len(rows)==len(grid)
    np.testing.assert_array_equal([r['source_frame_index'] for r in rows],grid.frame)
    np.testing.assert_array_equal([r['source_task_time_s'] for r in rows],grid.task_time_s)
    assert np.allclose(np.diff(grid.task_time_s),DT,atol=2e-9,rtol=0)
    for row in rows:
        assert row['source_episode']==entry['source_episode'] and row['source_path']==entry['source_path']
        assert row['source_session']==entry['source_session'] and row['split']==entry['split']
        assert row['eligible']==(not row['rejection_reasons'])
        assert abs(row['action_target_time_s']-row['source_task_time_s']-DT)<1e-12
        assert row['action_dt_s']==DT
    if entry['quarantined']:
        assert not any(r['eligible'] for r in rows)
        assert all('source_episode_clock_quarantine' in r['rejection_reasons'] for r in rows)
        return dict(video_grid_rows=len(rows),eligible_rows_verified=0,quarantined=True)

    go=unitree(raw,'sport_mode_state',['position_x','position_y','position_z','yaw','body_height','roll','pitch'])
    go.values[:,3]=np.unwrap(go.values[:,3])
    arm=unitree(raw,'piper_state',JOINTS);cmd=unitree(raw,'piper_gamepad',JOINTS)
    drone=pose_series(raw)
    flags=uav(raw,'uav1__prometheus__state',['connected','armed','odom_valid'])
    landed=uav(raw,'uav1__mavros__extended_state',['landed_state'])
    cameras=[]
    for camera,_ in CAMERAS:
        times=(mapping[f'{camera}_mapped_time_ns'].to_numpy(np.int64)-int(info['start_desktop_ns']))/1e9
        valid=mapping.get(f'{camera}_valid',pd.Series(np.ones(len(mapping),dtype=int))).astype(bool).to_numpy()
        frames=mapping.get(f'{camera}_video_frame',mapping.frame).to_numpy()
        indices=np.flatnonzero(valid & pd.notna(frames))
        order=indices[np.argsort(times[indices],kind='stable')]
        cameras.append((times,frames,order))
    eligible=[r for r in rows if r['eligible']]
    for row in eligible:
        t=row['source_task_time_s'];end=t+DT
        assert end<=float(info['duration_s'])+2e-9
        assert all(row['state_valid']) and all(row['action_valid']) and all(row['camera_valid'])
        assert row['action_is_kinematic_proxy']==[True]*3+[False]*7+[True]*4
        for c,(times,frames,order) in enumerate(cameras):
            pos=int(np.searchsorted(times[order],t+1e-10,side='right')-1)
            assert pos>=0
            j=int(order[pos]);age=t-times[j]
            assert -1e-9<=age<=.15+1e-9 and not bool(mapping.iloc[j].clock_warning)
            assert row['camera_mapping_rows'][c]==j
            assert row['camera_video_frames'][c]==int(frames[j])
            assert abs(row['camera_source_times_s'][c]-times[j])<1e-12
        assert np.ptp(row['camera_source_times_s'])<=.1+1e-9
        indices=[past_index(s,t,age) for s,age in [(go,.075),(arm,.075),(drone,.075),(flags,.15),(landed,.25)]]
        np.testing.assert_allclose(row['state_source_times_s'],[s.times[i] for s,i in zip([go,arm,drone,flags,landed],indices)],rtol=0,atol=1e-12)
        g,p,u,f,l=[s.values[i] for s,i in zip([go,arm,drone,flags,landed],indices)]
        assert g[4]>=.23 and np.all(f==1)
        go_xy=heading(g[:2]-go.values[0,:2],go.values[0,3]);go_yaw=g[3]-go.values[0,3]
        u_xy=heading(u[:2]-drone.values[0,:2],drone.values[0,5]);u_yaw=u[5]-drone.values[0,5]
        expected_state=[*go_xy,g[4],g[5],g[6],np.sin(go_yaw),np.cos(go_yaw),*past_velocity(go,t,3,False),
            *p,*u_xy,u[2],u[3],u[4],np.sin(u_yaw),np.cos(u_yaw),*past_velocity(drone,t,5,True),f[1],l[0]]
        ci=past_index(cmd,end,.10);command=cmd.df.iloc[ci]
        assert command.command_sent==1 and command.arm_enabled==1 and command.arm_connected==1 and command.low_level_mode=='joint'
        assert row['action_source_rows'][4]==ci and row['piper_command_source_time_s']==cmd.times[ci]
        expected_action=[*motion(go,t,end,3,False,row['action_source_rows'][:4]),*cmd.values[ci],
            *motion(drone,t,end,5,True,row['action_source_rows'][5:])]
        np.testing.assert_allclose(row['state'],np.asarray(expected_state,dtype=np.float32),atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(row['action'],np.asarray(expected_action,dtype=np.float32),atol=2e-6,rtol=2e-6)
        a=np.asarray(row['action']);assert np.all(np.abs(a[[0,1,10,11,12]])<=3) and np.all(np.abs(a[[2,13]])<=4)
        assert 0<=a[9]<=.075
    return dict(video_grid_rows=len(rows),eligible_rows_verified=len(eligible),
        video_timestamp_selection_verified=True,state_values_verified=True,action_values_verified=True)


def validate(args):
    root,source,reference=[p.expanduser().resolve() for p in [args.root,args.source,args.reference_root]]
    assert root!=reference and not root.is_relative_to(reference)
    schema=read(root/'configs/schema.json')
    assert schema['fps']==FPS and schema['horizon']==HORIZON and schema['action_dt_s']==DT
    inventory=read(root/'configs/source_inventory.json')
    original=read(reference/'configs/source_inventory.json')
    assert inventory['sources'][:32]==original['sources']
    assert len(inventory['sources'])==40 and sum(e['include'] for e in inventory['sources'])==39
    extra=inventory['sources'][32:]
    assert len(extra)==8 and all(e['source_path'].startswith('2026-10-18/') for e in extra)
    assert {int(Path(e['source_path']).name.split('_')[1]) for e in extra}=={1,2,3,5,6,7,8,11}
    assert sum(e['split']=='val' for e in extra)==3
    assert {e['source_session'] for e in extra if e['split']=='val'}=={'joint_20261018_152436_UTCp0800_f4b0'}
    sessions={s:{e['source_session'] for e in inventory['sources'] if e['split']==s} for s in ['train','val']}
    assert not sessions['train'] & sessions['val']
    manifest=read(root/'reports/episode_manifest.json');summary=read(root/'reports/alignment_summary.json')
    manifest_by_source=defaultdict(list)
    for e in manifest: manifest_by_source[e['source_episode']].append(e)
    audits={};raw_checks={}
    for entry in inventory['sources']:
        key=entry['source_episode'];raw=(source/entry['source_path']).resolve()
        assert raw.is_relative_to(source)
        rows=pq.read_table(root/'aligned'/f'{key}.parquet').to_pylist();audits[key]=rows
        raw_checks[key]=check_raw_alignment(raw,rows,entry)
        runs=[];current=[]
        for row in rows:
            if row['eligible']:
                if current:
                    assert row['source_frame_index']==current[-1]['source_frame_index']+1
                    assert abs(row['source_task_time_s']-current[-1]['source_task_time_s']-DT)<2e-9
                current.append(row)
            else:
                if len(current)>=HORIZON: runs.append(current)
                current=[]
        if len(current)>=HORIZON:runs.append(current)
        expected={(r[0]['source_task_time_s'],r[-1]['source_task_time_s'],len(r)) for r in runs}
        actual={(e['source_start_s'],e['source_last_s'],e['frames']) for e in manifest_by_source[key]}
        assert expected==actual,(key,'Segment coverage mismatch')
        report=read(root/'reports'/f'{key}.json')
        assert report['eligible_frames']==sum(r['eligible'] for r in rows)
        assert report['exported_segments']==len(runs) and report['exported_frames']==sum(map(len,runs))
        assert report['rejection_counts']==dict(Counter(reason for r in rows for reason in r['rejection_reasons']))
        if not entry['include']:assert not runs
        print('RAW_VERIFIED',key,len(rows),raw_checks[key]['eligible_rows_verified'],flush=True)
    expected_sources={k for k,v in manifest_by_source.items() if v}
    assert summary['included_sources_with_h15_segments']==len(expected_sources)
    assert set(summary['non_exportable_source_episodes'])=={e['source_episode'] for e in inventory['sources'] if e['include'] and e['source_episode'] not in expected_sources}
    result=dict(fps=FPS,horizon=HORIZON,original_v2_sources_and_splits_preserved=True,additional_upload_sources=8,no_source_session_leakage=True,
        independent_raw_alignment_checks=raw_checks,datasets={},videos=[],
        non_exportable_source_episodes=summary['non_exportable_source_episodes'])
    provenance={};train_arrays={'state':[],'actions':[]}
    for split in ['train','val']:
        directory=root/'lerobot/local'/f'heterovla_v2_15hz_{split}'
        info=read(directory/'meta/info.json');episodes=[e for e in manifest if e['split']==split]
        assert info['fps']==FPS and info['codebase_version']=='v2.1'
        assert info['total_episodes']==len(episodes) and info['total_videos']==3*len(episodes)
        assert [e['episode_index'] for e in episodes]==list(range(len(episodes)))
        frame_count=0
        for e in episodes:
            index,n=e['episode_index'],e['frames']
            assert n>=HORIZON and e['valid_h15_windows']==n-HORIZON+1
            data=pq.read_table(episode_file(directory,info,index)).to_pydict()
            rows=pq.read_table(root/'provenance'/f'{split}_{index:06d}.parquet').to_pylist()
            assert len(rows)==n and all(r['eligible'] for r in rows)
            for r in rows:
                assert r==audits[e['source_episode']][r['source_frame_index']]
                assert r['source_path']==e['source_path'] and r['source_session']==e['source_session'] and r['split']==split
            assert rows[0]['source_task_time_s']==e['source_start_s'] and rows[-1]['source_task_time_s']==e['source_last_s']
            state=np.asarray(data['observation.state'],np.float32);action=np.asarray(data['action'],np.float32)
            assert state.shape==(n,30) and action.shape==(n,14) and np.isfinite(state).all() and np.isfinite(action).all()
            np.testing.assert_array_equal(state,np.asarray([r['state'] for r in rows],np.float32))
            np.testing.assert_array_equal(action,np.asarray([r['action'] for r in rows],np.float32))
            assert np.asarray(data['action_valid']).shape==(n,14) and np.asarray(data['action_valid']).all()
            np.testing.assert_array_equal(np.asarray(data['source_task_time'],np.float32).reshape(-1),np.asarray([r['source_task_time_s'] for r in rows],np.float32))
            np.testing.assert_allclose(data['timestamp'],np.arange(n)/FPS,atol=1e-4,rtol=0)
            np.testing.assert_array_equal(data['frame_index'],np.arange(n))
            np.testing.assert_array_equal(data['episode_index'],np.full(n,index))
            np.testing.assert_array_equal(data['index'],np.arange(frame_count,frame_count+n))
            provenance[(split,index)]=rows
            for _,camera in CAMERAS:
                path=episode_file(directory,info,index,f'observation.images.{camera}')
                result['videos'].append(dict(path=str(path.relative_to(root)),**check_video(path,n)))
            if split=='train': train_arrays['state'].extend(state);train_arrays['actions'].extend(action)
            frame_count+=n
        assert frame_count==info['total_frames']
        tasks=[json.loads(line) for line in (directory/'meta/tasks.jsonl').read_text().splitlines()]
        assert len(tasks)==1 and tasks[0]['task']==TASK
        result['datasets'][split]=dict(frames=frame_count,episodes=len(episodes),sources=len({e['source_episode'] for e in episodes}))
        print('ALL_VIDEOS_VERIFIED',split,len(episodes)*3,frame_count,flush=True)

    # The release contains its exact strict loader; use the maintained loader for source-tree runs.
    sys.path.insert(0,str(root/'scripts'))
    if not (root/'scripts/dataset_15hz.py').exists():sys.path.insert(0,str(TRAIN_SCRIPTS))
    from dataset_15hz import JointV215HzDataset,as_openpi_sample
    from torch.utils.data import DataLoader
    import cv2
    for split in ['train','val']:
        ds=JointV215HzDataset(root,split,HORIZON)
        expected=sum(e['frames']-HORIZON+1 for e in manifest if e['split']==split)
        assert len(ds)==expected
        for i in np.unique(np.linspace(0,len(ds)-1,min(24,len(ds)),dtype=int)):
            sample=as_openpi_sample(ds[int(i)])
            assert sample['state'].shape==(30,) and sample['actions'].shape==(HORIZON,14) and sample['prompt']==TASK
            assert all(x.shape==(480,640,3) and x.dtype==np.uint8 for x in sample['image'].values())
        # Verify each episode's first and last complete window against its raw source pixels.
        errors=[]
        groups=defaultdict(list)
        for e in manifest:
            if e['split']==split:groups[e['source_path']].append(e)
        for source_path,episodes in groups.items():
            requests=defaultdict(list)
            for e in episodes:
                for offset in sorted({0,e['frames']-HORIZON}):
                    global_index=int(ds.dataset.episode_data_index['from'][e['episode_index']])+offset
                    loaded=ds.dataset[global_index]
                    row=provenance[(split,e['episode_index'])][offset]
                    for ci,(raw_camera,camera) in enumerate(CAMERAS):
                        pixels=np.asarray(loaded[f'observation.images.{camera}']).transpose(1,2,0)
                        requests[(raw_camera,row['camera_video_frames'][ci])].append(pixels)
            for raw_camera,_ in CAMERAS:
                wanted={idx for cam,idx in requests if cam==raw_camera}
                cap=cv2.VideoCapture(str(source/source_path/'videos'/f'{raw_camera}.mp4'));index=0
                while wanted:
                    ok,pixels=cap.read();assert ok,(source_path,raw_camera,index)
                    if index in wanted:
                        expected=cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB).astype(np.float32)/255
                        for actual in requests[(raw_camera,index)]:
                            error=float(np.mean(np.abs(actual-expected))*255)
                            assert error<8.,(source_path,raw_camera,index,error)
                            errors.append(error)
                        wanted.remove(index)
                    index+=1
                cap.release()
            print('LOADER_RGB_VERIFIED',split,source_path,flush=True)
        batch=next(iter(DataLoader(ds,batch_size=min(4,len(ds)),num_workers=0,shuffle=False)))
        assert tuple(batch['action'].shape)==(min(4,len(ds)),HORIZON,14)
        result['datasets'][split].update(h15_windows=len(ds),real_lerobot_loader_passed=True,
            dataloader_batch_action_shape=list(batch['action'].shape),source_frame_identity_rgb_check=True,
            source_frame_samples=len(errors),max_reencode_mean_absolute_pixel_error=max(errors))
    stats=read(root/'assets/heterovla_v2_15hz/norm_stats.json')['norm_stats']
    for key,values in train_arrays.items():
        x=np.asarray(values,np.float64)
        expected=dict(mean=x.mean(0),std=x.std(0),q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0))
        for name,arr in expected.items():np.testing.assert_allclose(stats[key][name],arr,atol=1e-8,rtol=1e-7)
    result.update(normalization_train_only=True,all_exported_videos_decoded=True,
        all_eligible_state_action_values_verified_against_raw=True,format_validation_passed=True,
        physical_alignment_independently_validated=False,full_vla_training_run=False)
    dump(root/'reports/validation.json',result)
    print(json.dumps(dict(status='passed',datasets=result['datasets']),ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DATA_ROOT)
    ap.add_argument('--source',type=Path,default=BENCHMARK_ROOT)
    ap.add_argument('--reference-root',type=Path,default=REFERENCE_ROOT)
    args=ap.parse_args()
    try:validate(args)
    except Exception as exc:
        root=args.root.expanduser().resolve();reference=args.reference_root.expanduser().resolve()
        if root.is_dir() and root!=reference and not root.is_relative_to(reference):
            dump(root/'reports/validation_failure.json',dict(format_validation_passed=False,error=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()

