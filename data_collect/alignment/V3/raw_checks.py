"""Independent timestamp and numerical checks for V3 source alignment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import av
from alignment_core import FPS, DT, JOINTS, unitree, uav, pose_series
CAMERAS = (("front", "front"), ("wrist", "wrist"), ("p450", "uav"))

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
            raw_index=row.get('camera_raw_frames',[-1]*3)[c]
            if raw_index>=0:
                camera=CAMERAS[c][0]
                raw_map=pd.read_csv(raw/'timing'/f'{camera}_raw_frames.csv')
                raw_times=raw_map.task_time_s.to_numpy(float)
                selected=int(np.searchsorted(raw_times,t+1e-10,side='right')-1)
                assert raw_index==selected and int(raw_map.iloc[selected].raw_frame)==selected
                assert not bool(raw_map.iloc[selected].clock_warning)
                assert 0<=t-raw_times[selected]<=.15
                assert row['camera_video_frames'][c]==-1 and row['camera_mapping_rows'][c]==-1
                assert abs(row['camera_source_times_s'][c]-raw_times[selected])<1e-12
                continue
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


