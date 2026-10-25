#!/usr/bin/env python3
"""Read-only benchmark -> auditable synchronized tables -> LeRobot v2.1.

The primary action is a *kinematic imitation target*, NOT a reconstructed
operator command: future Go2/UAV odometry displacement divided by dt, plus
Piper's logged commanded absolute joint/gripper target at the interval end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import shutil
import tempfile
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TASK = "抓取桌子上的纸箱内的方块"
TASK_EN = "Pick up the block inside the cardboard box on the table."
FPS, DT, HORIZON = 15, 1 / 15, 15
JOINTS = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
ACTION_NAMES = ["go2.vx_heading", "go2.vy_heading", "go2.yaw_rate"] + [
    f"piper.q{i}_target" for i in range(1, 7)
] + ["piper.gripper_target"] + ["uav.vx_heading", "uav.vy_heading", "uav.vz", "uav.yaw_rate"]
STATE_NAMES = ["go2.x_episode", "go2.y_episode", "go2.body_height", "go2.roll", "go2.pitch",
    "go2.sin_yaw_episode", "go2.cos_yaw_episode", "go2.vx_heading", "go2.vy_heading", "go2.yaw_rate"] + [
    f"piper.q{i}" for i in range(1, 7)
] + ["piper.gripper"] + ["uav.x_episode", "uav.y_episode", "uav.z_local", "uav.roll", "uav.pitch",
    "uav.sin_yaw_episode", "uav.cos_yaw_episode", "uav.vx_heading", "uav.vy_heading", "uav.vz",
    "uav.yaw_rate", "uav.armed", "uav.landed_state"]
CAMERAS = ["front", "wrist", "p450"]
CAM_KEYS = ["observation.images.front", "observation.images.wrist", "observation.images.uav"]


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def rotate_xy(xy, yaw):
    """Local/world xy -> heading-aligned forward/left xy (yaw-only FLU)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * xy[0] + s * xy[1], -s * xy[0] + c * xy[1]])


def q_to_rpy(x, y, z, w):
    q = np.array([x, y, z, w], dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or abs(n - 1) > 0.1:
        return np.full(3, np.nan)
    x, y, z, w = q / n
    return np.array([math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        math.asin(np.clip(2*(w*y-z*x), -1, 1)), math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


class Series:
    def __init__(self, df, times, columns, warnings=None, name=""):
        self.df = df
        self.times = np.asarray(times, dtype=float)
        self.values = df[columns].to_numpy(dtype=float)
        self.warnings = np.asarray(warnings if warnings is not None else np.zeros(len(df)), dtype=bool)
        self.columns, self.name = list(columns), name
        if len(self.times) != len(df) or np.any(np.diff(self.times) < 0):
            raise ValueError(f"Unordered or mismatched timestamps: {name}")

    def past(self, t, max_age=0.1):
        i = int(np.searchsorted(self.times, t + 1e-10, side="right") - 1)
        if i < 0:
            return None
        if t - self.times[i] > max_age or self.warnings[i] or not np.isfinite(self.values[i]).all():
            return None
        return self.values[i].copy(), i

    def interpolate_target(self, t, max_gap=0.12):
        """ONLY for future motion targets; never used for observation features."""
        j = int(np.searchsorted(self.times, t, side="left"))
        if j == 0 or j >= len(self.times):
            return None
        i = j - 1
        gap = self.times[j] - self.times[i]
        if gap <= 0 or gap > max_gap or self.warnings[i:j+1].any():
            return None
        if not np.isfinite(self.values[i:j+1]).all():
            return None
        f = (t - self.times[i]) / gap
        return self.values[i] * (1-f) + self.values[j] * f, [i, j]

    def interval_ok(self, a, b, max_gap=0.12):
        i = max(0, int(np.searchsorted(self.times, a, side="right")) - 1)
        j = min(len(self.times), int(np.searchsorted(self.times, b, side="left")) + 1)
        return j-i >= 2 and not self.warnings[i:j].any() and np.max(np.diff(self.times[i:j])) <= max_gap


def unitree(e, filename, columns):
    df = pd.read_csv(e / "states/unitree" / f"{filename}.csv")
    timing = pd.read_csv(e / "timing" / f"{filename}_timing.csv")
    if not np.array_equal(timing["row"], np.arange(len(df))):
        raise ValueError(f"Row map mismatch: {filename}")
    return Series(df, timing.task_time_s, columns, timing.clock_warning, filename)


def uav(e, filename, columns):
    df = pd.read_csv(e / "states/p450" / f"{filename}.csv")
    return Series(df, df.task_time_s, columns, df.clock_warning, filename)


def pose_series(e):
    s = uav(e, "uav1__mavros__local_position__pose", ["pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"])
    rpy = np.array([q_to_rpy(*v[3:]) for v in s.values])
    rpy[:, 2] = np.unwrap(rpy[:, 2])
    s.values = np.concatenate([s.values[:, :3], rpy], axis=1)
    return s


def kinematic_action(series, t, yaw_index, xyz_dims, include_z):
    p0 = series.interpolate_target(t)
    p1 = series.interpolate_target(t + DT)
    if p0 is None or p1 is None or not series.interval_ok(t, t + DT):
        return None
    v0, i0 = p0
    v1, i1 = p1
    xy = rotate_xy((v1[:2] - v0[:2]) / DT, v0[yaw_index])
    w = float(wrap(v1[yaw_index] - v0[yaw_index]) / DT)
    vals = [*xy, (v1[2]-v0[2])/DT, w] if include_z else [*xy, w]
    return np.array(vals), [*i0, *i1]


def causal_velocity(series, t, yaw_index, include_z):
    p0, p1 = series.past(t-DT, 0.075), series.past(t, 0.075)
    if p0 is None or p1 is None or not series.interval_ok(t-DT, t):
        return None
    a, i = p0
    b, j = p1
    dt = series.times[j] - series.times[i]
    if dt <= 0:
        return None
    xy = rotate_xy((b[:2]-a[:2])/dt, b[yaw_index])
    w = float(wrap(b[yaw_index]-a[yaw_index])/dt)
    return np.array([*xy, (b[2]-a[2])/dt, w] if include_z else [*xy, w])


def video_maps(e, start_ns):
    df = pd.read_csv(e / "timing/frame_mapping.csv")
    result = {}
    for cam in CAMERAS:
        valid = df.get(f"{cam}_valid", pd.Series(np.ones(len(df), dtype=int))).astype(bool)
        indices = df.get(f"{cam}_video_frame", df["frame"])
        # Subtract as integers BEFORE converting epoch-nanoseconds to float.
        times = (df[f"{cam}_mapped_time_ns"].to_numpy(np.int64) - start_ns) / 1e9
        rows = np.where(valid & indices.notna())[0]
        order = rows[np.argsort(times[rows], kind="stable")]
        result[cam] = dict(times=times[order], frames=indices.to_numpy()[order].astype(int),
            common_rows=order, warnings=df.clock_warning.to_numpy(dtype=bool)[order])
    return result


def causal_video(mapping, t, max_age=0.15):
    i = int(np.searchsorted(mapping["times"], t+1e-10, side="right") - 1)
    if i < 0 or t-mapping["times"][i] > max_age or mapping["warnings"][i]:
        return None
    return int(mapping["frames"][i]), float(mapping["times"][i]), int(mapping["common_rows"][i])


def video_timeline(e):
    """Use the existing common video timeline; never infer time from CSV row rate."""
    info = json.loads((e/'episode.json').read_text(encoding="utf-8-sig"))
    mapping = pd.read_csv(e/'timing/frame_mapping.csv')
    times = mapping.task_time_s.to_numpy(dtype=np.float64)
    frames = mapping.frame.to_numpy(dtype=np.int64)
    if (not len(times) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not np.array_equal(frames, np.arange(len(frames)))
            or not np.allclose(times, frames/FPS, atol=2e-9, rtol=0)):
        raise ValueError(f'Video timeline is not a valid {FPS} Hz grid: {e}')
    # Rounded MP4 duration cannot extend the source recording.
    keep = times < float(info['duration_s'])
    return [(int(k), float(t)) for k, t in zip(frames[keep], times[keep])]


def align_episode(e, out, entry):
    info = json.loads((e/"episode.json").read_text(encoding="utf-8-sig"))
    source = entry["source_session"]
    split = entry["split"]
    source_id = entry["source_episode"]
    go = unitree(e, "sport_mode_state", ["position_x", "position_y", "position_z", "yaw", "body_height", "roll", "pitch"])
    go.values[:, 3] = np.unwrap(go.values[:, 3])
    arm = unitree(e, "piper_state", JOINTS)
    cmd = unitree(e, "piper_gamepad", JOINTS)
    drone = pose_series(e)
    flags = uav(e, "uav1__prometheus__state", ["connected", "armed", "odom_valid"])
    landed = uav(e, "uav1__mavros__extended_state", ["landed_state"])
    targets = uav(e, "uav1__mavros__setpoint_raw__target_local", ["position.x", "position.y", "position.z", "velocity.x", "velocity.y", "velocity.z", "acceleration_or_force.x", "acceleration_or_force.y", "acceleration_or_force.z", "yaw", "yaw_rate"])
    mappings = video_maps(e, int(info["start_desktop_ns"]))
    # Local origin fixed from first pose record, never estimated using future episode statistics.
    go_origin, uav_origin = go.values[0].copy(), drone.values[0].copy()
    rows, reason_counts = [], Counter()
    for k, t in video_timeline(e):
        reasons = []
        if t + DT > float(info['duration_s']) + 2e-9:
            reasons.append('action_interval_outside_source_duration')
        state = np.zeros(30, np.float32)
        state_valid = np.zeros(30, bool)
        action = np.zeros(14, np.float32)
        action_valid = np.zeros(14, bool)
        image_frames, image_times, mapping_rows = [], [], []
        image_valid = []
        for cam in CAMERAS:
            v = causal_video(mappings[cam], t)
            image_frames.append(v[0] if v else -1)
            image_times.append(v[1] if v else -1.)
            mapping_rows.append(v[2] if v else -1)
            image_valid.append(v is not None)
        if not all(image_valid): reasons.append("image_missing_stale_or_clock_warning")
        if all(image_valid) and max(image_times)-min(image_times)>0.1:
            reasons.append("camera_skew_gt_100ms")
        g, p, u, f, land = go.past(t, .075), arm.past(t, .075), drone.past(t,.075), flags.past(t,.15), landed.past(t,.25)
        gv, uv = causal_velocity(go,t,3,False), causal_velocity(drone,t,5,True)
        state_times = [-1.,-1.,-1.,-1.,-1.]
        if g is not None and gv is not None:
            v, i = g
            xy = rotate_xy(v[:2]-go_origin[:2],go_origin[3])
            dyaw = v[3]-go_origin[3]
            state[:10] = [*xy,v[4],v[5],v[6],np.sin(dyaw),np.cos(dyaw),*gv]
            state_valid[:10] = True
            state_times[0] = go.times[i]
            # This 14-D task policy does not contain stand-up / stand-down commands.
            # Exclude posture setup instead of silently pretending planar commands explain it.
            if v[4] < .23: reasons.append("go2_posture_setup_not_in_14d_action")
        if p is not None:
            state[10:17] = p[0]; state_valid[10:17] = True; state_times[1] = arm.times[p[1]]
        if u is not None and uv is not None and f is not None and land is not None:
            v, i = u
            xy = rotate_xy(v[:2]-uav_origin[:2],uav_origin[5]);dyaw=v[5]-uav_origin[5]
            state[17:] = [*xy,v[2],v[3],v[4],np.sin(dyaw),np.cos(dyaw),*uv,f[0][1],land[0][0]]
            state_valid[17:] = True
            state_times[2:] = [drone.times[i],flags.times[f[1]],landed.times[land[1]]]
            if not np.all(f[0] == 1):reasons.append("uav_not_connected_armed_odom_valid")
        if not state_valid.all():reasons.append("state_missing_stale_or_clock_warning")
        ga = kinematic_action(go,t,3,2,False)
        ua = kinematic_action(drone,t,5,3,True)
        ca = cmd.past(t+DT,.10)
        action_source_rows = [-1]*9
        if ga is not None:
            action[:3] = ga[0];action_valid[:3]=True;action_source_rows[:4]=ga[1]
        if ua is not None:
            action[10:] = ua[0];action_valid[10:]=True;action_source_rows[5:]=ua[1]
        if ca is not None:
            c, i=ca; r=cmd.df.iloc[i]
            if r.command_sent==1 and r.arm_enabled==1 and r.low_level_mode=="joint" and r.arm_connected==1:
                action[3:10]=c;action_valid[3:10]=True;action_source_rows[4]=i
        if not action_valid.all():reasons.append("action_target_missing_or_invalid")
        # Reject gross odometry discontinuities; do not clip corrupt labels into plausible commands.
        if np.any(np.abs(action[[0,1,10,11,12]])>3) or np.any(np.abs(action[[2,13]])>4):
            reasons.append("kinematic_target_discontinuity")
        if ca is not None and not (0<=action[9]<=.075):reasons.append("invalid_gripper_target")
        if entry["quarantined"]:reasons.append("source_episode_clock_quarantine")
        # Preserve actual FCU echoed mixed-mode target separately, respecting every ignore bit.
        ti=int(np.searchsorted(targets.times,t+DT,side="right")-1)
        raw=np.zeros(11,np.float32);raw_mask=np.zeros(11,bool);type_mask=-1;coordinate_frame=-1
        if ti>=0 and t+DT-targets.times[ti]<=.15 and not targets.warnings[ti]:
            tr=targets.df.iloc[ti];type_mask=int(tr.type_mask);coordinate_frame=int(tr.coordinate_frame)
            bits=[1,2,4,8,16,32,64,128,256,1024,2048]
            vals=targets.values[ti]
            raw_mask=np.array([not(type_mask&bit) and np.isfinite(v) for bit,v in zip(bits,vals)])
            raw=np.where(raw_mask,vals,0).astype(np.float32)
        reason_counts.update(reasons)
        rows.append(dict(source_frame_index=k,source_task_time_s=t,source_episode=source_id,source_path=entry["source_path"],source_session=source,split=split,
            action_target_time_s=t+DT, action_dt_s=DT,
            piper_command_source_time_s=float(cmd.times[ca[1]]) if ca is not None else -1.,
            state=state.tolist(),state_valid=state_valid.tolist(),action=action.tolist(),action_valid=action_valid.tolist(),
            # True for motion-derived targets; false for the seven logged arm command dimensions.
            action_is_kinematic_proxy=[True]*3+[False]*7+[True]*4,
            camera_video_frames=image_frames,camera_source_times_s=image_times,camera_mapping_rows=mapping_rows,
            camera_valid=image_valid,state_source_times_s=state_times,action_source_rows=action_source_rows,
            uav_echoed_target=raw.tolist(),uav_echoed_target_valid=raw_mask.tolist(),uav_type_mask=type_mask,
            uav_coordinate_frame=coordinate_frame,eligible=not reasons,rejection_reasons=reasons))
    pq.write_table(pa.Table.from_pylist(rows),out/"aligned"/f"{source_id}.parquet",compression="zstd")
    report=dict(source_episode=source_id,source_path=entry["source_path"],source_session=source,split=split,total_grid_frames=len(rows),
        eligible_frames=sum(r['eligible'] for r in rows),rejection_counts=dict(reason_counts),
        go2_mode_counts={str(k):int(v) for k,v in go.df['mode'].value_counts().items()},
        go2_origin=go_origin.tolist(),uav_origin=uav_origin.tolist(),
        cross_device_alignment_physically_validated=False)
    return rows, report


def segments(rows):
    runs=[];current=[]
    for r in rows:
        if r['eligible']:
            if current and (r['source_frame_index'] != current[-1]['source_frame_index'] + 1
                    or abs(r['source_task_time_s'] - current[-1]['source_task_time_s'] - DT) > 2e-9):
                runs.append(current);current=[]
            current.append(r)
        else:
            if current:runs.append(current);current=[]
    if current:runs.append(current)
    # Never stitch across gaps. Very short runs cannot support H=15.
    return [r for r in runs if len(r)>=HORIZON]


def encode_h264(img_dir, video_path, fps, **kwargs):
    video_path=Path(video_path);video_path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y","-framerate",str(fps),
        "-i",str(Path(img_dir)/"frame_%06d.png"),"-c:v","libx264","-preset","fast","-crf","18",
        "-pix_fmt","yuv420p","-g",str(FPS),"-threads","2","-movflags","+faststart",str(video_path)],check=True)


def features():
    f={"observation.state":dict(dtype="float32",shape=(30,),names=STATE_NAMES),
       "action":dict(dtype="float32",shape=(14,),names=ACTION_NAMES),
       "action_valid":dict(dtype="bool",shape=(14,),names=ACTION_NAMES),
       "source_task_time":dict(dtype="float32",shape=(1,),names=None)}
    for key in CAM_KEYS:f[key]=dict(dtype="video",shape=(480,640,3),names=["height","width","channel"])
    return f


