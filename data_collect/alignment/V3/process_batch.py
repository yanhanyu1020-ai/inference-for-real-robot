"""Align a dated benchmark batch to V3 without modifying the active dataset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from alignment_core import align_episode, segments, dump
from raw_checks import check_raw_alignment


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def invalid_runs(rows):
    runs=[]
    begin=None
    for i in range(len(rows)+1):
        invalid=i<len(rows) and not rows[i]['eligible']
        if invalid and begin is None:
            begin=i
        if not invalid and begin is not None:
            selected=rows[begin:i]
            runs.append(dict(start_s=selected[0]['source_task_time_s'],end_exclusive_s=selected[-1]['source_task_time_s']+1/15,
                frames=len(selected),position='leading' if begin==0 else 'trailing' if i==len(rows) else 'internal',
                reasons=dict(Counter(reason for r in selected for reason in r['rejection_reasons']))))
            begin=None
    return runs


def stream_checks(source, duration):
    result={}
    specs=[('go2','states/unitree/sport_mode_state.csv','timing/sport_mode_state_timing.csv',.12),
        ('piper_state','states/unitree/piper_state.csv','timing/piper_state_timing.csv',.075),
        ('piper_command','states/unitree/piper_gamepad.csv','timing/piper_gamepad_timing.csv',.1),
        ('uav_pose','states/p450/uav1__mavros__local_position__pose.csv',None,.12)]
    for name,filename,timing,limit in specs:
        data=pd.read_csv(source/filename)
        time=data if timing is None else pd.read_csv(source/timing)
        t=time.task_time_s.to_numpy(float)
        assert len(t)==len(data) and np.isfinite(t).all()
        assert np.all(np.diff(t)>=0)
        if timing:
            np.testing.assert_array_equal(time['row'],np.arange(len(data)))
        gaps=np.diff(t)
        detail=dict(rows=len(data),first_s=float(t[0]),last_s=float(t[-1]),
            start_coverage_gap_s=max(0,float(t[0])),end_coverage_gap_s=max(0,float(duration-t[-1])),
            max_interval_s=float(gaps.max()),median_interval_s=float(np.median(gaps)),
            gaps_over_v3_reference_limit=int((gaps>limit).sum()),reference_limit_s=limit,
            clock_warning_rows=int(time.clock_warning.astype(bool).sum()),duplicate_timestamps=int((gaps==0).sum()),
            gap_intervals=[dict(start_s=float(t[i]),end_s=float(t[i+1]),gap_s=float(gaps[i])) for i in np.flatnonzero(gaps>limit)])
        if name=='piper_command':
            detail['command_sent_zero_rows']=int((data.command_sent!=1).sum())
            detail['arm_disabled_rows']=int((data.arm_enabled!=1).sum())
            detail['arm_disconnected_rows']=int((data.arm_connected!=1).sum())
        result[name]=detail
    return result


def align_batch(source, reference, out):
    assert not out.exists()
    for folder in ('configs','aligned','reports/sources','provenance','assets','scripts'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    entries=[]
    for task in sorted(source.glob('task_*')):
        info=read(task/'episode.json')
        entries.append(dict(source_episode=f'{source.name}__{task.name}',source_path=f'{source.name}/{task.name}',
            source_session=info['source_episode'],duration_s=info['duration_s'],task_number=info['task'],
            origin=f"upload_{source.name.replace('-', '')}",split='train',include=True,quarantined=False))
    declared=read(source/'export_manifest.json')['tasks']
    assert {e['task_number'] for e in entries}=={e['task'] for e in declared}
    assert sum(e['duration_s'] for e in entries)==sum(e['duration_s'] for e in declared)
    groups={'train':sorted({e['source_session'] for e in entries}),'val':[]}
    inventory=dict(sources=entries,raw_count=len(entries),included_count=len(entries),source_groups=groups,
        split_policy='Standalone V3-format batch, all eligible sources available as training data; no merge and no validation split.',capture_date=source.name)
    dump(out/'configs/source_inventory.json',inventory)
    schema=read(reference/'configs/schema.json')
    schema.update(version='V3',batch=source.name,source_groups=groups,no_source_session_leakage=True,
        overlapping_source_sessions=[],quarantine=[],quarantined_source_episodes=[],validation_imputation=None,
        additional_upload_count=len(entries),reference_release=str(reference),validation_horizon_at_export=20,
        dataset_ids={'train':'local/heterovla_v3_train'})
    dump(out/'configs/schema.json',schema)
    for name in ('task.json','action_chunk.json'):
        shutil.copy2(reference/'configs'/name,out/'configs'/name)
    dump(out/'configs/imputation.json',dict(method='none',splits=[],training_imputation=False))
    reports=[]
    for entry in entries:
        task=source.parent/entry['source_path']
        rows,report=align_episode(task,out,entry)
        verified=check_raw_alignment(task,rows,entry)
        runs=segments(rows)
        invalid=invalid_runs(rows)
        cmd=stream_checks(task,entry['duration_s'])
        original=read(task/'completeness.json')
        report.update(task_number=entry['task_number'],duration_s=entry['duration_s'],fps=15,action_horizon=20,
            aligned_values_independently_verified=verified,streams=cmd,
            upstream_recording_continuity_passed=original['meets_recording_continuity_checks'],
            upstream_recording_issues=original['issues'],
            exported_segments=len(runs),exported_frames=sum(map(len,runs)),
            longest_segment_frames=max(map(len,runs),default=0),h20_windows=sum(max(0,len(r)-19) for r in runs),
            excluded_short_segment_frames=sum(r['eligible'] for r in rows)-sum(map(len,runs)),
            invalid_runs=invalid,internal_invalid_runs=sum(r['position']=='internal' for r in invalid),
            internal_invalid_frames=sum(r['frames'] for r in invalid if r['position']=='internal'),
            all_actions_valid_frames=sum(all(r['action_valid']) for r in rows),
            all_cameras_valid_frames=sum(all(r['camera_valid']) for r in rows),
            invalid_action_frames_by_robot={name:sum(not all(r['action_valid'][a:b]) for r in rows)
                for name,a,b in [('go2',0,3),('piper',3,10),('uav',10,14)]})
        report['v3_continuous_interior']=report['internal_invalid_runs']==0
        dump(out/'reports/sources'/f"{entry['source_episode']}.json",report)
        reports.append(report)
        print('ALIGNED',entry['task_number'],'eligible',report['eligible_frames'],'exported',report['exported_frames'],
            'segments',len(runs),'H20',report['h20_windows'],'longest',report['longest_segment_frames'],flush=True)
    dump(out/'reports/alignment_summary.json',dict(sources=reports,source_count=len(entries),
        raw_grid_frames=sum(r['total_grid_frames'] for r in reports),exported_frames=sum(r['exported_frames'] for r in reports),
        exported_segments=sum(r['exported_segments'] for r in reports),h20_windows=sum(r['h20_windows'] for r in reports)))
    dump(out/'STATUS.json',dict(version='V3',batch=source.name,stage='aligned',published=False,
        active_v3_modified=False,training_started=False))


def export_batch(source,out):
    from export_training import export_datasets, norm_stats
    import export_training
    reports=read(out/'reports/alignment_summary.json')['sources']
    # Keep the configured instruction; the loader supplies its English translation.
    export_training.TASK=read(out/'configs/task.json')['instruction']
    aligned=[(pq.read_table(out/'aligned'/f"{r['source_episode']}.parquet").to_pylist(),r) for r in reports]
    manifest=export_datasets(source.parent,out,aligned)
    for e in manifest:
        e['h20_windows']=max(0,e['frames']-19)
    dump(out/'reports/episode_manifest.json',manifest)
    norm_stats(out)
    status=read(out/'STATUS.json');status['stage']='exported_pending_checks'
    dump(out/'STATUS.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--export-only',action='store_true')
    args=parser.parse_args()
    if args.export_only:
        export_batch(args.source.resolve(),args.out.resolve())
    else:
        align_batch(args.source.resolve(),args.reference.resolve(),args.out.resolve())
