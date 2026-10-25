"""Replace task02 with a causal raw-camera frame and reindex the standalone release."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alignment_core import dump, segments
from process_batch import invalid_runs
from raw_checks import check_raw_alignment, episode_file
from export_training import export_datasets, norm_stats


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def jsonlines(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))


def build(root,raw,stage,rebuild):
    assert not stage.exists()
    name='2026-10-20__task_02_0403-0515'
    episode=raw/'task_02_0403-0515'
    original=pq.read_table(root/'aligned'/f'{name}.parquet').to_pylist()
    rows=copy.deepcopy(original)
    for row in rows:row['camera_raw_frames']=[-1,-1,-1]
    target=rows[457]
    assert target['source_frame_index']==457 and target['rejection_reasons']==['camera_skew_gt_100ms']
    assert all(target['state_valid']) and all(target['action_valid'])
    mapping=pd.read_csv(episode/'timing/front_raw_frames.csv')
    i=int(np.searchsorted(mapping.task_time_s,target['source_task_time_s'],side='right')-1)
    chosen=mapping.iloc[i]
    assert int(chosen.raw_frame)==912 and int(chosen.source_frame)==8445
    target['camera_raw_frames'][0]=int(chosen.raw_frame)
    target['camera_video_frames'][0]=-1
    target['camera_mapping_rows'][0]=-1
    target['camera_source_times_s'][0]=float(chosen.task_time_s)
    assert max(target['camera_source_times_s'])-min(target['camera_source_times_s'])<.1
    target['rejection_reasons']=[];target['eligible']=True
    for old,new in zip(original,rows):
        assert new['state']==old['state'] and new['action']==old['action']
    entry=next(e for e in read(root/'configs/source_inventory.json')['sources'] if e['source_episode']==name)
    numerical_check=check_raw_alignment(episode,rows,entry)
    report=read(root/'reports/sources'/f'{name}.json')
    before=copy.deepcopy(report)
    runs=segments(rows)
    assert len(runs)==1 and len(runs[0])==1076
    report.update(eligible_frames=sum(r['eligible'] for r in rows),
        rejection_counts=dict(Counter(reason for r in rows for reason in r['rejection_reasons'])),
        exported_segments=1,exported_frames=1076,longest_segment_frames=1076,h20_windows=1057,
        invalid_runs=invalid_runs(rows),internal_invalid_runs=0,internal_invalid_frames=0,
        v3_continuous_interior=True,aligned_values_independently_verified=numerical_check,
        camera_repair='frame 457 front image read directly from raw MCAP frame 912')
    for folder in ('aligned','provenance','reports'):(rebuild/folder).mkdir(parents=True,exist_ok=True)
    import export_training
    export_training.TASK=read(root/'configs/task.json')['instruction']
    if (rebuild/'reports/episode_manifest.json').is_file():
        fixed_manifest=read(rebuild/'reports/episode_manifest.json')
        assert len(fixed_manifest)==1 and fixed_manifest[0]['source_episode']==name and fixed_manifest[0]['frames']==1076
        existing=pq.read_table(rebuild/'lerobot/local/heterovla_v3_train/data/chunk-000/episode_000000.parquet').to_pydict()
        np.testing.assert_array_equal(existing['action'],[r['action'] for r in runs[0]])
        assert pq.read_table(rebuild/'provenance/train_000000.parquet').to_pylist()==runs[0]
    else:
        fixed_manifest=export_datasets(raw.parent,rebuild,[(rows,report)])
    assert len(fixed_manifest)==1
    print('TASK02_REBUILT',1076,1057,flush=True)
    stage.mkdir()
    for p in root.iterdir():
        if p.name in ('lerobot','provenance','SHA256SUMS.txt'):continue
        if p.is_dir():shutil.copytree(p,stage/p.name)
        else:shutil.copy2(p,stage/p.name)
    (stage/'provenance').mkdir()
    pq.write_table(pa.Table.from_pylist(rows),stage/'aligned'/f'{name}.parquet',compression='zstd')
    dump(stage/'reports/sources'/f'{name}.json',report)
    correction=dict(source_episode=name,grid_frame=457,grid_time_s=target['source_task_time_s'],
        camera='front',raw_frame=912,source_frame=8445,raw_timestamp_s=float(chosen.task_time_s),
        previous_source_timestamp_s=original[457]['camera_source_times_s'][0],
        previous_camera_skew_ms=1000*np.ptp(original[457]['camera_source_times_s']),
        corrected_camera_skew_ms=1000*np.ptp(target['camera_source_times_s']),
        action_and_state_values_unchanged=True,no_imputation=True,raw_recording_unchanged=True,
        raw_image_crc_and_timing_verified=True,previous_frames=1075,corrected_frames=1076,
        previous_h20_windows=1037,corrected_h20_windows=1057)
    dump(stage/'configs/camera_overrides.json',{'overrides':[correction]})
    dump(stage/'reports/task02_camera_repair.json',correction)
    (stage/'reports/history').mkdir(exist_ok=True)
    dump(stage/'reports/history/task02_before_camera_repair.json',before)
    summary=read(stage/'reports/alignment_summary.json')
    summary['sources']=[report if r['source_episode']==name else r for r in summary['sources']]
    for key in ('exported_frames','exported_segments','h20_windows'):
        summary[key]=sum(r[key] for r in summary['sources'])
    dump(stage/'reports/alignment_summary.json',summary)
    old_manifest=read(root/'reports/episode_manifest.json')
    plan=[];replaced=False
    for item in old_manifest:
        if item['source_episode']==name:
            if not replaced:plan.append((rebuild,fixed_manifest[0]));replaced=True
        else:plan.append((root,item))
    prefix='lerobot/local/heterovla_v3_train'
    info=read(root/prefix/'meta/info.json')
    new_info=copy.deepcopy(info)
    new_info.update(total_frames=10735,total_episodes=395,total_videos=1185,splits={'train':'0:395'})
    dest=stage/prefix;(dest/'meta').mkdir(parents=True)
    dump(dest/'meta/info.json',new_info)
    shutil.copy2(root/prefix/'meta/tasks.jsonl',dest/'meta/tasks.jsonl')
    task=json.loads((dest/'meta/tasks.jsonl').read_text().splitlines()[0])['task']
    infos={str(p):read(p/prefix/'meta/info.json') for p in (root,rebuild)}
    stats={str(p):{x['episode_index']:x['stats'] for x in map(json.loads,(p/prefix/'meta/episodes_stats.jsonl').read_text().splitlines())} for p in (root,rebuild)}
    checksums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'SHA256SUMS.txt').read_text().splitlines()}
    metas=[];stat_rows=[];manifest=[];index_map=[];offset=0
    for index,(parent,item) in enumerate(plan):
        old_index=item['episode_index'];n=item['frames'];old_dir=parent/prefix;old_info=infos[str(parent)]
        previous=pq.read_table(episode_file(old_dir,old_info,old_index))
        table=previous
        for key,values in [('episode_index',np.full(n,index,np.int64)),('index',np.arange(offset,offset+n,dtype=np.int64))]:
            pos=table.schema.get_field_index(key)
            table=table.set_column(pos,table.schema.field(pos),pa.array(values,type=table.schema.field(pos).type))
        path=episode_file(dest,new_info,index);path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(table,path,compression='snappy')
        for attempt in range(5):
            try:
                saved=pq.read_table(path)
                break
            except pa.ArrowInvalid:
                if attempt==4:raise
                time.sleep(.5)
        for key in previous.column_names:
            if key not in ('episode_index','index'):assert saved[key].equals(previous[key])
        st=copy.deepcopy(stats[str(parent)][old_index])
        for key in ('episode_index','index'):
            x=saved[key].to_numpy();st[key]=dict(min=[int(x.min())],max=[int(x.max())],mean=[float(x.mean())],std=[float(x.std())],count=[n])
        stat_rows.append(dict(episode_index=index,stats=st))
        metas.append(dict(episode_index=index,tasks=[task],length=n))
        for cam in ('front','wrist','uav'):
            key=f'observation.images.{cam}'
            a=episode_file(old_dir,old_info,old_index,key);b=episode_file(dest,new_info,index,key)
            b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
            expected=checksums[str(a.relative_to(root))] if parent==root else sha(a)
            assert sha(b)==expected
        shutil.copy2(parent/'provenance'/f'train_{old_index:06d}.parquet',stage/'provenance'/f'train_{index:06d}.parquet')
        manifest.append({**item,'episode_index':index,'h20_windows':max(0,n-19)})
        index_map.append(dict(source_episode=item['source_episode'],previous_episode_index=old_index if parent==root else [1,2],episode_index=index))
        offset+=n
    assert offset==10735 and len(manifest)==395
    jsonlines(dest/'meta/episodes.jsonl',metas);jsonlines(dest/'meta/episodes_stats.jsonl',stat_rows)
    dump(stage/'reports/episode_manifest.json',manifest)
    dump(stage/'reports/task02_reindex_mapping.json',index_map)
    norm_stats(stage)
    status=read(stage/'STATUS.json');status.update(published=False,format_validation_passed=False,stage='task02_repaired_pending_verification')
    dump(stage/'STATUS.json',status)
    print('REPLACEMENT_STAGED',stage,offset,len(manifest),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','raw','stage','rebuild'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.root.resolve(),a.raw.resolve(),a.stage.resolve(),a.rebuild.resolve())
