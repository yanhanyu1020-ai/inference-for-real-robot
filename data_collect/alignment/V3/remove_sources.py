"""Stage removal of explicitly selected recordings, preserving retained samples and videos."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alignment_core import dump
from raw_checks import episode_file
from export_training import norm_stats
from repair_task02 import read,sha,jsonlines


def build(root,stage,numbers):
    assert not stage.exists() and stage.parent==root.parent
    inventory=read(root/'configs/source_inventory.json')
    deleted=[e for e in inventory['sources'] if e['task_number'] in numbers]
    assert {e['task_number'] for e in deleted}==set(numbers)
    names={e['source_episode'] for e in deleted}
    folders={Path(e['source_path']).name for e in deleted}
    inventory['sources']=[e for e in inventory['sources'] if e['source_episode'] not in names]
    old_manifest=read(root/'reports/episode_manifest.json')
    keep=[e for e in old_manifest if e['source_episode'] not in names]
    removed=[e for e in old_manifest if e['source_episode'] in names]
    stage.mkdir()
    for p in root.iterdir():
        if p.name in ('lerobot','provenance','SHA256SUMS.txt'):continue
        if p.is_dir():shutil.copytree(p,stage/p.name)
        else:shutil.copy2(p,stage/p.name)
    (stage/'provenance').mkdir()
    for name in names:
        (stage/'aligned'/f'{name}.parquet').unlink()
        (stage/'reports/sources'/f'{name}.json').unlink()
    for p in (stage/'reports').glob('tasks_06_07_08_continuity*'):p.unlink()
    log=stage/'reports/history/export_before_task02_repair.log'
    if log.exists():log.unlink()
    for key in ('raw_count','included_count'):inventory[key]=len(inventory['sources'])
    inventory['source_groups']={'train':sorted({e['source_session'] for e in inventory['sources']}),'val':[]}
    inventory['split_policy']='Standalone batch after user-requested deletion; remaining sources are train candidates.'
    dump(stage/'configs/source_inventory.json',inventory)
    schema=read(stage/'configs/schema.json')
    schema.update(additional_upload_count=len(inventory['sources']),source_groups=inventory['source_groups'])
    dump(stage/'configs/schema.json',schema)
    summary=read(stage/'reports/alignment_summary.json')
    summary['sources']=[r for r in summary['sources'] if r['source_episode'] not in names]
    summary.update(source_count=len(summary['sources']),raw_grid_frames=sum(r['total_grid_frames'] for r in summary['sources']))
    for key in ('exported_frames','exported_segments','h20_windows'):summary[key]=sum(r[key] for r in summary['sources'])
    dump(stage/'reports/alignment_summary.json',summary)
    upload=read(stage/'reports/upload_integrity.json')
    upload['tasks']={k:v for k,v in upload['tasks'].items() if k not in folders}
    upload['file_results']=[r for r in upload['file_results'] if r['path'].split('/')[0] not in folders]
    upload.update(manifest_files=len(upload['file_results']),manifest_bytes=sum(r['actual_bytes'] for r in upload['file_results']),
        raw_manifest_refresh_pending=True)
    dump(stage/'reports/upload_integrity.json',upload)
    prefix='lerobot/local/heterovla_v3_train';directory=root/prefix;dest=stage/prefix
    (dest/'meta').mkdir(parents=True)
    info=read(directory/'meta/info.json');new_info=copy.deepcopy(info)
    frames=sum(e['frames'] for e in keep)
    new_info.update(total_episodes=len(keep),total_frames=frames,total_videos=3*len(keep),
        total_chunks=(len(keep)+info['chunks_size']-1)//info['chunks_size'],splits={'train':f'0:{len(keep)}'})
    dump(dest/'meta/info.json',new_info)
    shutil.copy2(directory/'meta/tasks.jsonl',dest/'meta/tasks.jsonl')
    task=json.loads((dest/'meta/tasks.jsonl').read_text().splitlines()[0])['task']
    stats={r['episode_index']:r['stats'] for r in map(json.loads,(directory/'meta/episodes_stats.jsonl').read_text().splitlines())}
    checksums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'SHA256SUMS.txt').read_text().splitlines()}
    metas=[];stat_rows=[];manifest=[];mapping=[];offset=0
    for index,e in enumerate(keep):
        previous=pq.read_table(episode_file(directory,info,e['episode_index']))
        n=e['frames'];table=previous
        for key,values in [('episode_index',np.full(n,index,np.int64)),('index',np.arange(offset,offset+n,dtype=np.int64))]:
            pos=table.schema.get_field_index(key)
            table=table.set_column(pos,table.schema.field(pos),pa.array(values,type=table.schema.field(pos).type))
        p=episode_file(dest,new_info,index);p.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(table,p,compression='snappy')
        for attempt in range(5):
            try:saved=pq.read_table(p);break
            except pa.ArrowInvalid:
                if attempt==4:raise
                time.sleep(.5)
        for key in previous.column_names:
            if key not in ('index','episode_index'):assert saved[key].equals(previous[key])
        st=copy.deepcopy(stats[e['episode_index']])
        for key in ('episode_index','index'):
            x=saved[key].to_numpy();st[key]=dict(min=[int(x.min())],max=[int(x.max())],mean=[float(x.mean())],std=[float(x.std())],count=[n])
        stat_rows.append(dict(episode_index=index,stats=st));metas.append(dict(episode_index=index,tasks=[task],length=n))
        for camera in ('front','wrist','uav'):
            key=f'observation.images.{camera}'
            a=episode_file(directory,info,e['episode_index'],key);b=episode_file(dest,new_info,index,key)
            b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
            assert sha(b)==checksums[str(a.relative_to(root))]
        shutil.copy2(root/'provenance'/f"train_{e['episode_index']:06d}.parquet",stage/'provenance'/f'train_{index:06d}.parquet')
        manifest.append({**e,'episode_index':index})
        mapping.append(dict(source_episode=e['source_episode'],old_episode_index=e['episode_index'],episode_index=index))
        offset+=n
    jsonlines(dest/'meta/episodes.jsonl',metas);jsonlines(dest/'meta/episodes_stats.jsonl',stat_rows)
    dump(stage/'reports/episode_manifest.json',manifest)
    dump(stage/'reports/removal_index_mapping.json',mapping)
    norm_stats(stage)
    old_mapping=stage/'reports/task02_reindex_mapping.json'
    if old_mapping.exists():dump(old_mapping,[r for r in read(old_mapping) if r['source_episode'] not in names])
    record=dict(requested_task_numbers=sorted(numbers),removed_source_ids=sorted(names),raw_folders=sorted(folders),
        removed_episodes=len(removed),removed_frames=sum(e['frames'] for e in removed),
        retained_sources=len(inventory['sources']),retained_frames=frames,retained_episodes=len(keep),
        retained_h20_windows=sum(max(0,e['frames']-19) for e in keep),
        retained_samples_identical_except_indices=True,retained_video_bytes_identical=True,
        staged_at_utc=datetime.now(timezone.utc).isoformat(),raw_deletion_pending=True)
    dump(stage/'reports/source_removal.json',record)
    status=read(stage/'STATUS.json')
    status.update(source_count=len(inventory['sources']),published=False,format_validation_passed=False,stage='source_removal_pending_validation')
    dump(stage/'STATUS.json',status)
    print('REMOVAL_STAGED',json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--tasks',type=int,nargs='+',required=True)
    a=p.parse_args();build(a.root.resolve(),a.stage.resolve(),a.tasks)
