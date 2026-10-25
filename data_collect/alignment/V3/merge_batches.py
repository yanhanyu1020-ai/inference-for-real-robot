"""Merge verified V3 batches; both splits contain strict, unfilled source segments."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alignment_core import dump, segments
from export_training import norm_stats
from raw_checks import episode_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()


def jsonlines(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))


def read_table(path):
    for attempt in range(5):
        try:return pq.read_table(path)
        except pa.ArrowInvalid:
            if attempt==4:raise
            time.sleep(.5)


def append_copies(stage, split, plans, checksums):
    """Preserve payloads, changing only dataset indices and provenance split."""
    dest=stage/'lerobot/local'/f'heterovla_v3_{split}'
    manifest=[];mapping=[]
    if dest.exists():
        info=read(dest/'meta/info.json')
        metas=[json.loads(x) for x in (dest/'meta/episodes.jsonl').read_text().splitlines()]
        stats=[json.loads(x) for x in (dest/'meta/episodes_stats.jsonl').read_text().splitlines()]
        offset=info['total_frames']
    else:
        root,e=plans[0]
        src=root/'lerobot/local'/f"heterovla_v3_{e['split']}"
        info=read(src/'meta/info.json');metas=[];stats=[];offset=0
        (dest/'meta').mkdir(parents=True)
        jsonlines(dest/'meta/tasks.jsonl',[dict(task_index=0,task=read(stage/'configs/task.json')['instruction'])])
    tasks=[json.loads(x) for x in (dest/'meta/tasks.jsonl').read_text().splitlines()]
    assert len(tasks)==1
    cache={}
    for number,(root,e) in enumerate(plans):
        src=root/'lerobot/local'/f"heterovla_v3_{e['split']}"
        if str(src) not in cache:
            cache[str(src)]=(read(src/'meta/info.json'),{r['episode_index']:r['stats'] for r in
                map(json.loads,(src/'meta/episodes_stats.jsonl').read_text().splitlines())})
            parent_tasks=[json.loads(x) for x in (src/'meta/tasks.jsonl').read_text().splitlines()]
            assert len(parent_tasks)==1 and parent_tasks[0]['task_index']==0
            parent_task=read(root/'configs/task.json');current_task=read(stage/'configs/task.json')
            assert parent_task['same_task_for_all_included_recordings'] is True
            assert parent_task['english_translation']==current_task['english_translation']
        old_info,old_stats=cache[str(src)]
        assert info['features']==old_info['features']
        index=len(metas);n=e['frames'];old_index=e['episode_index']
        previous=read_table(episode_file(src,old_info,old_index));table=previous
        for key,values in [('episode_index',np.full(n,index,np.int64)),('index',np.arange(offset,offset+n,dtype=np.int64))]:
            pos=table.schema.get_field_index(key)
            table=table.set_column(pos,table.schema.field(pos),pa.array(values,type=table.schema.field(pos).type))
        target=episode_file(dest,info,index);target.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(table,target,compression='snappy');saved=read_table(target)
        for key in previous.column_names:
            if key not in ('index','episode_index'):assert saved[key].equals(previous[key]),(e,key)
        stat=copy.deepcopy(old_stats[old_index])
        for key in ('index','episode_index'):
            values=saved[key].to_numpy()
            stat[key]=dict(min=[int(values.min())],max=[int(values.max())],mean=[float(values.mean())],std=[float(values.std())],count=[n])
        for camera in ('front','wrist','uav'):
            key=f'observation.images.{camera}'
            a=episode_file(src,old_info,old_index,key);b=episode_file(dest,info,index,key)
            b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
            assert sha(b)==checksums[str(root)][a.relative_to(root).as_posix()]
        provenance=read_table(root/'provenance'/f"{e['split']}_{old_index:06d}.parquet").to_pylist()
        for row in provenance:
            row['split']=split
            assert row['eligible'] and all(row['state_valid']) and all(row['action_valid'])
            assert 'state_imputed' not in row and 'action_imputed' not in row
        pq.write_table(pa.Table.from_pylist(provenance),stage/'provenance'/f'{split}_{index:06d}.parquet',compression='zstd')
        metas.append(dict(episode_index=index,tasks=[tasks[0]['task']],length=n))
        stats.append(dict(episode_index=index,stats=stat))
        manifest.append({**e,'split':split,'episode_index':index,'valid_h15_windows':max(0,n-14),'h20_windows':max(0,n-19)})
        mapping.append(dict(parent_root=str(root),parent_split=e['split'],parent_episode_index=old_index,
            source_episode=e['source_episode'],split=split,episode_index=index,
            values_preserved_except_indices=True,video_bytes_preserved=True))
        offset+=n
        if number%50==0:print('COPIED',split,number+1,len(plans),flush=True)
    info.update(total_episodes=len(metas),total_frames=offset,total_videos=3*len(metas),
        total_chunks=(len(metas)+info['chunks_size']-1)//info['chunks_size'],splits={'train':f'0:{len(metas)}'})
    dump(dest/'meta/info.json',info);jsonlines(dest/'meta/episodes.jsonl',metas);jsonlines(dest/'meta/episodes_stats.jsonl',stats)
    return manifest,mapping


def build(main,batches,raw,stage,train_scripts,horizon):
    assert not stage.exists()
    roots=[main,*batches];checksums={};parents=[]
    for root in roots:
        checksum={rel:digest for digest,rel in (line.split('  ',1) for line in (root/'SHA256SUMS.txt').read_text().splitlines())}
        for rel,digest in checksum.items():assert sha(root/rel)==digest,(root,rel)
        checksums[str(root)]=checksum
        parents.append(dict(root=str(root),manifest_sha256=sha(root/'SHA256SUMS.txt'),files_verified=len(checksum)))
        print('PARENT_VERIFIED',root,len(checksum),flush=True)
    for folder in ('configs','aligned','provenance','reports/sources','reports/history/merge_parents','scripts'):
        (stage/folder).mkdir(parents=True,exist_ok=True)
    for p in (main/'configs').iterdir():shutil.copy2(p,stage/'configs'/p.name)
    inventory=read(main/'configs/source_inventory.json');entries=inventory['sources'];seen={e['source_episode'] for e in entries}
    entry_roots={e['source_episode']:main for e in entries};new_val=set()
    for root in batches:
        for item in read(root/'configs/source_inventory.json')['sources']:
            e=copy.deepcopy(item);name=e['source_episode'];assert name not in seen
            if e['source_path'].startswith('2026-10-20/') and e['task_number'] in (11,13,14,15):
                e['split']='val';new_val.add(name)
            else:e['split']='train'
            seen.add(name);entries.append(e);entry_roots[name]=root
    assert len(new_val)==4
    entry_map={e['source_episode']:e for e in entries}
    reports=[]
    for root in roots:
        history=stage/'reports/history/merge_parents'/root.name
        shutil.copytree(root/'reports',history/'reports')
        for name in ('STATUS.json','README.md','SHA256SUMS.txt'):shutil.copy2(root/name,history/name)
    for e in entries:
        root=entry_roots[e['source_episode']];name=e['source_episode']
        audit=read_table(root/'aligned'/f'{name}.parquet').to_pylist()
        for row in audit:row['split']=e['split']
        pq.write_table(pa.Table.from_pylist(audit),stage/'aligned'/f'{name}.parquet',compression='zstd')
        report=read(root/'reports/sources'/f'{name}.json');runs=segments(audit)
        report.update(split=e['split'],exported_segments=len(runs),exported_frames=sum(map(len,runs)),
            h15_windows=sum(max(0,len(r)-14) for r in runs),h20_windows=sum(max(0,len(r)-19) for r in runs),
            action_horizon=horizon,imputation_applied=False)
        dump(stage/'reports/sources'/f'{name}.json',report);reports.append(report)
    old_manifest=read(main/'reports/episode_manifest.json')
    original_vals={e['source_episode'] for e in old_manifest if e['split']=='val'}
    train_plans=[(main,e) for e in old_manifest if e['split']=='train']
    val_plans=[]
    for root in batches:
        for e in read(root/'reports/episode_manifest.json'):
            (val_plans if entry_map[e['source_episode']]['split']=='val' else train_plans).append((root,e))
    train_manifest,mapping=append_copies(stage,'train',train_plans,checksums)
    # The old validation videos contain held values; rebuild only their strict runs.
    import export_training
    export_training.TASK=read(stage/'configs/task.json')['instruction']
    rebuild=[]
    for report in reports:
        if report['source_episode'] in original_vals:
            rebuild.append((read_table(stage/'aligned'/f"{report['source_episode']}.parquet").to_pylist(),report))
    val_manifest=export_training.export_datasets(raw,stage,rebuild,splits=('val',))
    extra_val,extra_mapping=append_copies(stage,'val',val_plans,checksums)
    manifest=train_manifest+val_manifest+extra_val;mapping+=extra_mapping
    for e in manifest:
        e.update(valid_h15_windows=max(0,e['frames']-14),h20_windows=max(0,e['frames']-19),
            action_windows=max(0,e['frames']-horizon+1))
        assert 'imputation' not in e
    groups={s:sorted({e['source_session'] for e in manifest if e['split']==s}) for s in ('train','val')}
    overlap=sorted(set(groups['train']) & set(groups['val']))
    counts={s:dict(sources=len({e['source_episode'] for e in manifest if e['split']==s}),
        episodes=sum(e['split']==s for e in manifest),frames=sum(e['frames'] for e in manifest if e['split']==s),
        action_windows=sum(e['action_windows'] for e in manifest if e['split']==s)) for s in ('train','val')}
    assert counts['train']['sources']==55 and counts['val']['sources']==7
    assert not {e['source_episode'] for e in manifest if e['split']=='train'} & {e['source_episode'] for e in manifest if e['split']=='val'}
    inventory.update(raw_count=len(entries),included_count=sum(e['include'] for e in entries),source_groups=groups,
        split_policy='Preserve old source membership; add Oct20 tasks 11,13,14,15 to val and other Oct20/21 tasks to train. Both splits strict; no previous-value filling.',
        no_source_session_leakage=not overlap,overlapping_source_sessions=overlap,merged_upload_dates=['2026-10-20','2026-10-21'],
        current_exported_sources={s:c['sources'] for s,c in counts.items()})
    for key in ('previous_included_count','new_count','additional_upload_count','additional_upload_date','additional_capture_date','earlier_requested_source_count'):inventory.pop(key,None)
    dump(stage/'configs/source_inventory.json',inventory)
    dump(stage/'configs/imputation.json',dict(method='none',splits=[],training_imputation=False,validation_imputation=False,
        original_validity_preserved=True,invalid_rows='excluded; split into contiguous valid segments',
        source='User explicitly requested no previous-valid-value replacement in validation.'))
    chunk=read(stage/'configs/action_chunk.json');chunk.update(action_horizon=horizon,source='User selected action chunk 15 for the Oct20/21 merge.')
    dump(stage/'configs/action_chunk.json',chunk)
    schema=read(stage/'configs/schema.json');schema.update(source_groups=groups,validation_imputation=None,
        no_source_session_leakage=not overlap,overlapping_source_sessions=overlap,
        default_action_horizon=horizon,prediction_duration_s=horizon/15,validation_horizon_at_export=horizon,
        merged_upload_dates=['2026-10-20','2026-10-21'],additional_upload_count=24,
        rebuilt_all_sources_from_raw=False,verified_against_original_aligned_tables=True)
    dump(stage/'configs/schema.json',schema)
    dump(stage/'reports/episode_manifest.json',manifest)
    dump(stage/'reports/alignment_summary.json',dict(sources=reports,source_count=len(entries),exported_source_count=62,
        note='Both train and validation contain only original valid consecutive rows; historical imputation removed.'))
    dump(stage/'reports/merge_mapping.json',mapping)
    dump(stage/'reports/imputation.json',dict(method='none',splits=[],historical_validation_rebuilt_from_original_audit=sorted(original_vals)))
    dump(stage/'reports/merge.json',dict(parents=parents,new_validation_sources=sorted(new_val),
        task_metadata_unified_to=read(stage/'configs/task.json')['instruction'],english_prompt_unchanged=True,
        validation_sources_rebuilt_without_filling=sorted(original_vals),splits=counts,action_horizon=horizon,
        copied_episodes=len(mapping),copied_videos=3*len(mapping),all_copied_values_verified=True,
        all_copied_video_hashes_verified=True,overlapping_source_sessions=overlap,
        built_at_utc=datetime.now(timezone.utc).isoformat()))
    norm_stats(stage)
    for p in Path(__file__).parent.glob('*.py'):
        if not p.name.startswith('test_'):shutil.copy2(p,stage/'scripts'/p.name)
    for p in train_scripts.glob('*.py'):
        if not p.name.startswith('test_'):shutil.copy2(p,stage/'scripts'/p.name)
    dump(stage/'STATUS.json',dict(version='V3',stage='merged_pending_validation',published=False,validation_pending=True,
        fps=15,state_dim=30,action_dim=14,action_horizon=horizon,default_action_horizon=horizon,splits=counts,
        validation_imputation=None,no_source_session_leakage=not overlap,training_started=False))
    print('MERGED_PENDING_VALIDATION',json.dumps(counts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('main','raw','stage','train-scripts'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batches',type=Path,nargs='+',required=True)
    parser.add_argument('--horizon',type=int,default=15)
    args=parser.parse_args()
    build(args.main.resolve(),[p.resolve() for p in args.batches],args.raw.resolve(),args.stage.resolve(),args.train_scripts.resolve(),args.horizon)
