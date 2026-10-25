"""Repartition the verified 15 Hz release into 35 training / 3 validation sources."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VAL_SESSION = 'joint_20261018_152436_UTCp0800_f4b0'
PREFIX = 'heterovla_v2_15hz'


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def jsonlines(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))


def episode_path(root, info, index, camera=None):
    key='video_path' if camera else 'data_path'
    return root/info[key].format(episode_index=index,episode_chunk=index//info['chunks_size'],video_key=camera)


def split_for(entry):
    return 'val' if entry['source_session']==VAL_SESSION else 'train'


def build(source, out):
    source,out=source.resolve(),out.resolve()
    assert source!=out and source not in out.parents and out not in source.parents
    stage=out.parent/('.'+out.name+'-building')
    if out.exists() or stage.exists():raise FileExistsError('Output or staging already exists')
    parent_validation=read(source/'reports/validation.json')
    assert parent_validation['format_validation_passed'] and parent_validation['all_exported_videos_decoded']
    parent_sums={}
    for line in (source/'SHA256SUMS.txt').read_text().splitlines():
        digest,rel=line.split('  ',1);p=source/rel
        assert p.resolve().is_relative_to(source) and sha(p)==digest,rel
        parent_sums[rel]=digest
    parent_manifest_sha=sha(source/'SHA256SUMS.txt')
    print('PARENT_CHECKSUMS_VERIFIED',len(parent_sums),flush=True)
    stage.mkdir()
    for folder in ['aligned','provenance','reports','configs','scripts']:(stage/folder).mkdir()
    shutil.copy2(source/'scripts/dataset_15hz.py',stage/'scripts/dataset_15hz.py')
    shutil.copy2(Path(__file__),stage/'scripts/resplit_train35.py')
    write(stage/'reports/parent_validation.json',parent_validation)
    inventory=read(source/'configs/source_inventory.json')
    old_inventory=copy.deepcopy(inventory)
    for e in inventory['sources']:
        if e['include']:e['split']=split_for(e)
    inventory['source_groups']={s:sorted({e['source_session'] for e in inventory['sources'] if e['split']==s}) for s in ['train','val']}
    assert not set(inventory['source_groups']['train']) & set(inventory['source_groups']['val'])
    inventory['split_policy']='User requested 35 exportable training sources; reserve the three Oct18 f4b0-session sources as validation. All other included sources train; legacy quarantine unchanged.'
    write(stage/'configs/source_inventory.json',inventory)
    schema=read(source/'configs/schema.json')
    schema.update(version=out.name,reference_version='V2_15Hz',source_groups=inventory['source_groups'])
    assert schema['fps']==15 and schema['horizon']==15
    write(stage/'configs/schema.json',schema)
    shutil.copy2(source/'configs/task.json',stage/'configs/task.json')
    alignment=read(source/'reports/alignment_summary.json')
    entries={e['source_episode']:e for e in inventory['sources']}
    for report in alignment['sources']:report['split']=entries[report['source_episode']]['split']
    write(stage/'reports/alignment_summary.json',alignment)
    for entry in inventory['sources']:
        name=entry['source_episode']
        rows=pq.read_table(source/'aligned'/f'{name}.parquet').to_pylist()
        for row in rows:row['split']=entry['split']
        pq.write_table(pa.Table.from_pylist(rows),stage/'aligned'/f'{name}.parquet',compression='zstd')
        report=read(source/'reports'/f'{name}.json');report['split']=entry['split']
        write(stage/'reports'/f'{name}.json',report)

    old_manifest=read(source/'reports/episode_manifest.json')
    planned={s:[e for e in old_manifest if split_for(e)==s] for s in ['train','val']}
    assert {s:len({e['source_episode'] for e in es}) for s,es in planned.items()}=={'train':35,'val':3}
    infos={s:read(source/'lerobot/local'/f'{PREFIX}_{s}'/'meta/info.json') for s in ['train','val']}
    stats={s:{r['episode_index']:r['stats'] for r in map(json.loads,(source/'lerobot/local'/f'{PREFIX}_{s}'/'meta/episodes_stats.jsonl').read_text().splitlines())} for s in ['train','val']}
    manifest=[];mapping=[];train_arrays={'state':[],'actions':[]};summaries={};verified_videos=0
    for split,episodes in planned.items():
        dest=stage/'lerobot/local'/f'{PREFIX}_{split}';(dest/'meta').mkdir(parents=True)
        info=copy.deepcopy(infos['train'])
        total=sum(e['frames'] for e in episodes)
        info.update(total_episodes=len(episodes),total_frames=total,total_videos=3*len(episodes),
            total_chunks=(len(episodes)+info['chunks_size']-1)//info['chunks_size'],splits={'train':f'0:{len(episodes)}'})
        write(dest/'meta/info.json',info)
        shutil.copy2(source/'lerobot/local'/f'{PREFIX}_train'/'meta/tasks.jsonl',dest/'meta/tasks.jsonl')
        episode_meta=[];episode_stats=[];offset=0
        for index,old in enumerate(episodes):
            old_root=source/'lerobot/local'/f"{PREFIX}_{old['split']}";old_info=infos[old['split']]
            old_index=old['episode_index'];n=old['frames']
            old_table=pq.read_table(episode_path(old_root,old_info,old_index))
            table=old_table
            for key,values in [('episode_index',np.full(n,index,dtype=np.int64)),('index',np.arange(offset,offset+n,dtype=np.int64))]:
                position=table.schema.get_field_index(key)
                table=table.set_column(position,table.schema.field(position),pa.array(values,type=table.schema.field(position).type))
            path=episode_path(dest,info,index);path.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(table,path,compression='snappy')
            restored=pq.read_table(path)
            for field in old_table.column_names:
                if field not in ['episode_index','index']:assert restored[field].equals(old_table[field]),(old['source_episode'],field)
            np.testing.assert_array_equal(restored['index'].to_numpy(),np.arange(offset,offset+n))
            np.testing.assert_array_equal(restored['episode_index'].to_numpy(),np.full(n,index))
            st=copy.deepcopy(stats[old['split']][old_index])
            for key in ['episode_index','index']:
                arr=restored[key].to_numpy()
                st[key]={'min':[int(arr.min())],'max':[int(arr.max())],'mean':[float(arr.mean())],'std':[float(arr.std())],'count':[n]}
            episode_stats.append(dict(episode_index=index,stats=st))
            episode_meta.append(dict(episode_index=index,tasks=[read(source/'configs/task.json')['instruction']],length=n))
            for camera in ['front','wrist','uav']:
                key=f'observation.images.{camera}'
                a=episode_path(old_root,old_info,old_index,key);b=episode_path(dest,info,index,key)
                b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
                assert sha(b)==parent_sums[str(a.relative_to(source))]
                verified_videos+=1
            rows=pq.read_table(source/'provenance'/f"{old['split']}_{old_index:06d}.parquet").to_pylist()
            previous=copy.deepcopy(rows)
            for r in rows:r['split']=split
            pq.write_table(pa.Table.from_pylist(rows),stage/'provenance'/f'{split}_{index:06d}.parquet',compression='zstd')
            for a,b in zip(previous,rows):
                assert all(b[k]==v for k,v in a.items() if k!='split')
            record={**old,'split':split,'episode_index':index};manifest.append(record)
            mapping.append(dict(source_split=old['split'],source_episode_index=old_index,
                split=split,episode_index=index,source_episode=old['source_episode']))
            if split=='train':
                train_arrays['state'].extend(restored['observation.state'].to_pylist())
                train_arrays['actions'].extend(restored['action'].to_pylist())
            offset+=n
            if index%100==0:print('REPARTITIONED',split,index+1,len(episodes),offset,flush=True)
        jsonlines(dest/'meta/episodes.jsonl',episode_meta)
        jsonlines(dest/'meta/episodes_stats.jsonl',episode_stats)
        summaries[split]=dict(sources=len({e['source_episode'] for e in episodes}),episodes=len(episodes),frames=total,
            h15_windows=sum(e['frames']-14 for e in episodes))
    assert len(manifest)==len(old_manifest) and sum(e['frames'] for e in manifest)==31858
    write(stage/'reports/episode_manifest.json',manifest)
    write(stage/'reports/repartition_mapping.json',mapping)
    norm={}
    for key,values in train_arrays.items():
        x=np.asarray(values,np.float64)
        norm[key]={k:v.tolist() for k,v in dict(mean=x.mean(0),std=x.std(0),q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0)).items()}
    write(stage/'assets/heterovla_v2_15hz/norm_stats.json',{'norm_stats':norm})

    sys.path.insert(0,str(stage/'scripts'))
    from dataset_15hz import JointV215HzDataset,as_openpi_sample
    from torch.utils.data import DataLoader
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    for split in ['train','val']:
        ds=JointV215HzDataset(stage,split,horizon=15)
        assert len(ds)==summaries[split]['h15_windows']
        # Exercise first/last windows and every reindexed episode at its boundary.
        starts=set([0,len(ds)-1])
        lookup={v:i for i,v in enumerate(ds.starts)}
        for m in mapping:
            if m['split']==split and (m['source_split']!=split or m['source_episode_index']!=m['episode_index']):
                start=int(ds.dataset.episode_data_index['from'][m['episode_index']]);end=int(ds.dataset.episode_data_index['to'][m['episode_index']])
                starts.update([lookup[start],lookup[end-15]])
        for index in sorted(starts):
            sample=as_openpi_sample(ds[index])
            assert sample['state'].shape==(30,) and sample['actions'].shape==(15,14)
            assert all(v.shape==(480,640,3) for v in sample['image'].values())
        batch=next(iter(DataLoader(ds,batch_size=4,num_workers=0)))
        assert tuple(batch['action'].shape)==(4,15,14)
        meta=LeRobotDatasetMetadata(f'local/{PREFIX}_{split}',root=stage/'lerobot/local'/f'{PREFIX}_{split}')
        assert meta.total_frames==summaries[split]['frames'] and meta.total_episodes==summaries[split]['episodes']
        if split=='train':
            for field,key in [('observation.state','state'),('action','actions')]:
                np.testing.assert_allclose(meta.stats[field]['mean'],norm[key]['mean'],atol=1e-6,rtol=1e-6)
                np.testing.assert_allclose(meta.stats[field]['std'],norm[key]['std'],atol=1e-6,rtol=1e-6)
        summaries[split].update(real_lerobot_loader_passed=True,dataloader_batch_action_shape=[4,15,14],checked_boundary_windows=len(starts))
        print('LOADER_VERIFIED',split,summaries[split],flush=True)
    assert sha(source/'SHA256SUMS.txt')==parent_manifest_sha
    validation=dict(format_validation_passed=True,no_source_session_leakage=True,normalization_train_only=True,
        parent_release_checksums_verified=len(parent_sums),parent_manifest_sha256=parent_manifest_sha,
        state_action_and_timestamps_identical_to_parent=True,provenance_identical_except_split=True,
        video_files_byte_identical_to_parent=verified_videos,parent_all_videos_decoded=True,
        horizon_unchanged=15,fps=15,datasets=summaries,full_vla_training_run=False)
    write(stage/'reports/validation.json',validation)
    status=read(source/'STATUS.json')
    status.update(version=out.name,reference_version='V2_15Hz',splits=summaries,
        processing='split_only; all aligned state/action values and encoded video bytes preserved',
        parent_dataset=str(source),alignment_inherited_from_verified_parent=True)
    status.pop('all_sources_rebuilt_from_raw',None)
    write(stage/'STATUS.json',status)
    conversion=read(source/'reports/conversion.json')
    conversion.update(output=str(out),reference_version='V2_15Hz',operation='repartition_35_train_3_val')
    write(stage/'reports/conversion.json',conversion)
    text=['# V2_15Hz：35条训练、3条验证','',
        '按用户要求将38条有效示教划分为35条训练和3条验证。验证集仅保留2026-10-18的f4b0会话任务01、02、03；其余有效来源全部训练。原先验证集的5条来源转入训练集。采集会话无交叉。','',
        '本次仅重新划分和编号：15Hz、state30、action14保持；视频字节、状态、动作和源时间均与已验证的V2_15Hz父版本一致。训练归一化统计已重算。','',
        'action_horizon仍为15，即每次预测15步；在15Hz下对应约1秒。这是现有loader配置，不是15Hz数据或π0.5强制要求。预测长度与实际执行多少步后重新推理可以不同。若改用更长horizon，需要重新选择足够长的连续窗口，不能跨段或补造动作。','',
        '| split | 有效来源 | 连续片段 | 帧数 | H15窗口 |','|---|---:|---:|---:|---:|']
    for split,s in summaries.items():text.append(f"| {split} | {s['sources']} | {s['episodes']} | {s['frames']} | {s['h15_windows']} |")
    text+=['','## 读取','', '```python','import sys','from pathlib import Path',f'root = Path({str(out)!r})',
        "sys.path.insert(0, str(root/'scripts'))",'from dataset_15hz import JointV215HzDataset, as_openpi_sample',
        "data = JointV215HzDataset(root, split='train', horizon=15)",'sample = as_openpi_sample(data[0])','```','',
        'LeRobot v2.1目录为`lerobot/local/heterovla_v2_15hz_train`及`heterovla_v2_15hz_val`。新归一化统计位于`assets/heterovla_v2_15hz/norm_stats.json`。',
        '`reports/repartition_mapping.json`记录新旧片段映射；`reports/validation.json`记录本次一致性与loader校验；`reports/parent_validation.json`为父版本原始标签、全视频解码的校验留档。',
        '原10月14日task_03仍无有效Piper命令，仅保留审计；旧时钟隔离来源继续隔离。Go2/UAV为运动速度代理标签，跨设备物理曝光同步精度仍未独立测定。','本次没有启动模型训练。','']
    (stage/'README.md').write_text('\n'.join(text))
    sums=[]
    for p in sorted(stage.rglob('*')):
        if p.is_file() and not p.is_symlink() and not any(x.startswith('.') or x=='__pycache__' for x in p.relative_to(stage).parts):
            sums.append(f'{sha(p)}  {p.relative_to(stage)}')
    (stage/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n')
    stage.rename(out)
    print('COMPLETE',out,json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('/meta_eon_cfs/home/hph/heterovla-data/aligned/V2_15Hz'))
    ap.add_argument('--out',type=Path,default=Path('/meta_eon_cfs/home/hph/heterovla-data/aligned/V2_15Hz_train35'))
    args=ap.parse_args();build(args.source,args.out)
