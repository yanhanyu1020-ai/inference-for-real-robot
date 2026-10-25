"""Recover verified complete camera chunks into a NEW episode directory."""
import argparse
import json
import os
from pathlib import Path
import shutil

from episode_io import format_version, iter_mcap_frames, load_object
from mcap_storage import CameraStore,durable_json,sync_directory
from raw_image import image_schema


def recover_episode(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or source in destination.parents or destination in source.parents:
        raise ValueError('recovery destination must be separate from source')
    if destination.exists(): raise FileExistsError(str(destination))
    if format_version(source)!=2: raise ValueError('recovery requires v2 raw episode')
    destination.mkdir(parents=True)
    output=destination/'raw/cameras'
    output.mkdir(parents=True)
    report={'source':str(source),'destination':str(destination),'source_complete':True,
            'streams':{},'errors':[],'alignment_valid':None,'sensor_integrity_unknown':True}
    try:
        original_summary=load_object(source/'summary.json')
    except (OSError,ValueError):
        original_summary={}
    if not original_summary.get('durable_complete'):
        report['source_complete']=False
    streams={}
    for camera in ('front','wrist'):
        store=CameraStore(output,camera,'lz4')
        count=0
        error=None
        paths=sorted((source/'raw/cameras').glob(camera+'_*.mcap*'))
        try:
            for path in paths:
                if path.suffix not in ('.mcap','.active'): continue
                for metadata,image in iter_mcap_frames(path,camera):
                    original_seq=metadata['seq']
                    store.append({**metadata,'source_seq':original_seq,'source_shard':path.name,'seq':count+1},image)
                    count+=1
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)
            report['errors'].append(camera+': '+error)
            report['source_complete']=False
        streams[camera]={**store.close(),'received':count,'accepted':count,'rejected':0,'write_errors':0,'pending':0}
        declared=original_summary.get('streams',{}).get(camera,{})
        if not count or any(declared.get(key)!=count for key in ('received','accepted','written','durable')) or any(declared.get(key)!=0 for key in ('rejected','write_errors','pending')):
            report['source_complete']=False
        report['streams'][camera]={'recovered':count,'lost_upper_bound':'unknown','error':error}
    # Preserve state and provenance without pretending recovery validates training.
    for path in (source/'raw').glob('*'):
        if path.is_file() and path.suffix in ('.csv','.json'):
            copied=destination/'raw'/path.name
            shutil.copy2(path,copied)
            with copied.open('r+b') as handle:
                os.fsync(handle.fileno())
    sync_directory(destination/'raw')
    meta=load_object(source/'meta.json')
    durable_json(destination/'meta.json',{**meta,'recovered_from':str(source),'durable_complete':True})
    durable_json(output/'schema.json',image_schema())
    durable_json(destination/'summary.json',{'format_version':2,'streams':streams,
        'durable_complete':True,'camera_durable_complete':True,'quality_ok':False,'recovery':report})
    durable_json(destination/'capture_manifest.json',{'streams':streams,'recovery':report})
    durable_json(destination/'recovery_report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(recover_episode(args.episode,args.output),indent=2,ensure_ascii=False))
