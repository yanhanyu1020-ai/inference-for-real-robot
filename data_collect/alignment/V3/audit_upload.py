"""Read-only upload hash and full-video verification, independent of conversion."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from raw_checks import check_video


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def audit(source, output):
    manifest=read(source/'file_manifest.json')
    def check(row):
        relative=Path(row['path'].replace('\\','/'))
        assert not relative.is_absolute() and '..' not in relative.parts
        p=source/relative
        result=dict(path=relative.as_posix(),exists=p.is_file(),expected_bytes=row['bytes'],expected_sha256=row['sha256'])
        if p.is_file():
            result.update(actual_bytes=p.stat().st_size,actual_sha256=sha(p))
        result['passed']=result['exists'] and result['actual_bytes']==row['bytes'] and result['actual_sha256']==row['sha256']
        return result
    files=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(check,row) for row in manifest['files']]
        for future in as_completed(futures):
            files.append(future.result())
            if len(files)%100==0:
                print('HASH_CHECKED',len(files),len(futures),flush=True)
    files.sort(key=lambda r:r['path'])
    result=dict(source=str(source),manifest_files=len(files),manifest_bytes=manifest['bytes'],
        file_results=files,mismatches=[r for r in files if not r['passed']],tasks={})
    for task in sorted(source.glob('task_*')):
        info=read(task/'episode.json')
        listed=[r for r in files if r['path'].startswith(task.name+'/')]
        critical=[r for r in listed if '/raw/' in r['path'] or '/states/' in r['path'] or '/timing/' in r['path'] or '/videos/' in r['path'] or r['path'].endswith('/episode.json')]
        observed={str(p.relative_to(source)) for p in task.rglob('*') if p.is_file()}
        listed_paths={r['path'] for r in listed}
        missing_critical_from_manifest=sorted(p for p in observed-listed_paths if any('/'+s+'/' in p for s in ('raw','states','timing','videos')) or p.endswith('/episode.json'))
        detail=dict(task=info['task'],duration_s=info['duration_s'],source_session=info['source_episode'],
            manifest_files=len(listed),all_listed_files_match=all(r['passed'] for r in listed),
            critical_files_match=all(r['passed'] for r in critical) and not missing_critical_from_manifest,
            critical_files_not_in_manifest=missing_critical_from_manifest,
            unlisted_metadata=sorted(observed-listed_paths),videos={})
        for camera in ('front','wrist','p450'):
            try:
                detail['videos'][camera]={'passed':True,**check_video(task/'videos'/f'{camera}.mp4',int(info['video_frames']))}
            except Exception as exc:
                detail['videos'][camera]={'passed':False,'error':repr(exc)}
        detail['file_integrity_passed']=detail['critical_files_match'] and all(v['passed'] for v in detail['videos'].values())
        result['tasks'][task.name]=detail
        print('RAW_VIDEO_VERIFIED',task.name,detail['file_integrity_passed'],flush=True)
    result['all_task_artifacts_passed']=all(t['file_integrity_passed'] for t in result['tasks'].values())
    result['all_manifest_files_passed']=not result['mismatches']
    result['files_missing_from_manifest']=sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and str(p.relative_to(source)) not in {r['path'] for r in files} and p.name!='file_manifest.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('UPLOAD_AUDIT_COMPLETE',len(result['tasks']),'mismatches',len(result['mismatches']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.source.resolve(),args.output.resolve())
