"""Bind the deployment, data metadata and numerical tables before CPU preflight."""
import hashlib
import json
from pathlib import Path
import configuration as c

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    files={str(p.relative_to(c.HERE)):digest(p) for p in sorted(c.HERE.rglob('*.py'))
           if '__pycache__' not in p.parts and 'training_reports' not in p.parts and not p.name.startswith('._')}
    root=c.DATA.resolve()
    paths=[root/'SHA256SUMS.txt',Path(c.MODELS/'pretrained/pi05_base/params/_METADATA')]
    paths.extend((root/'configs').glob('*.json'))
    paths.extend((root/'assets').rglob('*.json'))
    paths.extend((root/'lerobot').rglob('*.json'))
    paths.extend((root/'lerobot').rglob('*.jsonl'))
    paths.extend((root/'lerobot').rglob('*.parquet'))
    status=json.loads((root/'STATUS.json').read_text())
    assert status['fps']==15 and status['action_horizon']==15
    assert status['splits']['train']['sources']==55 and status['splits']['val']['sources']==7
    assert status['splits']['train']['windows']==34013 and status['splits']['val']['windows']==743
    norm=root/'assets/heterovla_v3/norm_stats.json'
    dataset={'train_source_count':55,'val_source_count':7,'train_windows':34013,'val_windows':743,
        'norm_stats_path':str(norm),'norm_stats_sha256':digest(norm),
        'fps':15,'action_horizon':15,'imputation_applied':False}
    paths.extend([root/'STATUS.json',root/'reports/episode_manifest.json',root/'reports/validation.json'])
    paths.extend([c.HERE/'cache_memory_cpu_result.json'])
    result={'official_commit':c.OFFICIAL_COMMIT,'files':files,'dataset':dataset,
        'dataset_root':str(root),'runtime_files':{str(p):digest(p) for p in sorted(set(paths))}}
    (c.HERE/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'source_files':len(files),'runtime_files':len(result['runtime_files']),
                      'manifest_sha256':digest(c.HERE/'manifest.json')}))

if __name__=='__main__':main()
