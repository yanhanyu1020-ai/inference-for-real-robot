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
    files={str(p.relative_to(c.HERE)):digest(p) for p in sorted(c.HERE.rglob('*.py')) if '__pycache__' not in p.parts}
    root=c.DATA.resolve()
    paths=[root/'SHA256SUMS.txt',Path(c.MODELS/'pretrained/pi05_base/params/_METADATA')]
    paths.extend((root/'configs').glob('*.json'))
    paths.extend((root/'assets').rglob('*.json'))
    paths.extend((root/'lerobot').rglob('*.json'))
    paths.extend((root/'lerobot').rglob('*.jsonl'))
    paths.extend((root/'lerobot').rglob('*.parquet'))
    result={'official_commit':c.OFFICIAL_COMMIT,'files':files,
        'dataset_root':str(root),'runtime_files':{str(p):digest(p) for p in sorted(set(paths))}}
    (c.HERE/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'source_files':len(files),'runtime_files':len(result['runtime_files']),
                      'manifest_sha256':digest(c.HERE/'manifest.json')}))

if __name__=='__main__':main()
