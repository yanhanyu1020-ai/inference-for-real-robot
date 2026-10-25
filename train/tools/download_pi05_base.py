#!/usr/bin/env python3
"""Download the public OpenPI checkpoint with resumable files and GCS MD5 checks."""
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(os.environ.get('HETEROVLA_MODEL_ROOT', Path.home() / 'heterovla-model')).expanduser() / 'pretrained'
DEST = ROOT / 'pi05_base'
PREFIX = 'checkpoints/pi05_base/'
ROOT.mkdir(parents=True, exist_ok=True)
DEST.mkdir(exist_ok=True)


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=90)


def checksum(path):
    h = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return base64.b64encode(h.digest()).decode()


items = []
token = None
while True:
    query = {'prefix': PREFIX}
    if token:
        query['pageToken'] = token
    with request('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?' + urllib.parse.urlencode(query)) as r:
        listing = json.load(r)
    items.extend(x for x in listing.get('items', []) if not x['name'].endswith('/'))
    token = listing.get('nextPageToken')
    if not token:
        break
assert items and all(x.get('md5Hash') for x in items)
(ROOT / 'pi05_base.download_manifest.json').write_text(json.dumps(items, indent=2))
print('START', len(items), 'files', sum(int(x['size']) for x in items), 'bytes', flush=True)


def download(item):
    relative = item['name'][len(PREFIX):]
    path = DEST / relative
    assert path.resolve().is_relative_to(DEST.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    size = int(item['size'])
    expected = item['md5Hash']
    if path.exists():
        if path.stat().st_size == size and checksum(path) == expected:
            print('VERIFIED_EXISTING', relative, flush=True)
            return relative
        raise RuntimeError('Existing destination differs; refusing to overwrite: ' + str(path))
    partial = path.with_name(path.name + '.part')
    url = ('https://storage.googleapis.com/openpi-assets/' + urllib.parse.quote(item['name'], safe='/')
           + '?generation=' + item['generation'])
    for attempt in range(8):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise RuntimeError('Partial file exceeds expected size')
            if offset < size:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with request(url, headers) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected Content-Range')
                    with partial.open('ab' if append else 'wb') as f:
                        while True:
                            block = response.read(4 * 1024 * 1024)
                            if not block:
                                break
                            f.write(block)
            if partial.stat().st_size != size:
                raise RuntimeError('Incomplete download')
            if checksum(partial) != expected:
                partial.rename(partial.with_name(partial.name + f'.bad-{time.time_ns()}'))
                raise RuntimeError('Checksum mismatch; preserving corrupt partial separately')
            os.replace(partial, path)
            print('VERIFIED', relative, size, flush=True)
            return relative
        except Exception as exc:
            print('RETRY', attempt + 1, relative, repr(exc), flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))


started = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(download, x) for x in items]
    while futures:
        done, pending = concurrent.futures.wait(futures, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            future.result()
        futures = list(pending)
        if not done:
            downloaded = sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
            print('PROGRESS', downloaded, 'bytes', 'remaining_files', len(futures), flush=True)

report = {'source': 'gs://openpi-assets/checkpoints/pi05_base', 'directory': str(DEST),
          'files': len(items), 'bytes': sum(int(x['size']) for x in items),
          'all_official_md5_verified': True, 'elapsed_seconds': time.time() - started}
(ROOT / 'pi05_base.download_verified.json').write_text(json.dumps(report, indent=2))
print('COMPLETE', json.dumps(report), flush=True)
