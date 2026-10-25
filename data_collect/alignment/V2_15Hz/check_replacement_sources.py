"""Audit the three held-out sources before using them as training replacements."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from verify import check_raw_alignment, check_video, episode_file


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def audit(root, raw, output):
    inventory = json.loads((root / 'configs/source_inventory.json').read_text())
    manifest = json.loads((root / 'reports/episode_manifest.json').read_text())
    sums = {line.split('  ', 1)[1]: line.split('  ', 1)[0]
            for line in (root / 'SHA256SUMS.txt').read_text().splitlines()}
    result = {'dataset': str(root), 'sources': {}, 'file_corruption_detected': False}
    for entry in inventory['sources']:
        if entry['split'] != 'val':
            continue
        name = entry['source_episode']
        source = raw / entry['source_path']
        audit_path = root / 'aligned' / f'{name}.parquet'
        assert digest(audit_path) == sums[str(audit_path.relative_to(root))]
        rows = pq.read_table(audit_path).to_pylist()
        detail = check_raw_alignment(source, rows, entry)
        detail['rejection_counts'] = dict(Counter(reason for r in rows for reason in r['rejection_reasons']))
        detail['all_actions_valid_rows'] = sum(all(r['action_valid']) for r in rows)
        detail['all_cameras_valid_rows'] = sum(all(r['camera_valid']) for r in rows)
        detail['invalid_actions_by_robot'] = {
            key: sum(not all(r['action_valid'][a:b]) for r in rows)
            for key, a, b in [('go2', 0, 3), ('piper', 3, 10), ('uav', 10, 14)]}
        cmd = pd.read_csv(source / 'states/unitree/piper_gamepad.csv')
        detail['command_sent_zero_rows'] = int((cmd.command_sent == 0).sum())
        detail['raw_videos'] = {}
        for camera_index, camera in enumerate(('front', 'wrist', 'p450')):
            path = source / 'videos' / f'{camera}.mp4'
            count, last = 0, -float('inf')
            with av.open(str(path)) as container:
                container.streams.video[0].thread_count = 2
                for frame in container.decode(video=0):
                    assert (frame.width, frame.height) == (640, 480)
                    stamp = float(frame.pts * frame.time_base)
                    assert stamp > last
                    last = stamp
                    count += 1
            assert count > max(r['camera_video_frames'][camera_index] for r in rows)
            detail['raw_videos'][camera] = {'decoded_frames': count, 'sha256': digest(path)}
        exported = [e for e in manifest if e['source_episode'] == name]
        detail.update(exported_segments=len(exported), exported_frames=sum(e['frames'] for e in exported),
                      longest_segment=max(e['frames'] for e in exported),
                      h20_windows=sum(max(0, e['frames'] - 19) for e in exported),
                      h50_windows=sum(max(0, e['frames'] - 49) for e in exported))
        directory = root / 'lerobot/local/heterovla_v2_15hz_val'
        info = json.loads((directory / 'meta/info.json').read_text())
        for episode in exported:
            i, n = episode['episode_index'], episode['frames']
            p = episode_file(directory, info, i)
            assert digest(p) == sums[str(p.relative_to(root))]
            data = pq.read_table(p).to_pydict()
            provenance = pq.read_table(root / 'provenance' / f'val_{i:06d}.parquet').to_pylist()
            for field, key in [('observation.state', 'state'), ('action', 'action')]:
                values = np.asarray(data[field])
                assert np.isfinite(values).all()
                np.testing.assert_array_equal(values, [r[key] for r in provenance])
            assert np.asarray(data['action_valid']).all()
            np.testing.assert_allclose(np.diff(np.asarray(data['source_task_time']).reshape(-1)), 1/15, atol=1e-5, rtol=0)
            for camera in ('front', 'wrist', 'uav'):
                p = episode_file(directory, info, i, f'observation.images.{camera}')
                assert digest(p) == sums[str(p.relative_to(root))]
                check_video(p, n)
        detail['all_exported_tables_and_videos_passed'] = True
        result['sources'][name] = detail
        print(name, json.dumps(detail, ensure_ascii=False), flush=True)
    assert len(result['sources']) == 3
    result['passed'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.raw.resolve(), args.output.resolve())
