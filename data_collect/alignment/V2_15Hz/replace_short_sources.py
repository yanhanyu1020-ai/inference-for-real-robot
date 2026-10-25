"""Swap three short sources into validation and explicitly forward-fill their missing dimensions."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare import features, encode_h264, norm_stats
from resplit_train35 import read, write, sha, jsonlines, episode_path
from verify import check_video

VALIDATION_SOURCES = {
    '2026-10-13__task_12_1333-1430',
    '2026-10-13__task_13_1509-1606',
    '2026-10-15__task_04_1557-1648',
}
PREFIX = 'heterovla_v2_15hz'


def fill_rows(original):
    """Keep the source grid and measured-value masks; fill only invalid dimensions."""
    output = []
    previous = {}
    times = {}
    for item in original:
        row = copy.deepcopy(item)
        if not output and not (all(row['state_valid']) and all(row['action_valid']) and all(row['camera_valid'])):
            continue
        assert all(row['camera_valid']), 'Unexpected internal missing camera'
        assert set(row['rejection_reasons']) <= {'state_missing_stale_or_clock_warning', 'action_target_missing_or_invalid'}
        stamp = row['source_task_time_s']
        for key in ('state', 'action'):
            valid = np.asarray(row[f'{key}_valid'], bool)
            values = np.asarray(row[key], np.float32)
            if not output:
                assert valid.all()
                previous[key] = values.copy()
                times[key] = np.full(len(values), stamp)
            previous[key][valid] = values[valid]
            times[key][valid] = stamp
            row[key] = np.where(valid, values, previous[key]).tolist()
            row[f'{key}_imputed'] = (~valid).tolist()
            row[f'{key}_value_source_grid_time_s'] = times[key].tolist()
        row['original_eligible'] = row['eligible']
        row['original_rejection_reasons'] = row['rejection_reasons']
        row['eligible'] = True
        row['rejection_reasons'] = []
        row['split'] = 'val'
        output.append(row)
    assert len(output) >= 20
    np.testing.assert_allclose(np.diff([r['source_task_time_s'] for r in output]), 1/15, atol=2e-9, rtol=0)
    return output


class Frames:
    def __init__(self, path):
        self.container = av.open(str(path))
        self.container.streams.video[0].thread_count = 2
        self.frames = self.container.decode(video=0)
        self.index = -1
        self.image = None

    def get(self, index):
        assert index >= self.index
        while self.index < index:
            frame = next(self.frames)
            self.index += 1
            if self.index == index:
                self.image = frame.to_ndarray(format='rgb24')
        return self.image


def export_validation(stage, raw, sources, task):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ft = features()
    for name, size in [('state_valid', 30), ('state_imputed', 30), ('action_imputed', 14)]:
        ft[name] = dict(dtype='bool', shape=(size,), names=None)
    ds = LeRobotDataset.create(repo_id=f'local/{PREFIX}_val', fps=15,
        root=stage/'lerobot/local'/f'{PREFIX}_val', robot_type='go2_piper_p450_joint_kinematic_v2_15hz',
        features=ft, use_videos=True, image_writer_threads=4, video_backend='pyav')
    scratch = Path(tempfile.mkdtemp(prefix='heterovla-val-fill-', dir='/tmp'))
    def image_path(self, episode_index, image_key, frame_index):
        return scratch/f'episode_{episode_index:06d}'/image_key/f'frame_{frame_index:06d}.png'
    ds._get_image_file_path = types.MethodType(image_path, ds)
    def encode_cameras(self, episode_index):
        paths = {key: str(self.root/self.meta.get_video_file_path(episode_index,key)) for key in self.meta.video_keys}
        def encode(key):
            path = Path(paths[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            encode_h264(self._get_image_file_path(episode_index,key,0).parent, path, 15)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(encode, self.meta.video_keys))
        return paths
    ds.encode_episode_videos = types.MethodType(encode_cameras, ds)
    manifest, reports = [], {}
    for index, (entry, original) in enumerate(sources):
        rows = fill_rows(original)
        cursors = [Frames(raw/entry['source_path']/'videos'/f'{name}.mp4') for name in ('front','wrist','p450')]
        for row in rows:
            frame = {'task': task, 'observation.state': np.asarray(row['state'], np.float32),
                'action': np.asarray(row['action'], np.float32),
                'source_task_time': np.array([row['source_task_time_s']], np.float32)}
            for key in ('action_valid', 'state_valid', 'action_imputed', 'state_imputed'):
                frame[key] = np.asarray(row[key], bool)
            for i, camera in enumerate(('front','wrist','uav')):
                frame[f'observation.images.{camera}'] = cursors[i].get(row['camera_video_frames'][i])
            ds.add_frame(frame)
        ds.save_episode()
        for cursor in cursors:
            cursor.container.close()
        shutil.rmtree(scratch/f'episode_{index:06d}')
        pq.write_table(pa.Table.from_pylist(rows), stage/'provenance'/f'val_{index:06d}.parquet', compression='zstd')
        detail = {'frames': len(rows), 'leading_rows_without_previous_value_excluded': len(original)-len(rows)}
        for key in ('state','action'):
            mask = np.asarray([r[f'{key}_imputed'] for r in rows])
            ages = np.asarray([r['source_task_time_s'] for r in rows])[:,None] - np.asarray([r[f'{key}_value_source_grid_time_s'] for r in rows])
            detail[f'{key}_imputed_rows'] = int(mask.any(axis=1).sum())
            detail[f'{key}_imputed_values'] = int(mask.sum())
            detail[f'{key}_max_hold_seconds'] = float(ages.max())
        reports[entry['source_episode']] = detail
        manifest.append(dict(split='val', episode_index=index, source_episode=entry['source_episode'],
            source_path=entry['source_path'], source_session=entry['source_session'],
            source_start_s=rows[0]['source_task_time_s'], source_last_s=rows[-1]['source_task_time_s'],
            frames=len(rows), h20_windows=len(rows)-19, imputation='previous_valid_value'))
        print('VALIDATION_FILLED', entry['source_episode'], json.dumps(detail), flush=True)
    ds.stop_image_writer()
    shutil.rmtree(scratch)
    return manifest, reports


def build(source, raw, out, quality_report, train_scripts):
    stage = out.parent/('.'+out.name+'-building')
    assert not out.exists() and not stage.exists()
    quality = read(quality_report)
    assert quality['passed'] and not quality['file_corruption_detected']
    before = {}
    mismatches = []
    for line in (source/'SHA256SUMS.txt').read_text().splitlines():
        expected, rel = line.split('  ', 1)
        actual = sha(source/rel)
        if actual != expected:
            # Task metadata was edited before this operation; preserve it and
            # report the mismatch instead of silently blessing data changes.
            assert rel == 'configs/task.json', f'Unexpected parent checksum mismatch: {rel}'
            mismatches.append({'path': rel, 'manifest_sha256': expected, 'current_sha256': actual})
        before[rel] = actual
    print('PARENT_FILES_CHECKED', len(before), 'metadata_changes', mismatches, flush=True)
    stage.mkdir()
    for folder in ('configs','reports','scripts','aligned','provenance'):
        (stage/folder).mkdir()
    for p in (source/'configs').glob('*.json'):
        shutil.copy2(p, stage/'configs'/p.name)
    for name in ('dataset_15hz.py','chunk_settings.py','pi05_config.py','check_action_chunks.py'):
        shutil.copy2(train_scripts/name, stage/'scripts'/name)
    shutil.copy2(Path(__file__), stage/'scripts'/Path(__file__).name)
    shutil.copy2(quality_report, stage/'reports/replacement_source_validation.json')
    write(stage/'reports/parent_validation.json', read(source/'reports/validation.json'))
    inventory = read(source/'configs/source_inventory.json')
    for entry in inventory['sources']:
        if entry['include']:
            entry['split'] = 'val' if entry['source_episode'] in VALIDATION_SOURCES else 'train'
    groups = {s: sorted({e['source_session'] for e in inventory['sources'] if e['split']==s}) for s in ('train','val')}
    overlap = sorted(set(groups['train']) & set(groups['val']))
    assert len(overlap) == 2
    inventory.update(source_groups=groups, split_policy='User requested direct swap of three short sources with prior validation; validation invalid state/action dimensions hold previous valid values.',
        no_source_session_leakage=False, overlapping_source_sessions=overlap)
    write(stage/'configs/source_inventory.json', inventory)
    write(stage/'configs/imputation.json', dict(method='previous_valid_value', splits=['val'],
        fields=['state','action'], preserve_original_validity=True, leading_invalid_rows='exclude_until_first_complete_row',
        user_requested=True, training_imputation=False))
    schema = read(stage/'configs/schema.json')
    schema.update(version=out.name, reference_version=source.name, source_groups=groups,
        validation_imputation='previous_valid_value', no_source_session_leakage=False, overlapping_source_sessions=overlap)
    write(stage/'configs/schema.json', schema)
    aligned_reports, val_sources = [], []
    for entry in inventory['sources']:
        name = entry['source_episode']
        rows = pq.read_table(source/'aligned'/f'{name}.parquet').to_pylist()
        for row in rows:
            row['split'] = entry['split']
        pq.write_table(pa.Table.from_pylist(rows), stage/'aligned'/f'{name}.parquet', compression='zstd')
        report = read(source/'reports'/f'{name}.json')
        report['split'] = entry['split']
        write(stage/'reports'/f'{name}.json', report)
        aligned_reports.append(report)
        if name in VALIDATION_SOURCES:
            val_sources.append((entry, rows))
    write(stage/'reports/alignment_summary.json', {'sources':aligned_reports,'note':'Original unfilled audit rows; validation filling is recorded in provenance.'})
    old_manifest = read(source/'reports/episode_manifest.json')
    episodes = [e for e in old_manifest if e['source_episode'] not in VALIDATION_SOURCES]
    assert len({e['source_episode'] for e in episodes}) == 35
    infos = {s:read(source/'lerobot/local'/f'{PREFIX}_{s}'/'meta/info.json') for s in ('train','val')}
    stats = {s:{r['episode_index']:r['stats'] for r in map(json.loads,(source/'lerobot/local'/f'{PREFIX}_{s}'/'meta/episodes_stats.jsonl').read_text().splitlines())} for s in ('train','val')}
    dest = stage/'lerobot/local'/f'{PREFIX}_train'
    (dest/'meta').mkdir(parents=True)
    info = copy.deepcopy(infos['train'])
    total = sum(e['frames'] for e in episodes)
    info.update(total_episodes=len(episodes), total_frames=total, total_videos=3*len(episodes),
        total_chunks=(len(episodes)+info['chunks_size']-1)//info['chunks_size'], splits={'train':f'0:{len(episodes)}'})
    write(dest/'meta/info.json', info)
    task_rows = [json.loads(line) for line in (source/'lerobot/local'/f'{PREFIX}_train'/'meta/tasks.jsonl').read_text().splitlines()]
    assert len(task_rows) == 1
    task = task_rows[0]['task']
    jsonlines(dest/'meta/tasks.jsonl', task_rows)
    manifest, mapping, metas, stat_rows = [], [], [], []
    offset = 0
    for index, old in enumerate(episodes):
        old_root = source/'lerobot/local'/f"{PREFIX}_{old['split']}"
        old_index, n = old['episode_index'], old['frames']
        old_table = pq.read_table(episode_path(old_root, infos[old['split']], old_index))
        table = old_table
        for key, values in [('episode_index', np.full(n,index,np.int64)), ('index',np.arange(offset,offset+n,dtype=np.int64))]:
            pos = table.schema.get_field_index(key)
            table = table.set_column(pos, table.schema.field(pos), pa.array(values,type=table.schema.field(pos).type))
        path = episode_path(dest, info, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression='snappy')
        saved = pq.read_table(path)
        for key in old_table.column_names:
            if key not in ('episode_index','index'):
                assert saved[key].equals(old_table[key])
        st = copy.deepcopy(stats[old['split']][old_index])
        for key in ('episode_index','index'):
            v = saved[key].to_numpy()
            st[key] = dict(min=[int(v.min())],max=[int(v.max())],mean=[float(v.mean())],std=[float(v.std())],count=[n])
        stat_rows.append(dict(episode_index=index, stats=st))
        metas.append(dict(episode_index=index, tasks=[task], length=n))
        for camera in ('front','wrist','uav'):
            key = f'observation.images.{camera}'
            a, b = episode_path(old_root, infos[old['split']], old_index,key), episode_path(dest,info,index,key)
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a,b)
            assert sha(b) == before[str(a.relative_to(source))]
        rows = pq.read_table(source/'provenance'/f"{old['split']}_{old_index:06d}.parquet").to_pylist()
        for row in rows:
            row['split'] = 'train'
        pq.write_table(pa.Table.from_pylist(rows),stage/'provenance'/f'train_{index:06d}.parquet',compression='zstd')
        manifest.append({**old,'split':'train','episode_index':index,'h20_windows':max(0,n-19)})
        mapping.append(dict(source_split=old['split'],source_episode_index=old_index,split='train',episode_index=index,source_episode=old['source_episode']))
        offset += n
        if index % 50 == 0:
            print('TRAIN_REINDEXED', index+1, len(episodes), flush=True)
    jsonlines(dest/'meta/episodes.jsonl',metas)
    jsonlines(dest/'meta/episodes_stats.jsonl',stat_rows)
    val_manifest, fill_report = export_validation(stage,raw,val_sources,task)
    manifest.extend(val_manifest)
    write(stage/'reports/episode_manifest.json',manifest)
    write(stage/'reports/repartition_mapping.json',mapping)
    write(stage/'reports/imputation.json',fill_report)
    norm_stats(stage)
    for e in val_manifest:
        for camera in ('front','wrist','uav'):
            vr = stage/'lerobot/local'/f'{PREFIX}_val'
            vi = read(vr/'meta/info.json')
            check_video(episode_path(vr,vi,e['episode_index'],f'observation.images.{camera}'),e['frames'])
    summaries = {s:dict(sources=len({e['source_episode'] for e in manifest if e['split']==s}),
        episodes=sum(e['split']==s for e in manifest),frames=sum(e['frames'] for e in manifest if e['split']==s),
        action_windows=sum(max(0,e['frames']-19) for e in manifest if e['split']==s)) for s in ('train','val')}
    write(stage/'reports/validation.json',dict(format_validation_passed=False, pending='loader and independent fill verification',
        no_source_session_leakage=False, overlapping_source_sessions=overlap, normalization_train_only=True,
        train_values_unchanged=True, train_videos_byte_identical_to_parent=3*len(episodes), val_videos_decoded=9,
        parent_metadata_checksum_differences=mismatches, datasets=summaries, full_vla_training_run=False))
    write(stage/'STATUS.json',dict(version=out.name, parent_dataset=str(source),fps=15,default_action_horizon=20,
        action_horizon=20,splits=summaries,validation_imputation='previous_valid_value',no_source_session_leakage=False,
        processing='train/val swap; validation forward fill only',validation_pending=True))
    assert sha(source/'configs/task.json') == before['configs/task.json']
    stage.rename(out)
    print('BUILT_PENDING_FINAL_CHECK',out,json.dumps(summaries),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','raw','out','quality-report','train-scripts'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    build(args.source.resolve(),args.raw.resolve(),args.out.resolve(),args.quality_report.resolve(),args.train_scripts.resolve())
