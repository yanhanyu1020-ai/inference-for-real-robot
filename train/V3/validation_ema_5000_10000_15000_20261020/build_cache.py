#!/usr/bin/env python3
"""CPU-only immutable V3 validation batches; no model/checkpoint arrays loaded.

The original data adapter and all original OpenPI input transforms are used.
NPZ keys are slash-separated paths rooted at ``observation`` plus ``actions``,
``window_indices`` and ``valid_mask``. Images are stored *after* the official
Observation.from_dict uint8-to-float32 conversion, not converted a second time.
"""
from __future__ import annotations

import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                  HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                  PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                  XLA_PYTHON_CLIENT_PREALLOCATE='false')

import argparse
import dataclasses
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback
import urllib.parse

sys.dont_write_bytecode = True
BATCH_SIZE = 16
EXPECTED_WINDOWS = 2426
HORIZON = 15
FPS = 15


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_only_cached_resource(url, *, force_download=False, **kwargs):
    """Resolve original resources without downloader mkdir/chmod/network writes."""
    if force_download:
        raise ValueError('Downloads are forbidden in the CPU cache builder')
    parsed = urllib.parse.urlparse(str(url))
    if not parsed.scheme:
        path = Path(url).expanduser().resolve()
    else:
        cache = Path(os.environ.get('OPENPI_DATA_HOME', '~/.cache/openpi')).expanduser().resolve()
        path = (cache / parsed.netloc / parsed.path.strip('/')).resolve()
        if not path.is_relative_to(cache):
            raise ValueError('Remote resource escapes the existing OpenPI cache')
    if not path.exists():
        raise FileNotFoundError(f'Required offline resource is missing: {url} -> {path}')
    return path


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def jsonable_scalar(value):
    import numpy as np
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f'Expected a scalar metadata feature, got {array.shape}')
    return array.reshape(-1)[0].item()


def flatten_arrays(value, prefix=''):
    import numpy as np
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if '/' in key:
                raise ValueError('Nested observation key contains reserved slash')
            result.update(flatten_arrays(child, f'{prefix}/{key}' if prefix else key))
    elif value is not None:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f'Object/string arrays are forbidden in cache: {prefix}')
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f'Nonfinite transformed array: {prefix}')
        result[prefix] = array
    return result


def source_contract(code_dir, config_module):
    """Verify the existing locked deployment and pin dataset numerical sources."""
    manifest_path = code_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest['files'].items():
        path = code_dir / relative
        if sha256(path) != expected:
            raise ValueError(f'Original code manifest mismatch: {path}')
    root = config_module.DATA.expanduser().resolve()
    locked_runtime = {}
    for name, expected in manifest['runtime_files'].items():
        path = Path(name)
        if sha256(path) != expected:
            raise ValueError(f'Original runtime manifest mismatch: {path}')
        locked_runtime[str(path)] = expected
    dataset_dir = root / 'lerobot/local/heterovla_v3_val'
    paths = set()
    for subdir in ('configs', 'assets'):
        paths.update(p for p in (root / subdir).rglob('*') if p.is_file())
    for subdir in ('meta', 'data'):
        paths.update(p for p in (dataset_dir / subdir).rglob('*') if p.is_file())
    for name in ('SHA256SUMS.txt', 'reports/episode_manifest.json'):
        if (root / name).is_file():
            paths.add(root / name)
    # Videos are used by the original adapter; pin them as well, without copying.
    paths.update(p for p in (dataset_dir / 'videos').rglob('*') if p.is_file())
    sources = {str(p): sha256(p) for p in sorted(paths)}
    norm_path = root / 'assets/heterovla_v3/norm_stats.json'
    if str(norm_path) not in sources:
        raise ValueError('Original V3 normalization asset not found')
    tokenizer = read_only_cached_resource('gs://big_vision/paligemma_tokenizer.model')
    return dict(schema_version=1, dataset_root=str(root), split='val',
                repo_id='local/heterovla_v3_val', windows=EXPECTED_WINDOWS,
                batch_size=BATCH_SIZE, fps=FPS, action_horizon=HORIZON,
                config_manifest_sha256=sha256(manifest_path),
                code_dir=str(code_dir), original_runtime_files=locked_runtime,
                source_files=sources, norm_sha256=sources[str(norm_path)],
                tokenizer_path=str(tokenizer), tokenizer_sha256=sha256(tokenizer),
                builder_sha256=sha256(__file__),
                observation_conversion='official Observation.from_dict before caching',
                cache_encoding='NPZ compressed, slash-separated nested keys',
                validation_transform='original transform_dataset, skip_norm_stats=False',
                imputation_policy='unchanged original dataset_v3.JointV3Dataset',
                padding_policy='repeat last row to B16; valid_mask excludes repeats')


def build_metadata(dataset, root):
    """Read exported Parquet metadata directly; never decode videos a second time."""
    import numpy as np
    import pyarrow.parquet as pq
    base = dataset.base
    directory = root / 'lerobot/local/heterovla_v3_val'
    info = json.loads((directory / 'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (directory / 'meta/episodes.jsonl').read_text().splitlines()
                if line.strip()]
    source_manifest = root / 'reports/episode_manifest.json'
    entries = {}
    if source_manifest.exists():
        raw = json.loads(source_manifest.read_text())
        if isinstance(raw, dict):
            raw = raw.get('episodes', raw.get('records', []))
        if not isinstance(raw, list):
            raise ValueError('Unrecognized episode provenance manifest format')
        entries = {int(e['episode_index']): e for e in raw if e.get('split') == 'val'}
    rows, rebuilt = [], []
    for episode in sorted(episodes, key=lambda x: int(x['episode_index'])):
        episode_index = int(episode['episode_index'])
        parquet = directory / info['data_path'].format(
            episode_chunk=episode_index // info['chunks_size'], episode_index=episode_index)
        columns = ['episode_index', 'frame_index', 'index', 'timestamp', 'source_task_time']
        table = pq.read_table(parquet, columns=columns)
        arrays = {name: np.asarray(table[name].to_pylist()).reshape(-1) for name in columns}
        n = int(episode['length'])
        if len(table) != n or not np.array_equal(arrays['frame_index'], np.arange(n)):
            raise ValueError('Episode metadata and parquet lengths/frame indices differ')
        if not np.all(arrays['episode_index'] == episode_index):
            raise ValueError('Unexpected episode id in validation parquet')
        times = arrays['source_task_time'].astype(np.float64)
        tolerance = max(1e-5, np.finfo(np.float32).eps * max(1., float(np.abs(times).max())) * 2)
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1 / FPS, atol=tolerance, rtol=0):
            raise ValueError('Source timestamps contain a gap or invalid value')
        source = {key: value for key, value in entries.get(episode_index, {}).items()
                  if key in ('source_episode', 'source_session', 'source_path', 'source_start_s', 'source_last_s')}
        for start in range(n - HORIZON + 1):
            dataset_index = int(arrays['index'][start])
            rebuilt.append(dataset_index)
            rows.append(dict(window_index=len(rows), episode_index=episode_index,
                             frame_index=int(arrays['frame_index'][start]),
                             dataset_frame_index=dataset_index,
                             timestamp=float(arrays['timestamp'][start]),
                             source_task_time=float(times[start]),
                             horizon_end_time=float(times[start + HORIZON - 1]),
                             horizon=HORIZON, fps=FPS, **source))
    if rebuilt != [int(x) for x in base.starts] or len(rows) != EXPECTED_WINDOWS:
        raise ValueError('Metadata windows do not exactly match strict original adapter starts')
    return rows


def validate_batch_file(output, record, np, schema=None):
    path = output / record['file']
    if not path.is_file() or sha256(path) != record['sha256']:
        raise ValueError(f'Cache batch hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as arrays:
        actual_schema = {key: dict(shape=list(arrays[key].shape), dtype=str(arrays[key].dtype))
                         for key in arrays.files}
        if schema is not None and actual_schema != schema:
            raise ValueError(f'Cache array schema mismatch: {path}')
        indices = arrays['window_indices']
        mask = arrays['valid_mask']
        expected = np.arange(record['window_start'], record['window_end_exclusive'])
        if not np.array_equal(indices[mask], expected) or int(mask.sum()) != record['count']:
            raise ValueError('Cache batch index coverage mismatch')
        if not np.all(indices[~mask] == -1):
            raise ValueError('Padded row index must be -1')
    return actual_schema


def run(args):
    started = time.monotonic()
    output = args.output_dir.expanduser().resolve()
    code_dir = args.code_dir.expanduser().resolve()
    if output.is_relative_to(code_dir):
        raise ValueError('Cache destination must be outside training code')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'build.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(code_dir))
    c = importlib.import_module('configuration')
    if Path(c.__file__).resolve() != code_dir / 'configuration.py':
        raise ValueError('Loaded configuration from unexpected directory')
    c.bootstrap()
    from openpi.shared import download
    download.maybe_download = read_only_cached_resource
    root = c.DATA.expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError('Cache destination must be outside immutable training data')
    contract = source_contract(code_dir, c)
    identity = canonical_digest(contract)
    # Establish ownership before writing any provenance or batch sidecar. A
    # different experiment/implementation must use a different output directory.
    ownership_path = output / 'cache_contract.json'
    if ownership_path.exists():
        ownership = json.loads(ownership_path.read_text())
        if ownership.get('contract_sha256') != identity or ownership.get('contract') != contract:
            raise ValueError('Cache directory is bound to a different source contract; refusing overwrite')
    else:
        leftovers = [p.name for p in output.iterdir() if p.name not in ('build.lock', 'build_error.json')]
        if leftovers:
            raise ValueError(f'Unowned nonempty cache directory; refusing overwrite: {leftovers}')
        atomic_json(ownership_path, dict(contract_sha256=identity, contract=contract, created_utc=utcnow()))
    import numpy as np
    import torch
    # This only constructs input transforms/tokenizer/data adapter, never a model.
    cfg = c.make_config()
    if cfg.model.action_horizon != HORIZON or cfg.model.action_dim != 32:
        raise ValueError('Unexpected original model input specification')
    c.install_data_adapter()
    from openpi.training import data_loader
    from openpi.models.model import Observation
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    dc = dataclasses.replace(dc, repo_id='local/heterovla_v3_val')
    dataset = data_loader.create_torch_dataset(dc, HORIZON, cfg.model)
    if len(dataset) != EXPECTED_WINDOWS or dataset.base.fps != FPS or dataset.base.horizon != HORIZON:
        raise ValueError('Validation dataset must be exactly 2426 complete 15Hz/H15 windows')
    transformed = data_loader.transform_dataset(dataset, dc)
    manifest_path, progress_path = output / 'manifest.json', output / 'progress.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get('contract_sha256') != identity or existing.get('complete') is not True:
            raise ValueError('Existing cache belongs to a different/incomplete contract; refusing overwrite')
        for record in existing['batches']:
            validate_batch_file(output, record, np, existing['array_schema'])
        for name, digest in existing['sidecar_files'].items():
            if sha256(output / name) != digest:
                raise ValueError('Cache metadata sidecar hash mismatch')
        print(json.dumps(dict(status='reused', complete=True, windows=EXPECTED_WINDOWS,
                              manifest=str(manifest_path), manifest_sha256=sha256(manifest_path))))
        return
    rows = build_metadata(dataset, root)
    metadata_path = output / 'windows.jsonl'
    descriptor, temporary = tempfile.mkstemp(prefix='windows.', dir=output)
    with os.fdopen(descriptor, 'w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, metadata_path)
    (output / 'batches').mkdir(exist_ok=True)
    records, schema = [], None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get('contract_sha256') != identity:
            raise ValueError('Partial cache belongs to a different contract; refusing overwrite')
        records, schema = progress.get('batches', []), progress.get('array_schema')
        for record in records:
            validate_batch_file(output, record, np, schema)
        if [r['batch_index'] for r in records] != list(range(len(records))):
            raise ValueError('Partial cache batch ledger is not contiguous')
    next_index = sum(record['count'] for record in records)
    # A finite torch DataLoader deliberately replaces the infinite/drop_last=True
    # training wrapper. All original sample transforms and collation are reused.
    subset = torch.utils.data.Subset(transformed, range(next_index, EXPECTED_WINDOWS))
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=0, drop_last=False, generator=generator,
                                       collate_fn=data_loader._collate_fn)
    for batch in loader:
        count = int(batch['actions'].shape[0])
        begin, end = next_index, next_index + count
        observation = Observation.from_dict(batch)
        arrays = flatten_arrays({'observation': observation.to_dict(), 'actions': batch['actions']})
        if arrays['actions'].shape != (count, HORIZON, 32):
            raise ValueError('Transformed action shape differs from official input')
        if arrays['observation/state'].shape != (count, 32):
            raise ValueError('Transformed state shape differs from official input')
        for key, array in list(arrays.items()):
            if array.shape[0] != count:
                raise ValueError(f'Unexpected observation batch dimension: {key}')
            if key.startswith('observation/image/'):
                if array.dtype != np.float32 or array.shape != (count, 224, 224, 3):
                    raise ValueError('Official RGB conversion/resize shape differs')
                if array.min() < -1.00001 or array.max() > 1.00001:
                    raise ValueError('Official converted image outside [-1,1]')
            if count < BATCH_SIZE:
                arrays[key] = np.concatenate([array, np.repeat(array[-1:], BATCH_SIZE - count, axis=0)], axis=0)
        arrays['window_indices'] = np.concatenate([np.arange(begin, end, dtype=np.int64),
                                                  np.full(BATCH_SIZE - count, -1, dtype=np.int64)])
        arrays['valid_mask'] = np.arange(BATCH_SIZE) < count
        actual_schema = {key: dict(shape=list(value.shape), dtype=str(value.dtype))
                         for key, value in arrays.items()}
        if schema is None:
            schema = actual_schema
        if actual_schema != schema:
            raise ValueError('Cache batch array schema changed')
        batch_index = len(records)
        relative = f'batches/batch_{batch_index:05d}.npz'
        destination = output / relative
        if destination.exists():
            # Only an interrupted write may be unrecorded. Never overwrite silently.
            with np.load(destination, allow_pickle=False) as old:
                if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                    raise ValueError('Unrecorded cache file differs; refusing overwrite')
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        record = dict(batch_index=batch_index, file=relative, sha256=sha256(destination),
                      bytes=destination.stat().st_size, count=count, padding=BATCH_SIZE - count,
                      window_start=begin, window_end_exclusive=end)
        records.append(record)
        next_index = end
        atomic_json(progress_path, dict(status='building', complete=False, updated_utc=utcnow(),
                                       contract_sha256=identity, windows_completed=next_index,
                                       total_windows=EXPECTED_WINDOWS, batches=records, array_schema=schema,
                                       gpu_used=False, model_arrays_loaded=False,
                                       elapsed_seconds=time.monotonic() - started))
        print(json.dumps(dict(status='building', batches=len(records), windows_completed=next_index,
                              total_windows=EXPECTED_WINDOWS)), flush=True)
    if next_index != EXPECTED_WINDOWS or len(records) != math.ceil(EXPECTED_WINDOWS / BATCH_SIZE):
        raise ValueError('Incomplete validation window coverage')
    indices = []
    for record in records:
        validate_batch_file(output, record, np, schema)
        indices.extend(range(record['window_start'], record['window_end_exclusive']))
    if indices != list(range(EXPECTED_WINDOWS)):
        raise ValueError('Duplicate, missing or reordered validation windows')
    if source_contract(code_dir, c) != contract:
        raise ValueError('Original code/dataset sources changed while constructing the cache')
    preflight = dict(passed=True, complete=True, windows=EXPECTED_WINDOWS, batch_size=BATCH_SIZE,
                     batches=len(records), fps=FPS, action_horizon=HORIZON,
                     indices_cover_full_validation_exactly_once=True,
                     padding_rows_excluded_by_mask=sum(r['padding'] for r in records),
                     imputation_policy_unchanged=True, normalization_unchanged=True,
                     official_observation_uint8_conversion=True, gpu_used=False,
                     model_arrays_loaded=False, checkpoint_arrays_loaded=False,
                     sources_unchanged=True, contract_sha256=identity,
                     elapsed_seconds=time.monotonic() - started, completed_utc=utcnow())
    atomic_json(output / 'cpu_preflight.json', preflight)
    result = dict(contract, contract_sha256=identity, complete=True,
                  windows=EXPECTED_WINDOWS, batches=records, array_schema=schema,
                  padding_rows=sum(r['padding'] for r in records),
                  sidecar_files={'windows.jsonl': sha256(metadata_path),
                                 'cpu_preflight.json': sha256(output / 'cpu_preflight.json')},
                  completed_utc=utcnow(), elapsed_seconds=time.monotonic() - started)
    atomic_json(manifest_path, result)
    atomic_json(progress_path, dict(status='completed', complete=True, windows_completed=EXPECTED_WINDOWS,
                                   total_windows=EXPECTED_WINDOWS, batches=records, array_schema=schema,
                                   contract_sha256=identity, manifest_sha256=sha256(manifest_path),
                                   updated_utc=utcnow(), gpu_used=False, model_arrays_loaded=False))
    print(json.dumps(dict(status='completed', complete=True, windows=EXPECTED_WINDOWS,
                          manifest=str(manifest_path), manifest_sha256=sha256(manifest_path))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        output = args.output_dir.expanduser().resolve()
        if output.is_dir():
            atomic_json(output / 'build_error.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                                        traceback=traceback.format_exc(), updated_utc=utcnow(),
                                                        gpu_used=False, model_arrays_loaded=False))
        raise


if __name__ == '__main__':
    main()
