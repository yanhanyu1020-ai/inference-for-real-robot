"""Compute exact training-window norms. Never change raw data or the old V3 assets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from inference.real_robot.relative14 import HORIZON, STATE_INDICES, STATE_NAMES, chunk_delta, contract_metadata, state14_from_state30
from . import configuration as c


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temporary.replace(path)


def episode_windows(state30, actions):
    state = state14_from_state30(state30)
    n = len(state)-HORIZON+1
    if n <= 0:
        raise ValueError('Episode shorter than H15')
    chunks = np.moveaxis(np.lib.stride_tricks.sliding_window_view(actions,HORIZON,axis=0),-1,-2)
    anchors = state[:n]
    return anchors, chunk_delta(anchors,chunks)


def stats(values):
    values = values.reshape(-1,14).astype(np.float64)
    return dict(mean=values.mean(axis=0).tolist(),std=values.std(axis=0).tolist(),
                q01=np.quantile(values,.01,axis=0).tolist(),q99=np.quantile(values,.99,axis=0).tolist())


def prepare():
    root = c.DATA/'lerobot/local/heterovla_v3_train'
    info = json.loads((root/'meta/info.json').read_text())
    original_names=info['features']['observation.state']['names']
    if tuple(original_names[i] for i in STATE_INDICES)!=STATE_NAMES:
        raise ValueError('Original state layout changed; refusing an implicit remapping')
    states, actions, sources = [], [], []
    frames = 0
    for path in sorted((root/'data').rglob('*.parquet')):
        table = pq.read_table(path,columns=['observation.state','action','frame_index','episode_index','action_valid'])
        if len(set(table['episode_index'].to_pylist()))!=1 or table['frame_index'].to_pylist()!=list(range(len(table))):
            raise ValueError('Episode layout mismatch: '+str(path))
        state30 = np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
        physical = np.asarray(table['action'].to_pylist(),dtype=np.float32)
        if physical.shape!=(len(table),14) or not np.asarray(table['action_valid'].to_pylist()).all():
            raise ValueError('Invalid action labels: '+str(path))
        anchor, delta = episode_windows(state30,physical)
        states.append(anchor);actions.append(delta)
        frames += len(table)
        sources.append(dict(path=str(path.relative_to(c.DATA)),sha256=digest(path),frames=len(table),windows=len(anchor)))
    state = np.concatenate(states)
    action = np.concatenate(actions)
    if frames!=info['total_frames'] or len(sources)!=info['total_episodes']:
        raise ValueError('Incomplete training data scan')
    if state.shape!=(34013,14) or action.shape!=(34013,15,14) or frames!=45997:
        raise ValueError('Training dataset differs from the verified V3 population')
    normalization = {'norm_stats':{'state':stats(state),'actions':stats(action)}}
    norm_path = c.ASSET_DIR/'norm_stats.json'
    if norm_path.exists() and json.loads(norm_path.read_text())!=normalization:
        raise RuntimeError('Existing relative14 assets differ; use a new version, never overwrite run semantics')
    atomic_json(norm_path,normalization)
    atomic_json(c.ASSET_DIR/'contract.json',contract_metadata())
    report = dict(train_frames=frames,train_episodes=len(sources),train_windows=len(state),
                  action_statistics_rows=int(np.prod(action.shape[:-1])),
                  state_statistics_rows=len(state),split='train',state_dim=14,action_dim=14,
                  action_horizon=15,normalization_sha256=digest(norm_path),
                  schema=contract_metadata(),sources=sources,raw_data_modified=False)
    atomic_json(c.ASSET_DIR/'provenance.json',report)
    return {k:v for k,v in report.items() if k not in ('sources','schema')}


def verify_assets():
    metadata = json.loads((c.ASSET_DIR/'contract.json').read_text())
    if metadata!=contract_metadata():
        raise ValueError('Relative14 asset semantics changed')
    report = json.loads((c.ASSET_DIR/'provenance.json').read_text())
    if digest(c.ASSET_DIR/'norm_stats.json')!=report['normalization_sha256']:
        raise ValueError('Relative14 normalization changed')
    norm = json.loads((c.ASSET_DIR/'norm_stats.json').read_text())['norm_stats']
    for group in ('state','actions'):
        for field in ('mean','std','q01','q99'):
            values = np.asarray(norm[group][field])
            if values.shape!=(14,) or not np.isfinite(values).all():
                raise ValueError('Invalid relative14 norm stats')
    return report


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(prepare(),indent=2))
    print('Assets:',c.ASSET_DIR)


if __name__=='__main__':main()
