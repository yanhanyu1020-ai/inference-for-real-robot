#!/usr/bin/env python3
"""Infer once from a local V3 sample; never connect to robot executors."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-root', type=Path,
                        default=Path(os.environ.get('HETEROVLA_DATA_ROOT') or root.parent / 'heterovla-data/aligned/V3'))
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    import numpy as np
    sys.path.insert(0, str(root / 'train/V3/dataset/scripts'))
    from dataset_v3 import OpenpiV3Dataset
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    dataset = OpenpiV3Dataset(args.data_root, args.split, horizon=15)
    sample = dataset[args.index]
    observation = {key: value for key, value in sample.items() if key != 'actions'}
    assert observation['state'].shape == (30,)
    assert len(observation['image']) == 3
    print(f'Loaded {args.split}[{args.index}]: state30, three RGB images; connecting to {args.host}:{args.port}', flush=True)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print('Server metadata:', json.dumps(policy.get_server_metadata(), ensure_ascii=False), flush=True)
    started = time.monotonic()
    response = policy.infer(observation)
    actions = np.asarray(response['actions'])
    if actions.shape != (15, 14) or not np.isfinite(actions).all():
        raise ValueError(f'Expected finite (15, 14) actions; received {actions.shape}')
    print(json.dumps(dict(passed=True, action_shape=list(actions.shape),
                          all_actions_finite=True, elapsed_seconds=round(time.monotonic()-started, 3),
                          robot_commands_sent=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
