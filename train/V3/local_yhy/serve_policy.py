"""Check or serve the copied V3 ViT-LoRA checkpoint using its training model."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path


def checkpoint_contract(config, checkpoint, training_norm):
    import flax.nnx as nnx
    import jax
    abstract = nnx.eval_shape(lambda: config.model.create(jax.random.key(42)))
    expected = {path: tuple(value.value.shape)
                for path, value in nnx.state(abstract, nnx.Param).flat_state().items()}
    metadata = json.loads((checkpoint / 'params/_METADATA').read_text())['tree_metadata']
    actual = {}
    for entry in metadata.values():
        path = tuple(key['key'] for key in entry['key_metadata'])
        if path and path[0] == 'params':
            path = path[1:]
        if path and path[-1] == 'value':
            path = path[:-1]
        actual[path] = tuple(entry['value_metadata']['write_shape'])
    if actual != expected:
        raise ValueError('Checkpoint keys/shapes do not match the V3 ViT-LoRA model')
    norm = checkpoint / 'assets/heterovla_v3/norm_stats.json'
    assert norm.is_file()
    digest = hashlib.sha256(norm.read_bytes()).hexdigest()
    assert hashlib.sha256(training_norm.read_bytes()).hexdigest() == '4d2329417c7ddcd93d75776f04373e21d823574b28af527f00c3afb5d037d33c'
    # Orbax rewrites JSON whitespace when saving the same numerical statistics.
    assert json.loads(norm.read_text()) == json.loads(training_norm.read_text())
    assert (checkpoint / 'params/manifest.ocdbt').is_file()
    return dict(checkpoint=str(checkpoint), parameter_leaves=len(actual),
                architecture_matches=True, normalization_sha256=digest,
                action_shape=[15, 14], check_only=True, weights_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.environ['JAX_PLATFORMS'] = 'cpu' if args.check_only else 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.check_only else args.gpu
    os.environ.setdefault('OPENPI_DATA_HOME', str(root / '.cache/openpi'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.90')
    import configuration as c
    checkpoint = args.checkpoint or (c.MODELS / 'V3/checkpoints/pi05_heterovla_v3_vitlora'
        / 'vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_data20261021_fresh2/19999')
    checkpoint = checkpoint.expanduser().resolve()
    config = c.make_config()
    result = checkpoint_contract(config, checkpoint, c.DATA / 'assets/heterovla_v3/norm_stats.json')
    print(json.dumps(result), flush=True)
    if args.check_only:
        return
    import jax
    if not all(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('Policy serving requires an allocated CUDA GPU')
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    prompt = json.loads((c.DATA / 'configs/task.json').read_text())['english_translation']
    policy = policy_config.create_trained_policy(config, checkpoint, default_prompt=prompt)
    WebsocketPolicyServer(policy, host=args.host, port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
