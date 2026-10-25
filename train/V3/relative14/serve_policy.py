"""Serve a trained relative14 checkpoint; return restored physical 14D commands."""
import argparse
import json
import os
from pathlib import Path


def verify_checkpoint(checkpoint):
    from . import configuration as c
    from .prepare import verify_assets
    from inference.real_robot.relative14 import contract_metadata
    verify_assets()
    asset=Path(checkpoint)/'assets'/c.ASSET_ID
    if not (asset/'contract.json').is_file():
        raise ValueError('Checkpoint has no relative14 semantic marker; old V3/base checkpoints are incompatible')
    if json.loads((asset/'contract.json').read_text())!=contract_metadata():
        raise ValueError('Checkpoint relative14 semantics do not match this implementation')
    if json.loads((asset/'norm_stats.json').read_text())!=json.loads((c.ASSET_DIR/'norm_stats.json').read_text()):
        raise ValueError('Checkpoint normalization differs from this relative14 dataset')
    if not (Path(checkpoint)/'params/manifest.ocdbt').is_file():
        raise FileNotFoundError('Checkpoint weights are missing')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8001)
    args=parser.parse_args()
    os.environ.update(JAX_PLATFORMS='cpu' if args.check_only else 'cuda',
                      CUDA_VISIBLE_DEVICES='' if args.check_only else args.gpu,
                      JAX_ENABLE_COMPILATION_CACHE='false')
    from . import configuration as c
    checkpoint=args.checkpoint.expanduser().resolve()
    verify_checkpoint(checkpoint)
    cfg=c.make_config()
    import flax.nnx as nnx
    import jax
    abstract=nnx.eval_shape(lambda:cfg.model.create(jax.random.key(42)))
    expected={path:tuple(value.value.shape) for path,value in nnx.state(abstract,nnx.Param).flat_state().items()}
    raw=json.loads((checkpoint/'params/_METADATA').read_text())['tree_metadata']
    actual={}
    for entry in raw.values():
        path=tuple(key['key'] for key in entry['key_metadata'])
        if path and path[0]=='params':path=path[1:]
        if path and path[-1]=='value':path=path[:-1]
        actual[path]=tuple(entry['value_metadata']['write_shape'])
    if actual!=expected:raise ValueError('Checkpoint architecture mismatch')
    print(json.dumps(dict(checkpoint=str(checkpoint),state_dim=14,action_shape=[15,14],
                          output_encoding=cfg.policy_metadata['returned_action_encoding'],check_only=args.check_only)))
    if args.check_only:return
    if any(d.platform!='gpu' for d in jax.devices()):raise RuntimeError('GPU required for serving')
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    prompt=json.loads((c.DATA/'configs/task.json').read_text())['english_translation']
    policy=policy_config.create_trained_policy(cfg,checkpoint,default_prompt=prompt)
    WebsocketPolicyServer(policy,host=args.host,port=args.port,metadata=policy.metadata).serve_forever()


if __name__=='__main__':main()
