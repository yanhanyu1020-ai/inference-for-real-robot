"""Read-only real-image/state -> V3 actions interface check, including grounded robots.

This entry has no prepare/commit/execute path. It retains the real armed flag,
records readiness issues and transport delay, and does not certify control timing.
"""
import argparse
import base64
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .controller import Device, Policy, atomic_json, parallel, save_observations
from .core import SLOTS, finite


def observation(packets, prompt):
    images = {}
    for name, packet in packets.items():
        if packet['kind'] != name or packet.get('diagnostic_only') is not True:
            raise ValueError('Expected a non-executable diagnostic observation')
        for camera, encoded in packet['images'].items():
            bgr = cv2.imdecode(np.frombuffer(base64.b64decode(encoded,validate=True),np.uint8),cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape!=(480,640,3):
                raise ValueError('Invalid camera frame: '+camera)
            images[SLOTS[camera]]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if set(images)!=set(SLOTS.values()):
        raise ValueError('All three real images are required')
    state=finite([*packets['unitree']['state'],*packets['p450']['state']],(30,)).astype(np.float32)
    return dict(state=state,image=images,image_mask={key:np.bool_(True) for key in images},prompt=prompt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/real_robot'))
    parser.add_argument('--cycles',type=int,default=2)
    parser.add_argument('--validate-actions',action='store_true',
                        help='Send predicted chunks to observe-only gateways for validation, never execution')
    args=parser.parse_args()
    if args.cycles<1:parser.error('--cycles must be positive')
    config=json.loads(args.config.read_text())
    token=args.token_file.read_text().strip()
    devices={name:Device(config[name],token) for name in ('unitree','p450')}
    output=args.output/(time.strftime('%Y%m%d_%H%M%S')+'_diagnostic')
    output.mkdir(parents=True,exist_ok=False)
    statuses=parallel(devices,'/status')
    if any(s['execution_enabled'] for s in statuses.values()):
        raise RuntimeError('Restart both gateways in observe-only mode for diagnostics')
    if args.validate_actions and any('validate_actions_v1' not in s.get('capabilities',[]) for s in statuses.values()):
        raise RuntimeError('Deploy the action-validation gateway on both devices first')
    parallel(devices,'/reset',{name:{} for name in devices})
    policy=Policy(config.get('policy','ws://127.0.0.1:8000'))
    try:
        for cycle in range(args.cycles):
            start=time.monotonic()
            packets=parallel(devices,'/diagnostic')
            transport_s=time.monotonic()-start
            obs=observation(packets,config['prompt'])
            model_obs=policy.prepare_observation(obs)
            save_observations(output/'latest',packets,{'diagnostic_only':True,'cycle':cycle})
            start=time.monotonic()
            actions=finite(policy.infer(model_obs),(15,14))
            record=dict(cycle=cycle,diagnostic_only=True,executed=False,
                        synchronized_for_execution=False,observation_request_s=transport_s,
                        inference_s=time.monotonic()-start,prompt=config['prompt'],
                        state=model_obs['state'].tolist(),state_dim=len(model_obs['state']),
                        policy_schema=policy.metadata.get('schema_id','v3_state30_absolute'),
                        actions=actions.tolist(),
                        readiness_issues={name:p['readiness_issues'] for name,p in packets.items()})
            if args.validate_actions:
                record.update(validate_action_downlink(devices,actions,config,output,cycle))
            atomic_json(output/'latest/actions.json',record)
            with (output/'cycles.jsonl').open('a') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps({key:record[key] for key in ('cycle','diagnostic_only','executed',
                  'observation_request_s','inference_s','readiness_issues')}),flush=True)
    finally:
        policy.close()
        print('Diagnostic output:',output,flush=True)


def validate_action_downlink(devices, actions, config, output, cycle):
    """Verify exact action echo, then acquire fresh images following both receipts."""
    steps=config.get('execute_steps',3)
    if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=15:
        raise ValueError('execute_steps must be 1..15')
    actions=finite(actions,(15,14))
    bodies={name:dict(request_id=uuid.uuid4().hex,
                     actions=(actions[:steps,:10] if name=='unitree' else actions[:steps,10:]).tolist())
            for name in devices}
    receipts=parallel(devices,'/validate_actions',bodies)
    for name,receipt in receipts.items():
        if (receipt.get('kind')!=name or receipt.get('request_id')!=bodies[name]['request_id']
            or receipt.get('received') is not True or receipt.get('executed') is not False
            or receipt.get('executable') is not False or receipt.get('validation_only') is not True
            or 'plan_id' in receipt):
            raise RuntimeError('Invalid non-executable action receipt: '+name)
        np.testing.assert_array_equal(receipt['received_actions'],bodies[name]['actions'])
        if not np.isfinite(receipt['received_at']):
            raise RuntimeError('Invalid receipt timestamp')
    # Keep each cycle's receipts even if fetching subsequent frames fails.
    atomic_json(output/('downlink_%03d.json'%cycle),dict(executed=False,receipts=receipts))
    deadline=time.monotonic()+3
    while True:
        after=parallel(devices,'/diagnostic')
        if all(min(after[name]['image_stamps'].values())>receipts[name]['received_at'] for name in devices):
            break
        if time.monotonic()>deadline:
            raise RuntimeError('No fresh images after action-validation receipt')
        time.sleep(.05)
    save_observations(output/'latest',after,{'diagnostic_only':True,'cycle':cycle,'phase':'after_action_validation'})
    statuses=parallel(devices,'/status')
    if any(s['execution_enabled'] or s['active'] or s['execution'] is not None for s in statuses.values()):
        raise RuntimeError('Gateway execution state changed during validation')
    return dict(action_downlink_verified=True,validated_steps=steps,action_receipts=receipts,
                post_validation_image_stamps={name:p['image_stamps'] for name,p in after.items()},
                post_validation_gateway_status=statuses)


if __name__=='__main__':main()
