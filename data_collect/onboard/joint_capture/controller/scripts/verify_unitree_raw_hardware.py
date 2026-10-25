#!/usr/bin/env python3
"""Static isolated capture acceptance; never sends Home or motion commands."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--segments',type=int,default=10)
    args=parser.parse_args(); root=args.root.resolve()
    if '.raw-capture-test-20260926' not in root.parts or not root.name.startswith('integration-'):
        parser.error('only an isolated integration root is allowed')
    if args.seconds<=0 or args.segments<0: parser.error('invalid duration/segments')
    production=Path('/home/unitree/heterovla-collection')
    if (production/'run/active_episode').exists(): raise RuntimeError('production capture active')
    sys.path.insert(0,str(root/'onboard'))
    from collection_manager import Collection
    from session_support import owned_process,stop_owned
    from mcap_storage import durable_json
    from episode_io import inspect_episode
    collection=Collection(root/'config/collection.json')
    report={'episodes':[],'errors':[],'motion_commands_sent':False}
    def identities():
        return {name:owned_process(collection.session/(name+'.pid'),collection.onboard/script)['pid']
            for name,script in [('camera','camera_session.py'),('gamepad','piper_gamepad_teleop.py')]}
    try:
        print('PREPARE',flush=True)
        with collection.lock(): collection.prepare()
        initial=identities(); report['initial_pids']=initial
        print('PREPARED',json.dumps(initial),flush=True)
        for index in range(args.segments+1):
            name='static' if index==0 else 'segment_{:02d}'.format(index)
            duration=args.seconds if index==0 else 3
            with collection.lock(): collection.start(name,'static test; no Home','raw_hardware_test')
            directory=collection.active(); begin=time.monotonic(); samples=[]
            print('RECORDING',name,str(directory),flush=True)
            while time.monotonic()-begin<duration:
                health=collection.camera_status()
                samples.append({'at':time.monotonic(),'health':health})
                if health.get('fault') or not health.get('running'):
                    raise RuntimeError('capture fault: '+json.dumps(health))
                if collection.gamepad_snapshot().get('arm_enabled'):
                    raise RuntimeError('arm enabled externally; stopping test recording')
                time.sleep(1)
            stop_at=time.monotonic()
            with collection.lock(): collection.stop()
            stop_s=time.monotonic()-stop_at
            summary=json.loads((directory/'summary.json').read_text())
            validation=inspect_episode(directory)
            item={'episode':str(directory),'stop_s':stop_s,'summary':summary,
                  'integrity':validation,'pids':identities()}
            durable_json(directory/'hardware_samples.json',samples)
            report['episodes'].append(item)
            durable_json(root/'hardware_verification.json',report)
            print('STOPPED',name,'seconds',stop_s,'integrity',json.dumps(validation),flush=True)
            if item['pids']!=initial: raise RuntimeError('warm process PID changed')
            if not summary.get('quality_ok'): raise RuntimeError('quality failed; see saved summary')
    except Exception as exc:
        report['errors'].append(str(exc)); print('FAILED',str(exc),flush=True)
    finally:
        try:
            if collection.active():
                with collection.lock(): collection.stop()
        except Exception as exc: report['errors'].append('stop: '+str(exc))
        # Only these isolated owned warm processes; never touch production owners.
        for name,script in [('camera','camera_session.py'),('gamepad','piper_gamepad_teleop.py')]:
            try: stop_owned(collection.session/(name+'.pid'),collection.onboard/script,timeout=65)
            except Exception as exc: report['errors'].append('cleanup '+name+': '+str(exc))
        durable_json(root/'hardware_verification.json',report)
    return 1 if report['errors'] else 0


if __name__=='__main__': raise SystemExit(main())
