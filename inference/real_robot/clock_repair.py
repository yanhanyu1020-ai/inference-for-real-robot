"""Operator-terminal P450 clock repair; requires sudo and a connected, disarmed FCU.

Stops only this deployment's observation processes before changing wall time,
then restarts them. Supply --unix from the operator's correctly set PC clock.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .service import identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unix',type=float,required=True)
    args=parser.parse_args()
    if not 1577836800 <= args.unix <= 4102444800:
        parser.error('Expected a UTC Unix timestamp between 2020 and 2100')
    anchor=time.monotonic()
    root=Path(__file__).resolve().parent.parent
    os.chdir(root)
    import rospy
    from mavros_msgs.msg import State
    rospy.init_node('heterovla_clock_preflight',anonymous=True,disable_signals=True)
    state=rospy.wait_for_message('/uav1/mavros/state',State,timeout=5)
    if not state.connected or state.armed or not 0 <= (rospy.Time.now()-state.header.stamp).to_sec() < 1.5:
        raise RuntimeError('Clock repair requires fresh connected, disarmed FCU feedback')
    camera_path=root/'camera_process.json'
    camera=json.loads(camera_path.read_text())
    camera_identity=identity(camera['pid'])
    if camera_identity is None or camera_identity['command'][-len(camera['command'][1:]):] != camera['command'][1:]:
        raise RuntimeError('Camera process identity mismatch; inspect before clock repair')
    if not any(Path(arg).name=='roslaunch' for arg in camera_identity['command']):
        raise RuntimeError('Camera owner is not roslaunch')
    # Obtain credentials in the operator's TTY before disrupting any data stream.
    subprocess.run(['sudo','-v'],check=True)
    def run(module,*arguments):
        subprocess.run([sys.executable,'-m','real_robot.'+module,*arguments],check=True)
    run('service','stop','--kind','p450')
    run('p450_sensors','stop')
    os.kill(camera['pid'],signal.SIGINT)
    deadline=time.monotonic()+20
    while identity(camera['pid'])==camera_identity and time.monotonic()<deadline:
        time.sleep(.1)
    if identity(camera['pid'])==camera_identity:
        raise RuntimeError('Camera still stopping; wall clock has not been changed')
    before=time.time()
    clock_error=None
    try:
        target=args.unix+(time.monotonic()-anchor)
        subprocess.run(['sudo','-n','date','-u','-s','@%.6f'%target],check=True)
    except Exception as exc:
        clock_error=str(exc)
    finally:
        with (root/'camera.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(camera['command'],stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True)
        camera_path.write_text(json.dumps(dict(pid=process.pid,command=camera['command'],
                               identity=identity(process.pid)))+'\n')
        run('p450_sensors','start')
        run('service','start','--kind','p450')
    result=dict(previous_unix=before,current_unix=time.time(),clock_error=clock_error,
                execution_enabled=False,requires_timestamp_recheck=True)
    (root/'clock_repair.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if clock_error:raise RuntimeError(clock_error)


if __name__=='__main__':main()
