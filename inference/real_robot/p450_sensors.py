"""Build/start/status/stop the isolated P450 observation stack, without UAV_controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .service import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('build','start','stop','status'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    directory = Path.home()/'.config/heterovla-live'
    directory.mkdir(parents=True,exist_ok=True)
    record_path = directory/'p450_sensors.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    alive = bool(record and record['identity'] == identity(record['pid']))
    if args.operation == 'status':
        print(json.dumps(dict(running=alive, record=record)))
        return
    if args.operation == 'stop':
        if alive:
            if str(root/'ros/p450_observe.launch') not in record['identity']['command']:
                raise RuntimeError('Refusing to stop an unrelated process')
            os.kill(record['pid'],signal.SIGINT)
            deadline = time.monotonic()+20
            while record['identity'] == identity(record['pid']) and time.monotonic()<deadline:
                time.sleep(.1)
            if record['identity'] == identity(record['pid']):
                raise RuntimeError('ROS stack is still stopping; inspect p450_sensors.log')
        print(json.dumps(dict(stopped=True)))
        return
    if alive:
        raise RuntimeError('Observation stack is already running')
    if args.operation == 'build':
        subprocess.run(['cmake','-S',str(root/'ros'),'-B',str(root/'ros/build'),
                        '-DPYTHON_EXECUTABLE=/usr/bin/python3'],check=True)
        subprocess.run(['cmake','--build',str(root/'ros/build'),'-j2'],check=True)
        return
    import rospy
    rospy.init_node('heterovla_sensors_preflight',anonymous=True,disable_signals=True)
    code,_,state=rospy.get_master().getSystemState()
    if code!=1:
        raise RuntimeError('ROS master unavailable')
    for topic,nodes in state[0]:
        if nodes and (topic.startswith('/uav1/mavros/') or topic in
                      ('/Odometry','/livox/lidar','/uav1/prometheus/state')):
            raise RuntimeError('Existing sensor stack owns '+topic)
    if not (root/'ros/build/p450_estimator').is_file():
        raise RuntimeError('Build the estimator wrapper first')
    environment=dict(os.environ)
    environment['ROS_PACKAGE_PATH']=str(root/'ros')+os.pathsep+environment.get('ROS_PACKAGE_PATH','')
    command=['roslaunch',str(root/'ros/p450_observe.launch')]
    with (root.parent/'p450_sensors.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                 start_new_session=True,env=environment)
    time.sleep(.5)
    ident=identity(process.pid)
    if process.poll() is not None or ident is None:
        raise RuntimeError('Observation launch failed; inspect p450_sensors.log')
    record_path.write_text(json.dumps(dict(pid=process.pid,identity=ident))+'\n')
    print(json.dumps(dict(started=True,pid=process.pid,command_output=False)))


if __name__=='__main__':main()
