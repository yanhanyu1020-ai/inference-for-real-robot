"""Disposable browser acceptance fixture. Never connects to a robot."""
import json
from pathlib import Path
import tempfile
import threading
import time
from jointctl.console import make_server
from jointctl.console_state import ConsoleState
from jointctl.models import EpisodeManifest, EpisodeState, RemoteStatus

folder=Path(tempfile.mkdtemp(prefix='joint-console-ui-'))
config=folder/'config.json'
config.write_text('{}')
phase={'recording':False,'frames':0}


def run(argv,job_dir):
    command=argv[argv.index('--manifest-root')+2]
    if command=='start':
        m=EpisodeManifest('joint_ui_test','模拟界面验收','joint',time.time_ns(),EpisodeState.STARTING,
            remote_directories={'p450':'/home/amov/p450_data/19700101_test/long-path-ui-validation',
                                'unitree':'/home/unitree/heterovla-data/datasets/joint/joint_ui_test'})
        state.store.create(m)
        time.sleep(3)
        state.store.update(m.episode_id,state=EpisodeState.RECORDING,t0_desktop_ns=time.time_ns())
        phase['recording']=True
    elif command=='stop':
        m=state.store.active()
        state.store.update(m.episode_id,state=EpisodeState.STOPPING,t1_desktop_ns=time.time_ns())
        phase['recording']=False
        time.sleep(3)
        m=state.store.update(m.episode_id,state=EpisodeState.COMPLETE,clock_monitor_closed_cleanly=True)
        time.sleep(3)
        (state.store.episode_dir(m.episode_id)/'alignment.json').write_text(json.dumps({
            'episode_id':m.episode_id,'quality':{'degraded':False,'data_validated':True,
            'estimated_error_ns':4400000,'degradation_reasons':[]},'valid_interval':{
            'start_inclusive_ns':m.t0_desktop_ns,'end_exclusive_ns':m.t1_desktop_ns}}))
    return 0


state=ConsoleState(config,folder/'episodes',runner=run)


def updates():
    while True:
        phase['frames']+=30
        for h in ('p450','unitree'):
            recording=phase['recording']
            state.sample_status(h,lambda h=h:RemoteStatus(h,True,'recording' if recording else 'idle',
                active=recording,episode_id='joint_ui_test' if recording else None,
                progress_name='session_bytes' if h=='p450' else 'frames_saved',
                progress_value=phase['frames']*100000 if h=='p450' else phase['frames']))
            state._record(h,'resources',{'memory':{'total_bytes':16*1024**3,'available_bytes':10*1024**3,
                'used_bytes':6*1024**3,'used_percent':37.5},'disk':{'total_bytes':128*1024**3,
                'available_bytes':80*1024**3,'used_percent':37.5}})
        time.sleep(1)


threading.Thread(target=updates,daemon=True).start()
print('Mock console only: http://127.0.0.1:18766',flush=True)
make_server(state,18766).serve_forever()
