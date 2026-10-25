"""Warm v2 episode orchestration; pairing/export remain offline."""
from pathlib import Path
import shutil
import time

from raw_capture import RawCapture
from mcap_storage import durable_json
from session_support import read_json, sync_episode_csv


def readiness(streams, piper, go2, gamepad, config, now_ns):
    def fresh(value, limit):
        return isinstance(value,dict) and 0 <= now_ns-value.get('monotonic_ns',0) <= int(limit*1e6)
    missing=[]
    for name in ('front','wrist'):
        stream=streams.get(name,{})
        if stream.get('written',0)<2 or not fresh({'monotonic_ns':stream.get('last_received_monotonic_ns') or 0},config.get('camera_stale_ms',500)):
            missing.append(name)
    if not fresh(piper,config.get('piper_stale_ms',250)):
        missing.append('piper')
    for name in ('sport_mode_state','low_state'):
        state=(go2 or {}).get(name,{})
        if not state.get('valid') or not fresh(state,config.get('go2_stale_ms',500)):
            missing.append('go2.'+name)
    if not fresh(gamepad,config.get('piper_gamepad_stale_ms',1000)):
        missing.append('gamepad_state')
    return {'ready':not missing,'missing':missing}


def run(args, config, *, cameras, stop_event, publish=lambda _:None, piper_factory=None):
    directory=Path(args.episode_dir)
    raw=directory/'raw'
    raw.mkdir(parents=True,exist_ok=True)
    capture=RawCapture(directory,config,cameras)
    piper=None
    begun=False
    failure=None
    state_gaps=0
    ever_ready=False
    started=time.monotonic_ns()
    started_wall=time.time_ns()
    last_report=0
    report={}
    try:
        if shutil.disk_usage(directory).free < 20*1024**3:
            raise RuntimeError('less than 20 GiB free disk space')
        if piper_factory is None:
            from hetero_pkl_recorder import PiperReader
            piper_factory=PiperReader
        piper=piper_factory(config['can_interface'],raw)
        piper.start()
        # The per-segment Piper SDK waits for its CAN receive thread before it
        # can write the first state row. Open camera subscriptions afterward so
        # the recorded image interval starts with state coverage. A dead Piper
        # cannot block raw image preservation indefinitely.
        deadline=time.monotonic()+config.get('piper_startup_wait_s',1.5)
        while not stop_event.is_set() and not piper.error and piper.snapshot() is None and time.monotonic()<deadline:
            stop_event.wait(.01)
        capture.start()
        begun=True
        while True:
            image=capture.status()
            health=readiness(image['cameras'],piper.snapshot(),read_json(raw/'go2_snapshot.json'),
                read_json(raw/'piper_gamepad_snapshot.json'),config,time.monotonic_ns())
            if ever_ready and not health['ready']:
                state_gaps+=1
            ever_ready=ever_ready or health['ready']
            report={**image,'streams':image['cameras'],'recording_state':image['phase'],
                'running':image['phase']=='recording','ready':health['ready'],
                'missing_states':health['missing'],'state_gap_ticks':state_gaps,
                'quality_ok':image['quality_ok'] and state_gaps==0,
                'fault':image['faults'],'camera_durable_complete':False,'durable_complete':False,
                'piper_rows':piper.rows,'elapsed_s':(time.monotonic_ns()-started)/1e9}
            publish(report)
            if time.monotonic()-last_report>=1:
                durable_json(directory/'status.json',report)
                last_report=time.monotonic()
            if stop_event.is_set() or image['faults'] or capture.wait_stopped(0):
                break
            if piper.error:
                raise RuntimeError('Piper reader: '+piper.error)
            stop_event.wait(.05)
    except Exception as exc:
        failure=str(exc)
    finally:
        # Keep state producers alive through image cutoff and disk drain.
        if begun:
            capture.request_stop()
            while not capture.wait_stopped(.1):
                pending=capture.status()
                publish({**report,'streams':pending['cameras'],'running':False,
                    'recording_state':'stopping','camera_durable_complete':False})
        piper_synced=True
        if piper is not None:
            try:
                piper.stop()
                sync_episode_csv(directory, names=('piper_state.csv','piper_status.csv'))
            except Exception as exc:
                piper_synced=False
                failure=(failure+'; ' if failure else '')+'Piper close: '+str(exc)
        image=capture.status()
        camera_durable=(not begun or image['phase']=='stopped') and piper_synced
        faults=image['faults']+([failure] if failure else [])
        report={**report,'format_version':2,'image_storage':'mcap','codec':capture.codec,
            'streams':image['cameras'],'cameras':image['cameras'],
            'running':False,'ready':False,'recording_state':'stopped' if camera_durable else 'cleanup_pending',
            'camera_durable_complete':camera_durable,'durable_complete':False,
            'quality_ok':not faults and ever_ready and not state_gaps,
            'fault':faults,'recording_error':'; '.join(faults) or None,
            'state_gap_ticks':state_gaps,'ever_ready':ever_ready,
            'start_wall_time_ns':started_wall,'end_wall_time_ns':time.time_ns(),
            'duration_s':(time.monotonic_ns()-started)/1e9,'sensor_completeness':'unverified'}
        # The manager confirms gamepad/Go2 CSV durability before external Stop ACK.
        durable_json(directory/'summary.json',report)
        durable_json(directory/'status.json',report)
        publish(report)
    return 1 if faults else 0
