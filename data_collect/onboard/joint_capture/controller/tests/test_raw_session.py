import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'remote/unitree'))
from camera_session import CameraSession
from session_support import atomic_json
from test_raw_capture import reader_type, capture_env, records


class Camera:
    serial,device,error='123','fake',None
    def snapshot(self): return {'monotonic_ns':time.monotonic_ns()}


@pytest.mark.parametrize('failure_at', ['episode', 'parent', 'marker'])
def test_stop_directory_sync_failure_retains_ownership(tmp_path, monkeypatch, failure_at):
    import session_support
    from collection_manager import Collection
    config=tmp_path/'config/collection.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'data_root':str(tmp_path/'data')}))
    ctl=Collection(config)
    directory=tmp_path/'data/task/one'
    (directory/'raw').mkdir(parents=True)
    atomic_json(directory/'meta.json', {'format_version':2,'warm_session':True})
    atomic_json(directory/'summary.json', {'camera_durable_complete':True})
    (ctl.run/'active_episode').write_text(str(directory))
    monkeypatch.setattr(ctl,'camera_status',lambda:None)
    monkeypatch.setattr(ctl,'select_gamepad',lambda _:None)
    target={'episode':directory, 'parent':directory.parent, 'marker':ctl.run}[failure_at]
    def sync(path):
        if Path(path)==target: raise OSError('directory sync failed')
    monkeypatch.setattr(session_support,'sync_directory',sync,raising=False)
    with pytest.raises((OSError,RuntimeError),match='directory sync failed'):
        ctl.stop()
    assert ctl.active()==directory
    monkeypatch.setattr(session_support,'sync_directory',lambda _:None)
    ctl.stop()
    assert ctl.active() is None


def test_start_requires_both_streams_and_states_but_does_not_gate_image_writes():
    from raw_episode import readiness
    now=1_000_000_000
    streams={n:{'written':2,'last_received_monotonic_ns':now} for n in ('front','wrist')}
    config={'camera_stale_ms':500,'go2_stale_ms':500,'piper_stale_ms':250,'piper_gamepad_stale_ms':1000}
    go2={n:{'valid':True,'monotonic_ns':now} for n in ('sport_mode_state','low_state')}
    piper={'monotonic_ns':now}
    gamepad={'monotonic_ns':now}
    assert not readiness(streams,None,go2,gamepad,config,now)['ready']
    assert readiness(streams,piper,go2,gamepad,config,now)['ready']
    assert not readiness({**streams,'wrist':{'written':0}},piper,go2,gamepad,config,now)['ready']
    assert not readiness(streams,piper,go2,gamepad,config,now+600_000_000)['ready']


def test_status_remains_responsive_during_stop(tmp_path):
    entered,release=threading.Event(),threading.Event()
    def record(args,config,*,cameras,stop_event):
        entered.set()
        stop_event.wait(3)
        release.wait(3)
    s=CameraSession({'data_root':str(tmp_path)}, {'front':Camera()}, record)
    directory=tmp_path/'one'
    directory.mkdir()
    s.start(directory)
    assert entered.wait(1)
    stopper=threading.Thread(target=s.stop,args=(directory,))
    stopper.start()
    assert s.stop_event.wait(1)
    try:
        before=time.perf_counter()
        assert s.status()['running']
        assert time.perf_counter()-before < .25
    finally:
        release.set()
        stopper.join(3)


def test_stop_timeout_retains_episode(tmp_path):
    release=threading.Event()
    def record(*args,**kwargs): release.wait(3)
    s=CameraSession({'data_root':str(tmp_path),'camera_stop_timeout_s':.02},{'front':Camera()},record)
    directory=tmp_path/'one'
    directory.mkdir()
    s.start(directory)
    try:
        with pytest.raises(RuntimeError,match='retry stop'): s.stop(directory)
        assert s.episode == str(directory)
    finally:
        release.set()
        s.thread.join(3)
    s.stop(directory)
    assert s.episode is None


def test_concurrent_stop_is_idempotent(tmp_path):
    def record(*args,stop_event,**kwargs): stop_event.wait(3)
    s=CameraSession({'data_root':str(tmp_path)},{'front':Camera()},record)
    directory=tmp_path/'one'
    directory.mkdir()
    s.start(directory)
    errors=[]
    def stop():
        try: s.stop(directory)
        except Exception as exc: errors.append(str(exc))
    a,b=threading.Thread(target=stop),threading.Thread(target=stop)
    a.start(); b.start(); a.join(3); b.join(3)
    assert not errors and s.episode is None


def test_stop_ack_requires_csv_and_image_durability(tmp_path,monkeypatch,capsys):
    from collection_manager import Collection
    config=tmp_path/'config/collection.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'data_root':str(tmp_path/'data')}))
    ctl=Collection(config)
    directory=tmp_path/'data/one'
    (directory/'raw').mkdir(parents=True)
    atomic_json(directory/'meta.json', {'format_version':2,'warm_session':True})
    atomic_json(directory/'summary.json',{'format_version':2,'camera_durable_complete':True,
        'quality_ok':False,'fault':['camera failed'],'streams':{}})
    (directory/'raw/piper_state.csv').write_text('seq\n1\n')
    (ctl.run/'active_episode').write_text(str(directory))
    monkeypatch.setattr(ctl,'camera_status',lambda:{'running':False})
    monkeypatch.setattr(ctl,'select_gamepad',lambda _:None)
    monkeypatch.setattr('collection_manager.request',lambda *a,**k:{})
    # Inject only the final durability operation, not process ownership or metadata.
    def failed_sync(_): raise OSError('CSV disk sync failed')
    monkeypatch.setattr('collection_manager.sync_episode_csv',failed_sync,raising=False)
    with pytest.raises(RuntimeError,match='CSV disk sync'): ctl.stop()
    assert ctl.active() == directory
    monkeypatch.setattr('collection_manager.sync_episode_csv',lambda _:['piper_state.csv'])
    ctl.stop()
    assert ctl.active() is None
    assert json.loads((directory/'meta.json').read_text())['durable_complete']
    assert json.loads((directory/'summary.json').read_text())['durable_complete']
    assert json.loads((directory/'status.json').read_text())['durable_complete']
    reports=[json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert reports and reports[-1]['raw_episode_id']=='one'
    assert reports[-1]['fault']==['camera failed']


def test_low_disk_refuses_before_subscription(tmp_path,monkeypatch):
    import raw_episode
    import raw_capture
    monkeypatch.setattr(raw_capture.shutil,'disk_usage',lambda _:SimpleNamespace(free=1))
    class NotSubscribed(Camera):
        def subscribe(self,_): raise AssertionError('must refuse first')
    s=CameraSession({'data_root':str(tmp_path),'format_version':2,'boot_id':'test'},
                    {n:NotSubscribed() for n in ('front','wrist')},lambda *a,**k:None)
    directory=tmp_path/'one'
    directory.mkdir()
    s.start(directory)
    s.thread.join(3)
    assert s.error and '20 GiB' in s.error


def test_socket_status_not_queued_behind_stop(tmp_path):
    import socket
    from camera_session import CameraRequestPool
    entered,release=threading.Event(),threading.Event()
    class Session:
        def stop(self,episode):
            entered.set()
            assert release.wait(3)
            return {'running':False}
        def status(self): return {'running':True}
    pool=CameraRequestPool(Session())
    stop_client,stop_server=socket.socketpair()
    status_client,status_server=socket.socketpair()
    try:
        pool.submit(stop_server)
        stop_client.sendall(b'{"action":"stop","episode":"one"}\n')
        assert entered.wait(1)
        pool.submit(status_server)
        status_client.settimeout(.25)
        status_client.sendall(b'{"action":"status"}\n')
        response=json.loads(status_client.recv(65536))
        assert response['ok'] and response['status']['running']
    finally:
        release.set()
        stop_client.close()
        status_client.close()


def test_v2_start_uses_ready_flag_and_records_format(tmp_path,monkeypatch):
    from collection_manager import Collection
    config=tmp_path/'config/collection.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'data_root':str(tmp_path/'data'),'format_version':2,
                                 'raw_codec':'lz4','network_interface':'eth0','boot_id':'test'}))
    ctl=Collection(config)
    monkeypatch.setenv('HOME',str(tmp_path))
    monkeypatch.setattr(ctl,'prepare',lambda:None)
    monkeypatch.setattr(ctl,'select_gamepad',lambda _:None)
    monkeypatch.setattr(ctl,'spawn',lambda *a,**k:None)
    monkeypatch.setattr(ctl,'camera_status',lambda:{'running':True})
    def request(sock, action, directory):
        atomic_json(Path(directory)/'status.json',{'format_version':2,'ready':True})
    monkeypatch.setattr('collection_manager.request',request)
    # Missing v2 readiness must fail promptly rather than spending 12 real seconds.
    import collection_manager
    ticks=iter([0,0,13])
    monkeypatch.setattr(collection_manager,'time',SimpleNamespace(monotonic=lambda:next(ticks),time_ns=time.time_ns,sleep=lambda _:None))
    ctl.start('one','instruction','task')
    directory=ctl.active()
    meta=json.loads((directory/'meta.json').read_text())
    assert meta['format_version'] == 2 and meta['image_storage'] == 'mcap'
    assert meta['boot_id']=='test' and meta['codec']=='lz4'


def test_raw_episode_keeps_images_when_state_absent_and_stops_csv_last(tmp_path,capture_env):
    import numpy as np
    from raw_episode import run
    raw,cameras,config=capture_env
    config={**config,'piper_startup_wait_s':0.02}
    started,stopping,release=threading.Event(),threading.Event(),threading.Event()
    stop=threading.Event()
    reports=[]
    class Piper:
        error=None
        rows=1
        def __init__(self,can,directory):
            self.directory=directory
        def start(self):
            for name in ('piper_state.csv','piper_status.csv'):
                (self.directory/name).write_text('seq\n1\n')
            started.set()
        def snapshot(self): return None
        def stop(self):
            stopping.set()
            assert release.wait(3)
    def record():
        run(SimpleNamespace(episode_dir=tmp_path),{**config,'can_interface':'can0'},
            cameras=cameras,stop_event=stop,publish=reports.append,piper_factory=Piper)
    thread=threading.Thread(target=record)
    thread.start()
    assert started.wait(2)
    try:
        deadline=time.monotonic()+2
        while not all(camera._subscribers for camera in cameras.values()) and time.monotonic()<deadline:
            time.sleep(.002)
        assert all(camera._subscribers for camera in cameras.values())
        for camera in cameras.values(): camera._publish(np.zeros((3,4,3),np.uint8))
        stop.set()
        assert stopping.wait(2)
        assert records(tmp_path,'front') and records(tmp_path,'wrist')
        assert not (tmp_path/'summary.json').exists()
    finally:
        release.set()
        thread.join(3)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['camera_durable_complete']
    assert not summary['durable_complete']  # manager must confirm other state producers
    assert not summary['quality_ok']  # image preservation does not imply training readiness


def test_camera_subscription_waits_for_first_piper_state_but_not_forever(tmp_path,capture_env):
    from raw_episode import run
    _,cameras,config=capture_env
    subscribed=[]
    ready_times=[]
    for camera in cameras.values():
        original=camera.subscribe
        def subscribe(callback,original=original):
            subscribed.append(time.monotonic())
            return original(callback)
        camera.subscribe=subscribe
    class Piper:
        rows=1
        error=None
        ready_at=None
        def __init__(self,can,directory): self.directory=directory
        def start(self):
            for name in ('piper_state.csv','piper_status.csv'):
                (self.directory/name).write_text('monotonic_ns,wall_time_ns\n')
            self.ready_at=time.monotonic()+.07
        def snapshot(self):
            if time.monotonic()>=self.ready_at:
                if not ready_times: ready_times.append(time.monotonic())
                return {'monotonic_ns':time.monotonic_ns()}
            return None
        def stop(self): pass
    stop=threading.Event()
    timer=threading.Timer(.2,stop.set)
    timer.start()
    try:
        run(SimpleNamespace(episode_dir=tmp_path),{**config,'can_interface':'can0',
            'piper_startup_wait_s':.15},cameras=cameras,stop_event=stop,piper_factory=Piper)
    finally: timer.join()
    assert subscribed and ready_times
    assert min(subscribed)>=ready_times[0]
