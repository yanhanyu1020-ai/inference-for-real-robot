from pathlib import Path
import sys
import threading
import itertools
import time
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'remote/unitree'))
from test_raw_storage import rows


@pytest.fixture
def reader_type(monkeypatch):
    # No camera hardware is opened here; exercise the real publisher/subscription.
    import importlib
    if importlib.util.find_spec('cv2') is None:
        monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace())
    recorder = importlib.import_module('hetero_pkl_recorder')
    # Windows monotonic clock can repeat within one millisecond; camera hardware
    # delivers at 30 Hz. Keep synthetic burst timestamps strictly ordered.
    ticks = itertools.count(time.monotonic_ns(), 1000)
    monkeypatch.setattr(recorder, 'time', SimpleNamespace(monotonic_ns=lambda:next(ticks),time_ns=time.time_ns))
    return recorder.CameraReader


@pytest.fixture
def capture_env(monkeypatch, reader_type):
    import raw_capture
    monkeypatch.setattr(raw_capture.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100*1024**3))
    cameras = {n: reader_type(n, n+'123', 4, 3, 30) for n in ('front','wrist')}
    return raw_capture, cameras, {'boot_id':'test', 'width':4, 'height':3, 'raw_codec':'lz4'}


def records(path, camera):
    return [r for p in sorted((path/'raw/cameras').glob(camera+'*.mcap')) for r in rows(p)]


def test_front_saved_without_wrist_or_state(tmp_path, capture_env):
    raw, cameras, config = capture_env
    c = raw.RawCapture(tmp_path, config, cameras)
    c.start()
    cameras['front']._publish(np.full((3,4,3), 17, np.uint8))
    c.request_stop()
    assert c.wait_stopped(3)
    assert len(records(tmp_path, 'front')) == 1
    assert records(tmp_path, 'wrist') == []
    assert c.status()['cameras']['front']['durable'] == 1


def test_no_latest_replay_at_start(tmp_path, capture_env):
    raw, cameras, config = capture_env
    cameras['front']._publish(np.zeros((3,4,3),np.uint8))
    c = raw.RawCapture(tmp_path,config,cameras)
    c.start()
    c.request_stop()
    assert c.wait_stopped(3)
    assert c.status()['cameras']['front']['received'] == 0


def test_two_episodes_have_disjoint_sequences(tmp_path, capture_env):
    raw,cameras,config = capture_env
    for name in ('a','b'):
        c=raw.RawCapture(tmp_path/name,config,cameras)
        c.start()
        cameras['front']._publish(np.zeros((3,4,3),np.uint8))
        c.request_stop()
        assert c.wait_stopped(3)
    assert [records(tmp_path/n,'front')[0][0]['reader_seq'] for n in ('a','b')] == [1,2]
    assert [records(tmp_path/n,'front')[0][0]['seq'] for n in ('a','b')] == [1,1]


def test_queue_full_preserves_old_frames_and_faults(tmp_path, capture_env):
    raw,cameras,config = capture_env
    from mcap_storage import CameraStore
    entered, release = threading.Event(), threading.Event()
    class SlowStore(CameraStore):
        def append(self, metadata, image):
            entered.set()
            assert release.wait(3)
            return super().append(metadata,image)
    c=raw.RawCapture(tmp_path, {**config,'raw_queue_bytes':100}, cameras,store_factory=SlowStore)
    c.start()
    try:
        cameras['front']._publish(np.full((3,4,3),1,np.uint8))
        assert entered.wait(2)
        cameras['front']._publish(np.full((3,4,3),2,np.uint8))
        assert c.status()['cameras']['front']['queue_warning']
        cameras['front']._publish(np.full((3,4,3),3,np.uint8))
        s=c.status()['cameras']['front']
        assert (s['received'],s['accepted'],s['rejected'],s['pending_bytes']) == (3,2,1,72)
    finally:
        release.set()
        c.request_stop()
        assert c.wait_stopped(3)
    assert [int(img[0,0,0]) for _,img in records(tmp_path,'front')] == [1,2]
    assert c.status()['faults']


def test_unsubscribe_race_has_no_late_append(reader_type):
    camera=reader_type('front','123',4,3,30)
    entered, release, removed = threading.Event(), threading.Event(), threading.Event()
    got=[]
    def callback(frame):
        entered.set()
        assert release.wait(2)
        got.append(frame['seq'])
    token=camera.subscribe(callback)
    publisher=threading.Thread(target=camera._publish,args=(np.zeros((3,4,3),np.uint8),))
    publisher.start()
    assert entered.wait(2)
    def unsubscribe():
        assert camera.unsubscribe(token) == 1
        removed.set()
    stopper=threading.Thread(target=unsubscribe)
    stopper.start()
    assert not removed.is_set()
    release.set()
    publisher.join(2)
    stopper.join(2)
    camera._publish(np.zeros((3,4,3),np.uint8))
    assert removed.is_set() and got == [1]


def test_camera_buffer_mutation_cannot_corrupt_saved_frame(tmp_path,capture_env):
    raw,cameras,config=capture_env
    from mcap_storage import CameraStore
    entered,release=threading.Event(),threading.Event()
    class SlowStore(CameraStore):
        def append(self,metadata,image):
            entered.set()
            assert release.wait(3)
            super().append(metadata,image)
    c=raw.RawCapture(tmp_path,config,cameras,store_factory=SlowStore)
    c.start()
    pixels=np.full((3,4,3),19,np.uint8)
    try:
        cameras['front']._publish(pixels)
        assert entered.wait(2)
        pixels[:]=99
    finally:
        release.set()
        c.request_stop()
        assert c.wait_stopped(3)
    assert np.all(records(tmp_path,'front')[0][1] == 19)


def test_queue_defaults_and_limits(tmp_path,capture_env):
    raw,cameras,config=capture_env
    c=raw.RawCapture(tmp_path,config,cameras)
    assert c.status()['cameras']['front']['capacity_bytes'] == 64*1024**2
    with pytest.raises(ValueError): raw.RawCapture(tmp_path, {**config,'raw_queue_bytes':64*1024**2+1},cameras)


def test_changed_frame_dimensions_rejected(tmp_path,capture_env):
    raw,cameras,config=capture_env
    c=raw.RawCapture(tmp_path,config,cameras)
    c.start()
    cameras['front']._publish(np.zeros((2,4,3),np.uint8))
    c.request_stop()
    assert c.wait_stopped(3)
    assert c.status()['cameras']['front']['rejected'] == 1
    assert c.status()['faults']
    assert not records(tmp_path,'front')


def test_monitor_failure_unsubscribes_before_close(tmp_path,capture_env,monkeypatch):
    raw,cameras,config=capture_env
    c=raw.RawCapture(tmp_path,config,cameras)
    c.start()
    # Simulate an unexpected monitor failure without waiting on wall-clock sleep.
    class BrokenCamera:
        serial='wrist123'
        @property
        def error(self): raise OSError('device status unavailable')
        def unsubscribe(self,token): return cameras['wrist'].unsubscribe(token)
    c.cameras['wrist']=BrokenCamera()
    assert c.wait_stopped(3)
    before=c.status()['cameras']['front']['received']
    cameras['front']._publish(np.zeros((3,4,3),np.uint8))
    assert c.status()['cameras']['front']['received'] == before


def test_low_disk_start_refused(tmp_path,capture_env,monkeypatch):
    raw,cameras,config=capture_env
    monkeypatch.setattr(raw.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*1024**3-1))
    c=raw.RawCapture(tmp_path,config,cameras)
    with pytest.raises(RuntimeError,match='20 GiB'): c.start()
    assert c.status()['phase'] == 'idle'


def test_camera_stall_closes_and_reports_fault(tmp_path,capture_env):
    raw,cameras,config=capture_env
    c=raw.RawCapture(tmp_path,config,cameras)
    c.start()
    assert c.wait_stopped(3)
    assert any('500 ms' in fault for fault in c.status()['faults'])


def test_status_exposes_rates_and_disk_budget(tmp_path,capture_env):
    raw,cameras,config=capture_env
    c=raw.RawCapture(tmp_path,config,cameras)
    c.start()
    cameras['front']._publish(np.zeros((3,4,3),np.uint8))
    c.request_stop()
    assert c.wait_stopped(3)
    report=c.status()
    assert report['disk_available_bytes']==100*1024**3
    assert report['remaining_minutes']>0
    assert report['cameras']['front']['received_fps']>0
    assert report['cameras']['front']['written_fps']>0
    assert report['cameras']['front']['write_bytes_per_s']>0


def test_failed_unsubscribe_cannot_accept_after_drain(tmp_path,capture_env,monkeypatch):
    raw,cameras,config=capture_env
    def broken(_): raise OSError('unsubscribe failed')
    monkeypatch.setattr(cameras['front'],'unsubscribe',broken)
    c=raw.RawCapture(tmp_path,config,cameras); c.start()
    cameras['front']._publish(np.zeros((3,4,3),np.uint8))
    c.request_stop(); assert c.wait_stopped(3)
    before=c.status()['cameras']['front']['received']
    cameras['front']._publish(np.zeros((3,4,3),np.uint8))
    assert c.status()['cameras']['front']['received']==before
    assert c.status()['phase']=='cleanup_pending'
    assert not c.status()['quality_ok']
