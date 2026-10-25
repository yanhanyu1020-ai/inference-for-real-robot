import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest

ONBOARD = Path(__file__).resolve().parents[1] / 'remote/unitree'
sys.path.insert(0, str(ONBOARD))
from session_support import SegmentCsv, atomic_json, process_info, owned_process
from camera_session import CameraSession
from collection_manager import Collection


def test_two_segments_flush_without_reusing_output(tmp_path):
    selector = tmp_path / 'segment.json'
    a, b = tmp_path / 'a', tmp_path / 'b'
    with SegmentCsv(selector, tmp_path, ['seq']) as writer:
        writer.writerow({'seq': 0})
        assert not list(tmp_path.glob('*/raw/*.csv'))
        atomic_json(selector, {'episode': str(a)})
        writer.writerow({'seq': 1})
        atomic_json(selector, {'episode': None})
        writer.writerow({'seq': 2})
        assert writer.episode is None
        assert (a / 'raw/piper_gamepad.csv').read_text() == 'seq\n1\n'
        atomic_json(selector, {'episode': str(b)})
        writer.writerow({'seq': 3})
    assert (b / 'raw/piper_gamepad.csv').read_text() == 'seq\n3\n'
    with SegmentCsv(selector, tmp_path, ['seq']) as writer:
        with pytest.raises(FileExistsError):
            writer.writerow({'seq': 4})


def test_zombie_is_stopped(tmp_path):
    proc = tmp_path / '12'
    proc.mkdir()
    (proc / 'stat').write_text('12 (worker name) Z ' + '0 ' * 30)
    assert process_info(12, tmp_path) is None


def test_reused_pid_is_not_owned(tmp_path, monkeypatch):
    import session_support as support
    pid = tmp_path / 'recorder.pid'
    pid.write_text('12')
    atomic_json(str(pid) + '.identity.json', {'pid': 12, 'start_ticks': 'old'})
    monkeypatch.setattr(support, 'process_info', lambda _: {'pid': 12, 'start_ticks': 'new', 'args': ['recorder']})
    with pytest.raises(RuntimeError, match='reused'):
        owned_process(pid, 'recorder')


class Camera:
    serial, device, error = '123', '/fake', None
    def snapshot(self):
        return {'monotonic_ns': time.monotonic_ns()}


def test_camera_owner_survives_two_recordings_and_refuses_overlap(tmp_path):
    calls = []
    def record(args, config, *, cameras, stop_event):
        calls.append(cameras['wrist'])
        stop_event.wait(3)
        return 0
    camera = Camera()
    session = CameraSession({'data_root': str(tmp_path)}, {'wrist': camera}, record)
    for name in ('first', 'second'):
        directory = tmp_path / name
        directory.mkdir()
        session.start(directory)
        with pytest.raises(RuntimeError, match='already active'):
            session.start(tmp_path / 'other')
        with pytest.raises(RuntimeError, match='another episode'):
            session.stop(tmp_path / 'other')
        session.stop(directory)
        assert not json.loads((directory / 'status.json').read_text())['running']
        session.stop(directory)  # retry after acknowledgement
    assert calls == [camera, camera]


def test_recorder_exception_is_stopped_and_keeps_error(tmp_path):
    def record(*args, **kwargs):
        raise RuntimeError('camera disconnected')
    directory = tmp_path / 'bad'
    directory.mkdir()
    session = CameraSession({'data_root': str(tmp_path)}, {'wrist': Camera()}, record)
    session.start(directory)
    session.stop(directory)
    assert 'camera disconnected' in json.loads((directory/'status.json').read_text())['recording_error']


def make_collection(tmp_path):
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    config = config_dir / 'collection.json'
    config.write_text(json.dumps({'data_root': str(tmp_path / 'data')}))
    return Collection(config)


def test_legacy_dead_process_recovery_is_idempotent_and_preserves_raw(tmp_path):
    ctl = make_collection(tmp_path)
    episode = tmp_path / 'data/task/episode'
    episode.mkdir(parents=True)
    atomic_json(episode / 'meta.json', {'stop_wall_time_ns': 123})
    atomic_json(episode / 'status.json', {'running': True, 'frames_saved': 99})
    (episode / 'raw.bin').write_bytes(b'original')
    (ctl.run / 'active_episode').write_text(str(episode))
    ctl.stop()
    ctl.stop()
    assert ctl.active() is None
    assert (episode / 'raw.bin').read_bytes() == b'original'
    assert json.loads((episode/'meta.json').read_text())['stop_wall_time_ns'] == 123
    assert not json.loads((episode/'status.json').read_text())['running']


def test_stop_failure_retains_marker_until_retry(tmp_path, monkeypatch):
    import collection_manager as manager
    ctl = make_collection(tmp_path)
    episode = tmp_path / 'data/task/episode'
    episode.mkdir(parents=True)
    atomic_json(episode / 'meta.json', {})
    (ctl.run / 'active_episode').write_text(str(episode))
    def fail(*args, **kwargs):
        raise RuntimeError('still saving')
    monkeypatch.setattr(manager, 'stop_owned', fail)
    with pytest.raises(RuntimeError, match='still saving'):
        ctl.stop()
    assert ctl.active() == episode
    monkeypatch.setattr(manager, 'stop_owned', lambda *args: None)
    ctl.stop()
    assert ctl.active() is None


def test_warm_stop_waits_for_writer_and_gamepad_flush_before_releasing(tmp_path, monkeypatch):
    import collection_manager as manager
    ctl = make_collection(tmp_path)
    episode = tmp_path / 'data/warm'
    episode.mkdir(parents=True)
    atomic_json(episode / 'meta.json', {'warm_session': True})
    (ctl.run / 'active_episode').write_text(str(episode))
    calls = []
    monkeypatch.setattr(ctl, 'camera_status', lambda: {'running': True})
    def pending(*args, **kwargs):
        raise RuntimeError('still saving raw data')
    monkeypatch.setattr(manager, 'request', pending)
    monkeypatch.setattr(ctl, 'select_gamepad', lambda value: calls.append(('gamepad', value)))
    monkeypatch.setattr(manager, 'stop_owned', lambda *args: calls.append(('bridge', args[0])))
    with pytest.raises(RuntimeError, match='still saving'):
        ctl.stop()
    assert ctl.active() == episode
    assert calls == []
    monkeypatch.setattr(manager, 'request', lambda *a: calls.append(('camera', a[1])))
    ctl.stop()
    assert [call[0] for call in calls] == ['camera', 'gamepad', 'bridge']
    assert ctl.active() is None


def test_piper_file_thread_is_joined_without_premature_ack(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace())
    recorder = importlib.import_module('hetero_pkl_recorder')
    reader = recorder.PiperReader('fake', tmp_path)
    waits = []
    reader._thread = SimpleNamespace(join=lambda **kw: waits.append(kw))
    reader.stop()
    assert reader._stop.is_set()
    assert waits == [{}]


def test_previous_recording_error_does_not_block_healthy_warm_devices(tmp_path, monkeypatch, capsys):
    ctl = make_collection(tmp_path)
    monkeypatch.setattr(ctl, 'camera_status', lambda: {
        'prepared': True, 'running': False, 'cameras': {}, 'error': 'previous segment failed'})
    monkeypatch.setattr(ctl, 'gamepad_alive', lambda: {'pid': 123})
    monkeypatch.setattr(ctl, 'gamepad_snapshot', lambda: {'gamepad': {'connected': True}})
    ctl.status()
    value = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert value['session_prepared']
    assert value['session_error'] is None
