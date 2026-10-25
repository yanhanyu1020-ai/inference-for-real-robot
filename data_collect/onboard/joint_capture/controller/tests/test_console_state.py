import json
from pathlib import Path
import threading
import pytest
from jointctl.manifest import ManifestStore
from jointctl.models import EpisodeManifest, EpisodeState, RemoteStatus


def make_state(tmp_path, **kwargs):
    from jointctl.console_state import ConsoleState
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'p450':{'host':'p450'},'unitree':{'host':'unitree'}}))
    return ConsoleState(config,tmp_path/'episodes',**kwargs)


def idle(state):
    for h in ('p450','unitree'):
        state.sample_status(h,lambda h=h:RemoteStatus(h,True,'idle'))


def test_unknown_initial_state_cannot_start(tmp_path):
    state=make_state(tmp_path)
    snap=state.snapshot()
    assert snap['hosts']['p450']['status']['stale']
    assert not snap['allowed']['start']


def test_camera_failure_blocks_recording_but_allows_reinitialization(tmp_path):
    state = make_state(tmp_path)
    idle(state)
    message = json.dumps({'session_prepared': False,
                          'camera_health': {'wrist': {'ready': False, 'error': 'USB disconnected'}},
                          'session_error': 'check camera'})
    state.sample_status('unitree', lambda: RemoteStatus('unitree', True, 'idle', message))
    snap = state.snapshot()
    assert not snap['allowed']['start']
    assert snap['allowed']['prepare']


def test_gamepad_speeds_are_exposed_and_invalidated_when_status_expires(tmp_path):
    clock = [10.0]
    state = make_state(tmp_path, monotonic=lambda: clock[0])
    message = json.dumps({'gamepad': {'connected': True},
                          'speed_factor': 0.5, 'movement_speed': 90,
                          'motor_enabled': [True] * 6, 'feedback_error': 'bad zero',
                          'command_inhibited': True, 'teleop_error': 'reconnect', 'up_level_mode': 'pose'})
    state.sample_status('unitree', lambda: RemoteStatus(
        'unitree', True, 'recording', message, active=True))
    device = state.snapshot()['hosts']['unitree']
    assert device['speed_factor'] == 0.5
    assert device['movement_speed'] == 90
    assert device['command_inhibited'] is True
    assert device['teleop_error'] == 'reconnect'
    assert device['teleop_mode'] == 'pose'
    assert device['motor_enabled'] == [True] * 6
    assert device['feedback_error'] == 'bad zero'
    clock[0] += 20
    device = state.snapshot()['hosts']['unitree']
    assert device['speed_factor'] is None
    assert device['movement_speed'] is None
    assert device['teleop_error'] is None
    assert device['motor_enabled'] is None
    assert device['feedback_error'] is None


def test_stale_and_failed_status_does_not_claim_idle(tmp_path):
    clock=[10.0]
    state=make_state(tmp_path,monotonic=lambda:clock[0])
    idle(state)
    assert state.snapshot()['allowed']['start']
    clock[0]=17
    assert state.snapshot()['hosts']['p450']['status']['state']=='unknown'
    assert not state.snapshot()['allowed']['start']
    state.sample_status('p450',lambda:RemoteStatus('p450',False,last_error='offline'))
    assert state.snapshot()['hosts']['p450']['status']['error']=='offline'


def test_complete_manifest_alone_is_not_validated(tmp_path):
    state=make_state(tmp_path)
    store=ManifestStore(state.root)
    store.create(EpisodeManifest('example','test','joint',1789364402353640400,EpisodeState.COMPLETE))
    store.clear_active('example')
    snap=state.snapshot()
    assert snap['episode']['created_desktop_ns']=='1789364402353640400'
    assert snap['validation']['state']=='unavailable'


def test_latest_episode_uses_creation_time_not_revalidation_mtime(tmp_path):
    state = make_state(tmp_path)
    store = ManifestStore(state.root)
    store.create(EpisodeManifest('older', 'test', 'joint', 100, EpisodeState.COMPLETE))
    store.clear_active('older')
    store.create(EpisodeManifest('newer', 'test', 'joint', 200, EpisodeState.COMPLETE))
    store.clear_active('newer')
    store.update('older', metadata={'postprocess': 'failed'})
    assert state.snapshot()['episode']['episode_id'] == 'newer'


def test_fast_stopped_episode_is_pending_and_can_be_processed_while_idle(tmp_path):
    state = make_state(tmp_path, runner=lambda _argv, _directory: 0)
    store = ManifestStore(state.root)
    store.create(EpisodeManifest('pending', 'test', 'joint', 1, EpisodeState.COMPLETE,
                                metadata={'postprocess': 'pending'}))
    store.clear_active('pending')
    idle(state)
    snap = state.snapshot()
    assert snap['validation']['state'] == 'pending'
    assert snap['postprocess']['pending_count'] == 1
    assert snap['allowed']['start']
    assert snap['allowed']['finalize']
    state.submit('finalize', {})
    state.close()


def test_failed_revalidation_overrides_older_alignment_pass(tmp_path):
    state = make_state(tmp_path)
    store = ManifestStore(state.root)
    store.create(EpisodeManifest('example', 'test', 'joint', 1, EpisodeState.COMPLETE,
                                 metadata={'postprocess': 'failed',
                                           'postprocess_error': 'Unitree frames_dropped=245'}))
    store.clear_active('example')
    (store.episode_dir('example') / 'alignment.json').write_text(json.dumps({
        'quality': {'data_validated': True, 'degraded': False},
        'valid_interval': {'start_inclusive_ns': 1, 'end_exclusive_ns': 2},
    }), encoding='utf-8')
    snapshot = state.snapshot()
    assert snapshot['validation']['state'] == 'failed'
    assert snapshot['validation']['report'] is None
    assert 'frames_dropped=245' in snapshot['validation']['error']


def test_postprocess_is_unavailable_during_recording(tmp_path):
    state = make_state(tmp_path)
    store = ManifestStore(state.root)
    store.create(EpisodeManifest('pending', 'test', 'joint', 1, EpisodeState.COMPLETE,
                                metadata={'postprocess': 'pending'}))
    store.clear_active('pending')
    store.create(EpisodeManifest('live', 'test', 'joint', 2, EpisodeState.RECORDING))
    assert not state.snapshot()['allowed']['finalize']


def test_job_async_excludes_duplicate_and_passes_literal_arguments(tmp_path):
    entered=threading.Event(); release=threading.Event()
    def runner(argv,job_dir):
        entered.set(); release.wait(3)
        assert argv[-4:]==['--instruction','x; echo nope','--task','joint']
        return 0
    state=make_state(tmp_path,runner=runner)
    idle(state)
    job=state.submit('start',{'instruction':'x; echo nope','task':'joint'})
    assert job['id']
    assert entered.wait(1)
    try:
        with pytest.raises(RuntimeError): state.submit('start',{'instruction':'x','task':'joint'})
    finally:
        release.set(); state.close()


@pytest.mark.parametrize('action,payload',[('shell',{}),('start',{'instruction':None,'task':'joint'}),
                                        ('start',{'instruction':'x','task':'joint','command':'rm'})])
def test_actions_reject_unapproved_or_invalid_fields(tmp_path,action,payload):
    state=make_state(tmp_path)
    with pytest.raises(ValueError): state.submit(action,payload)


def test_foreign_episode_prevents_stop(tmp_path):
    state=make_state(tmp_path)
    ManifestStore(state.root).create(EpisodeManifest('own','test','joint',1,EpisodeState.RECORDING))
    state.sample_status('p450',lambda:RemoteStatus('p450',True,'recording',active=True,episode_id='foreign'))
    assert not state.snapshot()['allowed']['stop']


def test_interrupted_revalidation_never_reuses_old_pass(tmp_path):
    state=make_state(tmp_path)
    store=ManifestStore(state.root)
    store.create(EpisodeManifest('example','test','joint',1,EpisodeState.COMPLETE))
    store.clear_active('example')
    (store.episode_dir('example')/'alignment.json').write_text(json.dumps({
        'quality':{'data_validated':True,'degraded':False},'valid_interval':{'start_inclusive_ns':1,'end_exclusive_ns':2}}))
    directory=state.jobs_root/'prior'; directory.mkdir()
    (directory/'job.json').write_text(json.dumps({'id':'prior','action':'align','state':'running'}))
    restarted=make_state(tmp_path)
    assert restarted.snapshot()['validation']['state']=='unavailable'
    assert restarted.snapshot()['validation']['report'] is None


def test_status_probe_allows_real_p450_initialization_latency(tmp_path,monkeypatch):
    from jointctl.models import CommandResult
    state=make_state(tmp_path)
    def response(host,command,timeout):
        if timeout < 4:
            return CommandResult(command,-1,stderr='status initialization exceeded timeout')
        return CommandResult(command,0,json.dumps({'recorder':{'active':False},'capture':None}))
    monkeypatch.setattr('jointctl.remote.run_ssh',response)
    state.sample_status('p450')
    assert state.snapshot()['hosts']['p450']['status']['state']=='idle'


def test_blank_fields_generate_task_and_instruction(tmp_path):
    captured = []
    state = make_state(tmp_path, runner=lambda argv, directory: captured.append(argv) or 0)
    idle(state)
    state.submit('start', {})
    state.close()
    assert captured[0][-1] == 'start'
    assert '--task' not in captured[0]
    assert '--instruction' not in captured[0]


def test_pending_remote_finalize_blocks_new_start(tmp_path):
    state = make_state(tmp_path)
    idle(state)
    state.sample_status('p450', lambda: RemoteStatus('p450', True, 'idle', episode_id='old'))
    assert not state.snapshot()['allowed']['start']
    assert state.snapshot()['allowed']['recover']


def test_write_stall_resets_on_episode_change(tmp_path):
    now = [10.0]
    state = make_state(tmp_path, monotonic=lambda: now[0])
    def probe(episode='one'):
        return RemoteStatus('unitree', True, 'recording', active=True,
                            episode_id=episode, progress_name='frames_saved', progress_value=105)
    state.sample_status('unitree', probe)
    now[0] += 16
    state.sample_status('unitree', probe)
    assert state.snapshot()['hosts']['unitree']['write_stalled']
    state.sample_status('unitree', lambda: probe('two'))
    assert not state.snapshot()['hosts']['unitree']['write_stalled']


def test_resources_have_independent_freshness_budget(tmp_path):
    now = [10.0]
    state = make_state(tmp_path, monotonic=lambda: now[0])
    state._record('p450', 'resources', {'memory': {}})
    now[0] += 15
    assert not state.snapshot()['hosts']['p450']['resources']['stale']


def test_interrupted_start_is_exposed_to_console(tmp_path):
    state = make_state(tmp_path)
    store = ManifestStore(state.root)
    store.create(EpisodeManifest('orphan', 'test', 'joint', 1, EpisodeState.STARTING,
                                 metadata={'starter_pid': 99999999}))
    idle(state)
    snap = state.snapshot()
    assert snap['starter_alive'] is False
    assert snap['allowed']['stop']
    assert not snap['allowed']['start']
