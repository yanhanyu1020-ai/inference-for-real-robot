import json
import threading
from types import SimpleNamespace
import pytest
from jointctl.models import RemoteStatus,EpisodeManifest,EpisodeState
from jointctl.remote import parse_unitree_status
from jointctl.controller import JointController
from jointctl.manifest import ManifestStore
from test_console_state import make_state


def raw_status(count=1,fault=None):
    payload={'format_version':2,'running':True,'ready':True,'quality_ok':not fault,
        'recording_state':'recording','fault':fault or [],'durable_complete':False,
        'streams':{n:{'received':count,'written':count,'durable':count,'pending_bytes':72,
            'capacity_bytes':100,'queue_warning':True,'last_received_monotonic_ns':1780000000000000001}
            for n in ('front','wrist')}}
    return parse_unitree_status('episode=/data/live\n'+json.dumps(payload))


def test_start_accepts_two_v2_streams_without_pkl(tmp_path):
    number=[0]
    def unitree(): number[0]+=1; return raw_status(number[0])
    ctl=JointController(SimpleNamespace(status=lambda:RemoteStatus('p450',True,'recording',active=True,
        episode_id='live',progress_name='session_bytes',progress_value=number[0]*100+100)),
        SimpleNamespace(status=unitree),ManifestStore(tmp_path),{'p450':None,'unitree':None},sleep=lambda _:None)
    status=ctl._wait_until_ready('live')['unitree']
    assert status.progress_name=='camera_records'
    assert status.streams['wrist']['written']>=3


def test_recovered_state_gap_can_start_without_erasing_quality_failure(tmp_path):
    number = [0]
    def unitree():
        number[0] += 1
        payload = json.loads(raw_status(number[0]).message.split('\n')[-1])
        payload.update(quality_ok=False, state_gap_ticks=38)
        return parse_unitree_status('episode=/data/live\n' + json.dumps(payload))
    ctl = JointController(SimpleNamespace(status=lambda:RemoteStatus('p450',True,'recording',active=True,
        episode_id='live',progress_name='session_bytes',progress_value=number[0]*100+100)),
        SimpleNamespace(status=unitree),ManifestStore(tmp_path),{'p450':None,'unitree':None},
        sleep=lambda _:None, readiness_timeout_s=1)
    status = ctl._wait_until_ready('live')['unitree']
    assert status.quality_ok is False
    assert status.state_gap_ticks == 38


def test_state_gap_does_not_excuse_rejected_camera_records(tmp_path):
    payload = json.loads(raw_status(4).message.split('\n')[-1])
    payload.update(quality_ok=False, state_gap_ticks=38)
    payload['streams']['front']['rejected'] = 1
    status = parse_unitree_status('episode=/data/live\n' + json.dumps(payload))
    ctl = JointController(SimpleNamespace(status=lambda:RemoteStatus('p450',True,'recording',active=True,
        episode_id='live',progress_name='session_bytes',progress_value=100)),
        SimpleNamespace(status=lambda:status),ManifestStore(tmp_path),{'p450':None,'unitree':None},
        sleep=lambda _:None, readiness_timeout_s=1)
    with pytest.raises(RuntimeError, match='rejected|写入|丢帧'):
        ctl._wait_until_ready('live')


@pytest.mark.parametrize('streams',[None,[],{'front':None,'wrist':{'written':1}}])
def test_malformed_v2_counters_are_reported_as_invalid_status(streams):
    with pytest.raises(ValueError,match='camera counters'):
        parse_unitree_status(json.dumps({'running':True,'format_version':2,'streams':streams}))


def test_start_rejects_one_stalled_raw_camera(tmp_path):
    number=[0]
    def unitree():
        number[0]+=1
        status=raw_status(number[0]); status.streams['wrist']['written']=1
        return status
    ctl=JointController(SimpleNamespace(status=lambda:RemoteStatus('p450',True,'recording',active=True,
        episode_id='live',progress_name='session_bytes',progress_value=number[0]*100+100)),
        SimpleNamespace(status=unitree),ManifestStore(tmp_path),{'p450':None,'unitree':None},sleep=lambda _:None,
        readiness_timeout_s=1)
    with pytest.raises(RuntimeError): ctl._wait_until_ready('live')


def test_remote_client_preserves_raw_health_after_ssh(monkeypatch):
    from jointctl.remote import RemoteClient
    from jointctl.models import CommandResult
    parsed=raw_status(7,['disk error'])
    monkeypatch.setattr('jointctl.remote.run_ssh',lambda *_:CommandResult('status',0,parsed.message,'',1))
    status=RemoteClient('test-unitree',kind='unitree').status()
    assert status.host=='test-unitree'
    assert status.format_version==2
    assert status.streams['front']['written']==7
    assert status.fault==['disk error']
    assert status.quality_ok is False


def fault_console(tmp_path,runner):
    state=make_state(tmp_path,runner=runner)
    state.store.create(EpisodeManifest('live','test','joint',1,EpisodeState.RECORDING))
    state.sample_status('p450',lambda:RemoteStatus('p450',True,'recording',active=True,episode_id='live'))
    state.sample_status('unitree',lambda:raw_status(3,['front writer: disk error']))
    return state


def test_fault_stops_joint_episode_once(tmp_path):
    calls=[]
    state=fault_console(tmp_path,lambda argv,directory:calls.append(argv) or 0)
    assert state.check_capture_fault()
    state.worker.join(3)
    assert not state.check_capture_fault()
    assert len(calls)==1 and 'stop' in calls[0]
    assert calls[0][-2:]==['--expected-episode','live']


def test_disconnected_peer_remains_unconfirmed(tmp_path):
    calls=[]
    state=fault_console(tmp_path,lambda argv,directory:calls.append(argv) or 3)
    state.sample_status('p450',lambda:RemoteStatus('p450',False,last_error='SSH timeout'))
    assert state.check_capture_fault()
    state.worker.join(3)
    snap=state.snapshot()
    assert snap['active']
    assert snap['hosts']['p450']['status']['active'] is None
    assert snap['job']['state']=='failed'
    assert not state.check_capture_fault()


def test_user_stop_race_is_idempotent(tmp_path):
    entered,release=threading.Event(),threading.Event()
    calls=[]
    def run(argv,directory):
        calls.append(argv); entered.set(); release.wait(3); return 0
    state=fault_console(tmp_path,run)
    state.submit('stop',{})
    assert entered.wait(1)
    try: assert not state.check_capture_fault()
    finally:
        release.set(); state.worker.join(3)
    assert len(calls)==1


def test_raw_closed_is_not_alignment_pass(tmp_path):
    state=make_state(tmp_path)
    status=raw_status(3)
    state.sample_status('unitree',lambda:status)
    snap=state.snapshot()
    assert snap['hosts']['unitree']['raw_capture']['format_version']==2
    assert snap['validation']['state']!='passed'


def test_fault_history_survives_idle_and_console_restart_without_leaking(tmp_path):
    state=fault_console(tmp_path,lambda *a:0)
    state.store.update('live',state=EpisodeState.STOPPING)
    state.store.update('live',state=EpisodeState.COMPLETE)
    state=make_state(tmp_path)
    state.sample_status('unitree',lambda:RemoteStatus('unitree',True,'idle',active=False))
    raw=state.snapshot()['hosts']['unitree'].get('raw_capture')
    assert raw and raw['fault']==['front writer: disk error']
    assert raw['historical'] and raw['quality_ok'] is False
    state.store.create(EpisodeManifest('next','test','joint',2,EpisodeState.STARTING))
    assert not state.snapshot()['hosts']['unitree'].get('raw_capture')


def test_stop_persists_final_raw_closure_and_quality(tmp_path):
    from test_stop_recovery import _Harness
    from jointctl.models import CommandResult
    harness=_Harness(tmp_path)
    final={'format_version':2,'raw_episode_id':'joint_active',
        'recording_state':'stopped','durable_complete':True,'quality_ok':False,
        'fault':['front writer: disk error'],'streams':{'front':{'durable':12}}}
    def stop(host): return CommandResult('stop',0,json.dumps(final) if host=='unitree' else '', '',1)
    harness.controller._stops=lambda targets:{h:stop(h) for h in targets}
    result=harness.controller.stop()
    assert result.metadata['raw_capture']['durable_complete']
    assert result.metadata['raw_capture']['quality_ok'] is False
    assert any('disk error' in d.message for d in result.diagnostics)


def test_ns_fields_not_roundtripped_as_js_numbers(tmp_path):
    state=make_state(tmp_path)
    state.sample_status('unitree',lambda:raw_status(3))
    stream=state.snapshot()['hosts']['unitree']['raw_capture']['streams']['front']
    assert stream['last_received_monotonic_ns']=='1780000000000000001'
    assert stream['pending_bytes']==72


def test_automatic_stop_cannot_stop_next_episode(tmp_path,monkeypatch):
    from jointctl.cli import main
    config=tmp_path/'config.json'
    config.write_text('{}')
    store=ManifestStore(tmp_path/'episodes')
    store.create(EpisodeManifest('next','test','joint',1,EpisodeState.RECORDING))
    def forbidden(*_): pytest.fail('stale automatic Stop must not reach remote controller')
    monkeypatch.setattr('jointctl.cli.make_controller',forbidden)
    assert main(['--config',str(config),'--manifest-root',str(store.root),'stop','--expected-episode','old'])==4


def test_v2_inspector_and_alignment_accept_camera_stream_names(tmp_path,monkeypatch):
    from dataclasses import replace
    from test_raw_reader import episode_v2
    from test_report import _run_remote_script
    from test_acceptance_quality import episode
    from jointctl.cli import build_alignment_report
    raw=episode_v2(tmp_path/'raw')
    summaries=_run_remote_script('unitree',raw,monkeypatch)
    assert {s['name'] for s in summaries}=={'camera:front','camera:wrist'}
    directory,streams=episode(tmp_path/'joint')
    streams['unitree']=[replace(s,name=s.name.replace('pkl:','camera:')) for s in streams['unitree']]
    assert build_alignment_report(directory,streams)['quality']['data_validated']
