from dataclasses import replace
import pytest

from jointctl.controller import JointController, EpisodeMismatch
from jointctl.manifest import ManifestStore
from jointctl.models import CommandResult, EpisodeManifest, EpisodeState, RemoteStatus


class Remote:
    def __init__(self, host):
        self.host = host
        self.current = RemoteStatus(host, True, 'recording', active=True, episode_id='episode')
        self.calls = 0
        self.fail = False

    def status(self):
        return self.current

    def stop(self):
        self.calls += 1
        if self.fail:
            return CommandResult('stop', -1, stderr='timeout')
        self.current = RemoteStatus(self.host, True, 'idle')
        return CommandResult('stop', 0)


def controller(tmp_path, state=EpisodeState.RECORDING, metadata=None):
    store = ManifestStore(tmp_path)
    store.create(EpisodeManifest('episode', 'test', 'joint', 1, state,
                                 t0_desktop_ns=10 if state == EpisodeState.RECORDING else None,
                                 metadata=metadata or {}))
    p, u = Remote('p450'), Remote('unitree')
    ctl = JointController(p, u, store, {'p450': object(), 'unitree': object()}, time_ns=lambda: 100)
    return ctl, p, u


def test_one_side_already_inactive_does_not_block_owned_survivor_stop(tmp_path):
    ctl, p, u = controller(tmp_path)
    p.current = RemoteStatus('p450', True, 'idle')
    result = ctl.stop()
    assert not u.current.active
    assert p.calls == 0 and u.calls == 1
    assert result.state == EpisodeState.COMPLETE
    assert any(d.phase == 'early_exit' for d in result.diagnostics)


def test_partial_stop_retains_owner_and_retry_only_stops_surviving_side(tmp_path):
    ctl, p, u = controller(tmp_path)
    u.fail = True
    assert ctl.stop().state == EpisodeState.PARTIAL
    assert ctl.store.active().episode_id == 'episode'
    u.fail = False
    ctl.time_ns = lambda: 200
    result = ctl.stop()
    assert result.state == EpisodeState.COMPLETE
    assert result.t1_desktop_ns == 100
    assert p.calls == 1 and u.calls == 2
    assert ctl.store.active() is None


def test_interrupted_start_can_be_cleaned_up(tmp_path):
    ctl, p, u = controller(tmp_path, EpisodeState.STARTING)
    result = ctl.stop()
    assert result.state == EpisodeState.COMPLETE
    assert not p.current.active and not u.current.active


def test_disconnected_side_does_not_block_reachable_owned_side(tmp_path):
    ctl, p, u = controller(tmp_path)
    p.current = RemoteStatus('p450', False, last_error='disconnected')
    result = ctl.stop()
    assert result.state == EpisodeState.PARTIAL
    assert p.calls == 0 and not u.current.active
    assert ctl.store.active().episode_id == 'episode'


def test_retry_preserves_prior_verified_idle_when_host_disconnects(tmp_path):
    ctl, p, u = controller(tmp_path)
    p.current = RemoteStatus('p450', False, last_error='p450 offline')
    first = ctl.stop()
    assert first.state == EpisodeState.PARTIAL
    assert first.stop_results['unitree'].ok

    p.current = RemoteStatus('p450', True, 'idle')
    u.current = RemoteStatus('unitree', False, last_error='unitree offline')
    second = ctl.stop()

    assert second.state == EpisodeState.COMPLETE
    assert second.stop_results['p450'].ok
    assert second.stop_results['unitree'].ok
    assert ctl.store.active() is None


def test_recovery_accepts_existing_owner_or_known_single_survivor(tmp_path):
    ctl, p, u = controller(tmp_path, EpisodeState.STARTING)
    assert ctl.recover().episode_id == 'episode'
    ctl.store.clear_active('episode')
    p.current = RemoteStatus('p450', True, 'idle')
    assert ctl.recover().episode_id == 'episode'
    assert ctl.stop().state == EpisodeState.COMPLETE


def test_other_episode_is_never_stopped(tmp_path):
    ctl, p, u = controller(tmp_path)
    p.current = replace(p.current, episode_id='foreign')
    with pytest.raises(EpisodeMismatch):
        ctl.stop()
    assert p.calls == u.calls == 0


def test_cleanup_does_not_race_a_live_start_process(tmp_path):
    ctl,p,u=controller(tmp_path,EpisodeState.STARTING,{'starter_pid':12345})
    ctl.pid_is_running=lambda pid: pid==12345
    with pytest.raises(EpisodeMismatch,match='startup is still running'):
        ctl.stop()
    assert p.calls == u.calls == 0


@pytest.mark.parametrize('pointer_exists',[True,False])
def test_recover_merges_observed_directories(tmp_path,pointer_exists,monkeypatch):
    ctl,p,u=controller(tmp_path,EpisodeState.STARTING)
    if not pointer_exists:
        ctl.store.clear_active('episode')
    monkeypatch.setattr(ctl,'_directories_from_statuses',lambda statuses:{'p450':'/p450/episode','unitree':'/unitree/episode'})
    assert ctl.recover().remote_directories == {'p450':'/p450/episode','unitree':'/unitree/episode'}


def test_clock_model_rejection_preserves_remote_status(tmp_path,monkeypatch):
    from jointctl.models import ClockSample
    ctl,p,u=controller(tmp_path)
    sample=ClockSample.from_exchange(host='p450',sequence=0,local_send_wall_ns=90,
        local_send_mono_ns=90,local_receive_wall_ns=100,local_receive_mono_ns=100,
        remote_receive_wall_ns=95,remote_send_wall_ns=95,remote_monotonic_ns=95)
    monkeypatch.setattr('jointctl.controller.read_clock_records',lambda *args:({'p450':[sample],'unitree':[]},{'p450':[],'unitree':[]}))
    def inconsistent(*args,**kwargs):
        raise ValueError('clock model inconsistent with probe sequence 34')
    monkeypatch.setattr('jointctl.controller.build_clock_timeline',inconsistent)
    report=ctl.status()
    assert len(report.remotes)==2
    assert report.timing_degraded
    assert any('inconsistent' in r for r in report.timing_degradation_reasons)
