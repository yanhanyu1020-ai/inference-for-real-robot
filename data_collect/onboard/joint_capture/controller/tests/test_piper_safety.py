"""Safety latches detect stale pairs, main-loop stalls and SDK logged failures."""
from types import SimpleNamespace
import pytest
from test_safe_teleop import Piper, controller
from piper_safety import ENABLE_IDS, FEEDBACK_IDS, SafetyGuard, monitored_piper


def test_sdk_none_return_with_error_log_is_a_send_failure():
    p = Piper(); guard = SafetyGuard(p)
    def failed(*args): p.logger.error('CAN transmission failed')
    p.JointCtrl = failed
    with pytest.raises(RuntimeError, match='CAN transmission failed'):
        guard.call('JointCtrl', *([0] * 6))
    guard.active = True; guard.poll()
    assert guard.inhibited and guard.stop_requested
    guard.close()


@pytest.mark.parametrize('missing', FEEDBACK_IDS + ENABLE_IDS)
def test_watchdog_stops_when_any_required_feedback_frame_is_stale(missing):
    p = Piper(); guard = SafetyGuard(p)
    guard.acknowledge(); guard.active = True
    p.fresh.remove(missing); guard.poll()
    assert guard.inhibited and guard.stop_requested
    assert [c[0] for c in p.calls] == ['MotionCtrl_1']
    guard.close()


def test_watchdog_stops_without_waiting_for_the_main_loop():
    p = Piper(); clock = [10.]
    guard = SafetyGuard(p, clock=lambda: clock[0])
    guard.acknowledge(); guard.active = True; clock[0] += .3
    guard.poll()
    assert guard.inhibited and guard.stop_requested
    assert guard.stop_attempts == 1
    guard.close()


def test_home_pause_never_disables_gravity_support_or_resets():
    ctl = controller(); ctl.interface.enabled = [True] * 6
    ctl._connect_and_enable_arm(); ctl._finish_enable(ctl.guard.clock())
    ctl.interface.calls.clear(); ctl._handle_button_events({'home': True})
    assert ctl.command_inhibited and ctl.guard.stop_requested
    assert [c[0] for c in ctl.interface.calls] == ['MotionCtrl_1']


def test_monitor_requires_each_valid_standard_feedback_frame():
    class Base:
        def ParseCANFrame(self, frame): pass
    p = monitored_piper(Base)()
    def frame(can_id, **kwargs):
        return SimpleNamespace(arbitration_id=can_id, data=b'\0' * 8,
            is_error_frame=False, is_remote_frame=False, is_extended_id=False, **kwargs)
    for i in FEEDBACK_IDS[:-1]: p.ParseCANFrame(frame(i))
    assert not p.feedback_fresh(FEEDBACK_IDS, .25)
    p.ParseCANFrame(frame(FEEDBACK_IDS[-1]))
    assert p.feedback_fresh(FEEDBACK_IDS, .25)
    p.feedback_received.clear()
    bad = frame(FEEDBACK_IDS[0]); bad.is_extended_id = True
    p.ParseCANFrame(bad)
    assert not p.feedback_fresh((bad.arbitration_id,), .25)
