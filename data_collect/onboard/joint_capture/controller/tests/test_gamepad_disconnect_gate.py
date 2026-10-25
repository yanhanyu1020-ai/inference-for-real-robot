"""The collection wrapper must not replay commands after the USB gamepad disappears."""

from piper_gamepad_teleop import send_command


class ForbiddenPiper:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected robot command: {name}")


def test_disconnected_gamepad_suppresses_commands_despite_stale_enabled_state():
    stale_state = {"arm_connected": True, "arm_enabled": True}
    assert send_command(ForbiddenPiper(), object(), stale_state,
                        gamepad_connected=False) is False


def test_reconnected_gamepad_does_not_clear_latched_inhibit():
    from types import SimpleNamespace
    state = {"arm_connected": True, "arm_enabled": True}
    assert send_command(ForbiddenPiper(), SimpleNamespace(command_inhibited=True), state,
                        gamepad_connected=True) is False
