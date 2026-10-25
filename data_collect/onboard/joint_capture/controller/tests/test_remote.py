from pathlib import Path
import json

import pytest

from jointctl.models import CommandResult
from jointctl.remote import (
    RemoteClient,
    parse_p450_status,
    parse_unitree_status,
    run_ssh,
    run_process,
)


@pytest.fixture
def fixtures():
    return Path(__file__).parent / "fixtures"


def test_parse_p450_status_extracts_episode_and_progress(fixtures):
    status = parse_p450_status((fixtures / "p450_status.json").read_text())
    assert status.active is True
    assert status.episode_id == "joint_03"
    assert status.progress_value > 0


def test_parse_unitree_status_extracts_episode_and_frames(fixtures):
    status = parse_unitree_status((fixtures / "unitree_status.txt").read_text())
    assert status.active is True
    assert status.episode_id == "joint_03"
    assert status.progress_name == "frames_saved"


def test_unitree_cleanup_pending_is_not_idle():
    status = parse_unitree_status('episode=/data/old\n' + json.dumps(
        {'running': False, 'cleanup_pending': True, 'frames_saved': 10}))
    assert status.active
    assert status.state == 'cleanup_pending'
    assert status.episode_id == 'old'


def test_run_process_keeps_stderr_separate(tmp_path):
    script = tmp_path / "emit.py"
    script.write_text("import sys; print('stdout', end=''); print('warning', file=sys.stderr, end='')")
    result = run_process(["python", str(script)], timeout_s=2)
    assert result.stdout == "stdout"
    assert result.stderr == "warning"
    assert result.exit_code == 0


def test_run_ssh_delegates_to_ssh_argv(monkeypatch):
    seen = {}
    monkeypatch.setattr("jointctl.remote.run_process", lambda argv, timeout_s: seen.update(argv=list(argv), timeout=timeout_s) or CommandResult("ssh", 0))
    result = run_ssh("p450", "echo status", 3)
    assert seen == {"argv": ["ssh", "-n", "-T", "-o", "BatchMode=yes", "-o",
                             "ConnectTimeout=5", "p450", "echo status"], "timeout": 3}
    assert result.ok


def test_run_ssh_allows_slow_banner_for_long_operations(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "jointctl.remote.run_process",
        lambda argv, timeout_s: seen.update(argv=list(argv), timeout=timeout_s)
        or CommandResult("ssh", 0),
    )

    run_ssh("unitree", "status", 30)

    assert "ConnectTimeout=20" in seen["argv"]
    assert seen["timeout"] == 30


def test_unitree_start_quotes_instruction_and_task(monkeypatch):
    seen = {}
    monkeypatch.setattr("jointctl.remote.run_ssh", lambda host, command, timeout: seen.update(host=host, command=command, timeout=timeout) or CommandResult(command, 0))
    RemoteClient("unitree", timeout_s=4).start("joint_03", "pick 'red'", "a; b")
    assert seen["command"] == "~/heterovla-collection/onboard/collection_ctl.sh start joint_03 'pick '\"'\"'red'\"'\"'' 'a; b'"


def test_p450_stop_uses_fast_command(monkeypatch):
    seen = {}
    monkeypatch.setattr("jointctl.remote.run_ssh", lambda host, command, timeout: seen.update(command=command) or CommandResult(command, 0))
    RemoteClient("p450").stop()
    assert seen["command"] == "/home/amov/bin/p450_capture stop-fast"


def test_postprocess_uses_long_timeout_and_quotes_directory(monkeypatch):
    seen = {}
    monkeypatch.setattr("jointctl.remote.run_ssh", lambda host, command, timeout: seen.update(command=command, timeout=timeout) or CommandResult(command, 0))
    RemoteClient("unitree", stop_timeout_s=300).finalize_raw("/data/episode one")
    assert seen["command"] == "~/heterovla-collection/onboard/collection_ctl.sh finalize '/data/episode one'"
    assert seen["timeout"] >= 3600


@pytest.mark.parametrize("returncode, reachable", [(1, True), (255, False), (-1, False)])
def test_status_distinguishes_remote_failure_from_transport_failure(monkeypatch, returncode, reachable):
    result = CommandResult("ssh p450 status", returncode, "partial", "status failed")
    monkeypatch.setattr("jointctl.remote.run_ssh", lambda *args: result)
    status = RemoteClient("p450").status()
    assert status.reachable is reachable
    assert status.active is False
    assert "status failed" in status.last_error
    assert str(returncode) in status.last_error


def test_status_parse_failure_is_reachable_remote_protocol_failure(monkeypatch):
    monkeypatch.setattr("jointctl.remote.run_ssh", lambda *args: CommandResult("ssh", 0, "not json"))
    status = RemoteClient("p450").status()
    assert status.reachable is True
    assert status.last_error is not None
    assert "protocol" in status.last_error


def test_real_unitree_pretty_status_and_idle():
    text = ('episode=/data/joint/live\ngo2_pid=123\n'
            + json.dumps({"valid": False, "gamepad": {"connected": False}}, indent=4)
            + '\n' + json.dumps({"running": True, "frames_saved": 42}, indent=4)
            + '\nFilesystem Size Used Avail\n')
    status = parse_unitree_status(text)
    assert status.active and status.episode_id == 'live' and status.progress_value == 42
    idle = parse_unitree_status('stopped\n')
    assert not idle.active and idle.episode_id is None and idle.last_error is None


def test_p450_status_with_ros_warning_preserves_directory(fixtures):
    from jointctl.controller import JointController
    payload = json.loads((fixtures / 'p450_status.json').read_text())
    text = 'Unable to register with master node [http://localhost:11311]\n' + json.dumps(payload, indent=2)
    status = parse_p450_status(text)
    assert status.active and status.episode_id == 'joint_03'
    assert JointController._directory_from_text('p450', text) == '/data/joint_03'


def test_status_rejects_conflicting_json_objects():
    with pytest.raises(ValueError):
        parse_unitree_status('episode=/data/x\n{"running":true}\n{"running":false}\n')


def test_p450_clean_idle_has_null_capture():
    status = parse_p450_status('{"capture":null,"recorder":{"active":false,"stale":false}}')
    assert not status.active and status.episode_id is None
