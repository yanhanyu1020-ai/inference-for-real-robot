from jointctl.remote import RemoteClient
from jointctl.models import CommandResult


def test_p450_prepares_before_recording(monkeypatch):
    calls=[]
    def run(host, command, timeout):
        calls.append(command)
        return CommandResult(command,0)
    monkeypatch.setattr('jointctl.remote.run_ssh',run)
    remote=RemoteClient('p450',prepare_before_start=True)
    assert remote.start('ep').ok
    assert calls == ['/home/amov/bin/p450_capture prepare', '/home/amov/bin/p450_capture start ep']


def test_can_service_failure_never_starts_recorder(monkeypatch):
    calls=[]
    def run(host, command, timeout):
        calls.append(command)
        return CommandResult(command,1,stderr='service failed')
    monkeypatch.setattr('jointctl.remote.run_ssh',run)
    result=RemoteClient('unitree',prepare_before_start=True).start('ep','test','test')
    assert not result.ok
    assert calls == ['~/heterovla-collection/onboard/collection_ctl.sh prepare']
    assert 'preparation failed' in result.stderr


def test_remote_utf8_logs_decode_on_windows():
    import sys
    from jointctl.remote import run_process
    result=run_process([sys.executable,'-c',"import sys; sys.stdout.buffer.write(bytes.fromhex('e8bf9ee68ea5'))"],5)
    assert result.ok
    assert result.stdout == '连接'


def test_prepare_cli_never_starts_recording(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from jointctl.cli import _run
    from jointctl.models import RemoteStatus
    calls = []
    ctl = SimpleNamespace(
        store=SimpleNamespace(active=lambda: None),
        _statuses=lambda: {h: RemoteStatus(h, True, 'idle') for h in ('p450', 'unitree')},
        remotes={h: SimpleNamespace(prepare=lambda h=h: calls.append(h) or CommandResult('prepare', 0))
                 for h in ('p450', 'unitree')})
    monkeypatch.setattr('jointctl.cli.make_controller', lambda args: ctl)
    assert _run(SimpleNamespace(command='prepare')) == 0
    assert sorted(calls) == ['p450', 'unitree']


def test_prepare_cli_refuses_existing_capture(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from jointctl.cli import _run
    ctl = SimpleNamespace(store=SimpleNamespace(active=lambda: object()))
    monkeypatch.setattr('jointctl.cli.make_controller', lambda args: ctl)
    with pytest.raises(ValueError, match='活动会话'):
        _run(SimpleNamespace(command='prepare'))
