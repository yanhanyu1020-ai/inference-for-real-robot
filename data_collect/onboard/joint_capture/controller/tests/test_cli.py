from __future__ import annotations

import json
import io
from pathlib import Path

import pytest

from jointctl.cli import main
from jointctl.controller import JointStartFailed
from jointctl.clock_sync import ClockProbeError
from jointctl.manifest import ManifestStore
from jointctl.models import EpisodeManifest, EpisodeState


def test_successful_stop_defers_alignment_and_video_export(tmp_path, monkeypatch, capsys):
    from jointctl.models import CommandResult
    from types import SimpleNamespace
    store=ManifestStore(tmp_path)
    manifest=EpisodeManifest('stopped', 'test', 'joint', 1, EpisodeState.COMPLETE,
        metadata={'postprocess':'pending'},
        stop_results={h:CommandResult('stop',0) for h in ('p450','unitree')},
        clock_monitor_closed_cleanly=True)
    store.create(manifest)
    monkeypatch.setattr('jointctl.cli.make_controller',lambda _:SimpleNamespace(stop=lambda:manifest))
    monkeypatch.setattr('jointctl.cli.write_alignment_report', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('alignment must be deferred')))
    assert main(['--manifest-root',str(tmp_path),'stop']) == 0
    output = capsys.readouterr().err
    assert '待后处理' in output
    assert '停止成功不代表验收通过' in output


def test_finalize_command_uses_oldest_pending_episode(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from jointctl.models import CommandResult, RemoteStatus
    store = ManifestStore(tmp_path)
    manifest = EpisodeManifest('old', 'test', 'joint', 1, EpisodeState.COMPLETE,
        metadata={'postprocess': 'pending'},
        remote_directories={'p450': '/raw/p450', 'unitree': '/raw/unitree'})
    store.create(manifest)
    store.clear_active('old')
    called = []
    class Remote:
        def __init__(self, host): self.host = host
        def status(self): return RemoteStatus(self.host, True, 'idle')
        def finalize_raw(self, path):
            called.append((self.host, path))
            return CommandResult('finalize', 0)
    monkeypatch.setattr('jointctl.cli.make_controller', lambda _:
        SimpleNamespace(remotes={host: Remote(host) for host in ('p450', 'unitree')}))
    monkeypatch.setattr('jointctl.cli._run_alignment_for_finalization', lambda _args, item: 0)
    assert main(['--manifest-root', str(tmp_path), 'finalize']) == 0
    assert called == [('p450', '/raw/p450'), ('unitree', '/raw/unitree')]
    assert store.load('old').metadata['postprocess'] == 'passed'


def test_controller_allows_sensor_start_and_export_longer_than_probe_timeout(tmp_path, monkeypatch):
    from argparse import Namespace
    from jointctl.cli import make_controller
    from jointctl.models import CommandResult
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'ssh_timeout_s': 3, 'start_timeout_s': 120, 'stop_timeout_s': 300}))
    controller = make_controller(Namespace(config=config, manifest_root=tmp_path/'episodes'))
    def simulated_command(host, command, timeout):
        required = 180 if command.endswith(' finish') or command.endswith(' stop') else 50
        return CommandResult(command, 0 if timeout >= required else -1)
    monkeypatch.setattr('jointctl.remote.run_ssh', simulated_command)
    assert controller.remotes['unitree'].start('test', 'test', 'joint').ok
    assert controller.remotes['p450'].stop().ok
    assert controller.clock_probes['p450'].timeout_s == 3


def test_status_returns_nonzero_for_remote_conflict(tmp_path, monkeypatch, capsys):
    store = ManifestStore(tmp_path)
    store.create(EpisodeManifest("joint_cli", "demo", "joint", 1, EpisodeState.RECORDING))

    class Remote:
        def __init__(self, host):
            self.host = host
        def status(self):
            from jointctl.models import RemoteStatus
            return RemoteStatus(self.host, True, "recording", active=True,
                                episode_id="foreign", progress_name="session_bytes")
        def start(self, *args): raise AssertionError
        def stop(self): raise AssertionError

    monkeypatch.setattr("jointctl.cli.make_controller", lambda _args: __import__(
        "jointctl.controller", fromlist=["JointController"]
    ).JointController(Remote("p450"), Remote("unitree"), store,
                      {"p450": object(), "unitree": object()}))
    result = main(["--manifest-root", str(tmp_path), "status"])
    captured = capsys.readouterr()
    assert result == 4
    assert "episode mismatch" in captured.err.lower()


def test_invalid_command_configuration_returns_usage_code(tmp_path, capsys):
    result = main(["--config", str(tmp_path / 'missing.json'),
                   "--manifest-root", str(tmp_path), "start"])
    captured = capsys.readouterr()
    assert result == 2
    assert captured.err


def test_inspector_transport_failure_is_connectivity_exit(monkeypatch, tmp_path, capsys):
    from jointctl.models import CommandResult
    from jointctl.inspectors import P450Inspector
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: object())
    monkeypatch.setattr("jointctl.cli.P450Inspector", lambda *args, **kwargs: P450Inspector(
        runner=lambda *_: CommandResult("ssh", 255, stderr="connection refused")
    ))
    episode = tmp_path / "joint"; episode.mkdir()
    (episode / "manifest.json").write_text(json.dumps(EpisodeManifest(
        "joint", "demo", "joint", 1, remote_directories={"p450": "/remote"}
    ).to_dict()), encoding="utf-8")
    assert main(["--manifest-root", str(tmp_path), "align", "--episode-dir", str(episode)]) == 3
    assert "connectivity" in capsys.readouterr().err.lower()


def test_inspector_remote_command_failure_is_remote_exit(monkeypatch, tmp_path, capsys):
    from jointctl.models import CommandResult
    from jointctl.inspectors import P450Inspector
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: object())
    monkeypatch.setattr("jointctl.cli.P450Inspector", lambda *args, **kwargs: P450Inspector(
        runner=lambda *_: CommandResult("ssh", 7, stderr="rosbag missing")
    ))
    episode = tmp_path / "joint"; episode.mkdir()
    (episode / "manifest.json").write_text(json.dumps(EpisodeManifest(
        "joint", "demo", "joint", 1, remote_directories={"p450": "/remote"}
    ).to_dict()), encoding="utf-8")
    assert main(["--manifest-root", str(tmp_path), "align", "--episode-dir", str(episode)]) == 6
    assert "remote command" in capsys.readouterr().err.lower()


def test_start_exit_uses_structured_failure_kind_not_message_text(monkeypatch, capsys):
    class Controller:
        def start(self, *_args):
            raise JointStartFailed("remote preflight failed", failure_kind="remote_command")
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: Controller())
    assert main(["start", "--instruction", "demo", "--task", "joint"]) == 6
    assert "remote operation failed" in capsys.readouterr().err.lower()


def test_status_protocol_failure_returns_remote_command_exit(monkeypatch, capsys):
    from jointctl.models import JointStatusReport, RemoteStatus
    class Controller:
        def status(self):
            return JointStatusReport(None, EpisodeState.PARTIAL, [
                RemoteStatus("p450", True, "unknown", last_error="status protocol parse failed"),
                RemoteStatus("unitree", True, "idle"),
            ])
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: Controller())
    assert main(["status"]) == 6
    assert "remote command" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("failure_kind, expected", [("connectivity", 3), ("remote_command", 6)])
def test_probe_exit_uses_structured_clock_probe_failure_kind(monkeypatch, capsys, failure_kind, expected):
    class Probe:
        def __init__(self, *_args, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def sample(self):
            raise ClockProbeError("fake probe failure", failure_kind=failure_kind)
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: object())
    monkeypatch.setattr("jointctl.cli.ClockProbe", Probe)
    assert main(["probe", "--host", "p450"]) == expected


def test_start_without_arguments_does_not_prompt_in_interactive_terminal(monkeypatch, capsys):
    received = {}

    class Interactive(io.StringIO):
        def isatty(self):
            return True

    class Controller:
        def start(self, instruction, task):
            received.update(instruction=instruction, task=task)
            from dataclasses import replace
            return replace(EpisodeManifest.new("joint_prompt", instruction, task),
                           state=EpisodeState.RECORDING, t0_desktop_ns=1)

    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: Controller())
    monkeypatch.setattr("jointctl.cli.sys.stdin", Interactive())
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("must not prompt"))

    assert main(["start"]) == 0
    assert received == {"instruction": "", "task": ""}
    output = capsys.readouterr()
    assert "joint_prompt" in output.out
    assert "联合采集已开始" in output.err


def test_start_without_arguments_works_when_not_interactive(monkeypatch, capsys):
    class NonInteractive(io.StringIO):
        def isatty(self):
            return False

    monkeypatch.setattr("jointctl.cli.sys.stdin", NonInteractive())
    received = {}
    class Controller:
        def start(self, instruction, task):
            received.update(instruction=instruction, task=task)
            from dataclasses import replace
            return replace(EpisodeManifest.new("joint_auto", instruction, task),
                           state=EpisodeState.RECORDING, t0_desktop_ns=1)
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: Controller())
    assert main(["start"]) == 0
    assert received == {"instruction": "", "task": ""}
    assert "Task:" not in capsys.readouterr().err


def test_multi_host_duration_probe_emits_quality_summary(monkeypatch, capsys):
    from jointctl.models import ClockSample

    def sample(host, sequence, remote_ns, offset_ns, rtt_ns):
        return ClockSample(
            host=host, sequence=sequence,
            local_send_wall_ns=remote_ns - offset_ns - rtt_ns // 2,
            local_send_mono_ns=remote_ns - rtt_ns // 2,
            remote_receive_wall_ns=remote_ns, remote_send_wall_ns=remote_ns,
            remote_monotonic_ns=remote_ns,
            local_receive_wall_ns=remote_ns - offset_ns + rtt_ns // 2,
            local_receive_mono_ns=remote_ns + rtt_ns // 2,
            rtt_ns=rtt_ns, offset_ns=offset_ns,
        )

    samples = {
        "p450": [
            sample("p450", 0, 1_000_000_000, 10_000_000, 2_000_000),
            sample("p450", 1, 11_000_000_000, 5_000_000, 3_000_000),
            sample("p450", 2, 21_000_000_000, 0, 50_000_000),
        ],
        "unitree": [
            sample("unitree", 0, 1_000_000_000, -2_000_000, 4_000_000),
            sample("unitree", 1, 11_000_000_000, -1_580_000, 5_000_000),
            sample("unitree", 2, 21_000_000_000, -1_160_000, 6_000_000),
        ],
    }
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: object())
    monkeypatch.setattr(
        "jointctl.cli.run_probe_session",
        lambda hosts, duration_s, interval_s: (samples, {host: [] for host in hosts}),
    )

    assert main(["probe", "--duration", "600", "--hosts", "p450", "unitree"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["duration_s"] == 600.0
    assert report["hosts"]["p450"]["sample_count"] == 3
    assert report["hosts"]["p450"]["rtt_ns"] == {
        "min": 2_000_000, "median": 3_000_000, "p95": 50_000_000,
    }
    assert report["hosts"]["p450"]["offset_drift_ns"] == -10_000_000
    assert report["hosts"]["p450"]["outlier_percentage"] > 0
    assert report["hosts"]["unitree"]["drift_ppm"] == 42
    assert report["hosts"]["unitree"]["fit_residual_ns"]["p95"] >= 0
    assert report["estimated_cross_host_uncertainty_ns"] >= 0


def test_duration_probe_keeps_hosts_independent_when_one_sample_fails(tmp_path):
    from jointctl.cli import run_probe_session
    from jointctl.models import ClockSample

    class Probe:
        counts = {"p450": 0, "unitree": 0}

        def __init__(self, host):
            self.host = host
        def open(self):
            return self
        def close(self):
            pass
        def sample(self):
            Probe.counts[self.host] += 1
            if self.host == "p450":
                raise ClockProbeError("simulated timeout", failure_kind="connectivity")
            n = Probe.counts[self.host]
            remote = n * 1_000_000
            return ClockSample(
                self.host, n, remote, remote, remote, remote, remote,
                remote, remote, 0, 0,
            )

    samples, errors = run_probe_session(
        ("p450", "unitree"), duration_s=0.03, interval_s=0.001,
        probe_factory=Probe,
    )
    assert not samples["p450"]
    assert len(errors["p450"]) >= 1
    assert len(samples["unitree"]) >= 2
