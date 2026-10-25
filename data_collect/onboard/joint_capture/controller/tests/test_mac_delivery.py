from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


CONTROLLER = Path(__file__).resolve().parents[1]
REPOSITORY = CONTROLLER.parent
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX macOS launchers")


@pytest.fixture
def mac_delivery(tmp_path):
    # Finder paths commonly contain both spaces and non-ASCII characters.
    repository = tmp_path / "采集项目 with spaces"
    app = repository / "controller"
    scripts = app / "scripts"
    scripts.mkdir(parents=True)
    for name in ("Install-JointCapture.command", "JointConsole.command"):
        shutil.copyfile(CONTROLLER / "scripts" / name, scripts / name)
    shutil.copyfile(REPOSITORY / "JointConsole-macOS.command", repository / "JointConsole-macOS.command")
    fake_python = tmp_path / "Python 工具" / "python"
    fake_python.parent.mkdir()
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, shutil, sys\n"
        "with open(os.environ['FAKE_PYTHON_LOG'], 'a', encoding='utf-8') as log:\n"
        "    log.write(json.dumps({'args':sys.argv[1:], 'pythonpath':os.environ.get('PYTHONPATH'), 'cwd':os.getcwd()}) + '\\n')\n"
        "if sys.argv[1] == '-c':\n"
        "    sys.exit(int(os.environ.get('FAKE_PYTHON_TOO_OLD', '0')))\n"
        "if sys.argv[1:3] == ['-m', 'venv']:\n"
        "    target = pathlib.Path(sys.argv[-1]) / 'bin' / 'python'\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    if pathlib.Path(sys.argv[0]).resolve() != target.resolve():\n"
        "        shutil.copyfile(sys.argv[0], target)\n"
        "        target.chmod(0o755)\n"
        "    sys.exit(0)\n"
        "if sys.argv[1:4] == ['-m', 'jointctl.macos', 'setup']:\n"
        "    app = pathlib.Path(sys.argv[sys.argv.index('--app-root') + 1])\n"
        "    (app / 'macos_paths.json').write_text('{}', encoding='utf-8')\n"
        "    sys.exit(0)\n"
        "sys.exit(int(os.environ.get('FAKE_LAUNCH_EXIT', '0')))\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    unrelated = tmp_path / "unrelated folder"
    unrelated.mkdir()
    env = {
        **os.environ,
        "JOINTCAPTURE_PYTHON": str(fake_python),
        "JOINTCTL_NO_PAUSE": "1",
        "FAKE_PYTHON_LOG": str(log),
        "PYTHONPATH": "existing-source-path",
    }
    return repository, app, unrelated, log, env


def run_script(script, cwd, env, *args):
    return subprocess.run(
        ["/bin/bash", str(script), *args], cwd=cwd, env=env,
        text=True, capture_output=True, timeout=15, check=False,
    )


def read_calls(log):
    return [json.loads(row) for row in log.read_text(encoding="utf-8").splitlines()]


def test_mac_installer_uses_local_venv_and_preserves_setup_arguments(mac_delivery):
    repository, app, cwd, log, env = mac_delivery
    config = str(cwd / "用户 config.json")
    manifests = str(cwd / "saved episodes")
    result = run_script(
        app / "scripts" / "Install-JointCapture.command", cwd, env,
        "--config", config, "--manifest-root", manifests,
    )
    assert result.returncode == 0, result.stderr
    calls = read_calls(log)
    assert calls[1]["args"] == ["-m", "venv", "--without-pip", str(app / ".venv-macos")]
    assert calls[2]["args"] == [
        "-m", "jointctl.macos", "setup", "--app-root", str(app),
        "--config", config, "--manifest-root", manifests,
    ]
    assert calls[2]["pythonpath"].split(os.pathsep)[0] == str(app / "src")
    assert all(call["cwd"] == str(cwd) for call in calls)
    assert (app / "macos_paths.json").is_file()
    assert (app / ".venv-macos" / "bin" / "python").is_file()


def test_mac_launcher_requires_setup_without_downloading_anything(mac_delivery):
    _, app, cwd, log, env = mac_delivery
    result = run_script(app / "scripts" / "JointConsole.command", cwd, env)
    assert result.returncode == 2
    assert "Install-JointCapture.command" in result.stderr
    assert not log.exists()


def test_mac_root_launcher_runs_setup_once_and_propagates_launch_failure(mac_delivery):
    repository, app, cwd, log, env = mac_delivery
    launcher = repository / "JointConsole-macOS.command"
    first = run_script(launcher, cwd, env, "--no-browser")
    assert first.returncode == 0, first.stderr
    calls = read_calls(log)
    assert calls[-1]["args"] == ["-m", "jointctl.macos", "launch", "--app-root", str(app), "--no-browser"]
    assert calls[-1]["pythonpath"].split(os.pathsep)[0] == str(app / "src")
    assert len([call for call in calls if call["args"][2:3] == ["setup"]]) == 1
    log.unlink()
    second = run_script(launcher, cwd, {**env, "FAKE_LAUNCH_EXIT": "17"}, "--no-browser")
    assert second.returncode == 17
    calls = read_calls(log)
    assert len(calls) == 1
    assert calls[0]["args"][2] == "launch"


def test_mac_installer_rejects_explicit_outdated_python_before_creating_state(mac_delivery):
    _, app, cwd, log, env = mac_delivery
    result = run_script(
        app / "scripts" / "Install-JointCapture.command", cwd,
        {**env, "FAKE_PYTHON_TOO_OLD": "1"},
    )
    assert result.returncode == 2
    assert "Python 3.10 or newer" in result.stderr
    assert len(read_calls(log)) == 1
    assert not (app / ".venv-macos").exists()
    assert not (app / "macos_paths.json").exists()
