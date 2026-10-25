from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[1]


def test_installer_backs_up_prior_install_preserves_data_and_writes_absolute_python(tmp_path):
    desktop = tmp_path / "desktop with spaces"
    prior = desktop / "JointCapture"
    prior.mkdir(parents=True)
    (prior / "old_counter.txt").write_text("41", encoding="utf-8")
    (desktop / "JointStart.bat").write_text("old start", encoding="utf-8")
    (desktop / "joint_start.ps1").write_text("old ps start", encoding="utf-8")
    (desktop / "Joint_episode_counter.txt").write_text("41", encoding="utf-8")
    raw = desktop / "joint_manifests" / "raw" / "old.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"unchanged")
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text("@exit /b 17\n", encoding="utf-8")

    result = subprocess.run(
        ["powershell", "-NoProfile", "-File", str(REPOSITORY / "scripts" / "Install-JointCapture.ps1"),
         "-TargetRoot", str(desktop), "-SourceRoot", str(REPOSITORY),
         "-PythonPath", str(fake_python)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    backups = list(desktop.glob("JointCaptureBackup-*"))
    assert len(backups) == 1
    assert (backups[0] / "JointCapture" / "old_counter.txt").read_text(encoding="utf-8") == "41"
    assert (backups[0] / "JointStart.bat").read_text(encoding="utf-8") == "old start"
    assert (backups[0] / "joint_start.ps1").read_text(encoding="utf-8") == "old ps start"
    assert (backups[0] / "Joint_episode_counter.txt").read_text(encoding="utf-8") == "41"
    assert (desktop / "Joint_episode_counter.txt").read_text(encoding="utf-8") == "41"
    assert raw.read_bytes() == b"unchanged"
    installed_python = (desktop / "JointCapture" / "python_path.txt").read_text(encoding="utf-8").strip()
    assert installed_python == str(fake_python.resolve())
    assert (desktop / "JointCapture" / "src" / "jointctl" / "__main__.py").is_file()
    assert all((desktop / name).is_file() for name in (
        "JointStart.bat", "JointStop.bat", "JointStatus.bat", "JointRecover.bat"
    ))


def test_installed_batch_wrappers_work_from_any_cwd_propagate_pythonpath_and_exit_code(tmp_path):
    desktop = tmp_path / "desktop with spaces"
    manifest_root = tmp_path / "local records"
    log = tmp_path / "fake-python.log"
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "@echo %PYTHONPATH%^|%*^|%CD%>>\"%FAKE_PYTHON_LOG%\"\n@exit /b 17\n",
        encoding="utf-8",
    )
    install = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(REPOSITORY / "scripts" / "Install-JointCapture.ps1"),
         "-TargetRoot", str(desktop), "-SourceRoot", str(REPOSITORY),
         "-PythonPath", str(fake_python), "-ManifestRoot", str(manifest_root)],
        text=True, capture_output=True, check=False,
    )
    assert install.returncode == 0, install.stderr
    assert (desktop / "JointCapture" / "manifest_root.txt").read_text(encoding="utf-8").strip() == str(manifest_root)
    unrelated = tmp_path / "unrelated cwd"
    unrelated.mkdir()
    env = {**os.environ, "JOINTCTL_NO_PAUSE": "1", "FAKE_PYTHON_LOG": str(log)}
    for name, command in (
        ("JointStart.bat", "start"), ("JointStop.bat", "stop"),
        ("JointStatus.bat", "status"), ("JointRecover.bat", "recover"),
    ):
        result = subprocess.run(
            ["cmd", "/d", "/c", str(desktop / name)], cwd=unrelated, env=env,
            text=True, capture_output=True, check=False,
        )
        assert result.returncode == 17, (name, result.stdout, result.stderr)
    rows = log.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    for row, command in zip(rows, ("start", "stop", "status", "recover")):
        pythonpath, arguments, cwd = row.split("|", 2)
        assert pythonpath.split(os.pathsep)[0] == str(desktop / "JointCapture" / "src")
        assert "-m jointctl" in arguments
        assert f'--manifest-root "{manifest_root}"' in arguments
        assert arguments.rstrip().endswith(command)
        assert Path(cwd) == unrelated
