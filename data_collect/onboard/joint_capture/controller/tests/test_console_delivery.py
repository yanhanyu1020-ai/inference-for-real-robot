import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.skipif(os.name != 'nt',reason='Windows installer')
def test_installer_delivers_console_and_preserves_existing_data(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    target=tmp_path/'desktop'; target.mkdir()
    records=target/'joint_manifests'; records.mkdir()
    (records/'sentinel').write_text('keep')
    script=repo/'scripts/Install-JointCapture.ps1'
    args=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),
          '-TargetRoot',str(target),'-PythonPath',sys.executable]
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (target/'JointConsole.bat').is_file()
    assert (target/'JointCapture/src/jointctl/console.py').is_file()
    assert (target/'JointCapture/src/jointctl/preview.py').is_file()
    assert (target/'JointCapture/src/jointctl/console_web/preview.js').is_file()
    assert (records/'sentinel').read_text()=='keep'


def test_launcher_refuses_another_configuration(tmp_path,monkeypatch):
    from jointctl.console import launch
    monkeypatch.setattr('jointctl.console.health',lambda port:{'identity':'joint-capture-console-v1',
        'root':'other-root','config':'other-config'})
    with pytest.raises(RuntimeError):
        launch(tmp_path/'config',tmp_path/'root',8765,open_browser=False)


def test_cli_busy_lock_prevents_remote_mutation(tmp_path,monkeypatch):
    from jointctl.cli import main
    from jointctl.operation_lock import operation_lock
    def forbidden(*args):
        pytest.fail('remote controller must not be created when lock is held')
    monkeypatch.setattr('jointctl.cli.make_controller',forbidden)
    with operation_lock(tmp_path):
        assert main(['--manifest-root',str(tmp_path),'stop'])==4


@pytest.mark.skipif(os.name != 'nt',reason='Windows batch')
def test_console_wrapper_works_outside_desktop_and_propagates_failure(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    desktop=tmp_path/'desktop with spaces'; desktop.mkdir()
    app=desktop/'JointCapture'; app.mkdir()
    fake=tmp_path/'fake-python.cmd'
    fake.write_text('@echo %*\n@exit /b 17\n')
    (app/'python_path.txt').write_text(str(fake))
    manifest_root=tmp_path/'external records'
    (app/'manifest_root.txt').write_text(str(manifest_root))
    import shutil
    shutil.copyfile(repo/'scripts/JointConsole.bat',desktop/'JointConsole.bat')
    result=subprocess.run(['cmd','/d','/c',str(desktop/'JointConsole.bat')],cwd=tmp_path,
        env={**os.environ,'JOINTCTL_NO_PAUSE':'1'},capture_output=True,text=True,timeout=10)
    assert result.returncode==17
    assert '-m jointctl.console' in result.stdout
    assert str(manifest_root) in result.stdout
