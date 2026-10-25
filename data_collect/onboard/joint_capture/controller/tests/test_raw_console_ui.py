from pathlib import Path
import subprocess


def test_raw_console_states():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['node',str(root/'tests/raw_console_dom.cjs'),
        str(root/'src/jointctl/console_web/app.js')],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
