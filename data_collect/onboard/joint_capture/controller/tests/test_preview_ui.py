from pathlib import Path
import subprocess


def test_live_preview_lifecycle():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['node', str(root/'tests/preview_dom.cjs'),
        str(root/'src/jointctl/console_web/preview.js')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
