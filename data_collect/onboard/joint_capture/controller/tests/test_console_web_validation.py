"""Browser-level regression: a failed recheck must show its actual reason."""

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from jointctl.console import make_server


CHROME = next((candidate for candidate in (
    shutil.which("chrome"),
    shutil.which("google-chrome"),
    shutil.which("chromium"),
    shutil.which("chromium-browser"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    str(Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
) if candidate and Path(candidate).is_file()), None)


def stop_browser(process):
    """Clean up only the isolated browser process group created by this test."""
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=5)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=3)


def browser_html(tmp_path, snapshot):
    class State:
        root = tmp_path
        config_path = tmp_path / "config.json"

        def snapshot(self):
            return snapshot

    server = make_server(State(), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        argv = [
            CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
            "--disable-background-networking", "--disable-component-update",
            "--no-default-browser-check", f"--user-data-dir={tmp_path / 'chrome'}",
            *(["--use-mock-keychain"] if sys.platform == "darwin" else []),
            "--virtual-time-budget=3000", "--dump-dom",
            f"http://127.0.0.1:{server.server_port}/",
        ]
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                   if os.name == "nt" else {"start_new_session": True})
        dom = queue.Queue()
        errors = tmp_path / "chrome.stderr"
        with errors.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr,
                                       text=True, encoding="utf-8", errors="replace", **options)

            def read_dom():
                lines = []
                for line in process.stdout:
                    lines.append(line)
                    if line.rstrip().lower().endswith("</html>"):
                        break
                dom.put("".join(lines))

            reader = threading.Thread(target=read_dom, daemon=True)
            reader.start()
            try:
                try:
                    html = dom.get(timeout=20)
                except queue.Empty:
                    pytest.fail("Chrome did not produce a complete DOM within 20 seconds")
                exit_code = process.poll()
            finally:
                # macOS Chrome can keep running after --dump-dom has finished.
                # The complete DOM is the result; browser teardown is our job.
                stop_browser(process)
                reader.join(timeout=3)
                process.stdout.close()
        diagnostic = errors.read_text(encoding="utf-8", errors="replace")[-1000:]
        assert html.rstrip().lower().endswith("</html>"), diagnostic
        assert exit_code in (None, 0), diagnostic
        return html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.skipif(CHROME is None, reason="Chrome not installed")
def test_failed_validation_reason_visible_without_opening_logs(tmp_path):
    html = browser_html(tmp_path, {
                "active": False,
                "operation_busy": False,
                "hosts": {name: {"status": {"stale": False, "active": False,
                                              "state": "idle", "reachable": True},
                                 "resources": {"stale": True}}
                          for name in ("p450", "unitree")},
                "episode": {"episode_id": "joint_test", "state": "complete",
                            "metadata": {"postprocess": "failed"}},
                "job": None,
                "allowed": {name: False for name in
                            ("start", "stop", "recover", "finalize", "align", "prepare")},
                "postprocess": {"pending_count": 1},
                "clock": {"degraded": True, "reasons": []},
                "validation": {"state": "failed", "report": None, "path": None,
                               "error": "summary.json frames_dropped must be zero"},
                "elapsed_s": None, "remaining_s": None,
                "disk_warning_bytes": 10 * 1024 ** 3,
            })
    assert "summary.json frames_dropped must be zero" in html


@pytest.mark.skipif(CHROME is None, reason="Chrome not installed")
def test_legacy_recorder_drops_warn_during_recording(tmp_path):
    hosts = {name: {"status": {"stale": False, "active": True,
                               "state": "recording", "reachable": True,
                               "episode_id": "joint_test"},
                    "resources": {"stale": True}, "write_stalled": False}
             for name in ("p450", "unitree")}
    hosts["unitree"]["frames_dropped"] = 12
    hosts["unitree"]["save_errors"] = 0
    html = browser_html(tmp_path, {
        "active": True, "operation_busy": False, "hosts": hosts,
        "episode": {"episode_id": "joint_test", "state": "recording",
                    "t0_desktop_ns": "1790431394494818200", "metadata": {}},
        "job": None, "allowed": {name: False for name in
                             ("start", "stop", "recover", "finalize", "align", "prepare")},
        "postprocess": {"pending_count": 0},
        "clock": {"degraded": False, "reasons": [], "estimated_error_ms": 3.5},
        "validation": {"state": "pending", "report": None, "path": None},
        "elapsed_s": 10, "remaining_s": 1700, "disk_warning_bytes": 10 * 1024 ** 3,
    })
    assert "Dropped: 12" in html


@pytest.mark.skipif(CHROME is None, reason='Chrome not installed')
def test_powered_arm_and_blocked_teleop_are_distinct_in_the_console(tmp_path):
    hosts = {name: {'status': {'stale': False, 'active': False,
                              'state': 'idle', 'reachable': True},
                    'resources': {'stale': True}} for name in ('p450', 'unitree')}
    hosts['unitree'].update(gamepad={'connected': True}, arm_enabled=False,
        motor_enabled=[True] * 6, command_inhibited=True,
        feedback_error='原始关节反馈超限：J2=-2.494deg')
    html = browser_html(tmp_path, {
        'active': False, 'operation_busy': False, 'hosts': hosts,
        'episode': None, 'job': None,
        'allowed': {name: False for name in ('start', 'stop', 'recover', 'finalize', 'align', 'prepare')},
        'postprocess': {'pending_count': 0}, 'clock': {'degraded': True, 'reasons': []},
        'validation': {'state': 'unavailable', 'report': None, 'path': None},
        'elapsed_s': None, 'remaining_s': None, 'disk_warning_bytes': 10 * 1024 ** 3})
    assert '已使能 · 保持位置' in html
    assert '手柄遥操' in html and '未启用' in html
    assert '原始关节反馈超限：J2=-2.494deg' in html
    assert 'id="session-pill" class="pill green"' not in html
