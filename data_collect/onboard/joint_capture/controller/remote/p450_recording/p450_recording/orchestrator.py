import csv
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .exporter import export_session
from .common import normalize_bag_names, pending_bags
from .manager import DEFAULT_DATA_ROOT, RecorderManager
from .process_tracking import (process_exists as _linux_process_exists,
    process_identity as _process_identity, descendants as _descendants,
    command_matches, list_component_processes)


@dataclass(frozen=True)
class Component:
    name: str
    command: tuple
    launch_file: str


COMPONENTS = (
    Component(
        "mid360_driver",
        ("/usr/bin/env", "roslaunch", "p450_experiment", "msg_MID360.launch"),
        "msg_MID360.launch",
    ),
    Component(
        "flight_mid360",
        (
            "/usr/bin/env",
            "roslaunch",
            "p450_experiment",
            "p450_onboard_mid360.launch",
        ),
        "p450_onboard_mid360.launch",
    ),
    Component(
        "fast_lio",
        (
            "/usr/bin/env",
            "roslaunch",
            "p450_experiment",
            "mapping_mid360.launch",
            "rviz:=false",
        ),
        "mapping_mid360.launch",
    ),
    Component(
        "d435i",
        ("/usr/bin/env", "roslaunch", "p450_experiment", "rs_camera_d435i.launch"),
        "rs_camera_d435i.launch",
    ),
)


def _process_exists(pid):
    return _linux_process_exists(pid)


def _find_component_process(component):
    pids = list_component_processes(component)
    return pids[0] if pids else None


def _vehicle_state(timeout_seconds=1.0):
    import rospy
    from prometheus_msgs.msg import UAVState

    if not rospy.core.is_initialized():
        rospy.init_node("p450_capture", anonymous=True, disable_signals=True)
    try:
        message = rospy.wait_for_message(
            "/uav1/prometheus/state", UAVState, timeout=timeout_seconds
        )
    except rospy.ROSException:
        return None
    return {
        "connected": bool(message.connected),
        "armed": bool(message.armed),
        "mode": message.mode,
        "location_source": int(message.location_source),
        "odom_valid": bool(message.odom_valid),
    }


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _alignment_summary(path):
    count = 0
    valid = 0
    delta_sum = 0.0
    delta_max = 0.0
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            count += 1
            delta = float(row["delta_ms"])
            delta_sum += delta
            delta_max = max(delta_max, delta)
            valid += int(row["valid"])
    return {
        "alignment_count": count,
        "valid_alignment_count": valid,
        "valid_alignment_percent": 100.0 * valid / count if count else 0.0,
        "mean_delta_ms": delta_sum / count if count else 0.0,
        "max_delta_ms_observed": delta_max,
    }


def _bag_is_ready(session_dir):
    raw = Path(session_dir) / "raw"
    return not pending_bags(raw) and any(raw.glob("*.bag"))


def _rosbag_reindex(active_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["rosbag", "reindex", "--output-dir", str(output_dir), str(active_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "rosbag reindex failed for {}: {}".format(
                active_path, result.stdout.strip() or "exit {}".format(result.returncode)
            )
        )


def recover_interrupted_bags(session_dir, reindexer=None):
    """Publish reindexed copies while retaining every interrupted source."""
    session_dir = Path(session_dir)
    raw_dir = session_dir / "raw"
    recovery_dir = session_dir / "recovery"
    reindexer = reindexer or _rosbag_reindex
    recovered = []
    for source in pending_bags(raw_dir):
        target = source.with_suffix("")
        recovery_dir.mkdir(parents=True, exist_ok=True)
        # A failed reindex may leave a nonempty but unindexed copy. Never
        # treat file existence as evidence that a previous attempt succeeded.
        attempt = recovery_dir / (".reindex-" + uuid.uuid4().hex)
        temporary = target.with_suffix(target.suffix + ".recovering")
        try:
            reindexer(source, attempt)
            reindexed = attempt / source.name
            if not reindexed.is_file() or reindexed.stat().st_size <= 0:
                raise RuntimeError(
                    "rosbag reindex produced no usable output for {}".format(source)
                )
            shutil.copy2(str(reindexed), str(temporary))
            os.replace(str(temporary), str(target))
        finally:
            if temporary.exists():
                temporary.unlink()
            if attempt.exists():
                shutil.rmtree(str(attempt))
        recovered.append(target)
    return recovered


class CaptureOrchestrator:
    def __init__(
        self,
        data_root=DEFAULT_DATA_ROOT,
        recorder=None,
        exporter=None,
        process_finder=None,
        popen_factory=None,
        vehicle_state_provider=None,
        readiness_provider=None,
        bag_ready_provider=None,
        bag_reindexer=None,
        proc_cmdline_reader=None,
        killpg_func=None,
        process_exists=None,
        sleep_func=None,
        now_provider=None,
        readiness_timeout=45.0,
        bag_finalize_timeout=15.0,
        process_lister=None,
        process_identity=None,
        descendant_provider=None,
        getpgid_func=None,
        kill_func=None,
        cleanup_timeout=25.0,
    ):
        self.data_root = Path(data_root)
        self.runtime_dir = self.data_root / ".p450_capture"
        self.logs_dir = self.runtime_dir / "logs"
        self.stack_state_path = self.runtime_dir / "stack_state.json"
        self.capture_state_path = self.runtime_dir / "capture_state.json"
        self.prepare_lock_path = self.runtime_dir / "prepare.lock"
        self.recorder = recorder or RecorderManager(data_root=self.data_root)
        self.exporter = exporter or export_session
        self.process_finder = process_finder or _find_component_process
        self.process_lister = process_lister or (list_component_processes if process_finder is None
            else lambda component: [pid] if (pid := self.process_finder(component)) else [])
        self.process_identity = process_identity or _process_identity
        self.descendant_provider = descendant_provider or _descendants
        self.getpgid_func = getpgid_func or getattr(os, "getpgid", None)
        self.kill_func = kill_func or os.kill
        self.cleanup_timeout = cleanup_timeout
        self._children = []  # Reap roslaunch children while this CLI is alive.
        self.popen_factory = popen_factory or subprocess.Popen
        self.vehicle_state_provider = vehicle_state_provider or _vehicle_state
        self.readiness_provider = readiness_provider
        self.bag_ready_provider = bag_ready_provider or _bag_is_ready
        self.bag_reindexer = bag_reindexer or _rosbag_reindex
        self.proc_cmdline_reader = proc_cmdline_reader or (
            lambda pid: Path(f"/proc/{pid}/cmdline").read_bytes()
        )
        self.killpg_func = killpg_func or os.killpg
        self.process_exists = process_exists or _process_exists
        self.sleep_func = sleep_func or time.sleep
        self.now_provider = now_provider or (
            lambda: time.strftime("%Y%m%d_%H%M%S")
        )
        self.readiness_timeout = readiness_timeout
        self.bag_finalize_timeout = bag_finalize_timeout

    @contextmanager
    def _prepare_lock(self):
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        value = json.dumps({"pid": os.getpid(), "token": token})
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.prepare_lock_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                try:
                    existing = self._read_json(self.prepare_lock_path) or {}
                    owner_pid = int(existing.get("pid", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    owner_pid = 0
                if owner_pid and self.process_exists(owner_pid):
                    raise RuntimeError(
                        f"another prepare is already running (PID {owner_pid})"
                    )
                try:
                    self.prepare_lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(value)
                break
        else:
            raise RuntimeError("could not acquire prepare lock")

        try:
            yield
        finally:
            try:
                current = self._read_json(self.prepare_lock_path) or {}
                if current.get("token") == token:
                    self.prepare_lock_path.unlink()
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    def _read_json(self, path):
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _launch(self, component):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{component.name}.log"
        with log_path.open("ab", buffering=0) as log:
            process = self.popen_factory(
                list(component.command),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        self._children.append(process)
        return {
            "name": component.name,
            "pid": int(process.pid),
            "pgid": int(process.pid),
            "command": list(component.command),
            "launch_file": component.launch_file,
            "log_path": str(log_path),
            "owned": True,
            "identity": self.process_identity(process.pid),
            "children": [],
        }

    def _signal_owned(self, process):
        pid = int(process["pid"])
        if not self.process_exists(pid):
            return False
        cmdline = self.proc_cmdline_reader(pid)
        if not process.get("identity") or self.process_identity(pid) != process["identity"]:
            raise RuntimeError(f"refusing to signal PID {pid}: process identity changed or missing")
        if not command_matches(cmdline, process["launch_file"]):
            raise RuntimeError(
                f"refusing to signal PID {pid}: command does not match "
                f"{process['launch_file']}"
            )
        if self.getpgid_func(pid) != int(process["pgid"]) or int(process["pgid"]) != pid:
            raise RuntimeError(f"refusing to signal PID {pid}: process group changed")
        self.killpg_func(pid, signal.SIGINT)
        return True

    def _save_stack(self, processes, phase):
        for process in processes:
            pid = int(process['pid'])
            if self.process_exists(pid) and process.get('identity'):
                try:
                    if self.process_identity(pid) == process['identity']:
                        children = {p['pid']: p for p in process.get('children', [])}
                        children.update({p['pid']: p for p in self.descendant_provider(pid)})
                        process['children'] = list(children.values())
                except (OSError, ValueError):
                    pass
        _atomic_json(self.stack_state_path, {'processes': processes, 'phase': phase,
                                           'updated_at': self.now_provider()})

    def _cleanup(self, processes):
        self._save_stack(processes, 'cleanup_pending')
        errors = []
        stopped = []
        for process in reversed(processes):
            if not process.get('owned'):
                errors.append('unowned process record')
                continue
            try:
                if self._signal_owned(process):
                    stopped.append(process['name'])
                for child in process.get('children', []):
                    pid = int(child['pid'])
                    if self.process_exists(pid):
                        if self.process_identity(pid) != child['identity']:
                            raise RuntimeError(f'child PID {pid} identity changed')
                        self.kill_func(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            except (OSError, RuntimeError) as exc:
                errors.append(str(exc))
        deadline = time.monotonic() + self.cleanup_timeout
        while True:
            for child in self._children:
                if hasattr(child, 'poll'):
                    child.poll()
            alive = [p['pid'] for root in processes
                     for p in [root] + root.get('children', [])
                     if self.process_exists(int(p['pid']))]
            if not alive or time.monotonic() >= deadline:
                break
            self.sleep_func(0.1)
        if alive or errors:
            raise RuntimeError('cleanup pending; no new nodes launched: PIDs {}; {}'.format(alive, '; '.join(errors)))
        self.stack_state_path.unlink()
        return stopped

    def _tracked_alive(self, processes):
        return any(self.process_exists(p['pid']) for root in processes
                   for p in [root] + root.get('children', []))

    def _validate_vehicle(self, state):
        if not state:
            raise RuntimeError("Prometheus vehicle state is unavailable")
        if state.get("armed"):
            raise RuntimeError("refusing automatic setup while vehicle is armed")
        if not state.get("connected"):
            raise RuntimeError("PX4 is not connected")
        if int(state.get("location_source", -1)) != 10:
            raise RuntimeError("Prometheus location_source is not MID360 (10)")
        if not state.get("odom_valid"):
            raise RuntimeError("MID360 odometry is not valid")

    def _wait_ready(self, raw_rgb=False):
        if self.readiness_provider is not None:
            report = self.readiness_provider()
            if not report.get("ok"):
                raise RuntimeError("; ".join(report.get("errors", ["not ready"])))
            self._validate_vehicle(self.vehicle_state_provider())
            return report

        deadline = time.monotonic() + self.readiness_timeout
        last_errors = []
        while time.monotonic() < deadline:
            saved = self._read_json(self.stack_state_path)
            if saved and saved.get('phase') == 'launching':
                self._save_stack(saved['processes'], 'launching')
            report = self.recorder.check(raw_rgb=raw_rgb)
            state = self.vehicle_state_provider(1.0)
            if report.get("ok"):
                try:
                    self._validate_vehicle(state)
                    return report
                except RuntimeError as exc:
                    last_errors = [str(exc)]
            else:
                last_errors = list(report.get("errors", []))
            self.sleep_func(1.0)
        raise RuntimeError(
            "capture chain did not become ready: "
            + "; ".join(last_errors or ["timeout"])
        )

    def start(self, session_name, raw_rgb=False, max_minutes=None):
        with self._prepare_lock():
            return self._start(session_name, raw_rgb, max_minutes)

    def _start(self, session_name, raw_rgb=False, max_minutes=None):
        stack = self._read_json(self.stack_state_path) or {}
        if stack.get('phase') in ('launching', 'cleanup_pending'):
            raise RuntimeError('stack cleanup pending; run prepare before recording')
        duplicates = [c.name for c in COMPONENTS if len(self.process_lister(c)) > 1]
        if duplicates:
            raise RuntimeError('duplicate launch processes: ' + ', '.join(duplicates))
        recorder_state = self.recorder.status()
        if recorder_state.get("active"):
            raise RuntimeError("a recording is already active")
        if self.capture_state_path.exists():
            raise RuntimeError(
                f"capture state already exists at {self.capture_state_path}"
            )

        preflight = self.vehicle_state_provider(0.5)
        if preflight and preflight.get("armed"):
            raise RuntimeError("refusing automatic setup while vehicle is armed")

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        readiness = self._wait_ready(raw_rgb=raw_rgb)
        running = [
            component.name
            for component in COMPONENTS
            if self.process_finder(component) is not None
        ]
        session_dir = self.recorder.start(
            session_name, raw_rgb=raw_rgb, max_minutes=max_minutes
        )
        capture_state = {
            "session_name": session_name,
            "session_dir": str(session_dir),
            "started_at": self.now_provider(),
            "raw_rgb": bool(raw_rgb),
            "max_minutes": max_minutes,
        }
        _atomic_json(self.capture_state_path, capture_state)
        return {
            "started": True,
            "session_dir": str(session_dir),
            "started_components": [],
            "reused_components": running,
            "readiness": readiness,
        }

    def prepare(self):
        with self._prepare_lock():
            if self.recorder.status().get('active') or self.capture_state_path.exists():
                raise RuntimeError('cannot prepare stack while recording or capture cleanup is pending')
            preflight = self.vehicle_state_provider(0.5)
            if preflight and preflight.get("armed"):
                raise RuntimeError(
                    "refusing automatic setup while vehicle is armed"
                )

            saved = self._read_json(self.stack_state_path) or {}
            processes = saved.get('processes', [])
            if saved.get('phase') in ('launching', 'cleanup_pending'):
                if self._tracked_alive(processes) and (not preflight or not preflight.get('connected') or preflight.get('armed') is not False):
                    raise RuntimeError('cleanup pending: require connected, disarmed vehicle before recovery')
                self._cleanup(processes)
                processes = []
            listed = {component.name: self.process_lister(component) for component in COMPONENTS}
            running = {name: pids[0] if pids else None for name, pids in listed.items()}
            running_names = [name for name, pid in running.items() if pid]
            invalid = any(len(pids) > 1 for pids in listed.values()) or (running_names and len(running_names) != len(COMPONENTS))
            if invalid:
                owned = {p['pid'] for p in processes if p.get('owned') and p.get('identity')}
                current = {pid for pids in listed.values() for pid in pids}
                if current <= owned and preflight and preflight.get('connected') and preflight.get('armed') is False:
                    self._cleanup(processes)
                    processes = []
                    listed = {c.name: self.process_lister(c) for c in COMPONENTS}
                    if any(listed.values()):
                        raise RuntimeError('cleanup pending: launch processes still present')
                    running_names = []
                else:
                    raise RuntimeError('partial capture stack or duplicate launch detected; ownership not verified; refusing duplicate launch or automatic cleanup')
            if len(running_names) == len(COMPONENTS):
                readiness = self._wait_ready()
                if processes:
                    self._save_stack(processes, 'ready')
                return {
                    "prepared": True,
                    "started_components": [],
                    "reused_components": running_names,
                    "readiness": readiness,
                }

            # Never overwrite living orphan ownership just because roslaunch exited.
            if self._tracked_alive(processes):
                raise RuntimeError('cleanup pending: tracked processes remain; use shutdown before prepare')
            started = []
            try:
                for component in COMPONENTS:
                    started.append(self._launch(component))
                    self._save_stack(started, 'launching')
                    self.sleep_func(1.0)
                self._save_stack(started, 'launching')
                readiness = self._wait_ready()
                self._save_stack(started, 'ready')
            except Exception as original:
                latest_vehicle = self.vehicle_state_provider(0.5)
                if latest_vehicle and latest_vehicle.get('armed'):
                    self._save_stack(started, 'cleanup_pending')
                    raise RuntimeError(f'{original}; cleanup pending: vehicle is armed') from original
                # Include descendants discovered while waiting for sensor topics.
                saved = self._read_json(self.stack_state_path) or {}
                started = saved.get('processes', started)
                try:
                    self._cleanup(started)
                except RuntimeError as cleanup_error:
                    raise RuntimeError(f'{original}; {cleanup_error}') from original
                raise

            return {
                "prepared": True,
                "started_components": [
                    process["name"] for process in started
                ],
                "reused_components": [],
                "readiness": readiness,
            }

    def stop_fast(self):
        """Stop owned recording and release the marker without exporting data."""
        capture_state = self._read_json(self.capture_state_path)
        if not capture_state:
            raise RuntimeError("no active capture state")
        recorder_state = self.recorder.status()
        if recorder_state.get("active") or recorder_state.get("stale"):
            stop_result = self.recorder.stop()
            session_dir = stop_result["session_dir"]
        else:
            session_dir = capture_state["session_dir"]
        self.capture_state_path.unlink()
        return {"stopped": True, "session_dir": str(session_dir), "postprocess": "pending"}

    def finalize_raw(self, session_dir):
        """Wait for/recover the bag only; video export is a separate command."""
        session_dir = Path(session_dir).resolve()
        if not session_dir.is_dir():
            raise RuntimeError(f"session directory does not exist: {session_dir}")
        if self.recorder.status().get("active"):
            raise RuntimeError("cannot finalize raw data while recording")
        deadline = time.monotonic() + self.bag_finalize_timeout
        recovered_bags = []
        while pending_bags(session_dir / "raw") or not self.bag_ready_provider(session_dir):
            if time.monotonic() >= deadline:
                recovered_bags = recover_interrupted_bags(
                    session_dir, reindexer=self.bag_reindexer
                )
                if pending_bags(session_dir / "raw") or not self.bag_ready_provider(session_dir):
                    raise RuntimeError(
                        f"rosbag did not finalize within "
                        f"{self.bag_finalize_timeout:.1f} seconds: {session_dir}"
                    )
                break
            self.sleep_func(0.1)
        renamed = normalize_bag_names(session_dir / "raw")
        return {"session_dir": str(session_dir),
                "recovered_bags": [str(renamed.get(path, path)) for path in recovered_bags],
                "renamed_bags": {str(source): str(target) for source, target in renamed.items()}}

    def export(self, session_dir):
        raw = self.finalize_raw(session_dir)
        result = self.exporter(Path(raw["session_dir"]))
        result.update(_alignment_summary(result["alignment_csv"]))
        return {**raw, **result}

    def finish(self):
        capture_state = self._read_json(self.capture_state_path)
        if not capture_state:
            raise RuntimeError("no active capture state")
        recorder_state = self.recorder.status()
        if recorder_state.get("active") or recorder_state.get("stale"):
            stop_result = self.recorder.stop()
            session_dir = stop_result["session_dir"]
        else:
            session_dir = capture_state["session_dir"]

        result = self.export(session_dir)
        self.capture_state_path.unlink()
        return {"finished": True, **result}

    def status(self):
        stack_state = self._read_json(self.stack_state_path) or {"processes": []}
        components = {}
        for component in COMPONENTS:
            pid = self.process_finder(component)
            components[component.name] = {
                "running": pid is not None,
                "pid": pid,
            }
        phase = stack_state.get('phase', 'legacy')
        if phase == 'ready' and not all(c['running'] for c in components.values()):
            phase = 'not_ready'
        return {
            "components": components,
            "vehicle": self.vehicle_state_provider(0.5),
            "recorder": self.recorder.status(),
            "capture": self._read_json(self.capture_state_path),
            "owned_processes": stack_state["processes"],
            "stack_phase": phase,
        }

    def shutdown(self):
        with self._prepare_lock():
            return self._shutdown()

    def _shutdown(self):
        if self.recorder.status().get("active"):
            raise RuntimeError("cannot shut down stack while recording")
        stack_state = self._read_json(self.stack_state_path)
        if not stack_state:
            return {"stopped_components": []}
        state = self.vehicle_state_provider(0.5)
        if self._tracked_alive(stack_state['processes']) and (not state or not state.get('connected') or state.get('armed') is not False):
            raise RuntimeError('require connected, disarmed vehicle before shutdown')
        stopped = self._cleanup(stack_state['processes'])
        return {"stopped_components": stopped}
