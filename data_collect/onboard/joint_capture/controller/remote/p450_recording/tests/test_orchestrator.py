import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("cv2", mock.Mock())
sys.modules.setdefault("rosbag", mock.Mock())
sys.modules.setdefault("cv_bridge", mock.Mock(CvBridge=mock.Mock))

from p450_recording.orchestrator import (
    CaptureOrchestrator,
    COMPONENTS,
    recover_interrupted_bags,
)


class FakeProcess:
    next_pid = 5000

    def __init__(self):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / "test_scratch_orchestrator"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir()
        (self.root / "session").mkdir()
        self.calls = []
        self.recorder = mock.Mock()
        self.recorder.status.return_value = {"active": False, "stale": False}
        self.recorder.start.return_value = self.root / "session"
        self.recorder.stop.side_effect = self._stop
        self.exporter = mock.Mock(side_effect=self._export)
        self.popen = mock.Mock(side_effect=lambda *args, **kwargs: FakeProcess())
        self.finder = mock.Mock(return_value=None)
        self.state_provider = mock.Mock(
            return_value={
                "connected": True,
                "armed": False,
                "location_source": 10,
                "odom_valid": True,
            }
        )
        self.ready_provider = mock.Mock(return_value={"ok": True})
        self.bag_ready = mock.Mock(return_value=True)
        self.kill = mock.Mock()
        self.dead = set()
        self.kill.side_effect = lambda pid, sig: self.dead.add(pid)
        self.listed = {component.name: [] for component in COMPONENTS}
        self.orchestrator = CaptureOrchestrator(
            data_root=self.root,
            recorder=self.recorder,
            exporter=self.exporter,
            process_finder=self.finder,
            popen_factory=self.popen,
            vehicle_state_provider=self.state_provider,
            readiness_provider=self.ready_provider,
            bag_ready_provider=self.bag_ready,
            proc_cmdline_reader=lambda pid: (
                b"roslaunch\x00p450_experiment\x00"
                b"msg_MID360.launch\x00p450_onboard_mid360.launch\x00"
                b"mapping_mid360.launch\x00rs_camera_d435i.launch\x00"
            ),
            killpg_func=self.kill,
            process_exists=lambda pid: pid not in self.dead,
            process_identity=lambda pid: {"start_ticks": str(pid), "boot_id": "test"},
            descendant_provider=lambda pid: [],
            getpgid_func=lambda pid: pid,
            cleanup_timeout=0,
            sleep_func=lambda _: None,
            now_provider=lambda: "20260725_200000",
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _own_listed(self):
        self.orchestrator.runtime_dir.mkdir(parents=True, exist_ok=True)
        processes = [dict(name=c.name, pid=pid, pgid=pid,
                          launch_file=c.launch_file, owned=True,
                          identity={"start_ticks": str(pid), "boot_id": "test"})
                     for c in COMPONENTS for pid in self.listed[c.name]]
        self.orchestrator.stack_state_path.write_text(json.dumps({"processes": processes}))
        self.orchestrator.process_exists = lambda pid: any(pid in p for p in self.listed.values())

    def test_failed_cleanup_preserves_ownership_and_blocks_relaunch(self):
        self.ready_provider.return_value = {"ok": False, "errors": ["camera timeout"]}
        self.kill.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "cleanup pending"):
            self.orchestrator.prepare()
        state = json.loads(self.orchestrator.stack_state_path.read_text())
        self.assertEqual(state["phase"], "cleanup_pending")
        self.assertEqual(len(state["processes"]), 4)
        with self.assertRaisesRegex(RuntimeError, "cleanup pending"):
            self.orchestrator.prepare()
        self.assertEqual(self.popen.call_count, 4)

    def test_status_does_not_reuse_ready_after_processes_disappear(self):
        self.orchestrator.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.orchestrator.stack_state_path.write_text(json.dumps({'phase': 'ready', 'processes': []}))
        self.finder.return_value = None
        status = self.orchestrator.status()
        self.assertEqual(status['stack_phase'], 'not_ready')

    def test_status_preserves_cleanup_pending_with_missing_components(self):
        self.orchestrator.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.orchestrator.stack_state_path.write_text(json.dumps({'phase': 'cleanup_pending', 'processes': []}))
        self.assertEqual(self.orchestrator.status()['stack_phase'], 'cleanup_pending')

    def test_ownership_is_saved_before_next_launch(self):
        def launch(*args, **kwargs):
            if self.popen.call_count == 2:
                state = json.loads(self.orchestrator.stack_state_path.read_text())
                self.assertEqual(len(state["processes"]), 1)
            return FakeProcess()
        self.popen.side_effect = launch
        self.orchestrator.prepare()

    def test_shutdown_refuses_reused_pid(self):
        self.listed["d435i"] = [7000]
        self._own_listed()
        self.orchestrator.process_identity = lambda pid: {"start_ticks": "different", "boot_id": "test"}
        with self.assertRaisesRegex(RuntimeError, "cleanup pending"):
            self.orchestrator.shutdown()
        self.kill.assert_not_called()
        self.assertTrue(self.orchestrator.stack_state_path.exists())

    def test_shutdown_waits_for_child_after_launch_parent_exits(self):
        self.listed['d435i'] = [7000]
        self._own_listed()
        state = json.loads(self.orchestrator.stack_state_path.read_text())
        state['processes'][0]['children'] = [{'pid': 7001, 'identity': {'start_ticks': '7001', 'boot_id': 'test'}}]
        self.orchestrator.stack_state_path.write_text(json.dumps(state))
        self.orchestrator.process_exists = lambda pid: pid == 7001
        self.orchestrator.kill_func = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'cleanup pending'):
            self.orchestrator.shutdown()
        self.orchestrator.kill_func.assert_called_once_with(7001, mock.ANY)
        self.assertTrue(self.orchestrator.stack_state_path.exists())

    def test_prepare_does_not_touch_active_recording(self):
        self.recorder.status.return_value = {'active': True}
        with self.assertRaisesRegex(RuntimeError, 'recording'):
            self.orchestrator.prepare()
        self.kill.assert_not_called()
        self.popen.assert_not_called()

    def test_unknown_duplicate_is_reported_without_killing(self):
        self.orchestrator.process_lister = lambda c: [7000, 7001]
        with self.assertRaisesRegex(RuntimeError, 'ownership not verified'):
            self.orchestrator.prepare()
        self.kill.assert_not_called()
        self.popen.assert_not_called()

    def test_stale_pending_records_after_reboot_do_not_need_vehicle(self):
        self.orchestrator.runtime_dir.mkdir()
        self.orchestrator.stack_state_path.write_text(json.dumps({'phase': 'cleanup_pending', 'processes': []}))
        self.state_provider.side_effect = [None, {'connected': True, 'armed': False, 'location_source': 10, 'odom_valid': True}]
        self.orchestrator.prepare()
        self.assertEqual(self.popen.call_count, 4)

    def _stop(self):
        self.calls.append("stop")
        self.recorder.status.return_value = {"active": False, "stale": False}
        return {"stopped": True, "session_dir": str(self.root / "session")}

    def _export(self, session_dir):
        self.calls.append("export")
        alignment = self.root / "alignment.csv"
        alignment.write_text(
            "frame_index,delta_ms,valid\n0,10.0,1\n1,20.0,1\n",
            encoding="utf-8",
        )
        return {
            "session_dir": str(session_dir),
            "frame_count": 42,
            "pose_count": 28,
            "alignment_csv": str(alignment),
        }

    def test_start_never_launches_missing_components(self):
        result = self.orchestrator.start("demo")
        self.popen.assert_not_called()
        self.recorder.start.assert_called_once_with(
            "demo", raw_rgb=False, max_minutes=None
        )
        self.assertEqual(result["session_dir"], str(self.root / "session"))
        self.assertEqual(result["started_components"], [])

    def test_start_refuses_pending_stack_cleanup(self):
        self.orchestrator.runtime_dir.mkdir()
        self.orchestrator.stack_state_path.write_text(json.dumps({'phase': 'cleanup_pending', 'processes': []}))
        with self.assertRaisesRegex(RuntimeError, 'cleanup pending'):
            self.orchestrator.start('demo')
        self.recorder.start.assert_not_called()

    def test_start_refuses_duplicate_camera_launch(self):
        self.orchestrator.process_lister = lambda c: [7000, 7001] if c.name == 'd435i' else []
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            self.orchestrator.start('demo')
        self.recorder.start.assert_not_called()

    def test_start_reuses_running_components(self):
        pids = {
            item.name: 7000 + index for index, item in enumerate(COMPONENTS)
        }
        self.finder.side_effect = lambda component: pids[component.name]
        result = self.orchestrator.start("demo")
        self.popen.assert_not_called()
        self.assertEqual(result["reused_components"], list(pids))

    def test_start_preserves_ownership_when_reusing_its_own_components(self):
        self.orchestrator.runtime_dir.mkdir(parents=True)
        processes = []
        pids = {}
        for index, component in enumerate(COMPONENTS):
            pid = 8000 + index
            pids[component.name] = pid
            processes.append(
                {
                    "name": component.name,
                    "pid": pid,
                    "pgid": pid,
                    "command": list(component.command),
                    "launch_file": component.launch_file,
                    "owned": True,
                }
            )
        self.orchestrator.stack_state_path.write_text(
            json.dumps({"created_at": "earlier", "processes": processes}),
            encoding="utf-8",
        )
        self.finder.side_effect = lambda component: pids[component.name]
        self.orchestrator.start("demo")
        state = json.loads(
            self.orchestrator.stack_state_path.read_text(encoding="utf-8")
        )
        self.assertTrue(all(process["owned"] for process in state["processes"]))

    def test_start_refuses_armed_vehicle_before_launching(self):
        self.state_provider.return_value = {
            "connected": True,
            "armed": True,
            "location_source": 10,
            "odom_valid": True,
        }
        with self.assertRaisesRegex(RuntimeError, "armed"):
            self.orchestrator.start("demo")
        self.popen.assert_not_called()
        self.recorder.start.assert_not_called()

    def test_start_does_not_touch_processes_when_readiness_fails(self):
        self.ready_provider.return_value = {
            "ok": False,
            "errors": ["Odometry is inactive"],
        }
        with self.assertRaisesRegex(RuntimeError, "Odometry"):
            self.orchestrator.start("demo")
        self.popen.assert_not_called()
        self.kill.assert_not_called()
        self.recorder.start.assert_not_called()

    def test_prepare_launches_empty_stack_in_dependency_order(self):
        result = self.orchestrator.prepare()
        commands = [call.args[0] for call in self.popen.call_args_list]
        self.assertEqual(commands, [list(item.command) for item in COMPONENTS])
        self.assertEqual(
            result["started_components"], [item.name for item in COMPONENTS]
        )
        self.assertEqual(result["reused_components"], [])

    def test_prepare_reuses_complete_stack_without_launching(self):
        pids = {
            item.name: 7000 + index for index, item in enumerate(COMPONENTS)
        }
        self.finder.side_effect = lambda component: pids[component.name]
        result = self.orchestrator.prepare()
        self.popen.assert_not_called()
        self.assertEqual(result["started_components"], [])
        self.assertEqual(result["reused_components"], list(pids))

    def test_prepare_refuses_partial_stack_without_changing_processes(self):
        self.finder.side_effect = lambda component: (
            7000 if component is COMPONENTS[0] else None
        )
        with self.assertRaisesRegex(RuntimeError, "partial capture stack"):
            self.orchestrator.prepare()
        self.popen.assert_not_called()
        self.kill.assert_not_called()

    def test_prepare_replaces_duplicate_stack_before_relaunching(self):
        """A second exact roslaunch for one component must not survive prepare."""
        self.listed.update(
            {
                component.name: [7000 + index]
                for index, component in enumerate(COMPONENTS)
            }
        )
        self.listed["d435i"] = [7003, 7103]
        self._own_listed()
        self.orchestrator.process_lister = (
            lambda component: list(self.listed[component.name])
        )
        self.orchestrator.getpgid_func = lambda pid: pid
        self.kill.side_effect = lambda pgid, _signal: [
            values.remove(pgid) for values in self.listed.values() if pgid in values
        ]
        for component in COMPONENTS:
            self.finder.side_effect = lambda item: (
                self.listed[item.name][0] if self.listed[item.name] else None
            )

        result = self.orchestrator.prepare()

        self.assertEqual(self.kill.call_count, len(COMPONENTS) + 1)
        self.assertEqual(self.popen.call_count, len(COMPONENTS))
        self.assertEqual(
            result["started_components"], [item.name for item in COMPONENTS]
        )

    def test_prepare_repairs_partial_stack_before_relaunching(self):
        """A crashed partial stack is cleaned and rebuilt instead of blocking start."""
        self.listed["mid360_driver"] = [7000]
        self._own_listed()
        self.orchestrator.process_lister = (
            lambda component: list(self.listed[component.name])
        )
        self.orchestrator.getpgid_func = lambda pid: pid
        self.kill.side_effect = lambda pgid, _signal: [
            values.remove(pgid) for values in self.listed.values() if pgid in values
        ]
        self.finder.side_effect = lambda component: (
            self.listed[component.name][0] if self.listed[component.name] else None
        )

        result = self.orchestrator.prepare()

        self.kill.assert_called_once_with(7000, mock.ANY)
        self.assertEqual(self.popen.call_count, len(COMPONENTS))
        self.assertEqual(
            result["started_components"], [item.name for item in COMPONENTS]
        )

    def test_prepare_rolls_back_only_processes_from_failed_attempt(self):
        self.ready_provider.return_value = {
            "ok": False,
            "errors": ["Odometry is inactive"],
        }
        with self.assertRaisesRegex(RuntimeError, "Odometry"):
            self.orchestrator.prepare()
        self.assertEqual(self.kill.call_count, len(COMPONENTS))
        self.assertFalse(self.orchestrator.stack_state_path.exists())

    def test_prepare_refuses_when_another_prepare_holds_the_lock(self):
        self.orchestrator.runtime_dir.mkdir(parents=True)
        self.orchestrator.prepare_lock_path.write_text(
            json.dumps({"pid": 9000, "token": "other"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "already running"):
            self.orchestrator.prepare()
        self.popen.assert_not_called()

    def test_finish_stops_before_exporting(self):
        self.recorder.status.return_value = {"active": True}
        self.orchestrator.runtime_dir.mkdir(parents=True)
        self.orchestrator.capture_state_path.write_text(
            json.dumps({"session_dir": str(self.root / "session")}),
            encoding="utf-8",
        )
        result = self.orchestrator.finish()
        self.assertEqual(self.calls, ["stop", "export"])
        self.assertEqual(result["frame_count"], 42)
        self.assertFalse(self.orchestrator.capture_state_path.exists())

    def test_stop_fast_defers_export_and_releases_capture_marker(self):
        self.recorder.status.return_value = {"active": True, "stale": False}
        self.orchestrator.runtime_dir.mkdir(parents=True)
        self.orchestrator.capture_state_path.write_text(
            json.dumps({"session_dir": str(self.root / "session"), "session_name": "demo"}),
            encoding="utf-8",
        )
        result = self.orchestrator.stop_fast()
        self.assertEqual(self.calls, ["stop"])
        self.assertEqual(result["session_dir"], str(self.root / "session"))
        self.assertEqual(result["postprocess"], "pending")
        self.assertFalse(self.orchestrator.capture_state_path.exists())
        self.exporter.assert_not_called()
        self.kill.assert_not_called()

    def test_finalize_raw_does_not_encode_video(self):
        self.orchestrator.finalize_raw(self.root / "session")
        self.bag_ready.assert_called_once()
        self.exporter.assert_not_called()

    def test_finish_waits_for_final_bag_before_exporting(self):
        self.recorder.status.return_value = {"active": True}
        self.bag_ready.side_effect = [False, False, True]
        self.orchestrator.runtime_dir.mkdir(parents=True)
        self.orchestrator.capture_state_path.write_text(
            json.dumps({"session_dir": str(self.root / "session")}),
            encoding="utf-8",
        )
        result = self.orchestrator.finish()
        self.assertEqual(result["frame_count"], 42)
        self.assertEqual(self.bag_ready.call_count, 3)

    def test_finish_recovers_export_after_recording_already_stopped(self):
        self.recorder.status.return_value = {"active": False}
        self.orchestrator.runtime_dir.mkdir(parents=True)
        self.orchestrator.capture_state_path.write_text(
            json.dumps({"session_dir": str(self.root / "session")}),
            encoding="utf-8",
        )
        result = self.orchestrator.finish()
        self.recorder.stop.assert_not_called()
        self.assertEqual(result["frame_count"], 42)
        self.assertFalse(self.orchestrator.capture_state_path.exists())

    def test_recover_interrupted_bag_preserves_original_and_publishes_bag(self):
        session = self.root / "interrupted"
        raw = session / "raw"
        raw.mkdir(parents=True)
        source = raw / "flight_0.bag.active"
        source.write_bytes(b"original-unindexed")

        def reindex(active, output_dir):
            output_dir.mkdir(parents=True)
            (output_dir / active.name).write_bytes(b"reindexed")

        recovered = recover_interrupted_bags(session, reindexer=reindex)

        self.assertEqual(recovered, [raw / "flight_0.bag"])
        self.assertEqual(source.read_bytes(), b"original-unindexed")
        self.assertEqual((raw / "flight_0.bag").read_bytes(), b"reindexed")
        self.assertFalse(list((session / "recovery").glob(".reindex-*")))

    def test_finish_recovers_active_bag_before_export(self):
        session = self.root / "session"
        raw = session / "raw"
        raw.mkdir(parents=True)
        (raw / "flight_0.bag.active").write_bytes(b"original-unindexed")
        self.recorder.status.return_value = {"active": False, "stale": False}
        self.bag_ready.side_effect = lambda path: any(Path(path).joinpath("raw").glob("*.bag"))
        self.orchestrator.runtime_dir.mkdir(parents=True)
        self.orchestrator.capture_state_path.write_text(
            json.dumps({"session_dir": str(session)}), encoding="utf-8"
        )
        self.orchestrator.bag_finalize_timeout = 0
        self.orchestrator.bag_reindexer = lambda active, output: (
            output.mkdir(parents=True, exist_ok=True),
            (output / active.name).write_bytes(b"reindexed"),
        )

        result = self.orchestrator.finish()

        self.assertEqual(result["frame_count"], 42)
        self.assertEqual((raw / "flight_000000.bag").read_bytes(), b"reindexed")
        self.assertFalse(self.orchestrator.capture_state_path.exists())

    def test_shutdown_refuses_while_recording(self):
        self.recorder.status.return_value = {"active": True}
        with self.assertRaisesRegex(RuntimeError, "recording"):
            self.orchestrator.shutdown()
        self.kill.assert_not_called()

    def test_shutdown_never_kills_untracked_launch_processes(self):
        """Matching a command is not sufficient proof of ownership."""
        self.listed["d435i"] = [8123]
        self.orchestrator.process_lister = (
            lambda component: list(self.listed[component.name])
        )
        self.orchestrator.getpgid_func = lambda pid: pid
        self.kill.side_effect = lambda pgid, _signal: [
            values.remove(pgid) for values in self.listed.values() if pgid in values
        ]

        result = self.orchestrator.shutdown()

        self.kill.assert_not_called()
        self.assertEqual(result["stopped_components"], [])


if __name__ == "__main__":
    unittest.main()
