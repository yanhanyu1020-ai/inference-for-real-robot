"""Run deployed inspector programs against recorder-shaped files."""
import io
import json
import pickle
import sys
import types

import pytest

from jointctl.inspectors import UnitreeInspector, _remote_script


def run_payload(kind, path, monkeypatch, modules=None):
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "argv", ["inspect", str(path)])
    monkeypatch.setattr(sys, "stdout", stdout)
    for name, module in (modules or {}).items():
        monkeypatch.setitem(sys.modules, name, module)
    exec(compile(_remote_script(kind), "<remote-inspector>", "exec"), {})
    return json.loads(stdout.getvalue())


def write_frame(path, front, wrist):
    path.write_bytes(pickle.dumps({
        "timestamp_ns": max(front, wrist),
        "camera": {name: {"wall_time_ns": stamp, "monotonic_ns": stamp + 100,
                          "rgb": b"encoded PNG", "serial": name}
                   for name, stamp in (("front", front), ("wrist", wrist))},
        "diagnostics": {"skew_ns": abs(front - wrist)},
    }, protocol=5))


def test_unitree_reads_each_camera_timestamp_and_internal_gap(monkeypatch, tmp_path):
    # Filename order must be numeric; using packet max hides front delay.
    write_frame(tmp_path / "frame_2.pkl", 100, 120)
    write_frame(tmp_path / "frame_10.pkl", 110, 140)
    write_frame(tmp_path / "frame_11.pkl", 190, 160)
    rows = {row["name"]: row for row in run_payload("unitree", tmp_path, monkeypatch)}
    assert set(rows) == {"pkl:front", "pkl:wrist"}
    assert (rows["pkl:front"]["first_ns"], rows["pkl:front"]["last_ns"]) == (100, 190)
    assert rows["pkl:front"]["max_gap_ns"] == 80
    assert rows["pkl:wrist"]["max_gap_ns"] == 20
    assert rows["pkl:wrist"]["nonmonotonic_count"] == 0
    assert rows["pkl:front"]["timestamp_source"] == "camera.wall_time_ns"


def test_unitree_inspector_exposes_recorder_drops(monkeypatch, tmp_path):
    write_frame(tmp_path / "frame_1.pkl", 100, 100)
    write_frame(tmp_path / "frame_2.pkl", 200, 200)
    (tmp_path / "summary.json").write_text(json.dumps({
        "frames_saved": 2, "frames_dropped": 1,
    }), encoding="utf-8")
    rows = {row["name"]: row for row in run_payload("unitree", tmp_path, monkeypatch)}
    assert rows["pkl:front"]["producer_dropped"] == 1
    assert rows["pkl:wrist"]["producer_dropped"] == 1


@pytest.mark.parametrize("content", [b"", pickle.dumps({"timestamp_ns": 123}),
                                    pickle.dumps({"camera": {"front": {"wall_time_ns": "invalid"}}})])
def test_unitree_malformed_pickle_fails_instead_of_filename_fallback(monkeypatch, tmp_path, content):
    (tmp_path / "frame_1700000000000.pkl").write_bytes(content)
    with pytest.raises(ValueError, match="pkl"):
        run_payload("unitree", tmp_path, monkeypatch)


def test_unitree_preserves_backwards_and_duplicate_camera_timestamps(monkeypatch, tmp_path):
    write_frame(tmp_path / "frame_1.pkl", 100, 110)
    write_frame(tmp_path / "frame_2.pkl", 90, 110)
    rows = run_payload("unitree", tmp_path, monkeypatch)
    assert all(row["nonmonotonic_count"] == 1 for row in rows)
    assert rows[0]["first_ns"] == 90


@pytest.mark.parametrize("stamps,count,gap,nonmonotonic", [([100], 1, 0, 0), ([100, 150, 120, 120], 4, 50, 2)])
def test_csv_reports_singletons_and_nonmonotonic_capture_order(monkeypatch, tmp_path, stamps, count, gap, nonmonotonic):
    (tmp_path / "frames.csv").write_text("wall_time_ns\n" + "\n".join(map(str, stamps)), encoding="utf-8")
    row, = run_payload("unitree", tmp_path, monkeypatch)
    assert row["count"] == count
    assert row["max_gap_ns"] == gap
    assert row["nonmonotonic_count"] == nonmonotonic
    assert row["last_ns"] == max(stamps)


def test_p450_reports_per_topic_gaps_across_numeric_bag_volumes(monkeypatch, tmp_path):
    for name in ("capture_2.bag", "capture_10.bag"):
        (tmp_path / name).touch()
    class Bag:
        def __init__(self, name): self.name = name
        def get_type_and_topic_info(self): return None, {"/cam": object(), "/imu": object()}
        def read_messages(self, topics):
            stamps = ([100, 140] if self.name.endswith("_2.bag") else [200, 200]) if topics == ["/cam"] else [100, 110]
            for stamp in stamps:
                yield topics[0], object(), types.SimpleNamespace(to_nsec=lambda s=stamp: s)
        def close(self): pass
    rows = {row["name"]: row for row in run_payload("p450", tmp_path, monkeypatch, {"rosbag": types.SimpleNamespace(Bag=Bag)})}
    assert rows["/cam"]["max_gap_ns"] == 60
    assert rows["/cam"]["nonmonotonic_count"] == 1
    assert rows["/imu"]["max_gap_ns"] == 10
    assert rows["/imu"]["nonmonotonic_count"] == 1


def test_inspector_parses_quality_fields_and_leaves_legacy_unknown():
    rows = [{"name": "front", "count": 3, "first_ns": 100, "last_ns": 190,
             "max_gap_ns": 80, "nonmonotonic_count": 1},
            {"name": "legacy", "count": 1, "first_ns": 1, "last_ns": 1}]
    inspector = UnitreeInspector(runner=lambda *args: types.SimpleNamespace(ok=True, stdout=json.dumps(rows)))
    quality, legacy = inspector.summarize("/data")
    assert quality.max_gap_ns == 80
    assert quality.nonmonotonic_count == 1
    assert legacy.max_gap_ns is None
    assert legacy.nonmonotonic_count is None


def test_inspector_retains_reported_producer_drops():
    rows = [{"name": "pkl:front", "count": 2, "first_ns": 100, "last_ns": 200,
             "timestamp_source": "camera.wall_time_ns", "max_gap_ns": 100,
             "nonmonotonic_count": 0, "producer_dropped": 1}]
    inspector = UnitreeInspector(runner=lambda *args: types.SimpleNamespace(ok=True, stdout=json.dumps(rows)))
    stream, = inspector.summarize("/data")
    assert stream.producer_dropped == 1
