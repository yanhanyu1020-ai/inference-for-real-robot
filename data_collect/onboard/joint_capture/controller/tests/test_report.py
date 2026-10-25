from __future__ import annotations

import json
from pathlib import Path
import io
import sys
import types
import shlex
import pickle
from dataclasses import replace

import pytest

from jointctl.cli import build_alignment_report, main, write_alignment_report
from jointctl.inspectors import P450Inspector, StreamSummary, UnitreeInspector, _remote_script
from jointctl.models import ClockSample, EpisodeManifest, EpisodeState
from jointctl.manifest import ManifestStore


def test_p450_inspector_sources_ros_and_preserves_directory_argument():
    directory = "/data/episode with spaces/'quoted'"
    calls = []

    def runner(host, command, timeout):
        calls.append((host, command, timeout))
        return types.SimpleNamespace(ok=True, stdout="[]")

    assert P450Inspector(runner=runner).summarize(directory) == []
    outer = shlex.split(calls[0][1])
    assert outer[:2] == ["bash", "-c"]
    assert outer[2].startswith("source /opt/ros/noetic/setup.bash && ")
    assert "source /home/amov/p450_experiment/devel/setup.bash" in outer[2]
    python_command = outer[2].split("fi && ", 1)[1]
    assert shlex.split(python_command) == ["python3", "-c", _remote_script("p450"), directory]


def _sample(host: str, remote_ns: int, offset_ns: int = 100) -> ClockSample:
    return ClockSample(
        host=host,
        sequence=remote_ns,
        local_send_wall_ns=remote_ns - offset_ns,
        local_send_mono_ns=remote_ns,
        remote_receive_wall_ns=remote_ns,
        remote_send_wall_ns=remote_ns,
        remote_monotonic_ns=remote_ns,
        local_receive_wall_ns=remote_ns - offset_ns,
        local_receive_mono_ns=remote_ns,
        rtt_ns=0,
        offset_ns=offset_ns,
    )


def _fixture(tmp_path: Path) -> Path:
    store = ManifestStore(tmp_path)
    manifest = EpisodeManifest(
        episode_id="joint_report",
        label="demo",
        mode="joint",
        created_desktop_ns=1,
        state=EpisodeState.RECORDING,
        t0_desktop_ns=1_000,
        t1_desktop_ns=5_000,
        clock_samples={"p450": _sample("p450", 1_100), "unitree": _sample("unitree", 1_200)},
    )
    store.create(manifest)
    episode = store.episode_dir(manifest.episode_id)
    (episode / "clock_p450.jsonl").write_text(
        json.dumps(manifest.clock_samples["p450"].to_dict()) + "\n", encoding="utf-8"
    )
    (episode / "clock_unitree.jsonl").write_text(
        json.dumps(manifest.clock_samples["unitree"].to_dict()) + "\n", encoding="utf-8"
    )
    return episode


def test_alignment_report_contains_required_time_sources(tmp_path):
    episode = _fixture(tmp_path)
    report = build_alignment_report(episode)
    assert report["valid_interval"] is None
    assert report["requested_interval"] == {
        "start_inclusive_ns": 1_000, "end_exclusive_ns": 5_000,
    }
    assert report["quality"]["degraded"] is True
    assert "no stream summaries" in report["quality"]["degradation_reasons"]
    assert report["p450"]["preferred_timestamp"] == "ros_header"
    assert report["unitree"]["frame_timestamp"] == "camera.wall_time_ns"
    assert "estimated_error_ns" in report["quality"]
    assert report["quality"]["estimated_error_ns"] is None
    assert report["quality"]["data_endpoint_estimated_error_ns"] is None


def test_alignment_report_rejects_negative_t0_guard(tmp_path):
    with pytest.raises(ValueError, match="t0_guard_ns"):
        build_alignment_report(_fixture(tmp_path), t0_guard_ns=-1)


def test_write_alignment_report_is_atomic_and_preserves_raw_jsonl(tmp_path):
    episode = _fixture(tmp_path)
    raw = (episode / "clock_p450.jsonl").read_text(encoding="utf-8")
    output = write_alignment_report(episode)
    assert output == episode / "alignment.json"
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["episode_id"] == "joint_report"
    assert persisted["mappings"]["p450"]["anchors"]
    assert "valid_domain" in persisted["mappings"]["unitree"]
    assert (episode / "clock_p450.jsonl").read_text(encoding="utf-8") == raw
    assert not list(episode.glob("alignment.json.tmp"))


def test_remote_inspectors_parse_json_without_local_rosbag_or_unitree(tmp_path):
    responses = {
        "p450": json.dumps([{"name": "/cam", "count": 2, "first_ns": 10, "last_ns": 20}]),
        "unitree": json.dumps([{"name": "frames.csv", "count": 3, "first_ns": 30, "last_ns": 40}]),
    }

    def runner(host, command, timeout):
        return type("Result", (), {"returncode": 0, "stdout": responses[host], "stderr": ""})()

    p450 = P450Inspector(runner=runner).summarize("/data/p450_episode")
    unitree = UnitreeInspector(runner=runner).summarize("/data/unitree_episode")
    assert p450[0].count == 2 and p450[0].first_ns <= p450[0].last_ns
    assert unitree[0].count == 3 and unitree[0].first_ns <= unitree[0].last_ns


def _run_remote_script(kind: str, path: Path, monkeypatch, modules: dict[str, object] | None = None):
    """Execute the exact SSH payload locally with fake remote-only imports."""
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "argv", ["remote-inspector", str(path)])
    monkeypatch.setattr(sys, "stdout", stdout)
    for name, module in (modules or {}).items():
        monkeypatch.setitem(sys.modules, name, module)
    exec(compile(_remote_script(kind), f"<{kind}-inspector>", "exec"), {"__name__": "__main__"})
    return json.loads(stdout.getvalue())


@pytest.mark.parametrize("backward", [False, True])
def test_p450_remote_payload_aggregates_topics_across_bag_volumes_and_falls_back(monkeypatch, tmp_path, backward):
    first_bag = tmp_path / "capture_0.bag"
    second_bag = tmp_path / "capture_1.bag"
    first_bag.touch(); second_bag.touch()

    class Stamp:
        def __init__(self, value): self.value = value
        def to_nsec(self): return self.value

    class Message:
        def __init__(self, header_ns):
            self.header = types.SimpleNamespace(stamp=Stamp(header_ns)) if header_ns is not None else None

    class BagTime:
        def __init__(self, value): self.value = value
        def to_nsec(self): return self.value

    records = {
        str(first_bag): {"/camera": [(Message(1_000_000_000_000), BagTime(1_000_080_000_000)), (Message(0), BagTime(1_000_200_000_000))]},
        str(second_bag): {"/camera": [(Message(1_000_500_000_000), BagTime(1_000_580_000_000))], "/imu": [(Message(None), BagTime(400))]},
    }
    if backward:
        records[str(second_bag)]["/camera"] = [(Message(999_900_000_000), BagTime(1_000_580_000_000))]
    class Bag:
        def __init__(self, name): self.name = name
        def get_type_and_topic_info(self): return (None, {topic: object() for topic in records[self.name]})
        def read_messages(self, topics):
            for item in records[self.name][topics[0]]: yield (topics[0], *item)
        def close(self): pass

    if backward:
        with pytest.raises(ValueError, match="header timestamp moved backwards"):
            _run_remote_script("p450", tmp_path, monkeypatch, {"rosbag": types.SimpleNamespace(Bag=Bag)})
        return
    payload = _run_remote_script("p450", tmp_path, monkeypatch,
                                 {"rosbag": types.SimpleNamespace(Bag=Bag)})
    by_name = {item["name"]: item for item in payload}
    assert by_name["/camera"] == {
        "name": "/camera", "count": 3, "first_ns": 1_000_000_000_000, "last_ns": 1_000_500_000_000,
        "timestamp_source": "mixed",
        "max_gap_ns": 300_000_000, "nonmonotonic_count": 0,
    }
    assert by_name["/imu"]["count"] == 1
    assert by_name["/imu"]["timestamp_source"] == "bag_timestamp"


def test_unitree_remote_payload_aggregates_multiple_pkls_by_stream(monkeypatch, tmp_path):
    (tmp_path / "front").mkdir(); (tmp_path / "rear").mkdir()
    for name in ("front/frame_1700000000000.pkl", "front/frame_1700000000200.pkl",
                 "rear/frame_1700000000100.pkl"):
        camera = name.split('/')[0]
        timestamp = int(Path(name).stem.split('_')[-1])
        (tmp_path / name).write_bytes(pickle.dumps({'camera': {camera: {'wall_time_ns': timestamp}}}))
    payload = _run_remote_script("unitree", tmp_path, monkeypatch)
    by_name = {item["name"]: item for item in payload}
    assert by_name["pkl:front"] == {
        "name": "pkl:front", "count": 2, "first_ns": 1_700_000_000_000,
        "last_ns": 1_700_000_000_200, "timestamp_source": "camera.wall_time_ns",
        "max_gap_ns": 200, "nonmonotonic_count": 0,
    }
    assert by_name["pkl:rear"]["count"] == 1


def test_report_short_episode_persists_complete_mapping_and_marks_data_endpoint_error(tmp_path):
    episode = _fixture(tmp_path)
    report = build_alignment_report(episode, stream_summaries={
        "p450": [StreamSummary("/cam", 1, -19_998_900, -19_998_899)],
        "unitree": [StreamSummary("pkl:front", 1, -19_998_800, -19_998_799)],
    })
    assert report["valid_interval"] is None
    assert "stream coverage has no common interval" in report["quality"]["degradation_reasons"]
    assert report["mappings"]["p450"]["anchors"]
    assert report["mappings"]["p450"]["valid_domain"]["remote_start_ns"] == 1_100
    assert report["quality"]["data_endpoint_estimated_error_ns"] > 10_000_000
    assert report["quality"]["degraded"] is True


def test_short_episode_reference_estimate_uses_degraded_single_anchor_mapping(tmp_path):
    episode = _fixture(tmp_path)
    for host, remote_ns in (("p450", 1_200), ("unitree", 1_300)):
        sample = _sample(host, remote_ns)
        with (episode / f"clock_{host}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample.to_dict()) + "\n")
    report = build_alignment_report(episode, stream_summaries={
        "p450": [StreamSummary("/cam", 1, 1_100, 1_200)],
        "unitree": [StreamSummary("pkl:front", 1, 1_200, 1_300)],
    })
    for host in ("p450", "unitree"):
        assert report["mappings"][host]["reference_estimate"]["degraded"] is True
    assert "constant-offset mapping has no drift evidence" in report["quality"]["degradation_reasons"]


def test_report_with_streams_without_common_coverage_keeps_requested_interval_invalid(tmp_path):
    report = build_alignment_report(_fixture(tmp_path), stream_summaries={
        "p450": [StreamSummary("/cam", 1, 1_100, 1_101)],
        "unitree": [StreamSummary("pkl:front", 1, 1_500, 1_501)],
    })
    assert report["valid_interval"] is None
    assert report["requested_interval"] == {
        "start_inclusive_ns": 1_000, "end_exclusive_ns": 5_000,
    }
    assert "stream coverage has no common interval" in report["quality"]["degradation_reasons"]


def test_report_with_missing_host_clock_data_keeps_mapping_and_interval_unavailable(tmp_path):
    episode = _fixture(tmp_path)
    payload = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    payload["clock_samples"].pop("unitree")
    (episode / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (episode / "clock_unitree.jsonl").unlink()
    report = build_alignment_report(episode, stream_summaries={
        "p450": [StreamSummary("/cam", 1, 1_100, 1_200)],
        "unitree": [StreamSummary("pkl:front", 1, 1_200, 1_300)],
    })
    assert report["valid_interval"] is None
    assert report["mappings"]["unitree"]["unavailable"] is True
    assert "missing clock samples for unitree" in report["quality"]["degradation_reasons"]
    assert report["quality"]["estimated_error_ns"] is None
    assert report["quality"]["clock_estimated_error_ns"] is None


def test_report_serializes_rtt_residual_and_drift_diagnostics(tmp_path):
    episode = _fixture(tmp_path)
    for host, remote_ns, rtt_ns in (("p450", 5_000_001_100, 20), ("unitree", 5_000_001_200, 40)):
        sample = replace(_sample(host, remote_ns, offset_ns=5_000_100), rtt_ns=rtt_ns)
        with (episode / f"clock_{host}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample.to_dict()) + "\n")
    output = write_alignment_report(episode)
    diagnostics = json.loads(output.read_text(encoding="utf-8"))["clock_diagnostics"]
    assert diagnostics["p450"]["rtt_ns"] == {"min": 0, "median": 10, "p95": 20}
    assert diagnostics["unitree"]["fit_residual_ns"]["max"] == 0
    assert diagnostics["p450"]["drift_ppm"]["representative"] == 1000
    assert diagnostics["p450"]["uncertainty_components"] == {
        "network": "rtt_ns", "fit": "fit_residual_ns",
    }


def test_report_records_reason_for_degraded_extrapolation(tmp_path):
    episode = _fixture(tmp_path)
    for host, remote_ns in (("p450", 6_000_001_100), ("unitree", 6_000_001_200)):
        with (episode / f"clock_{host}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_sample(host, remote_ns).to_dict()) + "\n")
    report = build_alignment_report(episode, stream_summaries={
        "p450": [StreamSummary("/cam", 1, 20_000_001_100, 20_000_001_101)],
        "unitree": [StreamSummary("pkl:front", 1, 20_000_001_200, 20_000_001_201)],
    })
    assert "timestamp extrapolates beyond mapping domain" in report["quality"]["degradation_reasons"]


def test_valid_interval_uses_worst_middle_mapping_uncertainty_and_degrades(tmp_path):
    episode = _fixture(tmp_path)
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    manifest["t1_desktop_ns"] = 20_000_000_000
    (episode / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for host, first_ns in (("p450", 1_100), ("unitree", 1_200)):
        middle = replace(_sample(host, first_ns + 5_000_000_000), rtt_ns=30_000_000)
        last = _sample(host, first_ns + 10_000_000_000)
        with (episode / f"clock_{host}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(middle.to_dict()) + "\n")
            handle.write(json.dumps(last.to_dict()) + "\n")
    report = build_alignment_report(episode, stream_summaries={
        "p450": [StreamSummary("/cam", 3, 1_100, 10_000_001_100)],
        "unitree": [StreamSummary("pkl:front", 3, 1_200, 10_000_001_200)],
    })
    assert report["coverage_interval"] is not None
    assert report["valid_interval"] is None  # Incomplete synthetic camera set.
    assert report["quality"]["interval_mapping_estimated_error_ns"] >= 15_000_000
    assert report["quality"]["estimated_error_ns"] >= 15_000_000
    assert report["quality"]["degraded"] is True
    assert "valid interval mapping uncertainty exceeds 10 ms" in report["quality"]["degradation_reasons"]


def test_align_accepts_completed_episode_id_and_passes_timing_config(tmp_path, monkeypatch):
    episode = _fixture(tmp_path)
    store = ManifestStore(tmp_path)
    store.update("joint_report", state=EpisodeState.STOPPING)
    store.update("joint_report", state=EpisodeState.COMPLETE)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"manifest_root": str(tmp_path), "alignment_window_ns": 123, "t0_guard_ns": 7}), encoding="utf-8")
    received = {}
    def write_stub(path, **kwargs):
        received.update(kwargs)
        output = Path(path) / "alignment.json"
        output.write_text(json.dumps({"quality": {"degraded": False}}), encoding="utf-8")
        return output
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: object())
    monkeypatch.setattr("jointctl.cli.write_alignment_report", write_stub)
    assert main(["--config", str(config), "align", "joint_report", "--no-remote-inspect"]) == 0
    assert received["alignment_window_ns"] == 123
    assert received["t0_guard_ns"] == 7


@pytest.mark.parametrize("single_anchor, start_ns", [(True, 2_000), (False, 100), (False, 3_000)])
def test_valid_interval_outside_anchor_domain_keeps_numeric_uncertainty(tmp_path, single_anchor, start_ns):
    episode = _fixture(tmp_path)
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    manifest["t0_desktop_ns"] = start_ns
    (episode / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if not single_anchor:
        for host in ("p450", "unitree"):
            with (episode / f"clock_{host}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_sample(host, 2_100).to_dict()) + "\n")
    report = build_alignment_report(episode, alignment_window_ns=500, stream_summaries={
        host: [StreamSummary("stream", 2, start_ns + 100, start_ns + 600)]
        for host in ("p450", "unitree")
    })
    assert report["coverage_interval"] == {
        "start_inclusive_ns": start_ns, "end_exclusive_ns": start_ns + 500,
    }
    assert isinstance(report["quality"]["interval_mapping_estimated_error_ns"], int)
    assert report["quality"]["estimated_error_ns"] >= (2_900 if single_anchor else 2)


def test_relative_error_adds_host_bounds_and_triggers_timing_degradation(tmp_path):
    episode = _fixture(tmp_path)
    # Each host contributes 6 ms, so their relative alignment can be wrong by
    # 12 ms even though neither individual mapping exceeds 10 ms.
    for host in ("p450", "unitree"):
        rows = [replace(_sample(host, remote_ns), rtt_ns=12_000_000)
                for remote_ns in (1_100, 6_100)]
        (episode / f"clock_{host}.jsonl").write_text(
            "".join(json.dumps(row.to_dict()) + "\n" for row in rows), encoding="utf-8"
        )
    report = build_alignment_report(episode, alignment_window_ns=500, stream_summaries={
        host: [StreamSummary("stream", 2, 2_100, 5_100)]
        for host in ("p450", "unitree")
    })
    quality = report["quality"]
    assert quality["estimated_error_ns"] == 12_000_002
    assert quality["per_host_estimated_error_ns"] == {"p450": 6_000_001, "unitree": 6_000_001}
    assert quality["degraded"] is True
    assert "estimated error exceeds 10 ms" in quality["degradation_reasons"]


def test_align_returns_timing_exit_when_mapped_data_endpoint_exceeds_ten_ms(tmp_path, monkeypatch):
    episode = _fixture(tmp_path)
    manifest = EpisodeManifest.from_dict(json.loads((episode / "manifest.json").read_text(encoding="utf-8")))
    (episode / "manifest.json").write_text(json.dumps(EpisodeManifest(
        **{**manifest.to_dict(), "remote_directories": {"p450": "/p450", "unitree": "/unitree"}}
    ).to_dict()), encoding="utf-8")
    class Inspector:
        def __init__(self, *args, **kwargs): pass
        def summarize(self, _directory):
            return [StreamSummary("stream", 1, -19_998_900, -19_998_899)]
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: object())
    monkeypatch.setattr("jointctl.cli.P450Inspector", Inspector)
    monkeypatch.setattr("jointctl.cli.UnitreeInspector", Inspector)
    assert main(["--manifest-root", str(tmp_path), "align", "--episode-dir", str(episode)]) == 5


def test_align_returns_timing_exit_when_no_stream_summaries_exist(tmp_path, monkeypatch):
    episode = _fixture(tmp_path)
    monkeypatch.setattr("jointctl.cli.make_controller", lambda _: object())
    assert main(["--manifest-root", str(tmp_path), "align", "--episode-dir", str(episode),
                 "--no-remote-inspect"]) == 5
