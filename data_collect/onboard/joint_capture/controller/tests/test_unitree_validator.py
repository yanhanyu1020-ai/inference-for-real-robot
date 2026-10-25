"""Exercise the deployed Unitree validator on stopped recorder episodes."""
import importlib.util
import json
import io
import pickle
from pathlib import Path

import pytest
from PIL import Image


spec = importlib.util.spec_from_file_location(
    "unitree_validator", Path(__file__).parents[1] / "remote/unitree/validate_episode.py"
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
CONFIG = {"cameras": {"front": "front-serial", "wrist": "wrist-serial"},
          "fps": 10, "max_camera_skew_ms": 50}


def episode_fixture(tmp_path, timestamps=(20_000_000_000, 20_100_000_000)):
    (tmp_path / "raw").mkdir()
    (tmp_path / "frames").mkdir()
    for name in validator.REQUIRED_RAW_FILES:
        (tmp_path / "raw" / name).write_text(
            "wall_time_ns\n2000000000\n20000000000\n20100000000\n", encoding="utf-8"
        )
    png = io.BytesIO()
    Image.new('RGB', (2, 2), (1, 2, 3)).save(png, format='PNG')
    for index, stamp in enumerate(timestamps):
        record = {"timestamp_ns": stamp, "camera": {
            name: {"serial": serial, "wall_time_ns": stamp, "monotonic_ns": stamp,
                   "rgb": png.getvalue()}
            for name, serial in CONFIG["cameras"].items()},
            "diagnostics": {"camera_skew_ns": 0}}
        (tmp_path / "frames" / f"frame_{index:04}.pkl").write_bytes(pickle.dumps(record))
    (tmp_path / "summary.json").write_text(json.dumps({
        "frames_saved": len(timestamps), "save_errors": 0, "piper_error": None,
    }), encoding="utf-8")
    return tmp_path


def test_png_signature_does_not_prove_decodable_image(tmp_path):
    episode = episode_fixture(tmp_path)
    path = next((episode / 'frames').glob('*.pkl'))
    record = pickle.loads(path.read_bytes())
    record['camera']['front']['rgb'] = b'\x89PNG\r\n\x1a\nfixture'
    path.write_bytes(pickle.dumps(record))
    report = validator.validate(episode, CONFIG, False)
    assert not report['ok']
    assert any('PNG' in message for message in report['errors'])


def test_png_crc_corruption_rejected(tmp_path):
    episode = episode_fixture(tmp_path)
    path = next((episode / 'frames').glob('*.pkl'))
    record = pickle.loads(path.read_bytes())
    png = bytearray(record['camera']['front']['rgb'])
    png[29] ^= 1  # IHDR CRC, not the PNG signature.
    record['camera']['front']['rgb'] = bytes(png)
    path.write_bytes(pickle.dumps(record))
    assert not validator.validate(episode, CONFIG, False)['ok']


def test_single_frame_cannot_validate_as_complete(tmp_path):
    report = validator.validate(episode_fixture(tmp_path, (20_000_000_000,)), CONFIG, False)
    assert report["ok"] is False
    assert any("at least two" in error for error in report["errors"])
    assert any("rate too low" in error for error in report["errors"])


@pytest.mark.parametrize("summary", [None, {"frames_saved": 2, "save_errors": 1, "piper_error": None},
    {"frames_saved": 2, "save_errors": 0, "piper_error": "CAN read failed"},
    {"frames_saved": 3, "save_errors": 0, "piper_error": None}, {}])
def test_recorder_summary_must_confirm_successful_complete_save(tmp_path, summary):
    episode = episode_fixture(tmp_path)
    if summary is None:
        (episode / "summary.json").unlink()
    else:
        (episode / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = validator.validate(episode, CONFIG, False)
    assert report["ok"] is False
    assert any("summary" in error for error in report["errors"])


def test_recorder_reported_dropped_frames_fail_validation(tmp_path):
    episode = episode_fixture(tmp_path)
    summary = json.loads((episode / "summary.json").read_text(encoding="utf-8"))
    summary.update(frames_enqueued=3, frames_dropped=1)
    (episode / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = validator.validate(episode, CONFIG, False)
    assert report["ok"] is False
    assert any("frames_dropped" in error for error in report["errors"])


@pytest.mark.parametrize("name", ["sport_mode_state.csv", "low_state.csv", "piper_state.csv"])
@pytest.mark.parametrize("rows", ["2000000000\n19000000000", "21000000000\n22000000000"])
def test_critical_state_csv_must_cover_frame_interval(tmp_path, name, rows):
    episode = episode_fixture(tmp_path)
    (episode / "raw" / name).write_text("wall_time_ns\n" + rows + "\n", encoding="utf-8")
    report = validator.validate(episode, CONFIG, False)
    assert report["ok"] is False
    assert any(name in error and "cover" in error for error in report["errors"])


def test_valid_episode_allows_state_startup_lead_and_endpoint_tolerance(tmp_path):
    episode = episode_fixture(tmp_path)
    (episode / "raw/piper_state.csv").write_text(
        "wall_time_ns\n20250000000\n20300000000\n", encoding="utf-8"
    )
    report = validator.validate(episode, CONFIG, True)
    assert report["ok"] is True
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    assert "summary.json" in {item["path"] for item in manifest["files"]}
