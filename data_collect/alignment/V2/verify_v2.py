#!/usr/bin/env python3
"""Validate the V1+26 V2 release on CPU, without training or robot access."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


QUARANTINED_LEGACY_SOURCE = "task_03_1144-1300"
CAMERAS = (("front", "front"), ("wrist", "wrist"), ("p450", "uav"))


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("HETEROVLA_DATA_ROOT") or Path.home() / "heterovla-data/aligned/V2"))
    ap.add_argument("--source", type=Path, default=Path(os.environ.get("HETEROVLA_BENCHMARK_ROOT") or Path.home() / "heterovla-data/raw/benchmark"))
    ap.add_argument("--v1-root", type=Path, default=Path(os.environ.get("HETEROVLA_V1_ROOT") or Path.home() / "heterovla-data/aligned/V1"))
    ap.add_argument("--inventory", type=Path, help="Default: ROOT/configs/source_inventory.json")
    ap.add_argument("--train-scripts", type=Path, default=Path(os.environ.get("HETEROVLA_ROOT") or Path.home() / "heterovla") / "train/V2/scripts")
    ap.add_argument("--legacy-video-mae", type=float, default=1.0, help="Maximum old-V1/V2 per-frame mean absolute RGB error, in 0..255 pixels")
    ap.add_argument("--source-video-mae", type=float, default=8.0, help="Maximum raw/export sample mean absolute RGB error, in 0..255 pixels")
    args = ap.parse_args()
    if args.legacy_video_mae < 0 or args.source_video_mae <= 0:
        ap.error("Video error thresholds must be nonnegative (raw threshold strictly positive)")
    return args


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_directory(raw_root, source_path):
    relative = Path(source_path)
    assert not relative.is_absolute() and ".." not in relative.parts, source_path
    resolved = (raw_root / relative).resolve()
    assert resolved.is_relative_to(raw_root), source_path
    assert resolved.is_dir(), resolved
    return resolved


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_key(episode):
    return (episode["source_episode"], episode["split"], round(float(episode["source_start_s"]), 8), round(float(episode["source_last_s"]), 8), int(episode["frames"]))


def dataset_directory(root, split, version):
    return root / "lerobot/local" / f"heterovla_{version}_{split}"


def episode_file(directory, info, index, video_key=None):
    chunk = int(index) // int(info.get("chunks_size", 1000))
    if video_key is None:
        template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    else:
        template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return directory / template.format(episode_chunk=chunk, episode_index=int(index), video_key=video_key)


def check_video(path, expected_frames):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (640, 480), path
        count, last = 0, -1.0
        for frame in container.decode(video=0):
            timestamp = float(frame.pts * frame.time_base)
            assert timestamp > last, (path, count, timestamp)
            assert abs(timestamp - count / 10) <= 1e-4, (path, count, timestamp)
            last = timestamp
            count += 1
        assert count == expected_frames, (path, count, expected_frames)
    return {"frames": count, "width": 640, "height": 480}


def compare_legacy_video(old, new, expected_frames, maximum_mae):
    if file_sha256(old) == file_sha256(new):
        return {"sha256_equal": True, "decoded_pixels_equal": True, "frames": expected_frames, "max_frame_mae": 0.0}
    with av.open(str(old)) as left, av.open(str(new)) as right:
        old_frames, new_frames = iter(left.decode(video=0)), iter(right.decode(video=0))
        count, max_error, exact = 0, 0.0, True
        while True:
            a, b = next(old_frames, None), next(new_frames, None)
            assert (a is None) == (b is None), (old, new, count)
            if a is None:
                break
            aa, bb = a.to_ndarray(format="rgb24"), b.to_ndarray(format="rgb24")
            assert aa.shape == bb.shape
            same = np.array_equal(aa, bb)
            exact = exact and same
            error = float(np.mean(np.abs(aa.astype(np.int16) - bb.astype(np.int16))))
            assert error <= maximum_mae, (old, new, count, error, maximum_mae)
            max_error = max(max_error, error)
            count += 1
        assert count == expected_frames
    return {"sha256_equal": False, "decoded_pixels_equal": exact, "frames": count, "max_frame_mae": max_error}


def check_inventory(inventory, manifest, alignment_summary, root, v1_root, source):
    records = inventory["sources"]
    assert len(records) == 32, len(records)
    sources = {entry["source_episode"]: entry for entry in records}
    assert len(sources) == len(records), "Duplicate source_episode in inventory"
    assert len({entry["source_path"] for entry in records}) == len(records), "Duplicate raw source path"
    included = {key: entry for key, entry in sources.items() if entry["include"]}
    old = {key for key, entry in included.items() if entry["origin"] == "V1"}
    new = {key for key, entry in included.items() if entry["origin"] == "new"}
    assert (len(included), len(old), len(new)) == (31, 5, 26), (len(included), len(old), len(new))
    banned = sources[QUARANTINED_LEGACY_SOURCE]
    assert not banned["include"] and banned["quarantined"] and banned["split"] == "quarantine"
    assert QUARANTINED_LEGACY_SOURCE not in {entry["source_episode"] for entry in manifest}
    summary_records = alignment_summary["sources"]
    if isinstance(summary_records, dict):
        summaries = summary_records
    else:
        summaries = {entry["source_episode"]: entry for entry in summary_records}
    assert set(included).issubset(summaries), "A requested source is missing from the alignment summary"
    sessions = {split: set() for split in ("train", "val")}
    requested_split_counts = {split: sum(entry["split"] == split for entry in included.values()) for split in sessions}
    assert requested_split_counts == {"train": 26, "val": 5}, requested_split_counts
    new_validation = [entry for entry in included.values() if entry["origin"] == "new" and entry["split"] == "val"]
    assert len(new_validation) == 4
    assert {entry["source_session"] for entry in new_validation} == {"joint_20261012_155725_UTCp0800_faf6"}
    exportable, non_exportable = set(), []
    for key, entry in included.items():
        assert entry["split"] in sessions and not entry.get("quarantined", False), key
        assert entry["source_session"], key
        sessions[entry["split"]].add(entry["source_session"])
        raw = source_directory(source, entry["source_path"])
        aligned_path = root / "aligned" / f"{key}.parquet"
        assert aligned_path.is_file(), (key, "Requested source missing aligned audit rows")
        aligned_rows = pq.read_table(aligned_path).to_pylist()
        assert aligned_rows, (key, "Empty aligned audit table")
        assert all(row["source_episode"] == key and row["source_path"] == entry["source_path"] and row["source_session"] == entry["source_session"] and row["split"] == entry["split"] for row in aligned_rows)
        runs, current = [], []
        for row in aligned_rows:
            if row["eligible"]:
                if current:
                    assert abs(row["source_task_time_s"] - current[-1]["source_task_time_s"] - 0.1) < 1e-8
                current.append(row)
            else:
                if len(current) >= 10:
                    runs.append(current)
                current = []
        if len(current) >= 10:
            runs.append(current)
        expected_segments = {(round(run[0]["source_task_time_s"], 8), round(run[-1]["source_task_time_s"], 8), len(run)) for run in runs}
        actual_segments = {(round(episode["source_start_s"], 8), round(episode["source_last_s"], 8), int(episode["frames"])) for episode in manifest if episode["source_episode"] == key}
        assert actual_segments == expected_segments, (key, "Eligible aligned runs and export differ")
        summary = summaries[key]
        assert int(summary["eligible_frames"]) == sum(bool(row["eligible"]) for row in aligned_rows), key
        assert int(summary["exported_segments"]) == len(runs), key
        expected_frames = sum(len(run) for run in runs)
        summary_frames = summary.get("exported_frames", summary.get("frames"))
        assert summary_frames is not None and int(summary_frames) == expected_frames, key
        if runs:
            exportable.add(key)
        else:
            detail = {"source_episode": key, "source_path": entry["source_path"], "eligible_rows": sum(bool(row["eligible"]) for row in aligned_rows), "exported_segments": 0, "exported_frames": 0, "rejection_reasons": dict(Counter(reason for row in aligned_rows for reason in row["rejection_reasons"])), "reason": "No complete H=10 segment under unchanged V1 eligibility rules"}
            import pandas as pd
            command_csv = raw / "states/unitree/piper_gamepad.csv"
            commands = pd.read_csv(command_csv, usecols=["command_sent"])["command_sent"]
            detail.update(piper_gamepad_rows=len(commands), piper_command_sent_rows=int((commands == 1).sum()))
            if not detail["piper_command_sent_rows"]:
                detail["reason"] = "Piper command_sent has no valid commands; V1 rules keep every affected action invalid"
            non_exportable.append(detail)
        if key in old:
            old_rows = pq.read_table(v1_root / "aligned" / f"{key}.parquet").to_pylist()
            assert len(old_rows) == len(aligned_rows), key
            for previous, current_row in zip(old_rows, aligned_rows):
                for field, value in previous.items():
                    assert current_row[field] == value, (key, field, "Legacy aligned audit value changed")
    assert {entry["source_episode"] for entry in manifest} == exportable, "Exported source set differs from eligible H=10 sources"
    assert not sessions["train"] & sessions["val"], "Source session leakage in inventory"
    for entry in manifest:
        expected = included[entry["source_episode"]]
        for field in ("source_path", "source_session", "split"):
            assert entry[field] == expected[field], (entry["source_episode"], field)
    return sources, old, {"raw_sources": 32, "requested_sources": 31, "included_sources": 31, "exported_sources": len(exportable), "legacy_sources": 5, "new_sources": 26, "requested_split_sources": requested_split_counts, "non_exportable_source_episodes": non_exportable, "quarantined_sources": [QUARANTINED_LEGACY_SOURCE], "no_source_session_leakage": True}


def check_provenance(rows, episode, source):
    assert len(rows) == episode["frames"] and all(row["eligible"] for row in rows)
    times = np.asarray([row["source_task_time_s"] for row in rows], dtype=np.float64)
    assert np.isfinite(times).all() and np.allclose(np.diff(times), 0.1, atol=1e-8, rtol=0)
    assert abs(times[0] - episode["source_start_s"]) < 1e-8
    assert abs(times[-1] - episode["source_last_s"]) < 1e-8
    for row in rows:
        assert row["source_episode"] == episode["source_episode"]
        assert row["source_path"] == episode["source_path"]
        assert row["source_session"] == episode["source_session"]
        assert row["split"] == episode["split"]
        assert row["source_episode"] != QUARANTINED_LEGACY_SOURCE
        assert all(row["action_valid"]) and all(row["state_valid"]) and all(row["camera_valid"])
        assert not row["rejection_reasons"]
        assert row["action_is_kinematic_proxy"] == [True] * 3 + [False] * 7 + [True] * 4
        camera_times = np.asarray(row["camera_source_times_s"], dtype=np.float64)
        state_times = np.asarray(row["state_source_times_s"], dtype=np.float64)
        assert camera_times.shape == (3,) and state_times.shape == (5,)
        assert np.isfinite(camera_times).all() and np.isfinite(state_times).all()
        age = row["source_task_time_s"] - camera_times
        assert ((age >= -1e-9) & (age <= 0.15 + 1e-9)).all()
        assert float(np.ptp(camera_times)) <= 0.1 + 1e-9
        state_age = row["source_task_time_s"] - state_times
        assert (state_age >= -1e-9).all()
        assert (state_age <= np.asarray([0.075, 0.075, 0.075, 0.15, 0.25]) + 1e-9).all()
        assert all(int(frame) >= 0 for frame in row["camera_video_frames"])


def compare_legacy_episode(root, v1_root, new, old, new_data, new_rows, maximum_mae):
    split = new["split"]
    old_dir, new_dir = dataset_directory(v1_root, split, "v1"), dataset_directory(root, split, "v2")
    old_info, new_info = read_json(old_dir / "meta/info.json"), read_json(new_dir / "meta/info.json")
    old_data = pq.read_table(episode_file(old_dir, old_info, old["episode_index"])).to_pydict()
    numeric_fields = ["observation.state", "action", "action_valid", "source_task_time", "timestamp", "frame_index", "task_index"]
    for name in numeric_fields:
        assert np.array_equal(np.asarray(old_data[name]), np.asarray(new_data[name])), (new["source_episode"], name)
    old_rows = pq.read_table(v1_root / "provenance" / f"{split}_{old['episode_index']:06d}.parquet").to_pylist()
    assert len(old_rows) == len(new_rows)
    for old_row, new_row in zip(old_rows, new_rows):
        # The only added provenance field is source_path; every legacy field
        # retains its original numerical value and meaning.
        for name, value in old_row.items():
            assert new_row[name] == value, (new["source_episode"], name)
    videos = []
    for _, camera in CAMERAS:
        key = f"observation.images.{camera}"
        videos.append({"camera": camera, **compare_legacy_video(episode_file(old_dir, old_info, old["episode_index"], key), episode_file(new_dir, new_info, new["episode_index"], key), new["frames"], maximum_mae)})
    return {"source_episode": new["source_episode"], "split": split, "v1_episode_index": old["episode_index"], "v2_episode_index": new["episode_index"], "frames": new["frames"], "numerical_values_exact": True, "provenance_unchanged_except_source_path": True, "videos": videos}


def validate(args):
    global np, pq, av
    import numpy as np
    import pyarrow.parquet as pq
    import av
    from prepare_v2 import TASK

    root, source, v1_root = args.root.expanduser().resolve(), args.source.expanduser().resolve(), args.v1_root.expanduser().resolve()
    assert root != v1_root and not root.is_relative_to(v1_root), "V2 must be separate from immutable V1"
    # Reuse the mathematical invariants from V1's unchanged alignment methods.
    from prepare_v2 import rotate_xy, wrap, Series, causal_video
    import pandas as pd
    assert np.allclose(rotate_xy([0, 1], np.pi / 2), [1, 0], atol=1e-12)
    assert abs(wrap(-np.pi + 0.01 - (np.pi - 0.01)) - 0.02) < 1e-10
    series = Series(pd.DataFrame({"a": [0.0, 10.0, 20.0]}), [0.0, 0.1, 0.5], ["a"])
    assert series.past(0.099)[0][0] == 0 and series.past(-0.01) is None
    assert series.interpolate_target(0.3) is None and not series.interval_ok(0.1, 0.5)
    mapping = {"times": np.array([0.05, 0.15]), "frames": np.array([0, 1]), "common_rows": np.array([0, 1]), "warnings": np.array([False, False])}
    assert causal_video(mapping, 0.1)[0] == 0 and causal_video(mapping, 0.04) is None

    manifest = read_json(root / "reports/episode_manifest.json")
    inventory = read_json(args.inventory.expanduser().resolve() if args.inventory else root / "configs/source_inventory.json")
    alignment_summary = read_json(root / "reports/alignment_summary.json")
    sources, legacy_sources, inventory_result = check_inventory(inventory, manifest, alignment_summary, root, v1_root, source)
    legacy_manifest = read_json(v1_root / "reports/episode_manifest.json")
    assert {entry["source_episode"] for entry in legacy_manifest} == legacy_sources
    old_by_key = {episode_key(entry): entry for entry in legacy_manifest}
    assert len(old_by_key) == len(legacy_manifest)
    new_legacy_keys = {episode_key(entry) for entry in manifest if entry["source_episode"] in legacy_sources}
    assert new_legacy_keys == set(old_by_key), "A legacy V1 segment changed or disappeared"
    result = {"invariant_tests_passed": True, "source_inventory": inventory_result, "non_exportable_source_episodes": inventory_result["non_exportable_source_episodes"], "no_source_session_leakage": True, "datasets": {}, "videos": [], "legacy_v1_equivalence": [], "thresholds": {"legacy_video_frame_mae": args.legacy_video_mae, "raw_video_sample_mae": args.source_video_mae}}
    provenance = {}
    train_arrays = {"state": [], "actions": []}
    for split in ("train", "val"):
        directory = dataset_directory(root, split, "v2")
        info = read_json(directory / "meta/info.json")
        episodes = [entry for entry in manifest if entry["split"] == split]
        assert {int(entry["episode_index"]) for entry in episodes} == set(range(len(episodes)))
        assert len(episodes) == int(info["total_episodes"])
        frames = 0
        for episode in episodes:
            index, n = int(episode["episode_index"]), int(episode["frames"])
            assert n >= 10 and episode["valid_h10_windows"] == n - 9
            data = pq.read_table(episode_file(directory, info, index)).to_pydict()
            assert len(data["index"]) == n
            state, action = np.asarray(data["observation.state"]), np.asarray(data["action"])
            assert state.shape == (n, 30) and action.shape == (n, 14)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            assert np.asarray(data["action_valid"]).shape == (n, 14) and np.asarray(data["action_valid"]).all()
            assert np.allclose(np.asarray(data["timestamp"]), np.arange(n) / 10, atol=1e-4, rtol=0)
            rows = pq.read_table(root / "provenance" / f"{split}_{index:06d}.parquet").to_pylist()
            check_provenance(rows, episode, sources[episode["source_episode"]])
            assert np.array_equal(state.astype(np.float32), np.asarray([row["state"] for row in rows], dtype=np.float32))
            assert np.array_equal(action.astype(np.float32), np.asarray([row["action"] for row in rows], dtype=np.float32))
            assert np.array_equal(np.asarray(data["source_task_time"], dtype=np.float32).reshape(-1), np.asarray([row["source_task_time_s"] for row in rows], dtype=np.float32))
            provenance[(split, index)] = rows
            for _, camera in CAMERAS:
                video = episode_file(directory, info, index, f"observation.images.{camera}")
                result["videos"].append({"path": str(video.relative_to(root)), **check_video(video, n)})
            if episode["source_episode"] in legacy_sources:
                result["legacy_v1_equivalence"].append(compare_legacy_episode(root, v1_root, episode, old_by_key[episode_key(episode)], data, rows, args.legacy_video_mae))
            if split == "train":
                train_arrays["state"].extend(state)
                train_arrays["actions"].extend(action)
            frames += n
        assert frames == info["total_frames"]
        tasks = [json.loads(line) for line in (directory / "meta/tasks.jsonl").read_text().splitlines()]
        assert len(tasks) == 1 and tasks[0]["task"] == TASK
        result["datasets"][split] = {"frames": frames, "episodes": len(episodes), "sources": len({entry["source_episode"] for entry in episodes})}

    sys.path.insert(0, str(args.train_scripts.expanduser().resolve()))
    from dataset_v2 import JointV2Dataset, as_openpi_sample
    from torch.utils.data import DataLoader
    import cv2
    for split in ("train", "val"):
        dataset = JointV2Dataset(root, split, 10)
        expected_windows = sum(entry["frames"] - 9 for entry in manifest if entry["split"] == split)
        assert len(dataset) == expected_windows
        for index in np.unique(np.linspace(0, len(dataset) - 1, min(24, len(dataset)), dtype=int)):
            sample = as_openpi_sample(dataset[int(index)])
            assert sample["state"].shape == (30,) and sample["actions"].shape == (10, 14)
            assert sample["prompt"] == TASK
            assert all(array.shape == (480, 640, 3) and array.dtype == np.uint8 for array in sample["image"].values())
        errors = []
        for episode in [entry for entry in manifest if entry["split"] == split]:
            index, n = int(episode["episode_index"]), int(episode["frames"])
            raw = source_directory(source, episode["source_path"])
            for offset in sorted({0, min(n // 2, n - 10), n - 10}):
                global_index = int(dataset.dataset.episode_data_index["from"][index]) + offset
                loaded = dataset.dataset[global_index]
                row = provenance[(split, index)][offset]
                for camera_index, (source_camera, exported_camera) in enumerate(CAMERAS):
                    path = raw / "videos" / f"{source_camera}.mp4"
                    capture = cv2.VideoCapture(str(path))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, row["camera_video_frames"][camera_index])
                    ok, image = capture.read()
                    capture.release()
                    assert ok, (path, row["camera_video_frames"][camera_index])
                    expected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                    actual = np.asarray(loaded[f"observation.images.{exported_camera}"]).transpose(1, 2, 0)
                    error = float(np.mean(np.abs(expected - actual)) * 255)
                    assert error < args.source_video_mae, (path, offset, error, "Wrong source frame or RGB channel order")
                    errors.append(error)
        batch_size = min(4, len(dataset))
        batch = next(iter(DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)))
        assert tuple(batch["action"].shape) == (batch_size, 10, 14)
        result["datasets"][split].update(h10_windows=len(dataset), real_lerobot_loader_passed=True, dataloader_batch_action_shape=list(batch["action"].shape), source_frame_identity_rgb_check=True, source_frame_samples=len(errors), max_reencode_mean_absolute_pixel_error=max(errors))

    stats = read_json(root / "assets/heterovla_v2/norm_stats.json")["norm_stats"]
    for name, values in train_arrays.items():
        array = np.asarray(values, dtype=np.float64)
        expected = {"mean": array.mean(axis=0), "std": array.std(axis=0), "q01": np.quantile(array, 0.01, axis=0), "q99": np.quantile(array, 0.99, axis=0)}
        for key, value in expected.items():
            assert np.allclose(stats[name][key], value, atol=1e-8, rtol=1e-7), (name, key)
    result.update(normalization_train_only=True, normalization_fields_verified=["mean", "std", "q01", "q99"], legacy_v1_numeric_exact=True, legacy_v1_image_equivalent=True, format_validation_passed=True, full_vla_training_run=False, physical_alignment_independently_validated=False, action_semantics="kinematic proxy for Go2/UAV; logged target for Piper")
    atomic_json(root / "reports/validation.json", result)
    print(json.dumps({"status": "passed", "root": str(root), "source_inventory": inventory_result, "datasets": result["datasets"], "old_v1_segments_verified": len(result["legacy_v1_equivalence"])}, ensure_ascii=False, indent=2))


def main():
    args = arguments()
    try:
        validate(args)
    except Exception as error:
        root, v1_root = args.root.expanduser().resolve(), args.v1_root.expanduser().resolve()
        if root != v1_root and root.is_dir() and not root.is_relative_to(v1_root):
            atomic_json(root / "reports/validation_failure.json", {"format_validation_passed": False, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "full_vla_training_run": False})
        raise


if __name__ == "__main__":
    main()
