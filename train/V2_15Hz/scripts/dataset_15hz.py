"""Configurable chunks, with explicitly declared previous-value filling for validation."""
import json
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from chunk_settings import resolve_action_horizon


DATASET_PREFIX = "local/heterovla_v2_15hz"
ASSET_ID = "heterovla_v2_15hz"


class JointV215HzDataset:
    def __init__(self, root, split="train", horizon=None):
        if split not in ("train", "val"):
            raise ValueError(f"Unexpected V2_15Hz split: {split}")
        self.root = Path(root).expanduser().resolve()
        self.horizon = resolve_action_horizon(self.root, horizon)
        directory = self.root / "lerobot" / f"{DATASET_PREFIX}_{split}"
        task_config = json.loads((self.root / "configs/task.json").read_text())
        tasks = [json.loads(line) for line in (directory / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
        if task_config.get("same_task_for_all_included_recordings") is not True or len(tasks) != 1:
            raise ValueError("The shared English prompt requires a declared single-task dataset")
        translation = task_config.get("english_translation")
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError("configs/task.json must provide a nonempty english_translation")
        self.prompt = translation.strip()
        self.fps = json.loads((directory/'meta/info.json').read_text())['fps']
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)) or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Dataset fps must be finite and positive')
        offsets = [i / self.fps for i in range(self.horizon)]
        imputation_path = self.root / "configs/imputation.json"
        imputation = json.loads(imputation_path.read_text()) if imputation_path.exists() else {}
        self.allow_imputed = split == "val" and split in imputation.get("splits", [])
        if self.allow_imputed and imputation.get("method") != "previous_valid_value":
            raise ValueError("Unsupported validation imputation method")
        delta_timestamps = {"action": offsets, "action_valid": offsets, "source_task_time": offsets}
        if self.allow_imputed:
            delta_timestamps["action_imputed"] = offsets
        self.dataset = LeRobotDataset(
            f"{DATASET_PREFIX}_{split}",
            root=directory,
            video_backend="pyav",
            delta_timestamps=delta_timestamps,
        )
        if self.dataset.fps != self.fps:
            raise ValueError('LeRobot fps differs from dataset metadata')
        self.starts = []
        for start, end in zip(
            self.dataset.episode_data_index["from"],
            self.dataset.episode_data_index["to"],
        ):
            self.starts.extend(range(int(start), int(end) - self.horizon + 1))
        if not self.starts:
            raise ValueError(f'No complete H={self.horizon} action windows in {split}')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        sample = self.dataset[self.starts[index]]
        actions = np.asarray(sample["action"])
        valid = np.asarray(sample["action_valid"])
        if actions.shape != (self.horizon, 14) or valid.shape != actions.shape or not np.isfinite(actions).all():
            raise ValueError(f'Actions must be finite ({self.horizon}, 14) arrays')
        accepted = valid.astype(bool)
        if self.allow_imputed:
            imputed = np.asarray(sample["action_imputed"], dtype=bool)
            if imputed.shape != valid.shape or np.any(imputed & accepted):
                raise ValueError("Invalid validation imputation mask")
            accepted = accepted | imputed
            state = np.asarray(sample["observation.state"])
            state_valid = np.asarray(sample["state_valid"], dtype=bool)
            state_imputed = np.asarray(sample["state_imputed"], dtype=bool)
            if (state.shape != (30,) or state_valid.shape != (30,) or state_imputed.shape != (30,)
                    or not np.isfinite(state).all() or np.any(state_valid & state_imputed)
                    or not (state_valid | state_imputed).all()):
                raise ValueError("Invalid validation state imputation")
        if sample["action_is_pad"].any() or not accepted.all():
            raise ValueError("Invalid or padded action reached the strict V2_15Hz loader")
        times = np.asarray(sample["source_task_time"], dtype=np.float64).reshape(-1)
        if times.shape != (self.horizon,) or not np.isfinite(times).all():
            raise ValueError(f'Source timestamps must cover the full H={self.horizon} window')
        padded = sample.get("source_task_time_is_pad")
        if padded is not None and padded.any():
            raise ValueError("Padded source timestamps reached the strict V2_15Hz loader")
        # The exported source_task_time feature is float32. Its ULP grows with
        # episode time, while provenance retains the original float64 timestamps.
        tolerance = max(1e-5, np.finfo(np.float32).eps * max(1.0, float(np.abs(times).max())) * 2)
        if tolerance >= 0.25 / self.fps or not np.allclose(np.diff(times), 1 / self.fps, atol=tolerance, rtol=0):
            raise ValueError("A V2_15Hz action window crosses a source gap")
        # Keep LeRobot's original task text for provenance; model input uses
        # the explicitly configured translation for this single-task dataset.
        return {**sample, "prompt": self.prompt}


def as_openpi_sample(sample):
    """LeRobot CHW float RGB -> the same openpi mapping used by V1."""
    image = {}
    for name, slot in (
        ("front", "base_0_rgb"),
        ("wrist", "left_wrist_0_rgb"),
        ("uav", "right_wrist_0_rgb"),
    ):
        array = np.asarray(sample[f"observation.images.{name}"])
        if array.ndim != 3:
            raise ValueError(f"Unexpected image shape: {array.shape}")
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        image[slot] = array
    result = {
        "state": np.asarray(sample["observation.state"], dtype=np.float32),
        "image": image,
        "image_mask": {key: np.bool_(True) for key in image},
        "prompt": sample["prompt"],
    }
    if "action" in sample:
        result["actions"] = np.asarray(sample["action"], dtype=np.float32)
    return result


class OpenpiV215HzDataset:
    def __init__(self, root, split="train", horizon=None):
        self.base = JointV215HzDataset(root, split, horizon)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return as_openpi_sample(self.base[index])
