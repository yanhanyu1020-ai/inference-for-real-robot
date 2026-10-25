"""Strict V2 H=10 loader: no padded actions, source gaps or episode crossings."""
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


DATASET_PREFIX = "local/heterovla_v2"
ASSET_ID = "heterovla_v2"
HORIZON = 10
FPS = 10


class JointV2Dataset:
    def __init__(self, root, split="train", horizon=HORIZON):
        if split not in ("train", "val"):
            raise ValueError(f"Unexpected V2 split: {split}")
        if horizon != HORIZON:
            raise ValueError("V2 requires complete H=10 action windows")
        self.root = Path(root).expanduser().resolve()
        self.horizon = horizon
        offsets = [i / FPS for i in range(horizon)]
        self.dataset = LeRobotDataset(
            f"{DATASET_PREFIX}_{split}",
            root=self.root / "lerobot" / f"{DATASET_PREFIX}_{split}",
            video_backend="pyav",
            delta_timestamps={
                "action": offsets,
                "action_valid": offsets,
                "source_task_time": offsets,
            },
        )
        self.starts = []
        for start, end in zip(
            self.dataset.episode_data_index["from"],
            self.dataset.episode_data_index["to"],
        ):
            self.starts.extend(range(int(start), int(end) - horizon + 1))
        if not self.starts:
            raise ValueError("No complete V2 action windows")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        sample = self.dataset[self.starts[index]]
        actions = np.asarray(sample["action"])
        valid = np.asarray(sample["action_valid"])
        if actions.shape != (self.horizon, 14) or valid.shape != actions.shape or not np.isfinite(actions).all():
            raise ValueError("V2 actions must be finite H=10 by 14 arrays")
        if sample["action_is_pad"].any() or not sample["action_valid"].all():
            raise ValueError("Invalid or padded action reached the strict V2 loader")
        times = np.asarray(sample["source_task_time"], dtype=np.float64).reshape(-1)
        if times.shape != (self.horizon,) or not np.isfinite(times).all():
            raise ValueError("V2 source timestamps must cover the full H=10 window")
        padded = sample.get("source_task_time_is_pad")
        if padded is not None and padded.any():
            raise ValueError("Padded source timestamps reached the strict V2 loader")
        # The exported source_task_time feature is float32. Its ULP grows with
        # episode time, while provenance retains the original float64 timestamps.
        tolerance = max(1e-5, np.finfo(np.float32).eps * max(1.0, float(np.abs(times).max())) * 2)
        if tolerance >= 0.025 or not np.allclose(np.diff(times), 1 / FPS, atol=tolerance, rtol=0):
            raise ValueError("A V2 action window crosses a source gap")
        return sample


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
        "prompt": sample["task"],
    }
    if "action" in sample:
        result["actions"] = np.asarray(sample["action"], dtype=np.float32)
    return result


class OpenpiV2Dataset:
    def __init__(self, root, split="train", horizon=HORIZON):
        self.base = JointV2Dataset(root, split, horizon)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return as_openpi_sample(self.base[index])
