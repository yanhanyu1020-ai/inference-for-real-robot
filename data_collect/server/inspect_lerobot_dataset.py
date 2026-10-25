#!/usr/bin/env python3
"""Print a compact structural summary of a converted LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    info = root / "meta" / "info.json"
    if not info.is_file():
        raise SystemExit(f"missing LeRobot metadata: {info}")
    payload = json.loads(info.read_text(encoding="utf-8"))
    parquet = list((root / "data").rglob("*.parquet"))
    videos = list((root / "videos").rglob("*.mp4"))
    print(json.dumps({"root": str(root), "info": payload, "parquet_files": len(parquet), "video_files": len(videos)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
