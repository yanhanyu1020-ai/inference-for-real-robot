#!/usr/bin/env python3
"""Validate a transferred episode against its manifest and optionally promote it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def validate(episode: Path) -> list[str]:
    manifest_path = episode / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    expected_paths = set()
    for item in manifest.get("files", []):
        relative = item["path"]
        expected_paths.add(relative)
        path = episode / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        if path.stat().st_size != int(item["size"]):
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256(path) != item["sha256"]:
            errors.append(f"sha256 mismatch: {relative}")
    if len(expected_paths) != int(manifest.get("file_count", -1)):
        errors.append("manifest file_count does not match entries")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--promote-to", type=Path)
    args = parser.parse_args()
    episode = args.episode.resolve()
    errors = validate(episode)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    if args.promote_to is not None:
        destination = args.promote_to.expanduser().resolve()
        if destination.exists():
            raise SystemExit(f"destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(episode, destination)
    print(
        json.dumps(
            {
                "ok": True,
                "episode": str(args.promote_to or episode),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
