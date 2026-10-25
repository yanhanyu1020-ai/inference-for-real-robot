"""Explicit, retryable raw-data validation after a fast joint stop."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from .manifest import ManifestStore
from .models import EpisodeManifest


def pending_episodes(store: ManifestStore) -> list[EpisodeManifest]:
    items = []
    for path in store.root.glob("*/manifest.json"):
        episode = store.load(path.parent.name)
        if (episode.state.value == "complete"
                and episode.metadata.get("postprocess") in {"pending", "failed"}):
            items.append(episode)
    return sorted(items, key=lambda item: (item.created_desktop_ns, item.episode_id))


def finalize_pending(store: ManifestStore, remotes: Mapping, align: Callable,
                     episode_id: str | None = None) -> EpisodeManifest:
    if store.active() is not None:
        raise RuntimeError("an active capture must be stopped before post-processing")
    statuses = {host: remote.status() for host, remote in remotes.items()}
    for host in ("p450", "unitree"):
        status = statuses[host]
        if not status.reachable or status.last_error or status.active or status.episode_id:
            raise RuntimeError(f"{host} is not confirmed idle; post-processing refused")
    if episode_id is None:
        selected = next(iter(pending_episodes(store)), None)
        if selected is None:
            raise ValueError("no pending completed episode to process")
    else:
        selected = store.load(episode_id)
        if selected.state.value != "complete":
            raise ValueError("only stopped episodes can be revalidated")
    try:
        for host in ("p450", "unitree"):
            directory = selected.remote_directories.get(host)
            if not directory:
                raise RuntimeError(f"{host} raw data directory is missing")
            result = remotes[host].finalize_raw(directory)
            if not result.ok:
                detail = "\n".join(part for part in (result.stderr.strip(), result.stdout.strip()) if part)
                raise RuntimeError(f"{host} raw validation failed: {(detail or str(result.returncode))[-8000:]}")
        code = align(selected)
        if code != 0:
            raise RuntimeError(f"joint alignment validation failed (exit code {code})")
    except Exception as exc:
        store.update(selected.episode_id, metadata={**selected.metadata,
                     "postprocess": "failed", "postprocess_error": str(exc)})
        raise
    return store.update(selected.episode_id, metadata={**selected.metadata,
                        "postprocess": "passed", "postprocess_error": None})
