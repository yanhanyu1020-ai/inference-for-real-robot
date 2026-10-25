"""Durable episode manifests and the single active-episode pointer."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

from .models import (
    ClockEstimate,
    ClockSample,
    CommandResult,
    DiagnosticRecord,
    EpisodeManifest,
    EpisodeState,
    RemoteStatus,
)


class InvalidTransition(ValueError):
    """Raised when an episode state change is not part of the state machine."""


class ActiveEpisodeConflict(RuntimeError):
    """Raised when another episode already owns the active pointer."""


_TRANSITIONS: dict[EpisodeState, frozenset[EpisodeState]] = {
    EpisodeState.STARTING: frozenset(
        (EpisodeState.RECORDING, EpisodeState.START_FAILED, EpisodeState.PARTIAL, EpisodeState.STOPPING)
    ),
    EpisodeState.RECORDING: frozenset((EpisodeState.STOPPING,)),
    EpisodeState.STOPPING: frozenset(
        (EpisodeState.COMPLETE, EpisodeState.PARTIAL, EpisodeState.STOP_FAILED)
    ),
    # Recovery is deliberately active, so operators can finish stopping it.
    EpisodeState.RECOVERED: frozenset((EpisodeState.STOPPING,)),
    EpisodeState.PARTIAL: frozenset((EpisodeState.STOPPING,)),
    EpisodeState.STOP_FAILED: frozenset((EpisodeState.STOPPING,)),
}


class ManifestStore:
    """Persist manifests below *root* and coordinate one active episode."""

    _active_name = "active"
    _active_lock_name = "active.lock"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, episode_id: str) -> Path:
        return self.root / episode_id / "manifest.json"

    def episode_dir(self, episode_id: str) -> Path:
        return self._manifest_path(episode_id).parent

    def manifest_path(self, episode_id: str) -> Path:
        return self._manifest_path(episode_id)

    @contextmanager
    def _active_lock(self) -> Iterator[None]:
        """Serialize active-pointer mutations across processes.

        The lock file is permanent; the operating system releases its byte-range
        or advisory lock when the owning process exits, including after a crash.
        """
        path = self.root / self._active_lock_name
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def create(self, manifest: EpisodeManifest) -> EpisodeManifest:
        """Write a new manifest, then claim the active pointer exclusively."""
        path = self._manifest_path(manifest.episode_id)
        path.parent.mkdir(parents=True, exist_ok=False)
        self._atomic_json(path, manifest.to_dict())
        try:
            self.set_active(manifest.episode_id)
        except Exception:
            # The manifest is intentionally retained for recovery/audit.
            raise
        return manifest

    def restore(self, manifest: EpisodeManifest) -> EpisodeManifest:
        """Rebuild and activate a recovered manifest, including after pointer loss."""
        if manifest.state != EpisodeState.RECOVERED:
            raise ValueError("only recovered manifests can be restored")
        path = self._manifest_path(manifest.episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_json(path, manifest.to_dict())
        self.set_active(manifest.episode_id)
        return manifest

    def load(self, episode_id: str) -> EpisodeManifest:
        path = self._manifest_path(episode_id)
        with path.open("r", encoding="utf-8") as handle:
            return EpisodeManifest.from_dict(json.load(handle))

    def update(
        self,
        episode_id: str,
        *,
        state: EpisodeState | str | None = None,
        metadata: dict[str, Any] | None = None,
        start_results: Mapping[str, CommandResult] | None = None,
        remote_directories: Mapping[str, str] | None = None,
        clock_monitor_pid: int | None = None,
        t0_desktop_ns: int | None = None,
        t1_desktop_ns: int | None = None,
        clock_samples: Mapping[str, ClockSample] | None = None,
        clock_estimates: Mapping[str, ClockEstimate] | None = None,
        stop_results: Mapping[str, CommandResult] | None = None,
        clock_monitor_closed_cleanly: bool | None = None,
        rollback_statuses: Mapping[str, RemoteStatus] | None = None,
        rollback_results: Mapping[str, CommandResult] | None = None,
        diagnostics: list[DiagnosticRecord] | None = None,
    ) -> EpisodeManifest:
        current = self.load(episode_id)
        target = current.state if state is None else EpisodeState(state)
        if target != current.state and target not in _TRANSITIONS.get(current.state, frozenset()):
            raise InvalidTransition(f"{current.state.value} -> {target.value}")
        updated = replace(
            current,
            state=target,
            metadata=current.metadata if metadata is None else metadata,
            start_results=current.start_results if start_results is None else dict(start_results),
            remote_directories=(current.remote_directories if remote_directories is None
                                else dict(remote_directories)),
            clock_monitor_pid=(current.clock_monitor_pid if clock_monitor_pid is None
                               else clock_monitor_pid),
            t0_desktop_ns=current.t0_desktop_ns if t0_desktop_ns is None else t0_desktop_ns,
            t1_desktop_ns=current.t1_desktop_ns if t1_desktop_ns is None else t1_desktop_ns,
            clock_samples=current.clock_samples if clock_samples is None else dict(clock_samples),
            clock_estimates=(current.clock_estimates if clock_estimates is None
                             else dict(clock_estimates)),
            stop_results=current.stop_results if stop_results is None else dict(stop_results),
            clock_monitor_closed_cleanly=(
                current.clock_monitor_closed_cleanly
                if clock_monitor_closed_cleanly is None
                else clock_monitor_closed_cleanly
            ),
            rollback_statuses=(current.rollback_statuses if rollback_statuses is None
                               else dict(rollback_statuses)),
            rollback_results=(current.rollback_results if rollback_results is None
                              else dict(rollback_results)),
            diagnostics=current.diagnostics if diagnostics is None else list(diagnostics),
        )
        self._atomic_json(self._manifest_path(episode_id), updated.to_dict())
        if target in {
            EpisodeState.COMPLETE,
            EpisodeState.START_FAILED,
        }:
            self.clear_active(episode_id)
        return updated

    def active(self) -> EpisodeManifest | None:
        pointer = self.root / self._active_name
        try:
            episode_id = pointer.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not episode_id:
            return None
        return self.load(episode_id)

    def set_active(self, episode_id: str) -> None:
        self.load(episode_id)  # validate that the pointer never targets a ghost
        pointer = self.root / self._active_name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".active.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(episode_id + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            # The lock makes the ownership check and publication one mutation.
            # link() still guarantees readers only observe complete contents.
            with self._active_lock():
                try:
                    existing = pointer.read_text(encoding="utf-8").strip()
                except FileNotFoundError:
                    os.link(temporary, pointer)
                    return
                if existing == episode_id:
                    return
                raise ActiveEpisodeConflict(f"active episode already exists: {existing}")
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def clear_active(self, episode_id: str | None = None) -> None:
        pointer = self.root / self._active_name
        with self._active_lock():
            try:
                if (episode_id is not None
                        and pointer.read_text(encoding="utf-8").strip() != episode_id):
                    return
                pointer.unlink()
            except FileNotFoundError:
                return
