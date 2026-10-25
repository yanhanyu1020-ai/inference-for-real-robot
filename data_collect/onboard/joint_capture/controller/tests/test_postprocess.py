from pathlib import Path

import pytest

from jointctl.manifest import ManifestStore
from jointctl.models import CommandResult, EpisodeManifest, EpisodeState, RemoteStatus
from jointctl.postprocess import finalize_pending, pending_episodes


class Remote:
    def __init__(self, host):
        self.host = host
        self.calls = []
        self.result = CommandResult("finalize", 0)

    def status(self):
        return RemoteStatus(self.host, True, "idle")

    def finalize_raw(self, path):
        self.calls.append(path)
        return self.result


def episode(store, name, created):
    value = EpisodeManifest(name, name, "task", created, EpisodeState.COMPLETE,
                            metadata={"postprocess": "pending"},
                            remote_directories={"p450": f"/p450/{name}", "unitree": f"/unitree/{name}"})
    store.create(value)
    store.clear_active(name)
    return value


def test_finalize_oldest_pending_without_video_export(tmp_path):
    store = ManifestStore(tmp_path)
    episode(store, "new", 20)
    episode(store, "old", 10)
    remotes = {name: Remote(name) for name in ("p450", "unitree")}
    aligned = []
    result = finalize_pending(store, remotes, lambda item: aligned.append(item.episode_id) or 0)
    assert result.episode_id == "old"
    assert aligned == ["old"]
    assert remotes["p450"].calls == ["/p450/old"]
    assert remotes["unitree"].calls == ["/unitree/old"]
    assert store.load("old").metadata["postprocess"] == "passed"
    assert [item.episode_id for item in pending_episodes(store)] == ["new"]


def test_failed_finalization_keeps_retryable_episode(tmp_path):
    store = ManifestStore(tmp_path)
    episode(store, "one", 1)
    remotes = {name: Remote(name) for name in ("p450", "unitree")}
    remotes["unitree"].result = CommandResult("finalize", 1, stderr="CSV contains NUL")
    with pytest.raises(RuntimeError, match="CSV contains NUL"):
        finalize_pending(store, remotes, lambda _item: 0)
    assert store.load("one").metadata["postprocess"] == "failed"
    assert len(pending_episodes(store)) == 1


def test_finalization_error_keeps_validator_details(tmp_path):
    store = ManifestStore(tmp_path)
    episode(store, "one", 1)
    remotes = {name: Remote(name) for name in ("p450", "unitree")}
    remotes["unitree"].result = CommandResult(
        "finalize", 1,
        stdout='{"ok": false, "errors": ["summary.json frames_dropped must be zero"]}',
        stderr="collection error: validator exited 1")
    with pytest.raises(RuntimeError, match="frames_dropped"):
        finalize_pending(store, remotes, lambda _item: 0)


def test_explicit_revalidation_can_withdraw_older_pass(tmp_path):
    store = ManifestStore(tmp_path)
    episode(store, "old", 1)
    store.update("old", metadata={"postprocess": "passed"})
    remotes = {name: Remote(name) for name in ("p450", "unitree")}
    remotes["unitree"].result = CommandResult("finalize", 1, stderr="frames_dropped=245")
    with pytest.raises(RuntimeError, match="frames_dropped=245"):
        finalize_pending(store, remotes, lambda _item: 0, "old")
    assert store.load("old").metadata["postprocess"] == "failed"


def test_finalization_refuses_active_capture(tmp_path):
    store = ManifestStore(tmp_path)
    episode(store, "old", 1)
    store.create(EpisodeManifest("live", "live", "task", 2, EpisodeState.RECORDING))
    remotes = {name: Remote(name) for name in ("p450", "unitree")}
    with pytest.raises(RuntimeError, match="active"):
        finalize_pending(store, remotes, lambda _item: 0)
    assert not remotes["p450"].calls
