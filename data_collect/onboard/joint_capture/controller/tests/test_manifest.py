import json
import threading

import pytest

from jointctl.manifest import ActiveEpisodeConflict, InvalidTransition, ManifestStore
from jointctl.models import EpisodeManifest, EpisodeState


def test_create_writes_manifest_before_active_pointer(tmp_path):
    store = ManifestStore(tmp_path)
    manifest = EpisodeManifest.new("joint_x", "demo", "joint")
    store.create(manifest)
    assert store.active().episode_id == "joint_x"
    assert json.loads((tmp_path / "joint_x" / "manifest.json").read_text())["state"] == "starting"


def test_second_active_episode_is_rejected(tmp_path):
    store = ManifestStore(tmp_path)
    store.create(EpisodeManifest.new("joint_a", "demo", "joint"))
    with pytest.raises(ActiveEpisodeConflict):
        store.create(EpisodeManifest.new("joint_b", "demo", "joint"))


def test_complete_clears_active_pointer(tmp_path):
    store = ManifestStore(tmp_path)
    store.create(EpisodeManifest.new("joint_a", "demo", "joint"))
    store.update("joint_a", state=EpisodeState.RECORDING)
    store.update("joint_a", state=EpisodeState.STOPPING)
    store.update("joint_a", state=EpisodeState.COMPLETE)
    assert store.active() is None


def test_invalid_transition_is_rejected(tmp_path):
    store = ManifestStore(tmp_path)
    store.create(EpisodeManifest.new("joint_a", "demo", "joint"))
    with pytest.raises(InvalidTransition):
        store.update("joint_a", state=EpisodeState.COMPLETE)


def test_recovered_is_active_and_can_stop(tmp_path):
    store = ManifestStore(tmp_path)
    store.create(EpisodeManifest.new("joint_a", "demo", "joint"))
    store.update("joint_a", state=EpisodeState.RECORDING)
    store.update("joint_a", state=EpisodeState.STOPPING)
    store.update("joint_a", state=EpisodeState.PARTIAL)
    # A recovered manifest may be installed during restart recovery.
    store.clear_active()
    recovered = EpisodeManifest.new("joint_b", "demo", "joint")
    recovered = EpisodeManifest(**{**recovered.to_dict(), "state": EpisodeState.RECOVERED})
    store.create(recovered)
    assert store.active().episode_id == "joint_b"
    store.update("joint_b", state=EpisodeState.STOPPING)
    assert store.active().episode_id == "joint_b"


def test_set_active_is_exclusive(tmp_path):
    store = ManifestStore(tmp_path)
    store.create(EpisodeManifest.new("joint_a", "demo", "joint"))
    store.clear_active()
    store.set_active("joint_a")
    store._manifest_path("joint_b").parent.mkdir()
    store._manifest_path("joint_b").write_text(
        json.dumps(EpisodeManifest.new("joint_b", "demo", "joint").to_dict())
    )
    with pytest.raises(ActiveEpisodeConflict):
        store.set_active("joint_b")


def test_active_pointer_is_published_only_after_contents_are_ready(tmp_path, monkeypatch):
    store = ManifestStore(tmp_path)
    store._manifest_path("joint_a").parent.mkdir()
    store._manifest_path("joint_a").write_text(
        json.dumps(EpisodeManifest.new("joint_a", "demo", "joint").to_dict())
    )
    pointer = tmp_path / "active"
    import jointctl.manifest as manifest_module
    original_link = manifest_module.os.link

    def observe_then_link(source, destination):
        assert not pointer.exists()
        assert not list(tmp_path.glob(".active.*.tmp")) == []
        return original_link(source, destination)

    monkeypatch.setattr(manifest_module.os, "link", observe_then_link)
    store.set_active("joint_a")
    assert pointer.read_text(encoding="utf-8") == "joint_a\n"
    assert not list(tmp_path.glob(".active.*.tmp"))


def test_clear_cannot_delete_pointer_claimed_during_its_ownership_check(
    tmp_path, monkeypatch
):
    first = ManifestStore(tmp_path)
    second = ManifestStore(tmp_path)
    first.create(EpisodeManifest.new("joint_a", "demo", "joint"))
    first._manifest_path("joint_b").parent.mkdir()
    first._atomic_json(
        first._manifest_path("joint_b"),
        EpisodeManifest.new("joint_b", "demo", "joint").to_dict(),
    )
    pointer = tmp_path / "active"
    first_clear_reached_unlink = threading.Event()
    allow_first_clear = threading.Event()
    original_unlink = type(pointer).unlink

    def pause_first_clear(path, *args, **kwargs):
        if path == pointer and threading.current_thread().name == "first-clear":
            first_clear_reached_unlink.set()
            assert allow_first_clear.wait(timeout=2)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(pointer), "unlink", pause_first_clear)
    clear_thread = threading.Thread(
        target=first.clear_active, args=("joint_a",), name="first-clear"
    )
    clear_thread.start()
    assert first_clear_reached_unlink.wait(timeout=2)

    successor_done = threading.Event()

    def publish_successor():
        second.clear_active("joint_a")
        second.set_active("joint_b")
        successor_done.set()

    successor_thread = threading.Thread(target=publish_successor)
    successor_thread.start()
    assert not successor_done.wait(timeout=0.05)
    allow_first_clear.set()
    clear_thread.join(timeout=2)
    successor_thread.join(timeout=2)

    assert not clear_thread.is_alive()
    assert not successor_thread.is_alive()
    assert second.active().episode_id == "joint_b"
