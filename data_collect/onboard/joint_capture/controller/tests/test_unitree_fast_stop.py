import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'remote/unitree'))
from collection_manager import Collection


def test_stop_leaves_validation_for_explicit_finalize(tmp_path, monkeypatch):
    config = tmp_path / 'config/collection.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'data_root': str(tmp_path / 'data')}))
    ctl = Collection(config)
    episode = tmp_path / 'data/episode'
    episode.mkdir(parents=True)
    (episode / 'meta.json').write_text('{}')
    (ctl.run / 'active_episode').write_text(str(episode))
    calls = []
    monkeypatch.setattr('collection_manager.subprocess.run', lambda *a, **kw: calls.append(a))
    with pytest.raises(RuntimeError, match='while recording'):
        ctl.finalize(episode)
    ctl.stop()
    assert not calls
    assert ctl.active() is None
    assert json.loads((episode / 'meta.json').read_text())['postprocess'] == 'pending'
    ctl.finalize(episode)
    assert len(calls) == 1
    assert str(ctl.onboard / 'validate_episode.py') in calls[0][0]
    assert '--write' in calls[0][0]


def test_legacy_stop_keeps_ownership_when_raw_fsync_fails(tmp_path, monkeypatch):
    config = tmp_path / 'config/collection.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'data_root': str(tmp_path / 'data')}))
    ctl = Collection(config)
    episode = tmp_path / 'data/episode'
    (episode / 'frames').mkdir(parents=True)
    (episode / 'raw').mkdir()
    (episode / 'frames/1.pkl').write_bytes(b'fixture')
    (episode / 'meta.json').write_text('{}')
    (ctl.run / 'active_episode').write_text(str(episode))
    real_fsync = __import__('os').fsync
    def fail_image_sync(fd):
        import os
        if os.fstat(fd).st_size == 7:
            raise OSError('image disk failure')
        return real_fsync(fd)
    monkeypatch.setattr('os.fsync', fail_image_sync)
    with pytest.raises(RuntimeError, match='image disk failure'):
        ctl.stop()
    assert ctl.active() == episode
