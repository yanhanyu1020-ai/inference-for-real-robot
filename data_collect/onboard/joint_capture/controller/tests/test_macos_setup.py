"""Mac delivery keeps local preferences and uses the existing console launcher."""
import json
from pathlib import Path

import pytest

from jointctl import macos


@pytest.fixture
def app(tmp_path):
    root = tmp_path / '中文 controller with spaces'
    (root / 'config').mkdir(parents=True)
    (root / 'src' / 'jointctl').mkdir(parents=True)
    (root / 'src' / 'jointctl' / 'console.py').touch()
    (root / 'config' / 'default.json').write_text(json.dumps({
        'p450': {'host': 'p450'}, 'unitree': {'host': 'unitree'},
        'ssh_timeout_s': 30, 'manifest_root': 'episodes',
    }))
    return root


def test_setup_preserves_config_and_records_on_repeat(app, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'home')
    first = macos.setup(app)
    data = Path(first['manifest_root'])
    assert data == tmp_path / 'home/Library/Application Support/JointCapture/episodes'
    config = Path(first['config'])
    custom = json.loads(config.read_text())
    custom['unitree']['host'] = 'my-unitree'
    config.write_text(json.dumps(custom))
    (data / 'saved.json').write_text('keep')
    assert macos.setup(app) == first
    assert json.loads(config.read_text()) == custom
    assert (data / 'saved.json').read_text() == 'keep'
    assert macos.settings(app) == first


def test_paths_do_not_depend_on_launcher_working_directory(app, tmp_path, monkeypatch):
    records = tmp_path / '本机 records'
    result = macos.setup(app, manifest_root=records)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    calls = []
    monkeypatch.setattr(macos, 'launch', lambda *args, **kwargs: calls.append((args, kwargs)) or 'http://127.0.0.1:8899/')
    assert macos.main(['launch', '--app-root', str(app), '--port', '8899', '--no-browser']) == 0
    assert calls == [((result['config'], str(records), 8899), {'open_browser': False})]
    assert not (elsewhere / 'episodes').exists()


def test_setup_keeps_explicit_external_paths(app, tmp_path):
    config = tmp_path / 'preferences/custom.json'
    data = tmp_path / 'records'
    expected = {'config': str(config), 'manifest_root': str(data)}
    assert macos.setup(app, config=config, manifest_root=data) == expected
    assert macos.setup(app) == expected


def test_invalid_existing_configuration_is_not_replaced(app, tmp_path):
    config = app / 'config' / 'local.macos.json'
    config.write_text('{broken')
    with pytest.raises(ValueError, match='无法读取配置'):
        macos.setup(app, manifest_root=tmp_path / 'records')
    assert config.read_text() == '{broken'
    assert not (app / 'macos_paths.json').exists()


def test_unconfigured_or_invalid_paths_never_launch(app, monkeypatch):
    monkeypatch.setattr(macos, 'launch', lambda *args, **kwargs: pytest.fail('must not launch'))
    assert macos.main(['launch', '--app-root', str(app)]) == 1
    (app / 'macos_paths.json').write_text(json.dumps({'config': 'relative', 'manifest_root': 'episodes'}))
    assert macos.main(['launch', '--app-root', str(app)]) == 1


def test_manifest_path_cannot_replace_existing_file(app, tmp_path):
    data = tmp_path / 'saved-data'
    data.write_text('original')
    with pytest.raises(ValueError, match='不是目录'):
        macos.setup(app, manifest_root=data)
    assert data.read_text() == 'original'
    assert not (app / 'config' / 'local.macos.json').exists()
