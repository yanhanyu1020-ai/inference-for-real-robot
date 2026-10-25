"""macOS desktop setup and launch; acquisition remains in the existing console."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .console import launch


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'无法读取配置 {path}：{exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'配置必须是 JSON 对象：{path}')
    return value


def _absolute(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _write_object(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_config(path: Path) -> None:
    config = _read_object(path)
    for name in ('p450', 'unitree'):
        host = config.get(name, {})
        if not isinstance(host, dict) or not isinstance(host.get('host', name), str) or not host.get('host', name).strip():
            raise ValueError(f'{path} 中 {name}.host 必须是非空 SSH 主机名或别名')


def settings(app_root: str | Path) -> dict:
    """Load explicit paths so Finder's working directory cannot affect data."""
    path = _absolute(app_root) / 'macos_paths.json'
    if not path.exists():
        raise ValueError('尚未准备 Mac 环境，请先运行 controller/scripts/Install-JointCapture.command')
    value = _read_object(path)
    for name in ('config', 'manifest_root'):
        if not isinstance(value.get(name), str) or not value[name] or not Path(value[name]).is_absolute():
            raise ValueError(f'{path} 中 {name} 必须是绝对路径，请重新运行安装入口')
    _validate_config(Path(value['config']))
    return value


def setup(app_root: str | Path, *, config: str | Path | None = None,
          manifest_root: str | Path | None = None) -> dict:
    root = _absolute(app_root)
    template = root / 'config' / 'default.json'
    if not template.is_file() or not (root / 'src' / 'jointctl' / 'console.py').is_file():
        raise ValueError(f'不是有效的 JointCapture 控制台目录：{root}')
    paths_file = root / 'macos_paths.json'
    previous = _read_object(paths_file) if paths_file.exists() else {}
    config_path = _absolute(config or previous.get('config') or root / 'config' / 'local.macos.json')
    data_path = _absolute(manifest_root or previous.get('manifest_root') or
                          Path.home() / 'Library' / 'Application Support' / 'JointCapture' / 'episodes')
    if data_path.exists() and not data_path.is_dir():
        raise ValueError(f'采集记录路径不是目录：{data_path}')
    if config_path == data_path:
        raise ValueError('配置文件与采集记录目录不能使用同一路径')
    if not config_path.exists():
        _write_object(config_path, _read_object(template))
    _validate_config(config_path)
    data_path.mkdir(parents=True, exist_ok=True)
    value = {'config': str(config_path), 'manifest_root': str(data_path)}
    _write_object(paths_file, value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='JointCapture macOS 桌面入口')
    commands = parser.add_subparsers(dest='command', required=True)
    default_root = Path(__file__).resolve().parents[2]
    install = commands.add_parser('setup', help='准备本机路径，不连接机器人')
    install.add_argument('--app-root', type=Path, default=default_root)
    install.add_argument('--config', type=Path)
    install.add_argument('--manifest-root', type=Path)
    start = commands.add_parser('launch', help='打开现有网页控制台')
    start.add_argument('--app-root', type=Path, default=default_root)
    start.add_argument('--port', type=int, default=8766)
    start.add_argument('--no-browser', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'launch' and not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    try:
        if args.command == 'setup':
            value = setup(args.app_root, config=args.config, manifest_root=args.manifest_root)
            print(f"Mac 环境已准备。\n配置：{value['config']}\n采集记录：{value['manifest_root']}")
        else:
            value = settings(args.app_root)
            print(launch(value['config'], value['manifest_root'], args.port,
                         open_browser=not args.no_browser))
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
