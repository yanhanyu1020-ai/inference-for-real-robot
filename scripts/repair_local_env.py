"""Repair paths in environments copied from hph; never modify the source tree."""
from pathlib import Path
import argparse
import os

ROOT = Path(__file__).resolve().parents[1]
OLD = '/meta_eon_cfs/home/hph'
NEW = str(ROOT.parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environments', nargs='+',
                        choices=('data-v1-python', 'openpi-pi05', 'data-v1'),
                        default=['data-v1-python', 'openpi-pi05', 'data-v1'])
    args = parser.parse_args()
    changed = []
    for name in args.environments:
        env = ROOT / '.venvs' / name
        if not env.is_dir():
            raise FileNotFoundError(env)
        pending = [env]
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_symlink():
                        target = os.readlink(path)
                        if OLD in target:
                            path.unlink()
                            path.symlink_to(target.replace(OLD, NEW))
                            changed.append(str(path.relative_to(ROOT)))
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append(path)
        candidates = [env / 'pyvenv.cfg', *list((env / 'bin').glob('*')),
                      *list((env / 'lib/python3.11/site-packages').glob('*.pth')),
                      *list((env / 'lib/python3.11/site-packages').glob('*.dist-info/direct_url.json'))]
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            content = path.read_bytes()
            if b'\x00' in content:
                continue
            if OLD.encode() in content:
                path.write_bytes(content.replace(OLD.encode(), NEW.encode()))
                changed.append(str(path.relative_to(ROOT)))
    (ROOT / '.venvs/data-v1-cache').mkdir(exist_ok=True)
    (ROOT.parent / 'heterovla-model/V1/runs').mkdir(parents=True, exist_ok=True)
    print(f'Repaired {len(changed)} environment paths under {ROOT}')


if __name__ == '__main__':
    main()
