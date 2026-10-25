#!/usr/bin/env python3
"""Package the CPU field client with its OpenPI NumPy wire codec, without credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build(destination):
    root = Path(__file__).resolve().parents[2]
    robot = root/'inference/real_robot'
    codec = root/'third_party/openpi/packages/openpi-client/src/openpi_client'
    sources = {f'real_robot/{name}': robot/name for name in (
        '__init__.py', 'controller.py', 'core.py', 'diagnostic.py', 'relative14.py', 'field_client.py')}
    sources.update({f'openpi_client/{name}': codec/name for name in ('__init__.py', 'msgpack_numpy.py')})
    sources.update({
        'controller.json': robot/'config/controller_field.json',
        'requirements.txt': robot/'requirements-field.txt',
        'README.md': robot/'README_FIELD_WINDOWS.md',
        'LICENSE.openpi': root/'third_party/openpi/LICENSE',
    })
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sources.items():
            archive.write(path, 'field-client/'+name)
        archive.writestr('field-client/MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    return dict(archive=str(destination), files=len(sources)+1,
                credentials_included=False, gpu_dependencies_included=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2]/'.cache/real_robot/field-client.zip')
    print(json.dumps(build(parser.parse_args().output)), flush=True)
