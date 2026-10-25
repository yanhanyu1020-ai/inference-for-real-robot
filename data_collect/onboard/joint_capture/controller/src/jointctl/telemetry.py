"""Read-only Linux resource telemetry and lossless browser serialization."""
import json
import shlex
from .remote import run_process


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if type(value) is int and abs(value) > 9007199254740991:
        return str(value)
    return value


def parse_meminfo(text):
    try:
        fields = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in text.splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
        total, available = fields['MemTotal'], fields['MemAvailable']
        if total <= 0 or not 0 <= available <= total:
            raise ValueError('invalid memory range')
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError('MemTotal/MemAvailable unavailable or invalid') from exc
    return {'total_bytes': total, 'available_bytes': available,
            'used_bytes': total - available, 'used_percent': (total - available) * 100 / total}


def resource_script(directory):
    # Path is a Python string literal, then shell quoted as part of the whole script.
    return """import json, os
from pathlib import Path
p=Path(%r)
while not p.exists() and p != p.parent: p=p.parent
s=os.statvfs(str(p))
print(json.dumps({'meminfo':Path('/proc/meminfo').read_text(),
 'joysticks':[x.name for x in Path('/dev/input').glob('js*')],
 'disk':{'path':str(p),'total_bytes':s.f_blocks*s.f_frsize,'available_bytes':s.f_bavail*s.f_frsize}}))
""" % str(directory)


def read_resources(host, directory, timeout_s=3):
    command = 'python3 -c ' + shlex.quote(resource_script(directory))
    result = run_process(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=3', host, command], timeout_s)
    if not result.ok:
        raise ConnectionError(result.stderr or f'resource probe exit {result.returncode}')
    payload = json.loads(result.stdout)
    disk = payload['disk']
    total, available = disk['total_bytes'], disk['available_bytes']
    if type(total) is not int or type(available) is not int or total <= 0 or not 0 <= available <= total:
        raise ValueError('invalid disk telemetry')
    disk['used_percent'] = (total - available) * 100 / total
    return {'memory': parse_meminfo(payload['meminfo']), 'disk': disk, 'joysticks': payload.get('joysticks')}
