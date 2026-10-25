"""Linux process identity checks; never infer ownership from a name alone."""
import os
from pathlib import Path


def process_stat(pid):
    text = Path('/proc/{}/stat'.format(pid)).read_text()
    fields = text[text.rfind(')') + 2:].split()
    return fields


def process_identity(pid):
    return {
        'start_ticks': process_stat(pid)[19],
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    }


def process_exists(pid):
    try:
        return process_stat(pid)[0] not in ('Z', 'X')
    except (FileNotFoundError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def command_matches(cmdline, launch_file):
    tokens = [os.path.basename(token.decode(errors='replace'))
              for token in cmdline.split(b'\0') if token]
    return 'roslaunch' in tokens and launch_file in tokens


def list_component_processes(component):
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if command_matches((entry / 'cmdline').read_bytes(), component.launch_file):
                if process_exists(int(entry.name)):
                    result.append(int(entry.name))
        except (OSError, ValueError):
            continue
    return sorted(result)


def descendants(pid):
    rows = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                rows[int(entry.name)] = int(process_stat(int(entry.name))[1])
            except (OSError, ValueError, IndexError):
                continue
    owned = {pid}
    while True:
        found = {child for child, parent in rows.items() if parent in owned} - owned
        if not found:
            break
        owned.update(found)
    result = []
    for child in sorted(owned - {pid}):
        try:
            result.append({'pid': child, 'identity': process_identity(child)})
        except OSError:
            continue
    return result
