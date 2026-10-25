"""Small, hardware-independent helpers for durable segmented capture."""
import csv
import json
import os
from pathlib import Path
import signal
import time


def sync_directory(path):
    if os.name == 'nt':
        return  # Production durability is provided by the Linux robot.
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_episode_parents(directory, data_root):
    path = episode_path(directory, data_root)
    root = Path(data_root).resolve()
    while True:
        sync_directory(path)
        if path == root.parent:
            break
        path = path.parent


def clear_active_marker(path):
    """Release only after directory sync; restore retry ownership on sync failure."""
    path = Path(path)
    saved = path.read_bytes()
    path.unlink()
    try:
        sync_directory(path.parent)
    except OSError:
        with path.open('wb') as handle:
            handle.write(saved)
            handle.flush()
            os.fsync(handle.fileno())
        # If directory I/O still fails, the live marker nevertheless blocks Start.
        # A restart may see either marker state; all episode data was synced first.
        try:
            sync_directory(path.parent)
        except OSError:
            pass
        raise


def atomic_json(path, value, *, durable=False):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if durable:
        sync_directory(path.parent)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def sync_episode_csv(directory, names=None):
    """Call only after producers have acknowledged close/exited."""
    raw = Path(directory) / 'raw'
    paths = [raw / name for name in names] if names is not None else sorted(raw.glob('*.csv'))
    synced = []
    for path in paths:
        with path.open('r+b') as handle:
            os.fsync(handle.fileno())
        synced.append(path.name)
    if os.name != 'nt':
        fd = os.open(str(raw), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return synced


def episode_path(path, data_root):
    result, root = Path(path).resolve(), Path(data_root).resolve()
    if result == root or root not in result.parents:
        raise RuntimeError('episode path is outside data root')
    return result


def process_info(pid, proc_root=Path('/proc')):
    try:
        folder = Path(proc_root) / str(int(pid))
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        args = (folder / 'cmdline').read_bytes().decode().split('\0')
        return {'pid': int(pid), 'start_ticks': fields[19], 'args': args}
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_process(pid_file, executable, episode=None):
    path = Path(pid_file)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        raise RuntimeError('invalid PID file: ' + str(path))
    current = process_info(pid)
    if current is None:
        return None
    saved = read_json(str(path) + '.identity.json')
    if saved and (saved['pid'] != pid or saved['start_ticks'] != current['start_ticks']):
        raise RuntimeError('PID reused; refusing to signal: ' + str(path))
    if str(executable) not in current['args'] or (episode and str(episode) not in current['args']):
        raise RuntimeError('process ownership mismatch: ' + str(path))
    return current


def save_pid(path, process):
    path = Path(path)
    path.write_text(str(process.pid) + '\n')
    current = process_info(process.pid)
    if current is None:
        raise RuntimeError('new process exited: ' + str(path))
    atomic_json(str(path) + '.identity.json', current)


def stop_owned(path, executable, episode=None, timeout=15):
    current = owned_process(path, executable, episode)
    if current is not None:
        try:
            os.kill(current['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            latest = process_info(current['pid'])
            if latest is None or latest['start_ticks'] != current['start_ticks']:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('process has not finished saving; retry stop: ' + str(path))
    Path(path).unlink(missing_ok=True)
    Path(str(path) + '.identity.json').unlink(missing_ok=True)


class SegmentCsv:
    """A resident producer logs only while a segment is selected.

    The caller publishes its snapshot after writerow, acknowledging that the
    previous file is flushed and closed before Stop releases the episode.
    """
    def __init__(self, selector, data_root, fields):
        self.selector = Path(selector)
        self.data_root = data_root
        self.fields = fields
        self.handle = None
        self.writer = None
        self.episode = None

    def writerow(self, row):
        selected = read_json(self.selector, {}).get('episode')
        if selected != self.episode:
            self.close()
            if selected:
                raw = episode_path(selected, self.data_root) / 'raw'
                raw.mkdir(parents=True, exist_ok=True)
                self.handle = (raw / 'piper_gamepad.csv').open('x', newline='')
                self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
                self.writer.writeheader()
            self.episode = selected
        if self.writer:
            self.writer.writerow(row)

    def flush(self):
        if self.handle:
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        self.handle = self.writer = None
        self.episode = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
