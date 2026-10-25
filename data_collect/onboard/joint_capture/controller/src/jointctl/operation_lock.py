"""Nonblocking, crash-released exclusion shared by all operator entry points."""
from contextlib import contextmanager
import os
from pathlib import Path


class OperationBusy(RuntimeError):
    pass


@contextmanager
def operation_lock(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'operation.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'\0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OperationBusy('another start/stop/recover/align operation is running') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def operation_busy(root: Path) -> bool:
    try:
        with operation_lock(root):
            return False
    except OperationBusy:
        return True
