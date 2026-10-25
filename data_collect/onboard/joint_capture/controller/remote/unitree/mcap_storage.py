"""Single-owner camera MCAP writer; submission is distinct from durability."""
import copy
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from mcap.writer import Writer, CompressionType
from raw_image import ENCODING, encode_image, image_schema


def sync_directory(path):
    if os.name == 'nt':  # Linux robot owns production durability semantics.
        return
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(temporary), str(path))
    sync_directory(path.parent)


class CameraStore:
    def __init__(self, directory, camera, codec, *, clock=time.monotonic,
                 io_hooks=None, chunk_size=4*1024**2, shard_size=512*1024**2):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', camera):
            raise ValueError('invalid camera name')
        if codec not in ('lz4', 'none') or chunk_size < 1 or shard_size < 1:
            raise ValueError('invalid storage configuration')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.camera, self.codec, self.clock = camera, codec, clock
        self.chunk_size, self.shard_size = chunk_size, shard_size
        hooks = io_hooks or {}
        self.fsync = hooks.get('fsync', os.fsync)
        self.rename = hooks.get('rename', os.rename)
        self.sync_dir = hooks.get('sync_dir', sync_directory)
        self._lock = threading.RLock()
        self._file = self._writer = None
        self._index = 0
        self._finished = self._renamed = False
        self._last_seq = 0
        self._last_mono = None
        self._boot = None
        self._submitted = self._written = self._durable = 0
        self._bytes_closed = self._input_bytes = 0
        self._closed = False
        self._fault = None
        self._shards = []
        self._last_wall = None
        self._wall_summary = {'count':0,'first_ns':None,'last_ns':None,'max_gap_ns':0,'nonmonotonic_count':0}

    def _open(self):
        self._index += 1
        self._final = self.directory / ('{}_{:06d}.mcap'.format(self.camera, self._index))
        self._active = self._final.with_suffix('.mcap.active')
        if self._final.exists():
            raise FileExistsError(str(self._final))
        self._file = self._active.open('xb')
        self._writer = Writer(self._file, chunk_size=self.chunk_size,
                              compression=CompressionType.LZ4 if self.codec == 'lz4' else CompressionType.NONE,
                              enable_crcs=True, enable_data_crcs=True)
        self._writer.start(library='heterovla-raw/2 mcap-python/1.3.0')
        schema = self._writer.register_schema(ENCODING, 'heterovla-layout-v1',
                    json.dumps(image_schema()).encode())
        self._channel = self._writer.register_channel('/camera/' + self.camera, ENCODING, schema)
        self._finished = self._renamed = False
        self._input_bytes = 0
        self._shard_start = self._submitted
        self._range = {}

    def append(self, metadata, image):
        data = encode_image(metadata, image)
        with self._lock:
            if self._closed or self._fault:
                raise RuntimeError('camera store is closed or faulted: ' + str(self._fault))
            if metadata['camera'] != self.camera or metadata['seq'] != self._last_seq + 1:
                raise ValueError('wrong camera or nonconsecutive episode sequence')
            if self._boot is not None and (metadata['boot_id'] != self._boot or metadata['monotonic_ns'] <= self._last_mono):
                raise ValueError('boot identity changed or non-increasing monotonic time')
            if self._file is None:
                self._open()
            # Rotate conservatively by uncompressed input as well as physical bytes.
            # Reserve a chunk/footer allowance rather than overrun a full 512 MiB file.
            reserve = min(self.shard_size // 4, self.chunk_size * 2)
            if self._input_bytes and self._input_bytes + len(data) + reserve >= self.shard_size:
                self._finish_shard()
                self._open()
            before = self._file.tell()
            try:
                self._writer.add_message(self._channel, metadata['wall_time_ns'], data,
                    metadata['wall_time_ns'], sequence=metadata['seq'] % 2**32)
            except Exception as exc:
                self._fault = str(exc)
                raise
            self._submitted += 1
            wall = metadata['wall_time_ns']
            timing = self._wall_summary
            if self._last_wall is not None:
                timing['max_gap_ns'] = max(timing['max_gap_ns'], wall-self._last_wall)
                timing['nonmonotonic_count'] += int(wall <= self._last_wall)
            timing['count'] += 1
            timing['first_ns'] = wall if timing['first_ns'] is None else min(timing['first_ns'],wall)
            timing['last_ns'] = wall if timing['last_ns'] is None else max(timing['last_ns'],wall)
            self._last_wall = wall
            self._input_bytes += len(data)
            self._last_seq, self._last_mono, self._boot = metadata['seq'], metadata['monotonic_ns'], metadata['boot_id']
            # Public synchronous add_message writes a complete chunk when tell advances.
            # Characterized against the locked MCAP version; no private flush calls.
            if self._file.tell() > before:
                self._written = self._submitted
            if not self._range:
                self._range = {'first_seq': metadata['seq'], 'first_monotonic_ns': metadata['monotonic_ns'],
                               'first_wall_time_ns': metadata['wall_time_ns']}
            self._range.update(last_seq=metadata['seq'], last_monotonic_ns=metadata['monotonic_ns'],
                               last_wall_time_ns=metadata['wall_time_ns'])

    def checkpoint(self):
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.flush()
                self.fsync(self._file.fileno())
                self._durable = self._written
            self._save_manifest()
            return self.stats()

    def _finish_shard(self):
        if self._file is None:
            return
        if self._fault:
            self._file.close()
            raise RuntimeError('partial MCAP requires recovery: ' + self._fault)
        if not self._finished:
            try:
                self._writer.finish()
            except Exception as exc:
                # A partly written chunk/footer cannot safely be finished twice.
                self._fault = str(exc)
                self._file.close()
                raise
            self._finished = True
            self._written = self._submitted
        if not self._file.closed:
            self._file.flush()
            self.fsync(self._file.fileno())
            self._durable = self._written
            self._file.close()
        if not self._renamed:
            if self._final.exists():
                raise FileExistsError(str(self._final))
            self.rename(str(self._active), str(self._final))
            self._renamed = True
        self.sync_dir(self.directory)
        size = self._final.stat().st_size
        self._shards.append({'path': self._final.name, 'bytes': size,
                             'frames': self._submitted-self._shard_start, **self._range})
        self._bytes_closed += size
        self._file = self._writer = None

    def close(self):
        with self._lock:
            if not self._closed:
                self._finish_shard()
                # Persist the closed report before exposing success to callers.
                self._save_manifest(closed=True)
                self._closed = True
            return self.stats()

    def _save_manifest(self, closed=None):
        report = self.stats()
        if closed is not None:
            report['closed'] = closed
        durable_json(self.directory / (self.camera + '.manifest.json'), report)

    def stats(self):
        with self._lock:
            size = self._bytes_closed
            if self._file is not None:
                path = self._final if self._renamed else self._active
                if path.exists():
                    size += path.stat().st_size
            return {'camera': self.camera, 'codec': self.codec, 'submitted': self._submitted,
                    'written': self._written, 'durable': self._durable,
                    'bytes': size, 'closed': self._closed, 'fault': self._fault,
                    'wall_summary': copy.deepcopy(self._wall_summary),
                    'shards': copy.deepcopy(self._shards), 'chunk_size': self.chunk_size,
                    'shard_size': self.shard_size}
