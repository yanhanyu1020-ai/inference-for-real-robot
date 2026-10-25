"""Independent camera queues: capture ownership ends before disk draining."""
from collections import deque
import copy
from pathlib import Path
import shutil
import threading
import time

from mcap_storage import CameraStore, durable_json
from raw_image import image_schema


class RawCapture:
    def __init__(self, episode_dir, config, cameras, *, store_factory=CameraStore):
        self.directory = Path(episode_dir)
        self.config = dict(config)
        self.cameras = dict(cameras)
        if set(cameras) != {'front', 'wrist'}:
            raise ValueError('front and wrist camera owners are required')
        self.capacity = config.get('raw_queue_bytes', 64*1024**2)
        if type(self.capacity) is not int or not 0 < self.capacity <= 64*1024**2:
            raise ValueError('queue capacity must be at most 64 MiB per camera')
        self.codec = config.get('raw_codec', 'lz4')
        self.factory = store_factory
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._stopped = threading.Event()
        self._draining = False
        self._started = False
        self._tokens, self._stores, self._workers = {}, {}, []
        self._queues = {n: deque() for n in cameras}
        self._faults = []
        self._phase = 'idle'
        self._disk_available = None
        self._rate_at = None
        self._rate_previous = {n: (0, 0, 0) for n in cameras}
        self._stats = {n: {'received':0, 'accepted':0, 'submitted':0, 'written':0,
            'durable':0, 'rejected':0, 'write_errors':0, 'pending':0, 'pending_bytes':0,
            'capacity_bytes':self.capacity, 'queue_warning':False, 'closed':False,
            'bytes':0, 'received_fps':None, 'written_fps':None, 'write_bytes_per_s':None,
            'shards':[], 'last_received_monotonic_ns':None,
            'cutoff_reader_seq':None} for n in cameras}

    def start(self):
        with self._cv:
            if self._started:
                raise RuntimeError('capture already started')
            self.directory.mkdir(parents=True, exist_ok=True)
            self._disk_available = shutil.disk_usage(self.directory).free
            if self._disk_available < 20*1024**3:
                raise RuntimeError('less than 20 GiB free disk space')
            self._boot = self.config.get('boot_id') or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            output = self.directory/'raw/cameras'
            if output.exists() and any(output.iterdir()):
                raise FileExistsError('camera output already exists')
            output.mkdir(parents=True, exist_ok=True)
            durable_json(output/'schema.json', image_schema())
            self._stores = {n:self.factory(output,n,self.codec) for n in self.cameras}
            self._started = True
            self._phase = 'recording'
            self._start_ns = time.monotonic_ns()
            self._rate_at = time.perf_counter()
        try:
            for name,camera in self.cameras.items():
                self._tokens[name] = camera.subscribe(lambda frame,n=name: self._receive(n,frame))
            for name in self.cameras:
                worker = threading.Thread(target=self._write, args=(name,), name='raw-writer-'+name, daemon=True)
                worker.start()
                self._workers.append(worker)
        except Exception as exc:
            self._fault('start: '+str(exc))
        self._coordinator = threading.Thread(target=self._coordinate, name='raw-capture-stop', daemon=True)
        self._coordinator.start()

    def _receive(self, name, frame):
        with self._cv:
            if self._draining:
                return
            stat = self._stats[name]
            stat['received'] += 1
            stat['last_received_monotonic_ns'] = frame['monotonic_ns']
            image = frame['image']
            expected = (self.config.get('height',480),self.config.get('width',640),3)
            if image.shape != expected or str(image.dtype) != 'uint8':
                stat['rejected'] += 1
                self._fault(name+': unexpected pixel dimensions or dtype')
                return
            if stat['pending_bytes'] + image.nbytes > self.capacity:
                stat['rejected'] += 1
                self._fault(name+': queue capacity exceeded')
                return
            # The driver may reuse its read buffer immediately after this callback.
            owned = image.copy(order='C')
            height,width = owned.shape[:2]
            metadata = {'camera':name, 'serial':self.cameras[name].serial,
                'seq':stat['accepted']+1, 'reader_seq':frame['seq'],
                'width':width, 'height':height, 'stride':width*3, 'dtype':str(owned.dtype),
                'pixel_format':'bgr8', 'monotonic_ns':frame['monotonic_ns'],
                'wall_time_ns':frame['wall_time_ns'], 'boot_id':self._boot,
                'timestamp_source':'host_receive', 'device_timestamp_ns':None, 'device_frame_number':None}
            self._queues[name].append((metadata,owned))
            stat['accepted'] += 1
            stat['pending'] += 1
            stat['pending_bytes'] += owned.nbytes
            stat['queue_warning'] = stat['pending_bytes']*10 >= self.capacity*7
            self._cv.notify_all()

    def _fault(self, message):
        with self._cv:
            if message not in self._faults:
                self._faults.append(message)
            self._stop.set()
            self._cv.notify_all()

    def _cache(self, name, report):
        with self._cv:
            for key in ('submitted','written','durable','bytes','shards','closed'):
                self._stats[name][key] = report[key]
            self._stats[name]['wall_summary'] = report.get('wall_summary')

    def _write(self, name):
        store = self._stores[name]
        checkpoint = time.monotonic()
        try:
            while True:
                with self._cv:
                    if not self._queues[name] and not self._draining:
                        self._cv.wait(0.1)
                    item = self._queues[name].popleft() if self._queues[name] else None
                    done = self._draining and item is None
                if done:
                    break
                if item is not None:
                    metadata,image = item
                    store.append(metadata,image)
                    self._cache(name, store.stats())
                    with self._cv:
                        stat = self._stats[name]
                        stat['pending'] -= 1
                        stat['pending_bytes'] -= image.nbytes
                        stat['queue_warning'] = stat['pending_bytes']*10 >= self.capacity*7
                if time.monotonic() - checkpoint >= 1:
                    self._cache(name,store.checkpoint())
                    checkpoint = time.monotonic()
            self._cache(name,store.close())
        except Exception as exc:
            with self._cv:
                self._stats[name]['write_errors'] += 1
            self._fault(name+': writer: '+str(exc))
            # Attempt to finish known-good submitted records; failure stays visible.
            try:
                self._cache(name,store.close())
            except Exception as close_exc:
                self._fault(name+': close: '+str(close_exc))

    def _update_rates(self):
        with self._cv:
            now = time.perf_counter()
            elapsed = now - self._rate_at
            if elapsed <= 0:
                return
            for name, stat in self._stats.items():
                current = (stat['received'], stat['written'], stat['bytes'])
                previous = self._rate_previous[name]
                for key, value, before in zip(
                        ('received_fps', 'written_fps', 'write_bytes_per_s'), current, previous):
                    stat[key] = max(0, value-before) / elapsed
                self._rate_previous[name] = current
            self._rate_at = now

    def _coordinate(self):
        disk_check = time.monotonic()
        try:
            while not self._stop.wait(0.05):
                now = time.monotonic_ns()
                for name,camera in self.cameras.items():
                    with self._cv:
                        last = self._stats[name]['last_received_monotonic_ns'] or self._start_ns
                    if camera.error or now-last > 500_000_000:
                        self._fault(name+': '+(camera.error or 'no frame for over 500 ms'))
                if time.monotonic()-disk_check >= 1:
                    available = shutil.disk_usage(self.directory).free
                    with self._cv:
                        self._disk_available = available
                    self._update_rates()
                    if available < 10*1024**3:
                        self._fault('less than 10 GiB free disk space')
                    disk_check = time.monotonic()
        except Exception as exc:
            self._fault('monitor: '+str(exc))
        finally:
            unsubscribed = True
            for name,token in self._tokens.items():
                try:
                    cutoff = self.cameras[name].unsubscribe(token)
                    with self._cv:
                        self._stats[name]['cutoff_reader_seq'] = cutoff
                except Exception as exc:
                    unsubscribed = False
                    self._fault(name+': unsubscribe: '+str(exc))
            with self._cv:
                self._phase = 'stopping'
                self._draining = True
                self._cv.notify_all()
            for worker in self._workers:
                worker.join()
            self._update_rates()
            with self._cv:
                self._phase = 'stopped' if unsubscribed and all(s['closed'] for s in self._stats.values()) else 'cleanup_pending'
            try:
                durable_json(self.directory/'capture_manifest.json', self.status())
            except Exception as exc:
                self._fault('manifest: '+str(exc))
                with self._cv:
                    self._phase = 'cleanup_pending'
            self._stopped.set()

    def request_stop(self):
        self._stop.set()

    def wait_stopped(self, timeout):
        return self._stopped.wait(timeout)

    def status(self):
        # Never acquire a writer/disk lock on the status request path.
        with self._cv:
            # Conservative uncompressed image budget, plus 1 MiB/s for state/metadata.
            budget = (2*self.config.get('width',640)*self.config.get('height',480)*3
                      *self.config.get('fps',30) + 1024**2)
            minutes = (max(0, self._disk_available-10*1024**3)/budget/60
                       if self._disk_available is not None else None)
            return {'format_version':2, 'image_storage':'mcap', 'codec':self.codec,
                'disk_available_bytes':self._disk_available, 'remaining_minutes':minutes,
                'phase':self._phase, 'faults':list(self._faults), 'cameras':copy.deepcopy(self._stats),
                'quality_ok':not self._faults, 'sensor_completeness':'unverified'}
