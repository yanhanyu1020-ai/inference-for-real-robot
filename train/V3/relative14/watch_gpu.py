"""Persistent, one-attempt GPU watcher for the relative14 experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import configuration as c
from train.V3.local_yhy.queue_training import (
    atomic_json, available, identity_matches, process_identity, query_gpus,
    read_json, sha256, utcnow,
)

QUEUE = c.MODELS / 'V3/relative14/watchers' / c.RUN_NAME
RUN = c.MODELS / 'V3/relative14/runs' / c.RUN_NAME
CHECKPOINT = c.MODELS / 'V3/checkpoints' / c.CONFIG_NAME / c.RUN_NAME


def validate():
    proof = read_json(c.MODELS / 'V3/relative14/preflight.json')
    if not proof or proof.get('passed') is not True or proof.get('gpu_used') is not False:
        raise RuntimeError('Run bash train/V3/relative14/train.sh --check first')
    for path, expected in proof['source_hashes'].items():
        if sha256(c.PROJECT / path) != expected:
            raise RuntimeError('Source changed since preflight: ' + path)
    for path in (RUN, CHECKPOINT):
        if path.exists():
            raise FileExistsError('Fresh training destination already exists: ' + str(path))


def command(gpu):
    return ['bash', 'train/V3/relative14/train.sh', '--gpu', gpu['uuid']]


def claim_attempt(queue, gpu):
    """O_EXCL is permanent even if launch fails or the supervisor crashes."""
    attempt = dict(created_utc=utcnow(), gpu=gpu, command=command(gpu))
    with (queue / 'attempt.json').open('x') as stream:
        json.dump(attempt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return attempt


def snapshot(queue=QUEUE):
    state = read_json(queue / 'state.json') or {'status': 'not_started'}
    state['watcher_alive'] = identity_matches(state.get('watcher_identity'))
    state['attempt_exists'] = (queue / 'attempt.json').exists()
    state['runtime'] = read_json(RUN / 'runtime_status.json')
    if state.get('training_identity'):
        state['training_alive'] = identity_matches(state['training_identity'])
    return state


def watch(interval, queue=QUEUE, probe=query_gpus, verify=validate, launch=subprocess.Popen):
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / 'watcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A watcher or its training process is already active') from None
        if (queue / 'attempt.json').exists():
            raise RuntimeError('An attempt already exists; automatic retry is disabled')
        state = dict(run_name=c.RUN_NAME, watcher_identity=process_identity(os.getpid()),
                     interval_seconds=interval, started_utc=utcnow())

        def publish(status, **values):
            state.update(status=status, updated_utc=utcnow(), **values)
            atomic_json(queue / 'state.json', state)
            print(json.dumps(state, ensure_ascii=False), flush=True)

        try:
            verify()
            previous_idle = set()
            while True:
                if (queue / 'stop_requested').exists():
                    publish('stopped')
                    return
                try:
                    devices = probe()
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    previous_idle.clear()
                    publish('waiting_probe_error', error=str(exc))
                    time.sleep(interval)
                    continue
                idle = [g for g in devices if available(g)]
                stable = [g for g in idle if g['uuid'] in previous_idle]
                publish('waiting', gpus=devices, idle_candidates=[g['uuid'] for g in idle], error=None)
                if stable:
                    verify()
                    # Fresh process/memory inventory immediately before the permanent claim.
                    try:
                        fresh = {g['uuid']: g for g in probe() if available(g)}
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        previous_idle.clear()
                        publish('waiting_probe_error', error=str(exc))
                        time.sleep(interval)
                        continue
                    gpu = next((fresh[g['uuid']] for g in stable if g['uuid'] in fresh), None)
                    if gpu and not (queue / 'stop_requested').exists():
                        attempt = claim_attempt(queue, gpu)
                        publish('launching', attempt=attempt)
                        with (queue / 'training.log').open('ab', buffering=0) as output:
                            child = launch(attempt['command'], cwd=c.PROJECT,
                                           stdin=subprocess.DEVNULL, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(lock.fileno(),),
                                           env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                            publish('running', training_pid=child.pid,
                                    training_identity=process_identity(child.pid))
                            # bash execs Python; refresh identity after exec and publish a heartbeat.
                            while child.poll() is None:
                                time.sleep(min(interval, 5))
                                publish('running', training_identity=process_identity(child.pid))
                            publish('complete' if child.returncode == 0 else 'failed',
                                    returncode=child.returncode)
                        return
                previous_idle = {g['uuid'] for g in idle}
                time.sleep(interval)
        except BaseException as exc:
            publish('failed', error=f'{type(exc).__name__}: {exc}')
            raise


def start(interval):
    QUEUE.mkdir(parents=True, exist_ok=True)
    # Serialize starts and stop requests without racing removal of the stop marker.
    with (QUEUE / 'control.lock').open('a') as control:
        fcntl.flock(control, fcntl.LOCK_EX)
        with (QUEUE / 'watcher.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps(snapshot(), indent=2, ensure_ascii=False))
                return
            if (QUEUE / 'attempt.json').exists():
                raise RuntimeError('Training already attempted; inspect status and training.log')
            validate()
            (QUEUE / 'stop_requested').unlink(missing_ok=True)
        with (QUEUE / 'watcher.log').open('ab', buffering=0) as output:
            child = subprocess.Popen(
                [sys.executable, '-m', 'train.V3.relative14.watch_gpu', 'run', '--interval', str(interval)],
                cwd=c.PROJECT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(100):
            state = snapshot()
            if state.get('watcher_identity', {}).get('pid') == child.pid and state['watcher_alive']:
                print(json.dumps(state, indent=2, ensure_ascii=False))
                return
            if child.poll() is not None:
                raise RuntimeError('Watcher exited; inspect ' + str(QUEUE / 'watcher.log'))
            time.sleep(.1)
        raise RuntimeError('Watcher startup unconfirmed; inspect status before restarting')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'run', 'status', 'stop'))
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error('--interval must be at least 5 seconds')
    if args.action == 'start':
        start(args.interval)
    elif args.action == 'run':
        watch(args.interval)
    elif args.action == 'status':
        print(json.dumps(snapshot(), indent=2, ensure_ascii=False))
    else:
        QUEUE.mkdir(parents=True, exist_ok=True)
        with (QUEUE / 'control.lock').open('a') as control:
            fcntl.flock(control, fcntl.LOCK_EX)
            if (QUEUE / 'attempt.json').exists():
                raise RuntimeError('Training has already been attempted; stop does not terminate training')
            (QUEUE / 'stop_requested').touch()
        print('Stop requested; waiting watcher will exit after its current check/sleep.')


if __name__ == '__main__':
    main()
