#!/usr/bin/env python3
"""Manually resume this one official V3 run; never poll or retry automatically.

Run on 16.7 with its openpi-pi05 Python. --gpu accepts auto, a physical index,
or a GPU UUID. --check is read-only; --prepare-only writes an isolated attempt
without launching a supervisor or initializing CUDA/JAX.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
TRAIN = Path('/meta_eon_cfs/home/hph/heterovla/train/V3')
MODELS = Path('/meta_eon_cfs/home/hph/heterovla-model/V3')
ORIGINAL = TRAIN / 'official_vitlora_r8_a16_h15_r2'
TEMPLATE = TRAIN / 'official_vitlora_r8_a16_h15_r2_resume_20261020_cachefix'
RUN = 'vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_20261019_r2'
RUN_DIR = MODELS / 'runs' / RUN
CHECKPOINTS = MODELS / 'checkpoints/pi05_heterovla_v3_vitlora' / RUN
LOCK = RUN_DIR / 'manual_resume.lock'
PYTHON = '/meta_eon_cfs/home/hph/heterovla/.venvs/openpi-pi05/bin/python'
CHECKPOINT_FILES = ('_CHECKPOINT_METADATA', 'params/_METADATA', 'params/manifest.ocdbt',
                    'train_state/_METADATA', 'train_state/manifest.ocdbt')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        uid = int(next(x for x in (proc / 'status').read_text().splitlines()
                       if x.startswith('Uid:')).split()[1])
        command = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        if stat[0] != 'Z' and command:
            return dict(pid=int(pid), uid=uid, start_ticks=stat[19], command=command,
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return None


def live_training():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        row = identity(proc.name)
        if row and row['uid'] == 1037 and RUN in ' '.join(row['command']):
            scripts = {Path(x).name for x in row['command'] if x.endswith('.py')}
            if scripts & {'run_training.py', 'queue_training.py'}:
                found.append(row)
    return found


def queue_module():
    spec = importlib.util.spec_from_file_location('manual_v3_queue', TEMPLATE / 'queue_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint():
    candidates = sorted((p for p in CHECKPOINTS.iterdir() if p.is_dir() and p.name.isdigit()),
                        key=lambda p: int(p.name), reverse=True)
    for cp in candidates:
        if not all((cp / name).is_file() for name in CHECKPOINT_FILES):
            continue
        try:
            if not read(cp / '_CHECKPOINT_METADATA').get('commit_timestamp_nsecs'):
                continue
        except (ValueError, OSError):
            continue
        # Only three tiny scalar arrays are read; no model/JAX/CUDA is loaded.
        import tensorstore as ts
        values = {}
        for key in ('step', 'opt_state.1.0.count', 'opt_state.1.2.count'):
            array = ts.open({'driver': 'zarr', 'kvstore': {'driver': 'ocdbt',
                'base': {'driver': 'file', 'path': str(cp / 'train_state') + '/'},
                'path': key + '/'}}, read=True, write=False).result()
            values[key] = int(array.read().result())
        expected = int(cp.name) + 1
        require(set(values.values()) == {expected}, f'Checkpoint step/Adam mismatch: {cp}: {values}')
        require(expected <= 20000, 'Checkpoint exceeds the 20000-update target')
        return cp, values
    raise RuntimeError('No committed complete checkpoint with params and train_state was found')


def gpu_candidate(q, requested):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, check=True, timeout=20)
    indices = {row[0].strip(): row[1].strip() for row in csv.reader(io.StringIO(result.stdout)) if row}
    chosen = indices.get(requested, requested) if requested != 'auto' else None
    require(requested == 'auto' or chosen in indices.values(), f'Unknown GPU: {requested}')
    candidates = [g for g in q.query_gpus() if q.available(g) and (chosen is None or g['uuid'] == chosen)]
    gpu = min(candidates, key=lambda g: int(next(i for i, u in indices.items() if u == g['uuid']))) if candidates else None
    return dict(gpu, index=int(next(i for i, u in indices.items() if u == gpu['uuid']))) if gpu else None


def prepare(q, entry, cp, scalars):
    require(entry.resolve() == (HERE / 'manual_run_training.py').resolve(), 'Use the deployed manual training template')
    original = q.verify_preflight(ORIGINAL)
    source = read(TEMPLATE / 'manifest.json')
    proof = read(TEMPLATE / 'preflight_report.json')
    require(proof.get('passed') and proof.get('manifest_sha256') == sha(TEMPLATE / 'manifest.json'),
            'The successful resume template lacks its bound CPU preflight')
    # Old checkpoint files can have been pruned by normal retention. Validate
    # immutable source/data bindings, then replace all old checkpoint bindings.
    for name, digest in source['files'].items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe template file')
        require(sha(TEMPLATE / rel) == digest, f'Template changed: {name}')
    runtime = {name: digest for name, digest in source['runtime_files'].items()
               if not Path(name).is_relative_to(CHECKPOINTS)}
    for name, digest in runtime.items():
        require(sha(name) == digest, f'Template runtime changed: {name}')
    global_proof = read(HERE / 'global_cache_cpu_result.json')
    disabled = global_proof.get('global_disabled_cpu_process', {})
    require(global_proof.get('passed') is True
            and global_proof.get('global_cache_disabled_before_jax_import') is True
            and global_proof.get('global_cache_flag_remained_false') is True
            and disabled.get('persistent_cache_reads') == 0
            and disabled.get('memory', {}).get('alias_size_in_bytes', 0) > 0
            and global_proof.get('test_sha256') == sha(HERE / 'test_global_cache.py'),
            'Global-cache CPU regression has not passed or its test changed')
    require(read(TEMPLATE / 'cpu_resume_contract.json').get('passed') is True, 'CPU resume regression has not passed')
    compile(entry.read_text(), str(entry), 'exec')
    token = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    code = TRAIN / (ORIGINAL.name + '_manual_' + token)
    queue = MODELS / 'queues' / (RUN + '__manual_' + token)
    code.mkdir()
    queue.mkdir(parents=True)
    for name in source['files']:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TEMPLATE / name, target)
    shutil.copy2(entry, code / 'run_training.py')
    for name in ('global_cache_cpu_result.json', 'test_global_cache.py'):
        shutil.copy2(HERE / name, code / name)
    plan = read(TEMPLATE / 'resume_plan.json')
    plan.pop('preferred_gpu', None)
    plan.update(resume_id='manual_' + token, source_loop_step=int(cp.name), source_actual_step=scalars['step'],
                source_checkpoint=str(cp), resume_code=str(code), queue_dir=str(queue),
                boot_id_at_inspection=identity(os.getpid())['boot_id'],
                original_manifest_sha256=original['manifest_sha256'], manual=True,
                authorization='Explicit manual invocation; one attempt, no automatic retry')
    atomic(code / 'resume_plan.json', plan)
    runtime.update({str(cp / name): sha(cp / name) for name in CHECKPOINT_FILES})
    manifest = dict(source, files={str(p.relative_to(code)): sha(p) for p in code.rglob('*') if p.is_file()},
                    runtime_files=runtime, resume_from_actual_step=scalars['step'])
    atomic(code / 'manifest.json', manifest)
    atomic(code / 'preflight_report.json', dict(passed=True, manifest_sha256=sha(code / 'manifest.json'),
        original_contract=original, source_loop_step=int(cp.name), source_actual_step=scalars['step'],
        checkpoint_scalars=scalars, cpu_resume_roundtrip_passed=True, global_cache_cpu_regression_passed=True,
        cpu_evidence_policy='Reused hash-bound tiny official roundtrip and global-cache CPU regression; no GPU test',
        gpu_used=False, gpu_memory_verified=False, large_checkpoint_arrays_loaded=False,
        target_updates=20000, manual_entry_sha256=sha(code / 'run_training.py')))
    bound = q.verify_preflight(code)
    state = dict(status='prepared', run_name=RUN, resume_id=plan['resume_id'], code_dir=str(code),
        queue_dir=str(queue), run_dir=str(RUN_DIR), source_checkpoint=str(cp),
        source_loop_step=int(cp.name), source_actual_step=scalars['step'], uid=1037, boot_id=plan['boot_id_at_inspection'],
        proof=bound, log=str(queue / 'training.log'), no_automatic_retry=True)
    atomic(queue / 'state.json', state)
    return state


def supervise(args):
    queue = args.queue_dir.resolve()
    require(queue.parent == (MODELS / 'queues') and queue.name.startswith(RUN + '__manual_'), 'Invalid queue')
    require(os.fstat(args.lock_fd).st_ino == LOCK.stat().st_ino, 'Wrong run lock')
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(queue / 'attempt.json')
    ack_sent = False
    def publish(**values):
        nonlocal ack_sent
        state.update(values, updated_utc=now())
        atomic(queue / 'state.json', state)
        if not ack_sent:
            try:
                os.write(args.ack_fd, json.dumps(state).encode())
            except BrokenPipeError:
                pass
            os.close(args.ack_fd)
            ack_sent = True
    try:
        q = queue_module()
        code = Path(state['code_dir'])
        require(code.parent == TRAIN and code.name.startswith(ORIGINAL.name + '_manual_'), 'Invalid code path')
        require(q.verify_preflight(code) == state['proof'], 'Prepared source changed')
        require(identity(os.getpid())['boot_id'] == state['boot_id'], 'Reboot occurred during launch')
        cp, scalars = checkpoint()
        require(str(cp) == state['source_checkpoint'] and scalars['step'] == state['source_actual_step'],
                'Latest committed checkpoint changed; invoke manually again')
        gpu = next((g for g in q.query_gpus() if g['uuid'] == state['gpu']['uuid']), None)
        require(gpu and q.available(gpu), 'Selected GPU became busy; no replacement GPU was selected')
        command = [PYTHON, '-B', str(code / 'run_training.py'), '--gpu', gpu['uuid'], '--run-name', RUN]
        env = dict(os.environ, JAX_ENABLE_COMPILATION_CACHE='false', WANDB_MODE='offline', PYTHONDONTWRITEBYTECODE='1')
        with Path(state['log']).open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, close_fds=True, pass_fds=(args.lock_fd,))
        child = identity(process.pid)
        require(child and child['uid'] == 1037 and child['command'] == command, 'Training identity mismatch')
        publish(status='running', started_utc=now(), started_unix=time.time(),
                training_identity=child, supervisor_identity=identity(os.getpid()), command=command)
        atomic(RUN_DIR / 'active_resume.json', dict(state, gpu=state['gpu']['uuid'], assigned_gpu=state['gpu']))
        result = process.wait()
        runtime = read(RUN_DIR / 'runtime_status.json')
        done = (result == 0 and runtime.get('pid') == child['pid'] and runtime.get('resume_id') == state['resume_id']
                and runtime.get('status') == 'completed' and runtime.get('step') == 20000)
        publish(status='completed' if done else 'failed' if result else 'finished_unverified',
                returncode=result, finished_utc=now(), runtime_status=runtime)
        return result
    except Exception as error:
        publish(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=now())
        return 1
    finally:
        os.close(args.lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('_action', nargs='?', choices=('_supervise',))
    parser.add_argument('--gpu', default='auto', help='auto, physical index (e.g. 3), or GPU UUID')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check', action='store_true')
    flags.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--template-entry', type=Path, default=HERE / 'manual_run_training.py')
    parser.add_argument('--queue-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--ack-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(os.getuid() == 1037 and socket.gethostname().split('.')[0] == 'meta-eon-a8000002',
            'Run with user hph on 16.7 (meta-eon-a8000002)')
    if args._action:
        return supervise(args)
    active = live_training()
    if active:
        print(json.dumps(dict(status='already_running', processes=active)))
        return 0
    q = queue_module()
    cp, scalars = checkpoint()
    if scalars['step'] == 20000:
        print(json.dumps(dict(status='completed', step=20000, checkpoint=str(cp))))
        return 0
    gpu = gpu_candidate(q, args.gpu)
    if args.check:
        print(json.dumps(dict(status='available' if gpu else 'waiting', gpu=gpu,
                              checkpoint=str(cp), optimizer_step=scalars['step'], read_only=True)))
        return 0
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='already_starting_or_running', lock=str(LOCK))))
            return 0
        require(not live_training(), 'Another process is already starting or running this experiment')
        if not gpu and not args.prepare_only:
            print(json.dumps(dict(status='waiting', requested_gpu=args.gpu, detail='No eligible idle 80GB GPU; no attempt created')))
            return 0
        state = prepare(q, args.template_entry, cp, scalars)
        if args.prepare_only:
            print(json.dumps(dict(state, status='prepared_only', model_started=False)))
            return 0
        rechecked = gpu_candidate(q, str(gpu['index']))
        require(rechecked and rechecked['uuid'] == gpu['uuid'], 'GPU availability changed; no model was started')
        gpu = rechecked
        state.update(status='claimed', attempt_utc=now(), gpu=gpu)
        claim = Path(state['queue_dir']) / 'attempt.json'
        claim_fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(claim_fd, 'w') as stream:
            json.dump(state, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        atomic(claim.parent / 'state.json', state)
        read_fd, write_fd = os.pipe()
        try:
            command = [PYTHON, '-B', str(Path(__file__).resolve()), '_supervise',
                       '--queue-dir', str(claim.parent), '--lock-fd', str(fd), '--ack-fd', str(write_fd)]
            with (claim.parent / 'supervisor.log').open('ab', buffering=0) as log:
                subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True, pass_fds=(fd, write_fd))
            os.close(write_fd)
            write_fd = -1
            if select.select([read_fd], [], [], 10)[0]:
                data = os.read(read_fd, 65536)
                if data:
                    state = json.loads(data)
            print(json.dumps({k: state.get(k) for k in
                ('status', 'gpu', 'source_checkpoint', 'source_actual_step', 'training_identity', 'log', 'error')}))
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
    finally:
        # Never LOCK_UN: descendants share and retain this open-file description.
        os.close(fd)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(status='blocked', error=f'{type(error).__name__}: {error}')), file=sys.stderr)
        raise SystemExit(1)
