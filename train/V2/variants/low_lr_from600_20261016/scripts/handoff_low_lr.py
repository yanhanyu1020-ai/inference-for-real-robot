#!/usr/bin/env python3
"""Prepare and transfer this user's existing GPU run to a lower-LR warm start.

The staged trainer performs CPU checks before it signals readiness. Only then
does this supervisor terminate the verified old PID, acquire its cooperative
UUID lock, and release the staged trainer into its GPU phase. No other PID is
signalled, and neither CUDA compute mode nor system-wide GPU settings change.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

_ACTIVE = {}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    folder = Path('/proc') / str(pid)
    status = folder.joinpath('stat').read_text()
    fields = status[status.rindex(')') + 2:].split()
    return dict(pid=pid, uid=folder.stat().st_uid, state=fields[0],
                start_ticks=fields[19],
                command=folder.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0'))


def verify_old(record, expected=None):
    identity = process_identity(int(record['pid']))
    if identity['uid'] != os.getuid() or identity['command'] != record['command']:
        raise RuntimeError('Old process UID/cmdline does not match its launch record; nothing stopped')
    if expected and identity['start_ticks'] != expected['start_ticks']:
        raise RuntimeError('Old PID was reused; nothing stopped')
    if identity['state'] == 'Z':
        raise RuntimeError('Old process is already a zombie; no handoff signal sent')
    return identity


def signal_verified_old(record, expected):
    """Signal only this PID, preferably through a kernel pidfd immune to reuse."""
    descriptor = os.pidfd_open(int(record['pid'])) if hasattr(os, 'pidfd_open') else None
    try:
        verify_old(record, expected)
        if descriptor is not None and hasattr(signal, 'pidfd_send_signal'):
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        else:
            os.kill(int(record['pid']), signal.SIGTERM)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def gpu_observation(uuid):
    row = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=index,uuid,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
    index, actual, total, free = [x.strip() for x in row.split(',')]
    if actual != uuid:
        raise RuntimeError('Selected GPU UUID changed')
    rows = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return dict(index=index, uuid=actual, total_mib=int(total), free_mib=int(free)), [int(x.strip()) for x in rows.splitlines() if x.strip().isdigit()]


def old_is_live(identity):
    try:
        current = process_identity(identity['pid'])
        return current['start_ticks'] == identity['start_ticks'] and current['state'] != 'Z'
    except FileNotFoundError:
        return False


def protect_source(params, destination, parent):
    """Preserve immutable parameter files independently of old best retention."""
    destination.mkdir(parents=True, exist_ok=False)
    copied, linked = 0, 0
    import shutil
    try:
        for source in sorted(params.rglob('*')):
            relative = source.relative_to(params)
            target = destination / 'params' / relative
            if source.is_symlink():
                raise RuntimeError('Unexpected symlink in warm-start checkpoint')
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                    linked += 1
                except OSError:
                    shutil.copy2(source, target)
                    copied += 1
        atomic_json(destination / 'provenance.json', dict(parent_run=parent, original_params=str(params),
            linked_files=linked, copied_files=copied, created_at=datetime.now().astimezone().isoformat()))
    except BaseException:
        # Leave a visible partial backup for inspection; it is never used as a
        # training source and never permits stopping the old trainer.
        raise
    return dict(path=str(destination), linked_files=linked, copied_files=copied)


def environment(project):
    env = dict(os.environ)
    home = Path.home()
    env.update(HETEROVLA_ROOT=str(project), OPENPI_ROOT=str(project / 'third_party/openpi'),
        HETEROVLA_DATA_ROOT=str(home / 'heterovla-data/aligned/V2'),
        HETEROVLA_RUN_ROOT=str(home / 'heterovla-model/V2/runs'),
        PI05_PARAMS=str(home / 'heterovla-model/pretrained/pi05_base/params'),
        OPENPI_DATA_HOME=str(home / 'heterovla-model/V2/cache/openpi'),
        HF_LEROBOT_HOME=str(home / 'heterovla-data/aligned/V2/lerobot'),
        WANDB_MODE='offline', WANDB_SILENT='true', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parent), str(project / 'third_party/openpi/src')])
    return env


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run', required=True)
    parser.add_argument('--gpu', required=True, help='Exact full GPU UUID from the old launch record')
    parser.add_argument('--name', required=True)
    parser.add_argument('--execute-handoff', action='store_true', help='Explicitly authorize SIGTERM to the verified old PID after CPU readiness')
    parser.add_argument('--source-step', type=int, default=600)
    parser.add_argument('--peak-lr', type=float, default=1e-5)
    parser.add_argument('--target-steps', type=int, default=20000)
    args = parser.parse_args()
    import re
    for name in (args.old_run, args.name):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', name):
            parser.error('Run names must be safe single directory names')
    if args.name == args.old_run or args.source_step != 600 or args.target_steps != 20000 or args.peak_lr != 1e-5:
        parser.error('This reviewed handoff is specifically best600 → full LR1e-5 through global20000 in a new run')
    home = Path.home()
    project = Path(os.environ.get('HETEROVLA_ROOT') or home / 'heterovla').resolve()
    runs = home / 'heterovla-model/V2/runs'
    old_dir = runs / args.old_run
    record = read_json(runs / '_launch_logs' / (args.old_run + '.json'))
    if record['name'] != args.old_run or record['gpu']['uuid'] != args.gpu or record['run_directory'] != str(old_dir):
        raise RuntimeError('Old launch identity/path/GPU mismatch; nothing stopped')
    identity = verify_old(record)
    best = read_json(old_dir / 'best.json')
    if best['best_step'] != args.source_step:
        raise RuntimeError('Expected protected best600 is no longer the recorded best; nothing stopped')
    source_ckpt = Path(best['best_checkpoint'])
    if source_ckpt != old_dir / 'best' / str(best.get('best_checkpoint_tag', args.source_step)):
        raise RuntimeError('Unexpected old best checkpoint path')
    params = source_ckpt / 'params'
    if not (source_ckpt / '_CHECKPOINT_METADATA').is_file() or not (params / '_METADATA').is_file():
        raise RuntimeError('Warm-start checkpoint is not complete; nothing stopped')
    checkpoint_root = home / 'heterovla-model/V2/checkpoints/pi05_heterovla_v2_full' / args.old_run
    recoverable = sorted(int(p.name) for p in checkpoint_root.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / '_CHECKPOINT_METADATA').is_file())
    if not recoverable:
        raise RuntimeError('No completed old regular checkpoint; nothing stopped')
    if (runs / args.name).exists():
        raise RuntimeError('New run directory already exists; nothing stopped')
    gpu, gpu_pids = gpu_observation(args.gpu)
    if set(gpu_pids) != {identity['pid']}:
        raise RuntimeError(f'Unexpected GPU users {gpu_pids}; nothing stopped')
    script = Path(__file__).resolve().with_name('train_full_v2.py')
    if not script.is_file():
        raise RuntimeError('Missing staged entry')
    logs = runs / '_launch_logs'
    logs.mkdir(exist_ok=True)
    handoff_dir = runs / '_handoffs' / args.name
    if handoff_dir.exists():
        raise RuntimeError('Handoff record already exists; use a new name')
    handoff_dir.mkdir(parents=True)
    release = handoff_dir / 'gpu_release.json'
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    log = logs / (args.name + '.' + timestamp + '.log')
    command = [sys.executable, '-u', str(script), '--execute', '--gpu', args.gpu, '--name', args.name,
        '--warm-start-params', str(params), '--source-step', str(args.source_step), '--parent-run', args.old_run,
        '--batch-size', '32', '--peak-lr', str(args.peak_lr), '--seed', '42', '--steps', str(args.target_steps),
        '--preallocate-for-handoff', '--handoff-release-file', str(release)]
    plan = dict(old_run=args.old_run, old_identity=identity, old_launch_record=record,
        source_checkpoint=str(source_ckpt), source_params=str(params), source_step=args.source_step,
        old_recoverable_steps=recoverable, name=args.name, gpu=gpu, command=command, log=str(log),
        handoff_directory=str(handoff_dir), target_global_steps=args.target_steps,
        remaining_optimizer_steps=args.target_steps - args.source_step, peak_lr=args.peak_lr,
        status='planned', old_stop_authorized=args.execute_handoff,
        scope='Only the verified old training PID may receive SIGTERM. No other GPU or process is changed.')
    plan['runtime_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(script.parent.glob('*.py'))}
    record_path = handoff_dir / 'handoff.json'
    _ACTIVE.update(plan=plan, record_path=record_path, release=release)

    def publish(**changes):
        plan.update(changes, updated_at=datetime.now().astimezone().isoformat())
        atomic_json(record_path, plan)
        print(json.dumps(changes, ensure_ascii=False), flush=True)

    publish(status='planned')
    if not args.execute_handoff:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    lock_root = runs / '_launch_locks'
    lock_root.mkdir(exist_ok=True)
    run_lock = (lock_root / (args.name + '.lock')).open('a')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    _ACTIVE.setdefault('locks', []).append(run_lock)
    backup = home / 'heterovla-model/V2/warm_start_backups' / (args.old_run + '_step600_for_' + args.name)
    publish(status='protecting_best600')
    source_backup = protect_source(params, backup, args.old_run)
    publish(status='preparing_cpu_stage', source_backup=source_backup)
    env = environment(project)
    with log.open('ab', buffering=0) as output:
        trainer = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, env=env, pass_fds=(run_lock.fileno(),))
    _ACTIVE['trainer'] = trainer
    launch = dict(name=args.name, gpu=gpu, resume=False, run_directory=str(runs / args.name),
        log=str(log), command=command, pid=trainer.pid, supervisor_pid=os.getpid(), batch_size=32,
        peak_lr=args.peak_lr, steps=args.target_steps, source_step=600, parent_run=args.old_run,
        initialization='best600 weights; new Adam phase0; global600..20000', handoff_record=str(record_path))
    launch_path = logs / (args.name + '.json')
    atomic_json(launch_path, launch)
    publish(new_pid=trainer.pid, status='waiting_cpu_readiness')
    ready_path = Path(str(release) + '.ready.json')
    deadline = time.monotonic() + 1800
    while not ready_path.is_file():
        if trainer.poll() is not None:
            raise RuntimeError(f'CPU preparation failed with code{trainer.returncode}; old run remains alive. Read {log}')
        if time.monotonic() > deadline:
            trainer.terminate()
            raise TimeoutError('CPU readiness timed out; old run was not stopped')
        time.sleep(0.25)
    ready = read_json(ready_path)
    if ready.get('pid') != trainer.pid:
        raise RuntimeError('CPU readiness PID mismatch; old run was not stopped')
    audit = ready.get('audit', ready)
    if audit.get('peak_lr') != 1e-5 or audit.get('batch_size') != 32 or audit.get('steps') != 20000:
        raise RuntimeError('CPU readiness configuration mismatch; old run was not stopped')
    publish(status='cpu_ready', readiness=ready)
    verify_old(record, identity)
    # Never interrupt an in-flight checkpoint save. Existing committed saves
    # remain the recovery points; SIGTERM does not promise a new checkpoint.
    deadline = time.monotonic() + 300
    while any('tmp' in p.name.lower() for root in (checkpoint_root, old_dir / 'best')
              for p in root.iterdir() if p.is_dir()):
        if time.monotonic() > deadline:
            raise TimeoutError('Old checkpoint save still in progress; old run was not stopped')
        verify_old(record, identity)
        time.sleep(1)
    gpu, gpu_pids = gpu_observation(args.gpu)
    if set(gpu_pids) != {identity['pid']}:
        raise RuntimeError('GPU users changed before cutover; old run was not stopped')
    gpu_lock_dir = home / 'heterovla-model/.gpu_locks'
    gpu_lock_dir.mkdir(exist_ok=True)
    gpu_lock = (gpu_lock_dir / (args.gpu + '.lock')).open('a')
    _ACTIVE.setdefault('locks', []).append(gpu_lock)
    acquired = threading.Event()
    lock_error = []

    def acquire_gpu_lock():
        try:
            fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX)
            acquired.set()
        except BaseException as exc:
            lock_error.append(str(exc))

    threading.Thread(target=acquire_gpu_lock, daemon=True).start()
    publish(status='stopping_verified_old', old_last_status=read_json(old_dir / 'status.json'))
    signal_verified_old(record, identity)
    stop_started = time.monotonic()
    while old_is_live(identity):
        if time.monotonic() - stop_started > 120:
            raise TimeoutError('Verified old trainer did not exit after SIGTERM; no other PID was signalled')
        time.sleep(0.05)
    if not acquired.wait(timeout=30) or lock_error:
        raise RuntimeError('Could not acquire the old cooperative GPU lock after exit')
    deadline = time.monotonic() + 30
    while True:
        gpu, gpu_pids = gpu_observation(args.gpu)
        if not gpu_pids:
            break
        if set(gpu_pids) - {identity['pid']}:
            raise RuntimeError(f'Another GPU process appeared: {gpu_pids}; no other PID was signalled')
        if time.monotonic() > deadline:
            raise TimeoutError('Old CUDA memory was not released')
        time.sleep(0.05)
    if gpu['free_mib'] < max(60 * 1024, int(0.90 * gpu['total_mib']) + 1024):
        raise RuntimeError('Insufficient free memory for new training allocator; no competing process stopped')
    publish(status='releasing_gpu_stage', gpu_after_old_exit=gpu,
        old_exit_to_release_seconds=time.monotonic() - stop_started)
    atomic_json(release, dict(pid=trainer.pid, gpu=args.gpu, old_pid=identity['pid'],
        old_start_ticks=identity['start_ticks'], authorized_by='user_requested_low_lr_handoff'))
    # The supervisor retains the cooperative GPU lock for the whole new run.
    # The trainer itself reserves CUDA memory through its JAX allocator.
    registered_gpu_argv = False
    while trainer.poll() is None:
        if not registered_gpu_argv:
            current = process_identity(trainer.pid)
            if current['command'] != command:
                launch.update(command=current['command'], staged_command=command)
                atomic_json(launch_path, launch)
                registered_gpu_argv = True
        state_path = runs / args.name / 'status.json'
        if state_path.exists():
            state = read_json(state_path)
            stage = state.get('status')
            if stage != plan.get('new_stage'):
                publish(status='new_trainer_running', new_stage=stage, new_status=state)
        time.sleep(1)
    publish(status='new_trainer_exited', new_exit_code=trainer.returncode)
    if trainer.returncode:
        raise RuntimeError(f'New trainer exited with code{trainer.returncode}; inspect {log}')


def main():
    try:
        return _main()
    except BaseException as error:
        trainer = _ACTIVE.get('trainer')
        release = _ACTIVE.get('release')
        if trainer is not None and release is not None and not release.exists() and trainer.poll() is None:
            # A staged CPU-only child must not linger after an aborted handoff.
            # Its identity is the Popen handle created by this supervisor.
            trainer.terminate()
            try:
                trainer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        if 'record_path' in _ACTIVE:
            plan = _ACTIVE['plan']
            plan.update(status='supervisor_failed', error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(), updated_at=datetime.now().astimezone().isoformat())
            atomic_json(_ACTIVE['record_path'], plan)
        raise
    finally:
        for lock in reversed(_ACTIVE.get('locks', [])):
            lock.close()


if __name__ == '__main__':
    main()
