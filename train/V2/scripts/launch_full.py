#!/usr/bin/env python3
"""Launch one explicit single-GPU V2 full fine-tuning run, or check it on CPU."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='CPU data/config check only; never queries or uses a GPU')
    mode.add_argument('--prepare-cache', action='store_true', help='CPU preflight plus deterministic validation/probe cache')
    p.add_argument('--gpu', help='One physical nvidia-smi GPU index or full GPU UUID')
    p.add_argument('--name', help='Unique experiment name; default includes Shanghai timestamp')
    p.add_argument('--resume', action='store_true', help='Resume only the named V2 run, including Adam state and step')
    p.add_argument('--foreground', action='store_true', help='Keep trainer in foreground, e.g. within a scheduler allocation')
    a = p.parse_args()
    if a.resume and not a.name:
        p.error('--resume requires the exact --name of an existing V2 run')
    if not (a.preflight or a.prepare_cache) and not a.gpu:
        p.error('Select the allocated GPU explicitly with --gpu 0 (or its full UUID)')
    if a.name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', a.name):
        p.error('Experiment name must use letters, digits, _, . or -; no paths')
    if a.gpu is not None and not re.fullmatch(r'(?:[0-9]+|GPU-[A-Fa-f0-9-]+)', a.gpu):
        p.error('--gpu must select one physical index or full GPU UUID')
    return a


def resolve_gpu(selector, environment):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    devices = []
    for line in output.splitlines():
        index, uuid, total, free = [x.strip() for x in line.split(',')]
        devices.append(dict(index=index, uuid=uuid, total_mib=int(total), free_mib=int(free)))
    selected = next((d for d in devices if selector in (d['index'], d['uuid'])), None)
    if selected is None:
        raise RuntimeError(f'GPU {selector} does not exist according to nvidia-smi')
    assigned = environment.get('CUDA_VISIBLE_DEVICES', '').strip()
    if assigned and assigned not in ('all',):
        visible = set(assigned.split(','))
        if selected['index'] not in visible and selected['uuid'] not in visible:
            raise RuntimeError('Selected GPU is outside CUDA_VISIBLE_DEVICES; select the GPU assigned to this shell/job')
    # Previous successful batch32 full run needed 58.55 GiB before weight load.
    # The trainer also checks its actual compiled demand before allocating weights.
    if selected['free_mib'] < 60 * 1024:
        raise RuntimeError(f"GPU {selected['index']} has {selected['free_mib']/1024:.1f} GiB free; batch32 full fine-tuning requires at least 60 GiB free to launch")
    return selected


def main():
    args = arguments()
    script = Path(__file__).resolve().with_name('train_full_v2.py')
    if not script.is_file():
        raise FileNotFoundError(script)
    environment = dict(os.environ)
    home = Path.home()
    project = Path(environment.get('HETEROVLA_ROOT') or home/'heterovla')
    openpi = Path(environment.get('OPENPI_ROOT') or project/'third_party/openpi')
    # Apply the same V2-specific overrides as env.sh for direct Python callers.
    environment.update(
        HETEROVLA_ROOT=str(project), OPENPI_ROOT=str(openpi),
        HETEROVLA_DATA_ROOT=environment.get('HETEROVLA_V2_DATA_ROOT') or str(home/'heterovla-data/aligned/V2'),
        HETEROVLA_RUN_ROOT=environment.get('HETEROVLA_V2_RUN_ROOT') or str(home/'heterovla-model/V2/runs'),
        PI05_PARAMS=environment.get('HETEROVLA_PI05_BASE_PARAMS') or str(home/'heterovla-model/pretrained/pi05_base/params'),
        OPENPI_DATA_HOME=environment.get('HETEROVLA_V2_OPENPI_CACHE') or str(home/'heterovla-model/V2/cache/openpi'),
        WANDB_MODE='offline', WANDB_GROUP='pi05-v2-full-20k', WANDB_PROJECT='heterovla-pi05',
        WANDB_DIR=environment.get('HETEROVLA_V2_WANDB_DIR') or str(home/'heterovla-model/V2/wandb'),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.90')
    environment['HF_LEROBOT_HOME'] = str(Path(environment['HETEROVLA_DATA_ROOT'])/'lerobot')
    environment['PYTHONPATH'] = os.pathsep.join([str(script.parent), str(openpi/'src'), environment.get('PYTHONPATH', '')])
    cpu_environment = dict(environment, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    if args.preflight or args.prepare_cache:
        mode = '--prepare-cache' if args.prepare_cache else '--preflight'
        return subprocess.call([sys.executable, '-u', str(script), mode], env=cpu_environment)
    run_root = Path(environment.get('HETEROVLA_RUN_ROOT') or Path.home()/'heterovla-model/V2/runs').expanduser().resolve()
    name = args.name or 'full_b32_lr5e-5_s42_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    directory = run_root/name
    if directory.exists() and not args.resume:
        raise FileExistsError(f'Run already exists; choose another name or explicitly --resume --name {name}')
    if args.resume and not directory.is_dir():
        raise FileNotFoundError(f'Cannot resume a missing V2 run: {directory}')
    gpu = resolve_gpu(args.gpu, environment)
    gpu_lock_root = home/'heterovla-model/.gpu_locks'
    gpu_lock_root.mkdir(parents=True, exist_ok=True)
    gpu_lock = (gpu_lock_root/f"{gpu['uuid']}.lock").open('a')
    try:
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another HeteroVLA launcher holds GPU {gpu['uuid']}") from None
    # A lightweight CPU check precedes every launch. It cannot allocate GPU memory.
    subprocess.run([sys.executable, '-u', str(script), '--preflight'], env=cpu_environment, check=True)
    locks = run_root/'_launch_locks'
    locks.mkdir(parents=True, exist_ok=True)
    lock = (locks/f'{name}.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'A process already holds the launch lock for {name}') from None
    log_root = run_root/'_launch_logs'
    log_root.mkdir(parents=True, exist_ok=True)
    log = log_root/f'{name}.{datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")}.log'
    environment.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], JAX_PLATFORMS='cuda')
    command = [sys.executable, '-u', str(script), '--execute', '--gpu', gpu['uuid'], '--name', name]
    if args.resume:
        command.append('--resume')
    launch_record = dict(name=name, gpu=gpu, resume=args.resume, run_directory=str(directory),
        log=str(log), command=command, batch_size=32, peak_lr=5e-5, steps=20000,
        initialization='V2 own checkpoint' if args.resume else 'official pi05_base, fresh optimizer at step 0')
    # Lock FD is deliberately inherited for the lifetime of the trainer.
    if args.foreground:
        print(json.dumps(launch_record, ensure_ascii=False, indent=2), flush=True)
        os.set_inheritable(lock.fileno(), True)
        os.set_inheritable(gpu_lock.fileno(), True)
        os.execve(sys.executable, command, environment)
    with log.open('ab', buffering=0) as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True, pass_fds=(lock.fileno(), gpu_lock.fileno()))
    launch_record['pid'] = process.pid
    (log_root/f'{name}.json').write_text(json.dumps(launch_record, ensure_ascii=False, indent=2)+'\n')
    # Check only immediate argument/import failures; do not wait for GPU training.
    time.sleep(2)
    code = process.poll()
    if code is not None:
        raise RuntimeError(f'Trainer exited during startup (code {code}). Read {log}')
    print(json.dumps(launch_record, ensure_ascii=False, indent=2))
    print(f'后台已启动。日志：{log}\n状态：{directory}/status.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
