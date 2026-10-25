"""Low-memory offline scan controller on user-authorized shared GPUs 0 and 5."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from path_config import RUN_ROOT as ROOT, SCRIPT_ROOT
SCAN = ROOT/'lr_scan_20261005'
GPUS = ['GPU-87fdd428-76b9-b8ca-7898-fe5ed5907150', 'GPU-651fd2c4-e79c-920e-9afc-db984458f55f']
BATCHES = [[('dual_lora_lr1e-5_s42_20261005', 1e-5), ('dual_lora_lr2p5e-5_s42_20261005', 2.5e-5)],
           [('dual_lora_lr5e-5_s42_20261005', 5e-5), ('dual_lora_lr1e-4_s42_20261005', 1e-4)]]


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def status(name):
    return read(SCAN/name/'status.json')


def gpu_available(gpu):
    free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
    return free >= 21*1024 and own_gpu_jobs() < 2 and own_gpu_jobs(gpu) == 0


def own_gpu_jobs(gpu=None):
    result = subprocess.check_output(['nvidia-smi', *(['-i', gpu] if gpu else []), '--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
    pids = set()
    for value in result.splitlines():
        path = Path('/proc')/value.strip()
        try:
            if path.stat().st_uid == os.getuid():
                pids.add(value.strip())
        except FileNotFoundError:
            pass
    return len(pids)


def process_alive(record):
    # Read-only; never signal any PID (including kill(pid, 0)).
    path = Path(f'/proc/{record["pid"]}/cmdline')
    return path.exists() and b'train_scan.py' in path.read_bytes() and record['name'].encode() in path.read_bytes()


def event(state, **kwargs):
    obj = {'status': state, 'updated_unix': time.time(), 'pid': os.getpid(), **kwargs}
    write(SCAN/'supervisor_status.json', obj)
    print(json.dumps(obj), flush=True)


def launch(pair, gpu, resume=False):
    name, lr = pair
    while not gpu_available(gpu):
        event('waiting_for_assigned_gpu', name=name, gpu=gpu)
        time.sleep(15)
    cmd = [sys.executable, '-u', str(SCRIPT_ROOT / 'train_scan.py'), '--gpu-uuid', gpu,
           '--peak-lr', str(lr), '--name', name]
    if resume:
        cmd += ['--resume', '--stage500-approved']
    with (ROOT/'logs'/f'{name}.log').open('a') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    event('launched_resume' if resume else 'launched', name=name, child_pid=proc.pid)
    # Stagger loading, and ensure the previous launch has reached a known state.
    launched = time.monotonic()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f'{name} exited during launch; inspect its preserved logs')
        path = SCAN/name/'status.json'
        if path.exists():
            record = read(path)
            if record.get('pid') == proc.pid and record['status'] == 'failed':
                raise RuntimeError(f'{name}: {record.get("stop_reason")}')
            if (record.get('pid') == proc.pid and record['status'] not in ['initializing','loading_weights']
                    and time.monotonic()-launched >= 90):
                return
        time.sleep(10)


def rows(name):
    return [json.loads(s) for s in (SCAN/name/'metrics.jsonl').read_text().splitlines()]


def review500(batch, index):
    result = {'batch': index, 'clear_trend_definition': 'last 100-step mean training loss is >=5% below first 100-step mean and fixed train probe improves', 'runs': {}}
    initial = []
    for name, lr in batch:
        data = rows(name)
        zero = next(r for r in data if r['step'] == 0)
        final = next((r for r in data if r['step'] == 500), None)
        if final is None:
            raise RuntimeError(f'{name} has no 500-step evaluation; manual inspection required')
        initial.append(zero['val/real14_loss'])
        train = [r for r in data if 'train/loss' in r and r['step'] <= 500]
        first = sum(r['train/loss'] for r in train[:4])/4
        last = sum(r['train/loss'] for r in train[-4:])/4
        clear = last <= .95*first and final['train_probe/real14_loss'] < zero['train_probe/real14_loss']
        result['runs'][name] = {'peak_lr': lr, 'first100_train_mean': first, 'last100_train_mean': last,
            'clear_training_decrease': clear, 'step0': zero, 'step500': final,
            'status': status(name), 'gradient_norm_min': min(r['train/grad_norm'] for r in train),
            'gradient_norm_max': max(r['train/grad_norm'] for r in train)}
    result['initial_val_relative_difference'] = abs(initial[0]-initial[1])/max(abs(initial[0]),1e-12)
    result['initial_val_relative_tolerance'] = 0.001  # BF16 GPU reductions are not bitwise deterministic.
    if result['initial_val_relative_difference'] > 0.001:
        raise RuntimeError('Initial validation differs by >0.1% between identical-seed runs; pause scan')
    result['decision'] = 'continue' if any(r['clear_training_decrease'] for r in result['runs'].values()) else 'pause_no_clear_training_decrease'
    write(SCAN/f'batch{index}_stage500_review.json', result)
    return result['decision'] == 'continue'


def wait_stage(batch):
    while True:
        records = [status(name) for name,_ in batch]
        for record in records:
            if record['status'] == 'failed':
                raise RuntimeError(f'{record["name"]}: {record["stop_reason"]}; scan paused')
            if record['status'] not in ['completed','awaiting_review'] and not process_alive(record):
                raise RuntimeError(f'{record["name"]} disappeared; scan paused without relaunch')
        if all(r['status'] in ['completed','awaiting_review'] for r in records):
            return records
        time.sleep(20)


def summarize():
    results = []
    for batch in BATCHES:
        for name, lr in batch:
            directory = SCAN/name
            if not (directory/'status.json').exists():
                results.append({'name': name, 'peak_lr': lr, 'status': 'not_started'})
                continue
            record = status(name)
            result = dict(record)
            if (directory/'best.json').exists():
                best = read(directory/'best.json')
                result.update(best_val_metrics=best['best_metrics'], best_train_probe=best['best_train_probe'])
                at_best = next((r for r in rows(name) if r['step'] == best['best_step']), {})
                result['training_loss_at_best_step'] = at_best.get('train/loss')
            results.append(result)
    write(SCAN/'summary.json', {'runs': results})
    return results


def main():
    lock = (SCAN/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    event('monitoring_batch1')
    try:
        for index, batch in enumerate(BATCHES, 1):
            if index == 2:
                write(SCAN/'batch2_approved.json', {'basis': 'batch1 completed without runtime/data failure; stage500 training trend checked', 'time': time.time(), 'first_batch': summarize()[:2]})
                for pair,gpu in zip(batch,GPUS):
                    path = SCAN/pair[0]/'status.json'
                    if not path.exists():
                        launch(pair,gpu)
            records = wait_stage(batch)
            if not review500(batch,index):
                event('paused', reason=f'batch{index}: both training losses lack clear decrease at step500')
                summarize()
                return
            for pair,gpu,record in zip(batch,GPUS,records):
                if record['status'] == 'awaiting_review':
                    launch(pair,gpu,resume=True)
            records = wait_stage(batch)
            assert all(r['status'] == 'completed' for r in records)
            summarize()
            event(f'batch{index}_completed', runs=records)
        results = summarize()
        best = min((r for r in results if r['status']=='completed'), key=lambda r:r['best_val'])
        while not gpu_available(GPUS[0]):
            time.sleep(15)
        event('comparing_sampled_actions', best_run=best['name'])
        with (ROOT/'logs/action_comparison.log').open('a') as f:
            subprocess.run([sys.executable,'-u',str(SCRIPT_ROOT / 'compare_scan.py'),
                '--gpu-uuid',GPUS[0],'--run-name',best['name']],stdout=f,stderr=subprocess.STDOUT,check=True)
        event('completed', summary=str(SCAN/'summary.json'), comparison=str(SCAN/'action_comparison'/best['name']))
    except BaseException as exc:
        # A scan-level failure must also stop this scan's remaining active run.
        # Check UID, exact run name and training script before signalling; never
        # touch a process merely because it is using an assigned GPU.
        for batch in BATCHES:
            for name, _ in batch:
                path = SCAN/name/'status.json'
                if not path.exists():
                    continue
                record = read(path)
                proc = Path(f'/proc/{record["pid"]}')
                try:
                    if (record['status'] not in ['failed','completed','awaiting_review'] and
                            proc.stat().st_uid == os.getuid() and process_alive(record)):
                        os.kill(record['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except FileNotFoundError:
                    pass
        summarize()
        event('paused', reason=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
