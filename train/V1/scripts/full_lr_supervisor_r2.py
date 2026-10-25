"""Two authorized full-tuning jobs; batch16 fallback once only after actual GPU OOM."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from path_config import RUN_ROOT as ROOT, SCRIPT_ROOT
SCAN = ROOT/'full_lr_scan_20261007'
JOBS = [
    {'peak_lr': 1e-4, 'lr_label': '1e-4', 'gpu_uuid': 'GPU-87fdd428-76b9-b8ca-7898-fe5ed5907150'},
    {'peak_lr': 5e-5, 'lr_label': '5e-5', 'gpu_uuid': 'GPU-70e1ae82-5564-4841-c645-aeef80244811'},
]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def name(job, batch):
    return f'full_b{batch}_lr{job["lr_label"]}_s42_20261007_r2'


def own_gpu_jobs():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True)
    ids = set()
    for pid in result.splitlines():
        try:
            if (Path('/proc')/pid.strip()).stat().st_uid == os.getuid():
                ids.add(pid.strip())
        except FileNotFoundError:
            pass
    return len(ids)


def event(status, **values):
    record = {'status': status, 'pid': os.getpid(), 'updated_unix': time.time(),
              'jobs': [{k:v for k,v in job.items() if k!='process'} for job in JOBS], **values}
    write(SCAN/'supervisor_status_r2.json', record)
    print(json.dumps(record), flush=True)


def summarize():
    runs = []
    for directory in sorted(SCAN.glob('full_b*_s42_20261007')):
        path=directory/'status.json'
        if path.exists():
            prior=read(path)
            prior['attempt_role']='preserved_pre_update_implementation_failure'
            runs.append(prior)
    for job in JOBS:
        for run_name in job.get('attempts', []):
            directory = SCAN/run_name
            path = directory/'status.json'
            status = read(path) if path.exists() else {'name': run_name, 'status': 'launching', 'step': 0}
            if (directory/'best.json').exists():
                best = read(directory/'best.json')
                status.update(best_val_metrics=best['best_metrics'], best_train_probe=best['best_train_probe'])
                rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
                at_best = next((row for row in rows if row['step']==best['best_step']), {})
                status['training_loss_at_best_step'] = at_best.get('train/loss')
            runs.append(status)
    previous = read(ROOT/'lr_scan_20261005/summary.json')
    lora = next(run for run in previous['runs'] if run['name']=='dual_lora_lr1e-4_s42_20261005')
    write(SCAN/'summary.json', {'runs': runs, 'lora_reference': lora,
          'preferred_batch_size': 32, 'fallback_batch_size': 16,
          'fallback_rule': 'Only actual GPU OOM, at most once per learning rate, fresh run from pretrained base',
          'precision_note': 'Full float32 parameters versus previous LoRA frozen bf16; batch32 versus earlier batch16'})


def launch(job, batch):
    run_name = name(job, batch)
    if (SCAN/run_name).exists():
        raise FileExistsError(f'{run_name} already exists; never duplicate or overwrite')
    if own_gpu_jobs() >= 2:
        raise RuntimeError('Two own GPU processes already active; no third job')
    command = [sys.executable, '-u', str(SCRIPT_ROOT / 'train_full_lr_scan_r2.py'),
               '--gpu-uuid', job['gpu_uuid'], '--peak-lr', str(job['peak_lr']),
               '--batch-size', str(batch), '--name', run_name]
    with (ROOT/'logs'/f'{run_name}.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    job.update(process=process, child_pid=process.pid, active_name=run_name,
               batch_size=batch, phase='training', launch_unix=time.time())
    job.setdefault('attempts', []).append(run_name)
    event('launched_training', name=run_name, child_pid=process.pid)


def actual_oom(job, status):
    text = status.get('stop_reason', '').lower()
    path = ROOT/'logs'/f'{job["active_name"]}.log'
    if path.exists():
        text += path.read_text(errors='replace')[-24000:].lower()
    return any(term in text for term in ['cuda_error_out_of_memory', 'out of memory',
                                         'out-of-memory'])


def after_training(job):
    run_name = job['active_name']
    path = SCAN/run_name/'status.json'
    status = read(path) if path.exists() else {'status': 'failed', 'stop_reason': 'Process exited before status was written'}
    if job['process'].returncode and job['batch_size']==32 and actual_oom(job, status):
        write(SCAN/run_name/'fallback_decision.json', {
            'decision': 'fresh_batch16_run', 'authorized_by': 'User: batch32, OOM only fallback16',
            'original_status': status, 'next_name': name(job, 16),
            'note': 'No checkpoint overwritten; batch16 starts from original base, not a batch32 resume'})
        event('oom_fallback_to_16', failed_name=run_name)
        launch(job, 16)
        return
    if job['process'].returncode or status['status']!='completed':
        raise RuntimeError(f'{run_name}: {status.get("stop_reason",status["status"])}; no further retry')
    job['phase'] = 'trained'
    event('training_completed', name=run_name, best_step=status['best_step'])


def compare(job):
    if own_gpu_jobs() >= 2:
        raise RuntimeError('Two own GPU processes already active; no third job')
    run_name = job['active_name']
    command = [sys.executable, '-u', str(SCRIPT_ROOT / 'compare_full_lr_scan.py'),
               '--gpu-uuid', job['gpu_uuid'], '--run-name', run_name]
    with (ROOT/'logs'/f'{run_name}_action_comparison.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    job.update(process=process, child_pid=process.pid, phase='comparison')
    event('comparing_actions', name=run_name)


def stop_own_children(reason):
    for job in JOBS:
        process = job.get('process')
        if process is None or process.poll() is not None:
            continue
        path = Path('/proc')/str(process.pid)
        try:
            cmd = (path/'cmdline').read_bytes()
            if path.stat().st_uid!=os.getuid() or job['active_name'].encode() not in cmd:
                continue
            if not any(script in cmd for script in [b'train_full_lr_scan_r2.py', b'compare_full_lr_scan.py']):
                continue
            write(SCAN/(job['active_name']+'_supervisor_stop.json'), {'reason': reason, 'pid': process.pid})
            process.send_signal(signal.SIGTERM)
        except FileNotFoundError:
            pass


def main():
    SCAN.mkdir(exist_ok=True)
    lock = (SCAN/'supervisor_r2.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for job in JOBS:
            if (SCAN/name(job,32)).exists():
                raise RuntimeError('Existing run found; inspect instead of restarting supervisor')
        launch(JOBS[0], 32)
        deadline = time.monotonic()+1200
        # The second job starts only after the first has completed weight loading.
        while True:
            job = JOBS[0]
            if job['process'].poll() is not None:
                after_training(job)
            path = SCAN/job['active_name']/'status.json'
            if path.exists() and read(path)['status'] in ['verifying_official_loss','validating_step0','training','completed']:
                break
            if time.monotonic()>deadline:
                raise TimeoutError('First job did not complete weight loading within 20 minutes')
            time.sleep(10)
        launch(JOBS[1], 32)
        while True:
            for job in JOBS:
                if job['phase']=='training' and job['process'].poll() is not None:
                    after_training(job)
                if job['phase']=='comparison' and job['process'].poll() is not None:
                    if job['process'].returncode:
                        raise RuntimeError(f'{job["active_name"]}: action comparison failed; inspect logs')
                    job['phase']='completed'
                    event('action_comparison_completed', name=job['active_name'])
            # Serialize comparison weight loading and wait until both trainers finish.
            if all(job['phase'] in ['trained','completed'] for job in JOBS):
                next_job = next((job for job in JOBS if job['phase']=='trained'), None)
                if next_job is not None:
                    compare(next_job)
            summarize()
            if all(job['phase']=='completed' for job in JOBS):
                event('completed', summary=str(SCAN/'summary.json'))
                break
            time.sleep(15)
    except BaseException as exc:
        reason=f'{type(exc).__name__}: {exc}'
        stop_own_children(reason)
        # Give signaled trainers a chance to preserve their own failure status.
        for job in JOBS:
            process=job.get('process')
            if process is not None:
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: pass
        summarize()
        event('paused', reason=reason)
        raise
    finally:
        subprocess.run([sys.executable, str(SCRIPT_ROOT / 'build_full_lr_report.py'), str(SCAN)], check=False)


if __name__=='__main__':
    main()
