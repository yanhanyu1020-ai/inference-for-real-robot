"""One authorized full-tuning experiment, then offline action comparison; no retries."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from path_config import RUN_ROOT as ROOT, SCRIPT_ROOT
SCAN = ROOT/'full_ft_20261006'
NAME = 'full_b2_lr1e-4_s42_20261006'
GPU = 'GPU-87fdd428-76b9-b8ca-7898-fe5ed5907150'


def event(status, **values):
    record = {'status': status, 'pid': os.getpid(), 'updated_unix': time.time(), **values}
    temp = SCAN/'supervisor_status.tmp'
    temp.write_text(json.dumps(record, indent=2, allow_nan=False))
    temp.replace(SCAN/'supervisor_status.json')
    print(json.dumps(record), flush=True)


def summarize():
    directory = SCAN/NAME
    status = json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {'name': NAME, 'status': 'waiting_for_resources', 'step': 0}
    if (directory/'best.json').exists():
        best = json.loads((directory/'best.json').read_text())
        status.update(best_val_metrics=best['best_metrics'], best_train_probe=best['best_train_probe'])
        rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        match = next((row for row in rows if row['step'] == best['best_step']), {})
        status['training_loss_at_best_step'] = match.get('train/loss')
    reference = json.loads((ROOT/'lr_scan_20261005/summary.json').read_text())
    lora = next(run for run in reference['runs'] if run['name'] == 'dual_lora_lr1e-4_s42_20261005')
    initial = SCAN/'full_lr1e-4_s42_20261006'
    check = {key: json.loads((initial/key).read_text()) for key in ['status.json','compiled_memory.json'] if (initial/key).exists()}
    (SCAN/'summary.json').write_text(json.dumps({'runs': [status], 'lora_reference': lora,
        'startup_resource_check': check}, indent=2, allow_nan=False))
    return status


def main():
    SCAN.mkdir(exist_ok=True)
    lock = (SCAN/'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        directory = SCAN/NAME
        if not (directory/'status.json').exists():
            event('launching_training', name=NAME, gpu_uuid=GPU)
            with (ROOT/'logs'/f'{NAME}.log').open('a') as log:
                code = subprocess.call([sys.executable, '-u', str(SCRIPT_ROOT / 'train_full_b2.py'),
                    '--gpu-uuid', GPU, '--peak-lr', '1e-4', '--name', NAME],
                    stdout=log, stderr=subprocess.STDOUT)
            if code:
                raise RuntimeError(f'Full training exited with code {code}; inspect preserved training log; no retry')
        status = summarize()
        if status['status'] != 'completed':
            raise RuntimeError(f'Full experiment stopped: {status.get("stop_reason", status["status"])}')
        comparison = SCAN/'action_comparison'/NAME
        if not (comparison/'metrics.json').exists():
            event('comparing_sampled_actions', best_step=status['best_step'])
            with (ROOT/'logs'/(NAME+'_action_comparison.log')).open('a') as log:
                subprocess.run([sys.executable, '-u', str(SCRIPT_ROOT / 'compare_full_b2.py'),
                    '--gpu-uuid', GPU, '--run-name', NAME], stdout=log, stderr=subprocess.STDOUT, check=True)
        event('completed', summary=str(SCAN/'summary.json'), comparison=str(comparison))
    except BaseException as exc:
        summarize()
        event('paused', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        subprocess.run([sys.executable, str(SCRIPT_ROOT / 'build_full_report_b2.py'), str(SCAN)], check=False)


if __name__ == '__main__':
    main()
