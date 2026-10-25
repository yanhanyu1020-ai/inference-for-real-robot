"""Sequential batch profiling on one GPU, stopping or narrowing at memory limit."""
import argparse
import json
import os
from pathlib import Path
from path_config import RUN_ROOT
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[16, 32, 64])
    args = p.parse_args()
    root = RUN_ROOT
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    pending = list(args.batches)
    previous = []
    for f in args.report_dir.glob('batch_*.json'):
        if f.stem.removeprefix('batch_').isdigit():
            d = json.loads(f.read_text())
            if d['status'] == 'passed':
                previous.append(d)
    last_good = max((d['batch_size'] for d in previous), default=0)
    refined = False
    results = []
    while pending:
        batch = pending.pop(0)
        destination = args.report_dir / f'batch_{batch}.json'
        if destination.exists():
            raise RuntimeError(f'Refusing to overwrite existing result: {destination}')
        log = root / 'logs' / f'benchmark_batch_{batch}.log'
        print('START_BATCH', batch, flush=True)
        with log.open('w') as f:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('benchmark_lora.py')),
                                        '--batch-size', str(batch), '--gpu-uuid', args.gpu_uuid,
                                        '--output', str(destination)], stdout=f, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            (args.report_dir / 'active_case.json').write_text(json.dumps({'pid': process.pid, 'batch_size': batch}))
            try:
                returncode = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = -999
        if destination.exists():
            result = json.loads(destination.read_text())
        else:
            result = {'batch_size': batch, 'status': 'failed_without_report', 'returncode': returncode,
                      'log_tail': log.read_text()[-3000:]}
            destination.write_text(json.dumps(result, indent=2))
        results.append(result)
        print('END_BATCH', batch, result['status'], flush=True)
        (args.report_dir / 'sweep_progress.json').write_text(json.dumps({
            'completed_batches': [r['batch_size'] for r in results],
            'remaining_batches': pending, 'last_status': result['status']}, indent=2))
        if result['status'] == 'passed':
            last_good = batch
            peak = result['peak_process_memory_bytes'] / 2**30
            print('METRICS', json.dumps({'batch': batch, 'peak_GiB': peak,
                  'compute_samples_s': result['compute']['samples_per_second'],
                  'pipeline_samples_s': result['pipeline']['samples_per_second']}), flush=True)
            if peak > 68:
                print('STOP: peak process memory already above 68 GiB', flush=True)
                break
        else:
            description = (result.get('error', '') + result.get('log_tail', '')).lower()
            memory_limit = result['status'] == 'skipped_memory_estimate' or any(
                word in description for word in ['resource_exhausted', 'out of memory', 'oom'])
            if memory_limit and last_good and not refined and batch - last_good >= 16:
                middle = ((batch + last_good) // 2 // 8) * 8
                pending = [middle]
                refined = True
                print('REFINE_MEMORY_BOUNDARY', middle, flush=True)
            else:
                print('STOP: unsuccessful case requires review', flush=True)
                break
        time.sleep(3)
    (args.report_dir / 'active_case.json').write_text(json.dumps({'pid': None, 'status': 'finished'}))
    print('SWEEP_FINISHED', flush=True)


if __name__ == '__main__':
    main()
