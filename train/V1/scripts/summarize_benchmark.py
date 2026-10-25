"""Summarize completed benchmarks and optionally log scalar results to W&B."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('report_dir', type=Path)
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()
    rows = []
    for f in args.report_dir.glob('batch_*.json'):
        if not f.stem.removeprefix('batch_').isdigit():
            continue
        d = json.loads(f.read_text())
        row = {'batch_size': d['batch_size'], 'status': d['status']}
        if d['status'] == 'passed':
            row.update(peak_process_memory_GiB=d['peak_process_memory_bytes'] / 2**30,
                       compute_samples_per_second=d['compute']['samples_per_second'],
                       pipeline_samples_per_second=d['pipeline']['samples_per_second'],
                       compute_step_seconds=d['compute']['step_seconds_mean'],
                       pipeline_step_seconds=d['pipeline']['step_seconds_mean'],
                       pipeline_data_wait_fraction=d['pipeline']['data_wait_fraction'],
                       compiler_live_memory_GiB=d['compiled_live_memory_estimate_bytes'] / 2**30,
                       benchmark_optimizer_steps=d['benchmark_optimizer_steps'])
        rows.append(row)
    rows.sort(key=lambda r: r['batch_size'])
    good = [r for r in rows if r['status'] == 'passed']
    if not good:
        raise RuntimeError('No successful cases')
    best = max(good, key=lambda r: r['pipeline_samples_per_second'])
    near_best = [r for r in good if r['pipeline_samples_per_second'] >= 0.95 * best['pipeline_samples_per_second']]
    recommended = min(near_best, key=lambda r: r['batch_size'])
    summary = {'variant': 'dual_lora', 'gpu': 'A800-SXM4-80GB', 'host': '16.7',
               'results': rows, 'best_pipeline_batch': best['batch_size'],
               'max_successfully_tested_batch': max(r['batch_size'] for r in good),
               'recommended_initial_batch': recommended['batch_size'],
               'recommendation_rule': 'Smallest measured batch within 95% of the best pipeline throughput.',
               'training_checkpoint_saved': False,
               'method': 'Same official initialization/train_step, 5 warmup, 15 cached-batch timed steps, '
                         '8 real-data pipeline steps per case; separate process and base-weight reset.',
               'caveats': ['Short single-GPU measurements, not model-quality comparisons.',
                           'NVML process peak includes the growing JAX allocator; preallocation disabled.',
                           'Compile and initialization are excluded from throughput.',
                           'Concurrent multi-experiment I/O contention was not tested.']}
    existing = args.report_dir / 'summary.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        if old.get('wandb_url'):
            summary['wandb_url'] = old['wandb_url']
    if args.wandb:
        import wandb
        run = wandb.init(project='heterovla-pi05', entity='huph7-sun-yat-sen-university',
                         name='dual-lora-batch-profile-' + args.report_dir.name,
                         job_type='benchmark', group='batch-size-profile',
                         config={'variant': 'dual_lora', 'horizon': 10, 'cameras': 3,
                                 'real_action_dim': 14, 'state_dim': 30, 'formal_training': False},
                         settings=wandb.Settings(disable_code=True))
        for i, row in enumerate(good):
            run.log({f'benchmark/{k}': v for k, v in row.items() if isinstance(v, (int, float))}, step=i)
        run.summary['best_pipeline_batch'] = summary['best_pipeline_batch']
        run.summary['max_successfully_tested_batch'] = summary['max_successfully_tested_batch']
        run.summary['recommended_initial_batch'] = summary['recommended_initial_batch']
        summary['wandb_url'] = run.url
        run.finish()
    (args.report_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
