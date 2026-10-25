#!/usr/bin/env python3
"""Compare bounded rematerialization benchmarks; never start training."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()
    reports = {path.stem: json.loads(path.read_text())
               for path in sorted(args.directory.glob('*_b16.json'))}
    baseline = reports['baseline_b16']
    initial = np.load(args.directory / 'baseline_b16.npz')
    summary = {'baseline': 'baseline_b16', 'batch_size': 16, 'cases': {},
               'scope': 'Short throughput and sampled-update check, not convergence or validation evaluation.'}
    for name, row in reports.items():
        case = {'status': row['status'], 'remat_policy': row['remat_policy'],
                'peak_memory_gib': row['peak_process_memory_bytes'] / 2**30,
                'compiled_live_memory_gib': row.get('compiled_live_memory_estimate_bytes', 0) / 2**30}
        if row['status'] == 'passed':
            sampled = np.load(args.directory / f'{name}.npz')
            delta_ref = initial['after_warmup'] - initial['initial']
            delta = sampled['after_warmup'] - sampled['initial']
            difference = delta - delta_ref
            case.update(
                pipeline_samples_per_second=row['pipeline']['samples_per_second'],
                compute_samples_per_second=row['compute']['samples_per_second'],
                speed_ratio=row['pipeline']['samples_per_second'] / baseline['pipeline']['samples_per_second'],
                same_batch_sha256=row['cached_batch_sha256'] == baseline['cached_batch_sha256'],
                same_initial_parameter_samples=bool(np.array_equal(sampled['initial'], initial['initial'])),
                sampled_parameter_count=int(delta.size),
                sampled_update_max_abs_difference=float(np.max(np.abs(difference))),
                sampled_update_relative_l2_difference=float(np.linalg.norm(difference) / max(np.linalg.norm(delta_ref), 1e-30)),
                first_step_metrics=row['first_step_metrics'],
                loss_sequence_max_abs_difference=float(np.max(np.abs(
                    np.array(row['loss_sequence']) - np.array(baseline['loss_sequence'])))),
                optimizer_steps=row['benchmark_optimizer_steps'],
            )
        summary['cases'][name] = case
    if args.wandb:
        import wandb
        with wandb.init(project='heterovla-pi05', entity='huph7-sun-yat-sen-university',
                        name='hph-eval-pi05-lora-remat', job_type='benchmark',
                        config={'batch_size': 16, 'formal_training': False, 'checkpoint_saved': False}) as run:
            for name, row in summary['cases'].items():
                for key, value in row.items():
                    if isinstance(value, (int, float, bool)):
                        run.summary[f'{name}/{key}'] = value
            summary['wandb_url'] = run.url
    (args.directory / 'comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
