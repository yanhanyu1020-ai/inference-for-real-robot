#!/usr/bin/env python3
"""Default: print configuration on CPU. Training requires an explicit --execute."""
import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
from path_config import OPENPI_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['action_lora', 'dual_lora', 'expert_full', 'full'], required=True)
    p.add_argument('--gpu', help='A GPU index or UUID assigned to this job')
    p.add_argument('--exp-name', default='prepared')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--peak-lr', type=float, default=2.5e-5)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--remat-policy', choices=['nothing_saveable', 'dots_saveable', 'dots_with_no_batch_dims_saveable', 'none'], default='nothing_saveable')
    args = p.parse_args()
    if args.execute and (not args.gpu or args.exp_name == 'prepared'):
        p.error('--execute requires --gpu and a unique --exp-name')
    if args.execute:
        if ',' in args.gpu:
            p.error('Each experiment must be bound to one assigned GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        os.environ['JAX_PLATFORMS'] = 'cuda'
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.90')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORMS'] = 'cpu'
    from experiment_configs import make_config, summary, install_dataset_adapter, PROJECT_ROOT
    config = make_config(args.variant, batch_size=args.batch_size, peak_lr=args.peak_lr,
                         steps=args.steps, exp_name=args.exp_name, remat_policy=args.remat_policy)
    print(json.dumps(summary(config), ensure_ascii=False, indent=2))
    if not args.execute:
        print('CONFIGURATION ONLY: no model training or GPU allocation performed.')
        return
    install_dataset_adapter()
    spec = importlib.util.spec_from_file_location('openpi_official_train',
        OPENPI_ROOT / 'scripts/train.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    logging.basicConfig(level=logging.INFO, force=True)
    trainer.main(config)


if __name__ == '__main__':
    main()
