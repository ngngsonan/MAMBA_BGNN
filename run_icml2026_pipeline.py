"""Entry point for the ICML 2026 regime-aware training pipeline."""

from __future__ import annotations

import argparse
import os

import torch

from mamba_bgnn import ModelArgs, data_processing
from hierarchical_mamba_bgnn import HierarchicalMambaBGNN
from risk_aware_training import (
    RiskAwareLossConfig,
    RiskAwareLoss,
    RiskAwareTrainer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICML 2026 regime-aware pipeline")
    parser.add_argument('--dataset', type=str, default='IXIC', choices=['IXIC', 'DJI', 'NYSE'],
                        help='Dataset identifier (default: IXIC)')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--kl_weight', type=float, default=5e-4)
    parser.add_argument('--risk_weight', type=float, default=0.05)
    parser.add_argument('--conformal_alpha', type=float, default=0.1)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--mc_train', type=int, default=4)
    parser.add_argument('--mc_eval', type=int, default=16)
    parser.add_argument('--cvar_level', type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.upper()
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Missing dataset at {data_path}')

    num_features, train_loader, val_loader, test_loader = data_processing(
        data_path,
        window=args.window,
        batch_size=args.batch_size,
    )

    model_args = ModelArgs(d_model=num_features, seq_len=args.window, d_state=128)
    model = HierarchicalMambaBGNN(
        model_args,
        mc_train=args.mc_train,
        mc_eval=args.mc_eval,
        cvar_level=args.cvar_level,
    )

    log_dir = args.log_dir or f'{dataset}_icml2026_log'
    run_args = {
        'epochs': args.epochs,
        'early_stop': True,
        'early_stop_patience': 20,
        'grad_norm': True,
        'max_grad_norm': 1.0,
        'log_dir': log_dir,
        'model_name': f'{dataset}_ICML2026',
    }

    config = RiskAwareLossConfig(
        kl_weight=args.kl_weight,
        risk_weight=args.risk_weight,
        alpha=args.conformal_alpha,
    )
    loss_fn = RiskAwareLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = RiskAwareTrainer(
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        args=run_args,
        lr_scheduler=scheduler,
    )
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()
