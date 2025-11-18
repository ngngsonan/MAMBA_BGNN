"""
Run GNN Ablation Study - Compare different graph layers with BIMamba

This script trains and compares:
- BIMamba + GCN
- BIMamba + GAT
- BIMamba + GraphSAGE
- BIMamba + MAGAC (baseline)

Usage:
    python run_gnn_study.py --dataset IXIC --epochs 50
    python run_gnn_study.py --dataset DJI --models GCN GAT MAGAC
    python run_gnn_study.py --all-datasets
"""

import argparse
import torch
import sys
import os

# Add models directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamba_gnn_study import train_gnn_comparison


def parse_args():
    parser = argparse.ArgumentParser(description='GNN Ablation Study with BIMamba')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='IXIC',
                        choices=['IXIC', 'DJI', 'NYSE'],
                        help='Dataset to use')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run on all datasets sequentially')

    # Model selection
    parser.add_argument('--models', type=str, nargs='+',
                        default=['GCN', 'GAT', 'GraphSAGE', 'MAGAC'],
                        choices=['GCN', 'GAT', 'GraphSAGE', 'MAGAC'],
                        help='Models to train')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--window', type=int, default=5,
                        help='Lookback window size')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')

    # Model architecture
    parser.add_argument('--R', type=int, default=3,
                        help='Number of BIMamba layers')
    parser.add_argument('--K', type=int, default=3,
                        help='Chebyshev polynomial order (for MAGAC)')
    parser.add_argument('--d-e', type=int, default=10,
                        help='Node embedding dimension')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads')

    # Training options
    parser.add_argument('--loss-type', type=str, default='auto',
                        choices=['auto', 'nll', 'mse', 'mae', 'huber'],
                        help='Loss function type')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Logging
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Base directory for logs')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to train on')

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*80)
    print("MAMBA-GNN ABLATION STUDY")
    print("="*80)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("="*80)

    # Determine datasets to run
    datasets = ['IXIC', 'DJI', 'NYSE'] if args.all_datasets else [args.dataset]

    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"RUNNING STUDY FOR {dataset}")
        print(f"{'='*80}\n")

        results = train_gnn_comparison(
            dataset=dataset,
            models=args.models,
            window=args.window,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            hidden_dim=args.hidden_dim,
            loss_type=args.loss_type,
            early_stop_patience=args.patience,
            R=args.R,
            K=args.K,
            d_e=args.d_e,
            heads=args.heads,
            verbose=not args.quiet,
            log_base_dir=args.log_dir,
            device=device
        )

        all_results[dataset] = results

    # Print summary if multiple datasets
    if len(datasets) > 1:
        print("\n" + "="*80)
        print("CROSS-DATASET SUMMARY")
        print("="*80)

        for model_key in ['BIMamba+GCN', 'BIMamba+GAT', 'BIMamba+GraphSAGE', 'BIMamba+MAGAC']:
            print(f"\n{model_key}:")
            for dataset in datasets:
                if model_key in all_results[dataset]:
                    res = all_results[dataset][model_key]
                    print(f"  {dataset:6s}: IC={res['ic']:.4f}, RIC={res['ric']:.4f}, "
                          f"MSE={res['mse']:.4f}, MAE={res['mae']:.4f}")

        print("="*80)

    print("\n✓ GNN Ablation Study completed!")
    print(f"  Results saved to: {args.log_dir}/mamba_gnn/")


if __name__ == "__main__":
    main()
