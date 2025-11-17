"""
Train all baseline models and save comprehensive metrics for comparison.

This script:
1. Loads data using the current data_processing pipeline
2. Trains each baseline model (Linear, LSTM, Transformer, AGCRN, TemporalGN)
3. Saves comprehensive metrics to separate log directories
4. Enables easy comparison across all models
"""

import torch
import torch.nn as nn
import os
import sys
from datetime import datetime
import argparse

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from baseline.baseline_models import create_baseline_models
from utils.data_processing import data_processing
from utils.trainer import Trainer


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood Loss for probabilistic predictions"""
    def __init__(self):
        super().__init__()

    def forward(self, mu, y, var):
        """
        Args:
            mu: predicted mean (B,)
            y: true values (B,)
            var: predicted variance (B,)
        Returns:
            NLL loss (scalar)
        """
        var = var.clamp(min=1e-6)  # Numerical stability
        nll = 0.5 * (torch.log(var) + ((y - mu) ** 2) / var)
        return nll.mean()


def train_single_model(model_name, model, train_loader, val_loader, test_loader,
                       num_features, dataset_name, args):
    """
    Train a single baseline model with comprehensive evaluation

    Args:
        model_name: Name of the model
        model: Model instance
        train_loader, val_loader, test_loader: Data loaders
        num_features: Number of input features
        dataset_name: Name of dataset for logging
        args: Training arguments

    Returns:
        Dictionary of test metrics
    """
    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)

    # Create log directory for this model
    log_dir = os.path.join(args['base_log_dir'], f"{dataset_name}_{model_name}")
    os.makedirs(log_dir, exist_ok=True)

    # Update args with model-specific info
    model_args = args.copy()
    model_args['log_dir'] = log_dir
    model_args['model_name'] = model_name

    # Setup loss function and optimizer
    loss_fn = GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    # Optional learning rate scheduler
    lr_scheduler = None
    if args.get('use_scheduler', False):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.get('scheduler_step', 10), gamma=0.5
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=model_args,
        lr_scheduler=lr_scheduler
    )

    # Add dataset info
    dataset_info = {
        'dataset_name': dataset_name,
        'num_features': num_features,
        'model_name': model_name,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset)
    }
    trainer.set_dataset_info(dataset_info)

    # Train
    print(f"\n[{model_name}] Starting training...")
    trainer.train()

    # Test
    print(f"\n[{model_name}] Evaluating on test set...")
    test_metrics = trainer.test()

    print(f"\n[{model_name}] Training completed!")
    print(f"[{model_name}] Results saved to: {log_dir}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train all baseline models')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='IXIC',
                       choices=['IXIC', 'DJI', 'NYSE'],
                       help='Dataset to use')
    parser.add_argument('--window', type=int, default=5,
                       help='Lookback window size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--early_stop', action='store_true', default=True,
                       help='Use early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--grad_norm', action='store_true', default=True,
                       help='Use gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                       help='Max gradient norm for clipping')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for models')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['Linear', 'LSTM', 'Transformer', 'AGCRN', 'TemporalGN'],
                       help='Models to train')

    # Scheduler arguments
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                       help='Use learning rate scheduler')
    parser.add_argument('--scheduler_step', type=int, default=10,
                       help='Scheduler step size')

    # Logging arguments
    parser.add_argument('--log_step', type=int, default=10,
                       help='Logging frequency')
    parser.add_argument('--base_log_dir', type=str, default='logs/baselines',
                       help='Base directory for logs')

    # Rolling window evaluation
    parser.add_argument('--rolling_window_size', type=int, default=63,
                       help='Rolling window size for evaluation (~3 months)')
    parser.add_argument('--rolling_step_size', type=int, default=21,
                       help='Rolling step size (~1 month)')

    args = parser.parse_args()

    # Convert to dict for easier handling
    args_dict = vars(args)

    print("="*80)
    print("BASELINE MODELS TRAINING")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Window: {args.window}, Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.learning_rate}")
    print(f"Log directory: {args.base_log_dir}")
    print("="*80)

    # Load data
    print("\nLoading and processing data...")
    data_path = f'Dataset/combined_dataframe_{args.dataset}.csv'
    num_features, train_loader, val_loader, test_loader = data_processing(
        data_path=data_path,
        window=args.window,
        batch_size=args.batch_size
    )

    print(f"\nData loaded successfully!")
    print(f"  Features: {num_features}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Create all baseline models
    print("\nCreating baseline models...")
    all_models = create_baseline_models(input_dim=num_features, seq_len=args.window)

    # Filter models based on args
    models_to_train = {name: model for name, model in all_models.items()
                      if name in args.models}

    print(f"Models to train: {list(models_to_train.keys())}")

    # Train each model
    results = {}
    for model_name, model in models_to_train.items():
        try:
            test_metrics = train_single_model(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_features=num_features,
                dataset_name=args.dataset,
                args=args_dict
            )
            results[model_name] = test_metrics

        except Exception as e:
            print(f"\n❌ ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}

    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    for model_name, metrics in results.items():
        if 'error' in metrics:
            print(f"\n{model_name}: ❌ FAILED - {metrics['error']}")
        else:
            print(f"\n{model_name}:")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE:  {metrics['mae']:.6f}")
            print(f"  IC:   {metrics['ic']:.6f}")
            print(f"  RIC:  {metrics['ric']:.6f}")
            print(f"  NLL:  {metrics['nll']:.6f}")
            print(f"  Dir Acc: {metrics['dir_acc']:.6f}")
            print(f"  Sharpe:  {metrics['sharpe_ratio']:.6f}")

    print("\n" + "="*80)
    print("ALL TRAINING COMPLETED!")
    print(f"Results saved to: {args.base_log_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
