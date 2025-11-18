"""
MAMBA Ablation Study - Ready-to-Run Script

This script provides a convenient interface to run ablation studies on MAMBA architecture variants.
Simply copy the code blocks below into a Jupyter notebook or run this file directly.

Available Models:
    1. MAMBA     - Single direction Mamba (forward only)
    2. BIMAMBA   - Bidirectional Mamba (forward + backward)
    3. MAMBA+    - Single direction Mamba-2 with SSD
    4. BIMAMBA+  - Bidirectional Mamba-2 with SSD + attention

Key Research Questions:
    Q1: Does bidirectional processing help? (MAMBA vs BIMAMBA, MAMBA+ vs BIMAMBA+)
    Q2: Does Mamba-2 SSD improve performance? (MAMBA vs MAMBA+, BIMAMBA vs BIMAMBA+)
    Q3: What's the best overall architecture? (Compare all 4)

Usage:
    # Option 1: Run all models
    python ablation_study.py --dataset IXIC --all

    # Option 2: Run specific models
    python ablation_study.py --dataset IXIC --models MAMBA BIMAMBA MAMBA+

    # Option 3: In Jupyter notebook - copy and run the example cells below
"""

import torch
import argparse
from mamba_models import train_ablation_models


# ============================================================================
# CONVENIENCE FUNCTIONS FOR JUPYTER NOTEBOOK
# ============================================================================

def run_full_ablation(dataset='IXIC', epochs=50, R=3, device='auto'):
    """
    Run complete ablation study on all 4 MAMBA variants

    Args:
        dataset: 'IXIC', 'DJI', or 'NYSE'
        epochs: Number of training epochs
        R: Number of layers in each model
        device: 'cpu', 'cuda', or 'auto'

    Returns:
        Dictionary with results for each model

    Example:
        ```python
        # Copy and run in notebook cell
        from ablation_study import run_full_ablation

        results = run_full_ablation(
            dataset='IXIC',
            epochs=50,
            R=3
        )
        ```
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("FULL ABLATION STUDY - All 4 MAMBA Variants")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Layers: {R}")
    print(f"Device: {device}")
    print("="*80)

    results = train_ablation_models(
        dataset=dataset,
        models=['MAMBA', 'BIMAMBA', 'MAMBA+', 'BIMAMBA+'],
        epochs=epochs,
        R=R,
        verbose=True,
        device=device
    )

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETED!")
    print("="*80)
    print("\nResults Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  IC:     {metrics['ic']:.6f}")
        print(f"  RIC:    {metrics['ric']:.6f}")
        print(f"  Sharpe: {metrics['sharpe']:.4f}")
        print(f"  Dir Acc: {metrics['dir_acc']:.6f}")

    return results


def run_direction_ablation(dataset='IXIC', epochs=50, R=3, use_mamba2=False, device='auto'):
    """
    Ablation study: Single vs Bidirectional processing

    Args:
        dataset: 'IXIC', 'DJI', or 'NYSE'
        epochs: Number of training epochs
        R: Number of layers
        use_mamba2: If True, use Mamba-2; otherwise use original Mamba
        device: 'cpu', 'cuda', or 'auto'

    Returns:
        Dictionary with results comparing single vs bidirectional

    Example:
        ```python
        # Test original Mamba: single vs bidirectional
        from ablation_study import run_direction_ablation

        results = run_direction_ablation(
            dataset='IXIC',
            use_mamba2=False  # Use original Mamba
        )

        # Test Mamba-2: single vs bidirectional
        results_v2 = run_direction_ablation(
            dataset='IXIC',
            use_mamba2=True  # Use Mamba-2
        )
        ```
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_mamba2:
        models = ['MAMBA+', 'BIMAMBA+']
        title = "DIRECTION ABLATION - Mamba-2 (Single vs Bidirectional)"
    else:
        models = ['MAMBA', 'BIMAMBA']
        title = "DIRECTION ABLATION - Mamba (Single vs Bidirectional)"

    print("="*80)
    print(title)
    print("="*80)

    results = train_ablation_models(
        dataset=dataset,
        models=models,
        epochs=epochs,
        R=R,
        verbose=True,
        device=device
    )

    return results


def run_architecture_ablation(dataset='IXIC', epochs=50, R=3, bidirectional=True, device='auto'):
    """
    Ablation study: Mamba vs Mamba-2 (SSD improvement)

    Args:
        dataset: 'IXIC', 'DJI', or 'NYSE'
        epochs: Number of training epochs
        R: Number of layers
        bidirectional: If True, compare BIMAMBA vs BIMAMBA+; otherwise MAMBA vs MAMBA+
        device: 'cpu', 'cuda', or 'auto'

    Returns:
        Dictionary with results comparing Mamba vs Mamba-2

    Example:
        ```python
        # Test bidirectional: Mamba vs Mamba-2
        from ablation_study import run_architecture_ablation

        results = run_architecture_ablation(
            dataset='IXIC',
            bidirectional=True
        )

        # Test single-direction: Mamba vs Mamba-2
        results_single = run_architecture_ablation(
            dataset='IXIC',
            bidirectional=False
        )
        ```
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if bidirectional:
        models = ['BIMAMBA', 'BIMAMBA+']
        title = "ARCHITECTURE ABLATION - Bidirectional (Mamba vs Mamba-2)"
    else:
        models = ['MAMBA', 'MAMBA+']
        title = "ARCHITECTURE ABLATION - Single Direction (Mamba vs Mamba-2)"

    print("="*80)
    print(title)
    print("="*80)

    results = train_ablation_models(
        dataset=dataset,
        models=models,
        epochs=epochs,
        R=R,
        verbose=True,
        device=device
    )

    return results


def run_quick_test(dataset='IXIC', epochs=10, device='auto'):
    """
    Quick test run with reduced epochs for debugging

    Args:
        dataset: 'IXIC', 'DJI', or 'NYSE'
        epochs: Number of epochs (default: 10 for quick testing)
        device: 'cpu', 'cuda', or 'auto'

    Example:
        ```python
        # Quick test before full run
        from ablation_study import run_quick_test

        results = run_quick_test(dataset='IXIC', epochs=10)
        ```
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("QUICK TEST RUN (Reduced Epochs)")
    print("="*80)

    results = train_ablation_models(
        dataset=dataset,
        models=['MAMBA', 'BIMAMBA+'],  # Test extremes
        epochs=epochs,
        R=2,  # Fewer layers for speed
        verbose=True,
        device=device
    )

    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for ablation study"""
    parser = argparse.ArgumentParser(
        description='MAMBA Ablation Study - Compare architecture variants'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='IXIC',
        choices=['IXIC', 'DJI', 'NYSE'],
        help='Dataset to use (default: IXIC)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        choices=['MAMBA', 'BIMAMBA', 'MAMBA+', 'BIMAMBA+'],
        help='Models to train (default: all 4 models)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Train all 4 models'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )

    parser.add_argument(
        '--R',
        type=int,
        default=3,
        help='Number of layers (default: 3)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='Hidden dimension (default: 64)'
    )

    parser.add_argument(
        '--loss',
        type=str,
        default='auto',
        choices=['auto', 'nll', 'mse', 'mae', 'huber'],
        help='Loss function (default: auto)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use (default: auto)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with 10 epochs'
    )

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Quick test mode
    if args.quick:
        print("Running quick test...")
        results = run_quick_test(dataset=args.dataset, epochs=10, device=device)
        return results

    # Determine models to train
    if args.all or args.models is None:
        models = ['MAMBA', 'BIMAMBA', 'MAMBA+', 'BIMAMBA+']
    else:
        models = args.models

    print("="*80)
    print("MAMBA ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Epochs: {args.epochs}")
    print(f"Layers: {args.R}")
    print(f"Device: {device}")
    print("="*80)

    # Run ablation study
    results = train_ablation_models(
        dataset=args.dataset,
        models=models,
        epochs=args.epochs,
        R=args.R,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        loss_type=args.loss,
        verbose=True,
        device=device
    )

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETED!")
    print("="*80)
    print("\nCheck logs/ablation/ for detailed results.")

    return results


# ============================================================================
# JUPYTER NOTEBOOK EXAMPLES
# ============================================================================

"""
# =============================================================================
# JUPYTER NOTEBOOK EXAMPLES - Copy and paste these cells
# =============================================================================

# Cell 1: Import
# ---------------
from ablation_study import (
    run_full_ablation,
    run_direction_ablation,
    run_architecture_ablation,
    run_quick_test
)

# Cell 2: Quick Test (Optional - to verify setup)
# -------------------------------------------------
# Run quick test with reduced epochs
results_test = run_quick_test(dataset='IXIC', epochs=10)

# Cell 3: Full Ablation Study (All 4 Models)
# --------------------------------------------
# Train all 4 MAMBA variants
results_full = run_full_ablation(
    dataset='IXIC',
    epochs=50,
    R=3
)

# Cell 4: Direction Ablation (Original Mamba)
# --------------------------------------------
# Compare single vs bidirectional (original Mamba)
results_dir_v1 = run_direction_ablation(
    dataset='IXIC',
    use_mamba2=False
)

# Cell 5: Direction Ablation (Mamba-2)
# --------------------------------------
# Compare single vs bidirectional (Mamba-2)
results_dir_v2 = run_direction_ablation(
    dataset='IXIC',
    use_mamba2=True
)

# Cell 6: Architecture Ablation (Bidirectional)
# ----------------------------------------------
# Compare Mamba vs Mamba-2 (bidirectional)
results_arch_bi = run_architecture_ablation(
    dataset='IXIC',
    bidirectional=True
)

# Cell 7: Architecture Ablation (Single Direction)
# -------------------------------------------------
# Compare Mamba vs Mamba-2 (single direction)
results_arch_single = run_architecture_ablation(
    dataset='IXIC',
    bidirectional=False
)

# Cell 8: Run on Multiple Datasets
# ---------------------------------
datasets = ['IXIC', 'DJI', 'NYSE']
all_results = {}

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"Running ablation study on {dataset}")
    print(f"{'='*80}\n")

    all_results[dataset] = run_full_ablation(
        dataset=dataset,
        epochs=50,
        R=3
    )

# Cell 9: Analyze Results
# ------------------------
import pandas as pd

# Create comparison table
comparison_data = []
for dataset, results in all_results.items():
    for model, metrics in results.items():
        comparison_data.append({
            'Dataset': dataset,
            'Model': model,
            'IC': metrics['ic'],
            'RIC': metrics['ric'],
            'Sharpe': metrics['sharpe'],
            'Dir_Acc': metrics['dir_acc'],
            'RMSE': metrics['rmse']
        })

df = pd.DataFrame(comparison_data)
print("\nComparison Table:")
print(df.to_string(index=False))

# Save to CSV
df.to_csv('logs/ablation/comparison_summary.csv', index=False)
print("\nResults saved to logs/ablation/comparison_summary.csv")

# =============================================================================
"""


if __name__ == "__main__":
    main()
