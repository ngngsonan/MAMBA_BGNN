"""
Results Comparison Utility

This module provides functions to read, compare, and visualize results from
multiple model training runs (MAMBA_BGNN and baselines).

Functions:
    - read_test_metrics: Read test metrics from a single run
    - compare_multiple_runs: Compare results from multiple runs
    - generate_comparison_report: Generate comprehensive comparison report
    - plot_metrics_comparison: Visualize metrics comparison (optional)
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def read_test_metrics(log_dir: str, model_name: str = None) -> Dict:
    """
    Read test metrics from a single training run

    Args:
        log_dir: Path to log directory
        model_name: Optional model name (extracted from path if not provided)

    Returns:
        Dictionary with test metrics and metadata
    """
    log_path = Path(log_dir)

    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Extract model name from path if not provided
    if model_name is None:
        model_name = log_path.name

    # Read test metrics CSV
    test_csv = log_path / 'test_metrics.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f"Test metrics CSV not found: {test_csv}")

    test_df = pd.read_csv(test_csv)
    if len(test_df) == 0:
        raise ValueError(f"Empty test metrics CSV: {test_csv}")

    # Get last row (latest test results)
    metrics = test_df.iloc[-1].to_dict()

    # Read validation metrics for additional info
    val_csv = log_path / 'val_metrics.csv'
    val_info = {}
    if val_csv.exists():
        val_df = pd.read_csv(val_csv)
        if len(val_df) > 0:
            # Find best epoch
            if 'nll' in val_df.columns:
                best_epoch = val_df['nll'].idxmin() + 1
                val_info['best_epoch'] = int(best_epoch)
                val_info['best_val_nll'] = float(val_df['nll'].min())
            elif 'loss' in val_df.columns:
                best_epoch = val_df['loss'].idxmin() + 1
                val_info['best_epoch'] = int(best_epoch)
                val_info['best_val_loss'] = float(val_df['loss'].min())

            val_info['total_epochs'] = len(val_df)

    # Read rolling window stats if available
    rolling_stats_csv = log_path / 'rolling_window_stats.csv'
    rolling_stats = {}
    if rolling_stats_csv.exists():
        rolling_df = pd.read_csv(rolling_stats_csv, index_col=0)
        rolling_stats = rolling_df.to_dict()

    # Combine all info
    result = {
        'model_name': model_name,
        'log_dir': str(log_dir),
        'test_metrics': metrics,
        'validation_info': val_info,
        'rolling_stats': rolling_stats
    }

    return result


def compare_multiple_runs(log_dirs: List[str], model_names: List[str] = None) -> pd.DataFrame:
    """
    Compare test metrics from multiple training runs

    Args:
        log_dirs: List of paths to log directories
        model_names: Optional list of model names (same length as log_dirs)

    Returns:
        DataFrame with comparison of all models
    """
    if model_names is None:
        model_names = [None] * len(log_dirs)

    if len(model_names) != len(log_dirs):
        raise ValueError("model_names must have same length as log_dirs")

    results = []

    for log_dir, model_name in zip(log_dirs, model_names):
        try:
            result = read_test_metrics(log_dir, model_name)

            # Flatten metrics for DataFrame
            row = {
                'Model': result['model_name'],
                'Log Dir': result['log_dir']
            }

            # Add test metrics
            test_metrics = result['test_metrics']
            for key, value in test_metrics.items():
                row[key.upper()] = value

            # Add validation info
            val_info = result['validation_info']
            if 'best_epoch' in val_info:
                row['Best Epoch'] = val_info['best_epoch']
            if 'total_epochs' in val_info:
                row['Total Epochs'] = val_info['total_epochs']

            results.append(row)

        except Exception as e:
            print(f"Warning: Failed to read {log_dir}: {e}")
            continue

    if not results:
        raise ValueError("No valid results found")

    df = pd.DataFrame(results)

    # Reorder columns for better readability
    priority_cols = ['Model', 'RMSE', 'MAE', 'IC', 'RIC']
    other_cols = [col for col in df.columns if col not in priority_cols and col != 'Log Dir']
    ordered_cols = priority_cols + other_cols + ['Best Epoch', 'Total Epochs', 'Log Dir']
    ordered_cols = [col for col in ordered_cols if col in df.columns]

    return df[ordered_cols]


def generate_comparison_report(
    log_dirs: List[str],
    model_names: List[str] = None,
    output_dir: str = None,
    dataset_name: str = "Comparison"
) -> Tuple[pd.DataFrame, str]:
    """
    Generate comprehensive comparison report similar to baseline_notebook.py

    Args:
        log_dirs: List of paths to log directories
        model_names: Optional list of model names
        output_dir: Output directory for report (default: current directory)
        dataset_name: Dataset name for report title

    Returns:
        Tuple of (comparison_df, report_path)
    """
    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    # Generate comparison DataFrame
    comparison_df = compare_multiple_runs(log_dirs, model_names)

    # Save comparison CSV
    comparison_csv = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_csv, index=False)

    # Generate text report
    report_txt = os.path.join(output_dir, 'comparison_summary.txt')

    with open(report_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODELS COMPARISON SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of models: {len(comparison_df)}\n")
        f.write(f"Models: {', '.join(comparison_df['Model'].values)}\n")
        f.write("="*80 + "\n\n")

        # Metrics comparison table
        f.write("METRICS COMPARISON:\n")
        f.write("-"*80 + "\n")

        # Select key metrics for display
        display_cols = ['Model', 'RMSE', 'MAE', 'IC', 'RIC']

        # Add additional metrics if available
        if 'NLL' in comparison_df.columns:
            display_cols.append('NLL')
        if 'CRPS' in comparison_df.columns:
            display_cols.extend(['CRPS', 'SHARP'])
        if 'DIR_ACC' in comparison_df.columns:
            display_cols.append('DIR_ACC')
        if 'SHARPE' in comparison_df.columns:
            display_cols.extend(['SHARPE', 'MAX_DRAWDOWN'])

        display_cols = [col for col in display_cols if col in comparison_df.columns]
        display_df = comparison_df[display_cols].copy()

        # Format numeric columns
        for col in display_df.columns:
            if col != 'Model' and display_df[col].dtype in [np.float64, np.float32]:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")

        f.write(display_df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")

        # Best model by each metric
        f.write("\nBEST MODELS BY METRIC:\n")
        f.write("-"*80 + "\n")

        metrics_to_check = {
            'RMSE': 'min',
            'MAE': 'min',
            'IC': 'max',
            'RIC': 'max'
        }

        # Add additional metrics based on availability
        if 'NLL' in comparison_df.columns:
            metrics_to_check['NLL'] = 'min'
        if 'CRPS' in comparison_df.columns:
            metrics_to_check['CRPS'] = 'min'
        if 'DIR_ACC' in comparison_df.columns:
            metrics_to_check['DIR_ACC'] = 'max'
        if 'SHARPE' in comparison_df.columns:
            metrics_to_check['SHARPE'] = 'max'

        for metric, opt in metrics_to_check.items():
            if metric in comparison_df.columns:
                if opt == 'min':
                    best_idx = comparison_df[metric].idxmin()
                else:
                    best_idx = comparison_df[metric].idxmax()

                best_model = comparison_df.loc[best_idx, 'Model']
                best_value = comparison_df.loc[best_idx, metric]

                f.write(f"{metric:15s}: {best_model:20s} ({best_value:.6f})\n")

        f.write("\n")

        # Training efficiency (if epoch info available)
        if 'Best Epoch' in comparison_df.columns and 'Total Epochs' in comparison_df.columns:
            f.write("\nTRAINING EFFICIENCY:\n")
            f.write("-"*80 + "\n")
            for idx, row in comparison_df.iterrows():
                if pd.notna(row.get('Best Epoch')) and pd.notna(row.get('Total Epochs')):
                    f.write(f"{row['Model']:20s}: {int(row['Best Epoch']):3d}/{int(row['Total Epochs']):3d} epochs\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("OUTPUT FILES:\n")
        f.write("-"*80 + "\n")
        f.write(f"Comparison table: {comparison_csv}\n")
        f.write(f"Summary report:   {report_txt}\n")
        f.write("="*80 + "\n")

    # Print summary to console
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Models compared: {len(comparison_df)}")
    print("-"*80)
    print(display_df.to_string(index=False))
    print("\n" + "="*80)
    print(f"✓ Comparison saved to: {output_dir}")
    print(f"  - Table: {comparison_csv}")
    print(f"  - Report: {report_txt}")
    print("="*80 + "\n")

    return comparison_df, report_txt


def find_model_logs(
    base_dir: str = 'logs',
    dataset: str = None,
    pattern: str = None
) -> List[str]:
    """
    Find all model log directories matching criteria

    Args:
        base_dir: Base directory to search (default: 'logs')
        dataset: Filter by dataset name (e.g., 'IXIC')
        pattern: Additional pattern to match in directory name

    Returns:
        List of log directory paths
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    log_dirs = []

    for item in base_path.rglob('test_metrics.csv'):
        log_dir = item.parent

        # Apply filters
        if dataset and dataset not in str(log_dir):
            continue

        if pattern and pattern not in str(log_dir):
            continue

        log_dirs.append(str(log_dir))

    return sorted(log_dirs)


def extract_metrics_for_plotting(comparison_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract metrics from comparison DataFrame for plotting

    Args:
        comparison_df: Comparison DataFrame from compare_multiple_runs

    Returns:
        Dictionary mapping metric names to arrays of values
    """
    metrics = {}
    models = comparison_df['Model'].values

    for col in comparison_df.columns:
        if col not in ['Model', 'Log Dir', 'Best Epoch', 'Total Epochs']:
            if comparison_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                metrics[col] = comparison_df[col].values

    metrics['models'] = models
    return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# if __name__ == "__main__":
#     """
#     Example usage of results comparison utilities
#     """

#     # Example 1: Compare specific runs
#     print("Example 1: Compare specific runs")
#     print("-" * 80)

#     log_dirs = [
#         'logs/IXIC_MAMBA_BGNN_20250101_120000',
#         'logs/baselines/IXIC_Linear_20250101_120000',
#         'logs/baselines/IXIC_LSTM_20250101_120000'
#     ]

#     model_names = ['MAMBA_BGNN', 'Linear', 'LSTM']

#     try:
#         comparison_df, report_path = generate_comparison_report(
#             log_dirs=log_dirs,
#             model_names=model_names,
#             output_dir='logs/comparison',
#             dataset_name='IXIC'
#         )
#         print(f"\nComparison report generated: {report_path}")
#     except Exception as e:
#         print(f"Error in Example 1: {e}")

#     print("\n" + "="*80 + "\n")

#     # Example 2: Auto-find and compare all runs for a dataset
#     print("Example 2: Auto-find logs for IXIC dataset")
#     print("-" * 80)

#     try:
#         found_logs = find_model_logs(base_dir='logs', dataset='IXIC')
#         print(f"Found {len(found_logs)} log directories:")
#         for log_dir in found_logs[:5]:  # Print first 5
#             print(f"  - {log_dir}")

#         if found_logs:
#             comparison_df = compare_multiple_runs(found_logs)
#             print("\nComparison table:")
#             print(comparison_df[['Model', 'RMSE', 'IC', 'RIC']].to_string(index=False))
#     except Exception as e:
#         print(f"Error in Example 2: {e}")
