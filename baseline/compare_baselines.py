"""
Compare baseline models by aggregating their comprehensive metrics.

This script:
1. Reads metrics from all baseline model log directories
2. Compiles them into a comparison table
3. Generates visualizations and rankings
4. Saves comprehensive comparison report
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_model_metrics(log_dir):
    """
    Load comprehensive metrics for a single model

    Args:
        log_dir: Path to model's log directory

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Load comprehensive metrics (main results)
    comprehensive_path = os.path.join(log_dir, 'comprehensive_metrics.csv')
    if os.path.exists(comprehensive_path):
        df = pd.read_csv(comprehensive_path)
        if len(df) > 0:
            # Get the last row (should only be one row for test results)
            metrics.update(df.iloc[-1].to_dict())

    # Load regime analysis
    regime_path = os.path.join(log_dir, 'regime_analysis.csv')
    if os.path.exists(regime_path):
        df = pd.read_csv(regime_path)
        if len(df) > 0:
            regime_metrics = df.iloc[-1].to_dict()
            metrics.update({f'regime_{k}': v for k, v in regime_metrics.items()})

    # Load stress test
    stress_path = os.path.join(log_dir, 'stress_test.csv')
    if os.path.exists(stress_path):
        df = pd.read_csv(stress_path)
        if len(df) > 0:
            stress_metrics = df.iloc[-1].to_dict()
            metrics.update({f'stress_{k}': v for k, v in stress_metrics.items()})

    # Load rolling window stats
    rolling_stats_path = os.path.join(log_dir, 'rolling_window_stats.csv')
    if os.path.exists(rolling_stats_path):
        df = pd.read_csv(rolling_stats_path, index_col=0)
        if len(df) > 0:
            # Add mean and std for key metrics
            for metric in ['rmse', 'mae', 'ic', 'ric', 'dir_acc']:
                if metric in df.columns:
                    metrics[f'rolling_{metric}_mean'] = df.loc['mean', metric]
                    metrics[f'rolling_{metric}_std'] = df.loc['std', metric]

    return metrics


def load_all_models(base_log_dir, dataset_name=None):
    """
    Load metrics from all baseline models

    Args:
        base_log_dir: Base directory containing model logs
        dataset_name: Optional dataset name to filter

    Returns:
        DataFrame with all model metrics
    """
    all_metrics = []

    if not os.path.exists(base_log_dir):
        print(f"Warning: {base_log_dir} does not exist")
        return pd.DataFrame()

    # Iterate through all subdirectories
    for subdir in os.listdir(base_log_dir):
        subdir_path = os.path.join(base_log_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        # Parse directory name (format: DATASET_MODELNAME)
        if '_' in subdir:
            parts = subdir.split('_', 1)
            dataset = parts[0]
            model_name = parts[1]

            # Filter by dataset if specified
            if dataset_name and dataset != dataset_name:
                continue

            # Load metrics
            metrics = load_model_metrics(subdir_path)

            if metrics:
                metrics['model'] = model_name
                metrics['dataset'] = dataset
                metrics['log_dir'] = subdir_path
                all_metrics.append(metrics)

    if not all_metrics:
        print(f"No metrics found in {base_log_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)

    # Sort by model name
    df = df.sort_values('model').reset_index(drop=True)

    return df


def create_comparison_table(df, metrics_to_compare=None):
    """
    Create comparison table for key metrics

    Args:
        df: DataFrame with all model metrics
        metrics_to_compare: List of metrics to include

    Returns:
        Formatted comparison DataFrame
    """
    if metrics_to_compare is None:
        # Default key metrics
        metrics_to_compare = [
            'rmse', 'mae', 'ic', 'ric', 'nll', 'crps',
            'dir_acc', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio',
            'info_ratio', 'hit_rate', 'tail_ratio',
            'aurc', 'picp90', 'picp95'
        ]

    # Filter to available metrics
    available_metrics = [m for m in metrics_to_compare if m in df.columns]

    # Create comparison table
    comparison = df[['model'] + available_metrics].copy()

    # Round numeric columns
    for col in available_metrics:
        if col in comparison.columns:
            comparison[col] = comparison[col].round(6)

    return comparison


def rank_models(df, metrics_config=None):
    """
    Rank models based on multiple metrics

    Args:
        df: DataFrame with model metrics
        metrics_config: Dict specifying metrics and whether higher is better
                       Format: {'metric_name': higher_is_better (bool)}

    Returns:
        DataFrame with rankings
    """
    if metrics_config is None:
        # Default configuration
        metrics_config = {
            'rmse': False,      # Lower is better
            'mae': False,
            'nll': False,
            'ic': True,         # Higher is better
            'ric': True,
            'dir_acc': True,
            'sharpe_ratio': True,
            'calmar_ratio': True,
            'info_ratio': True,
            'hit_rate': True
        }

    rankings = df[['model']].copy()

    # Calculate rank for each metric
    for metric, higher_is_better in metrics_config.items():
        if metric in df.columns:
            if higher_is_better:
                rankings[f'{metric}_rank'] = df[metric].rank(ascending=False)
            else:
                rankings[f'{metric}_rank'] = df[metric].rank(ascending=True)

    # Calculate average rank
    rank_columns = [col for col in rankings.columns if col.endswith('_rank')]
    rankings['avg_rank'] = rankings[rank_columns].mean(axis=1)

    # Sort by average rank
    rankings = rankings.sort_values('avg_rank').reset_index(drop=True)

    return rankings


def create_visualizations(df, output_dir):
    """
    Create comparison visualizations

    Args:
        df: DataFrame with model metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # 1. Probabilistic metrics comparison
    prob_metrics = ['rmse', 'mae', 'ic', 'ric', 'nll', 'crps']
    available_prob = [m for m in prob_metrics if m in df.columns]

    if available_prob:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(available_prob[:6]):
            if i < len(axes):
                ax = axes[i]
                df_sorted = df.sort_values(metric, ascending=(metric in ['rmse', 'mae', 'nll', 'crps']))
                ax.barh(df_sorted['model'], df_sorted[metric])
                ax.set_xlabel(metric.upper())
                ax.set_title(f'{metric.upper()} Comparison')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'probabilistic_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Financial metrics comparison
    fin_metrics = ['dir_acc', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'hit_rate', 'info_ratio']
    available_fin = [m for m in fin_metrics if m in df.columns]

    if available_fin:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(available_fin[:6]):
            if i < len(axes):
                ax = axes[i]
                df_sorted = df.sort_values(metric, ascending=(metric == 'max_drawdown'))
                ax.barh(df_sorted['model'], df_sorted[metric])
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'financial_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Radar chart for normalized metrics
    key_metrics = ['ic', 'ric', 'dir_acc', 'sharpe_ratio', 'hit_rate']
    available_radar = [m for m in key_metrics if m in df.columns]

    if available_radar and len(df) > 0:
        # Normalize metrics to 0-1 scale
        df_norm = df.copy()
        for metric in available_radar:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[metric] = 0.5

        # Create radar chart
        from math import pi

        categories = [m.upper() for m in available_radar]
        N = len(categories)

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for idx, row in df_norm.iterrows():
            values = [row[m] for m in available_radar]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Model Comparison - Normalized Metrics', size=16, y=1.08)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to: {output_dir}")


def generate_report(df, rankings, output_path):
    """
    Generate comprehensive comparison report

    Args:
        df: DataFrame with model metrics
        rankings: DataFrame with model rankings
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE MODELS COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        # Model rankings
        f.write("MODEL RANKINGS (by average rank across key metrics)\n")
        f.write("-"*80 + "\n")
        for idx, row in rankings.iterrows():
            f.write(f"{idx+1}. {row['model']:15s} - Avg Rank: {row['avg_rank']:.2f}\n")
        f.write("\n")

        # Best performers by metric category
        f.write("\nBEST PERFORMERS BY METRIC CATEGORY\n")
        f.write("-"*80 + "\n")

        # Probabilistic metrics
        prob_metrics = {
            'RMSE': ('rmse', False),
            'MAE': ('mae', False),
            'IC': ('ic', True),
            'RIC': ('ric', True),
            'NLL': ('nll', False),
            'CRPS': ('crps', False)
        }

        f.write("\nProbabilistic Metrics:\n")
        for metric_name, (metric_col, higher_is_better) in prob_metrics.items():
            if metric_col in df.columns:
                if higher_is_better:
                    best_idx = df[metric_col].idxmax()
                else:
                    best_idx = df[metric_col].idxmin()
                best_model = df.loc[best_idx, 'model']
                best_value = df.loc[best_idx, metric_col]
                f.write(f"  {metric_name:10s}: {best_model:15s} ({best_value:.6f})\n")

        # Financial metrics
        fin_metrics = {
            'Dir Acc': ('dir_acc', True),
            'Sharpe': ('sharpe_ratio', True),
            'Max DD': ('max_drawdown', False),
            'Calmar': ('calmar_ratio', True),
            'Hit Rate': ('hit_rate', True),
            'Info Ratio': ('info_ratio', True)
        }

        f.write("\nFinancial Metrics:\n")
        for metric_name, (metric_col, higher_is_better) in fin_metrics.items():
            if metric_col in df.columns:
                if higher_is_better:
                    best_idx = df[metric_col].idxmax()
                else:
                    best_idx = df[metric_col].idxmin()
                best_model = df.loc[best_idx, 'model']
                best_value = df.loc[best_idx, metric_col]
                f.write(f"  {metric_name:12s}: {best_model:15s} ({best_value:.6f})\n")

        # Full comparison table
        f.write("\n\nDETAILED METRICS COMPARISON\n")
        f.write("-"*80 + "\n")

        comparison = create_comparison_table(df)
        f.write(comparison.to_string(index=False))
        f.write("\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline model results')

    parser.add_argument('--base_log_dir', type=str, default='logs/baselines',
                       help='Base directory containing model logs')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Filter by dataset name (optional)')
    parser.add_argument('--output_dir', type=str, default='logs/baselines/comparison',
                       help='Output directory for comparison results')

    args = parser.parse_args()

    print("="*80)
    print("BASELINE MODELS COMPARISON")
    print("="*80)
    print(f"Log directory: {args.base_log_dir}")
    if args.dataset:
        print(f"Dataset filter: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Load all model metrics
    print("\nLoading model metrics...")
    df = load_all_models(args.base_log_dir, args.dataset)

    if df.empty:
        print("No metrics found. Please train models first.")
        return

    print(f"Found {len(df)} models: {', '.join(df['model'].tolist())}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create comparison table
    print("\nCreating comparison table...")
    comparison = create_comparison_table(df)
    comparison_path = os.path.join(args.output_dir, 'comparison_table.csv')
    comparison.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}")

    # Rank models
    print("\nRanking models...")
    rankings = rank_models(df)
    rankings_path = os.path.join(args.output_dir, 'model_rankings.csv')
    rankings.to_csv(rankings_path, index=False)
    print(f"Rankings saved to: {rankings_path}")

    # Create visualizations
    print("\nGenerating visualizations...")
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    create_visualizations(df, viz_dir)

    # Generate report
    print("\nGenerating comparison report...")
    report_path = os.path.join(args.output_dir, 'comparison_report.txt')
    generate_report(df, rankings, report_path)

    # Print summary
    print("\n" + "="*80)
    print("TOP 3 MODELS (by average rank):")
    print("="*80)
    for idx, row in rankings.head(3).iterrows():
        print(f"{idx+1}. {row['model']:15s} - Avg Rank: {row['avg_rank']:.2f}")

    print("\n" + "="*80)
    print("COMPARISON COMPLETED!")
    print(f"All results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
