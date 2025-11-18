"""
Result Plotting Utilities with Bayesian vs Non-Bayesian Comparison

This module provides comprehensive plotting functions for:
    1. Learning curves (loss, IC, CRPS)
    2. Calibration plots (PICP, reliability diagrams)
    3. Risk-coverage curves
    4. Bayesian vs Non-Bayesian comparison
    5. Market regime analysis plots
    6. Financial performance plots

Usage:
    from utils.result_plot import plot_bayesian_vs_nonbayesian_comparison

    plot_bayesian_vs_nonbayesian_comparison(results, output_dir='logs/comparison')
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from scipy.special import erfinv

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# Original Plotting Functions
# ============================================================================

def load_csv(path):
    """Load CSV file"""
    return pd.read_csv(path)


def load_preds(path):
    """Load predictions CSV"""
    return pd.read_csv(path)


def plot_learning_curves(val_df, out_dir):
    """Plot learning curves (NLL, CRPS, RMSE, MAE, Coverage)"""
    # NLL & CRPS (twin y-axis)
    fig, ax1 = plt.subplots()
    ax1.plot(val_df['epoch'], val_df['nll'], label='Val NLL', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('NLL', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(val_df['epoch'], val_df['crps'], '--', label='Val CRPS', color='red')
    ax2.set_ylabel('CRPS', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.suptitle('Learning Curves (Probabilistic)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'curve_nll_crps.png'), dpi=150)
    plt.close(fig)

    # RMSE / MAE
    plt.figure()
    plt.plot(val_df['epoch'], val_df['rmse'], label='Val RMSE')
    plt.plot(val_df['epoch'], val_df['mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Learning Curves (Point Estimates)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curve_rmse_mae.png'), dpi=150)
    plt.close()

    # Coverage gap (90/95)
    plt.figure()
    plt.plot(val_df['epoch'], val_df['gap90'], label='Gap@90%')
    plt.plot(val_df['epoch'], val_df['gap95'], label='Gap@95%')
    plt.xlabel('Epoch')
    plt.ylabel('|Observed - Nominal|')
    plt.legend()
    plt.title('Coverage Gap (Calibration)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curve_coverage_gap.png'), dpi=150)
    plt.close()


def plot_picp_from_preds(test_preds_csv, out_dir):
    """Plot Prediction Interval Coverage Probability"""
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)

    qs = [0.5, 0.8, 0.9, 0.95]
    nom, obs = [], []

    for q in qs:
        p = (1.0 + q) / 2.0
        z = np.sqrt(2.0) * erfinv(2.0 * p - 1.0)
        lo, hi = mu - z * sigma, mu + z * sigma
        oc = np.mean((y >= lo) & (y <= hi))
        nom.append(q)
        obs.append(oc)

    plt.figure()
    plt.plot(nom, nom, 'k--', label='Ideal', linewidth=2)
    plt.plot(nom, obs, 'o-', label='Observed', linewidth=2, markersize=8)
    plt.xlabel('Nominal Coverage')
    plt.ylabel('Observed Coverage')
    plt.legend()
    plt.title('Calibration: PICP (Test)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'calib_picp.png'), dpi=150)
    plt.close()


def plot_risk_coverage(test_preds_csv, out_dir):
    """Plot Risk-Coverage curve"""
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)

    idx = np.argsort(sigma)  # most certain first
    y, mu = y[idx], mu[idx]

    covs = np.linspace(0.1, 1.0, 10)
    rmses = []

    for c in covs:
        k = max(1, int(c * len(y)))
        rmses.append(np.sqrt(np.mean((y[:k] - mu[:k]) ** 2)))

    # AURC (area under risk-coverage curve)
    aurc = 0.0
    for i in range(1, len(covs)):
        h = covs[i] - covs[i - 1]
        aurc += 0.5 * h * (rmses[i] + rmses[i - 1])

    plt.figure()
    plt.plot(covs, rmses, 'o-', linewidth=2)
    plt.xlabel('Coverage')
    plt.ylabel('RMSE')
    plt.title(f'Risk-Coverage (Test) | AURC={aurc:.5f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'risk_coverage.png'), dpi=150)
    plt.close()


def plot_ic_by_sigma_decile(test_preds_csv, out_dir):
    """Plot IC by uncertainty decile"""
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)

    dec = pd.qcut(sigma, 10, labels=False, duplicates='drop')
    ic_vals = []

    for g in np.unique(dec):
        m = (dec == g)
        yy, mm = y[m], mu[m]

        if len(yy) > 2:
            vx, vy = yy - yy.mean(), mm - mm.mean()
            ic = (vx * vy).sum() / (np.sqrt((vx ** 2).sum()) * np.sqrt((vy ** 2).sum()) + 1e-12)
            ic_vals.append(ic)
        else:
            ic_vals.append(np.nan)

    plt.figure()
    plt.plot(range(1, len(ic_vals) + 1), ic_vals, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sigma Decile (1=lowest uncertainty)')
    plt.ylabel('IC')
    plt.title('IC by Uncertainty Decile (Test)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ic_by_sigma_decile.png'), dpi=150)
    plt.close()


def plot_analytics(log_dir):
    """Original analytics plotting"""
    os.makedirs(log_dir, exist_ok=True)

    val_csv = os.path.join(log_dir, 'val_metrics.csv')
    if os.path.exists(val_csv):
        val_df = load_csv(val_csv)
        plot_learning_curves(val_df, log_dir)

    test_preds_csv = os.path.join(log_dir, 'test_predictions.csv')
    if os.path.exists(test_preds_csv):
        plot_picp_from_preds(test_preds_csv, log_dir)
        plot_risk_coverage(test_preds_csv, log_dir)
        plot_ic_by_sigma_decile(test_preds_csv, log_dir)

    test_csv = os.path.join(log_dir, 'test_metrics.csv')
    if os.path.exists(test_csv):
        df_test = load_csv(test_csv)
        print('TEST summary:\n', df_test.tail(1).to_string(index=False))


# ============================================================================
# Bayesian vs Non-Bayesian Comparison Plots
# ============================================================================

def plot_bayesian_vs_nonbayesian_comparison(
    all_results: Dict,
    output_dir: str = 'logs/comparison'
):
    """
    Comprehensive comparison plots for Bayesian vs Non-Bayesian models

    Args:
        all_results: {dataset: {model_name: metrics, regime_analysis: {...}, financial_metrics: {...}}}
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Performance Metrics Comparison (Bar plot)
    plot_performance_comparison(all_results, output_dir)

    # 2. Regime Analysis (Grouped bar plot)
    plot_regime_analysis(all_results, output_dir)

    # 3. Financial Metrics (Radar plot)
    plot_financial_metrics_radar(all_results, output_dir)

    # 4. Uncertainty Quality (Bayesian only)
    plot_uncertainty_quality(all_results, output_dir)

    # 5. IC Time Series Comparison
    plot_ic_time_series(all_results, output_dir)

    print(f"✓ Comparison plots saved to: {output_dir}")


def plot_performance_comparison(all_results: Dict, output_dir: str):
    """Bar plot comparing IC, RIC, DA, Sharpe"""
    metrics = ['ic', 'ric', 'directional_accuracy', 'sharpe']
    metric_labels = ['IC', 'RIC', 'Directional Accuracy', 'Sharpe Ratio']

    datasets = list(all_results.keys())
    n_metrics = len(metrics)
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 5))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        bayes_vals = []
        nonbayes_vals = []

        for dataset in datasets:
            results = all_results[dataset]

            # Find bayesian and non-bayesian results
            bayes_key = [k for k in results.keys() if 'Bayesian' in k and 'Non' not in k]
            nonbayes_key = [k for k in results.keys() if 'NonBayesian' in k]

            if bayes_key and metric in results[bayes_key[0]]:
                bayes_vals.append(results[bayes_key[0]][metric])
            else:
                bayes_vals.append(0)

            if nonbayes_key and metric in results[nonbayes_key[0]]:
                nonbayes_vals.append(results[nonbayes_key[0]][metric])
            else:
                nonbayes_vals.append(0)

        x = np.arange(n_datasets)
        width = 0.35

        ax.bar(x - width/2, nonbayes_vals, width, label='Non-Bayesian', alpha=0.8)
        ax.bar(x + width/2, bayes_vals, width, label='Bayesian', alpha=0.8)

        ax.set_xlabel('Dataset')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=150)
    plt.close()


def plot_regime_analysis(all_results: Dict, output_dir: str):
    """Grouped bar plot for performance in different market regimes"""
    datasets = list(all_results.keys())

    # Check if regime analysis exists
    has_regime = any('regime_analysis' in all_results[ds] for ds in datasets)
    if not has_regime:
        print("Skipping regime analysis plot (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    regimes = ['Stable', 'Volatile']

    for regime_idx, regime in enumerate(regimes):
        ax = axes[regime_idx]

        ic_bayes = []
        ic_nonbayes = []
        dataset_labels = []

        for dataset in datasets:
            if 'regime_analysis' not in all_results[dataset]:
                continue

            regime_data = all_results[dataset]['regime_analysis']

            if regime not in regime_data:
                continue

            if 'Bayesian' in regime_data[regime]:
                ic_bayes.append(regime_data[regime]['Bayesian']['ic'])
            else:
                ic_bayes.append(0)

            if 'NonBayesian' in regime_data[regime]:
                ic_nonbayes.append(regime_data[regime]['NonBayesian']['ic'])
            else:
                ic_nonbayes.append(0)

            dataset_labels.append(dataset)

        if not dataset_labels:
            continue

        x = np.arange(len(dataset_labels))
        width = 0.35

        ax.bar(x - width/2, ic_nonbayes, width, label='Non-Bayesian', alpha=0.8)
        ax.bar(x + width/2, ic_bayes, width, label='Bayesian', alpha=0.8)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('IC')
        ax.set_title(f'{regime} Market Regime')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regime_analysis.png'), dpi=150)
    plt.close()


def plot_financial_metrics_radar(all_results: Dict, output_dir: str):
    """Radar plot for financial metrics"""
    datasets = list(all_results.keys())

    if 'financial_metrics' not in all_results[datasets[0]]:
        print("Skipping financial radar plot (no data)")
        return

    # Average across datasets
    metrics = ['sharpe_ratio', 'hit_rate', 'total_return']
    metric_labels = ['Sharpe Ratio', 'Hit Rate', 'Total Return']

    bayes_avg = {m: [] for m in metrics}
    nonbayes_avg = {m: [] for m in metrics}

    for dataset in datasets:
        if 'financial_metrics' not in all_results[dataset]:
            continue

        fin_metrics = all_results[dataset]['financial_metrics']

        # Find bayesian and non-bayesian
        bayes_key = [k for k in fin_metrics.keys() if 'Bayesian' in k and 'Non' not in k]
        nonbayes_key = [k for k in fin_metrics.keys() if 'NonBayesian' in k]

        if bayes_key:
            for m in metrics:
                if m in fin_metrics[bayes_key[0]]:
                    bayes_avg[m].append(fin_metrics[bayes_key[0]][m])

        if nonbayes_key:
            for m in metrics:
                if m in fin_metrics[nonbayes_key[0]]:
                    nonbayes_avg[m].append(fin_metrics[nonbayes_key[0]][m])

    # Compute averages
    bayes_values = [np.mean(bayes_avg[m]) if bayes_avg[m] else 0 for m in metrics]
    nonbayes_values = [np.mean(nonbayes_avg[m]) if nonbayes_avg[m] else 0 for m in metrics]

    # Normalize to [0, 1]
    all_values = bayes_values + nonbayes_values
    max_val = max(all_values) if max(all_values) > 0 else 1
    bayes_norm = [v / max_val for v in bayes_values]
    nonbayes_norm = [v / max_val for v in nonbayes_values]

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    bayes_norm += bayes_norm[:1]
    nonbayes_norm += nonbayes_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, nonbayes_norm, 'o-', linewidth=2, label='Non-Bayesian')
    ax.fill(angles, nonbayes_norm, alpha=0.25)

    ax.plot(angles, bayes_norm, 'o-', linewidth=2, label='Bayesian')
    ax.fill(angles, bayes_norm, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.set_title('Financial Metrics Comparison (Normalized)', pad=20)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'financial_radar.png'), dpi=150)
    plt.close()


def plot_uncertainty_quality(all_results: Dict, output_dir: str):
    """Plot uncertainty quality for Bayesian models"""
    # Load predictions for Bayesian models
    datasets = list(all_results.keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))

    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Find bayesian predictions file
        pred_file = f'logs/bayesian_vs_nonbayesian/{dataset}/{dataset}_Bayesian_predictions.csv'

        if not os.path.exists(pred_file):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{dataset} (No data)')
            continue

        df = pd.read_csv(pred_file)
        y = df['y'].values
        mu = df['mu'].values
        sigma = np.maximum(df['sigma'].values, 1e-12)

        # Plot prediction intervals
        sorted_idx = np.argsort(mu)
        y_sorted = y[sorted_idx]
        mu_sorted = mu[sorted_idx]
        sigma_sorted = sigma[sorted_idx]

        # Sample for visualization
        n_points = min(200, len(mu_sorted))
        step = len(mu_sorted) // n_points
        indices = np.arange(0, len(mu_sorted), step)[:n_points]

        ax.fill_between(
            indices,
            mu_sorted[indices] - 1.96 * sigma_sorted[indices],
            mu_sorted[indices] + 1.96 * sigma_sorted[indices],
            alpha=0.3,
            label='95% CI'
        )
        ax.plot(indices, mu_sorted[indices], 'b-', label='Prediction', linewidth=1)
        ax.scatter(indices, y_sorted[indices], c='red', s=10, alpha=0.5, label='Actual')

        ax.set_xlabel('Sample (sorted by prediction)')
        ax.set_ylabel('Value')
        ax.set_title(f'{dataset} Uncertainty Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_quality.png'), dpi=150)
    plt.close()


def plot_ic_time_series(all_results: Dict, output_dir: str):
    """Plot IC over time for both models"""
    datasets = list(all_results.keys())

    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 5 * len(datasets)))

    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Load predictions
        bayes_pred = f'logs/bayesian_vs_nonbayesian/{dataset}/{dataset}_Bayesian_predictions.csv'
        nonbayes_pred = f'logs/bayesian_vs_nonbayesian/{dataset}/{dataset}_NonBayesian_predictions.csv'

        if not (os.path.exists(bayes_pred) and os.path.exists(nonbayes_pred)):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{dataset} (No data)')
            continue

        df_bayes = pd.read_csv(bayes_pred)
        df_nonbayes = pd.read_csv(nonbayes_pred)

        # Compute rolling IC
        window = 50
        ic_bayes = []
        ic_nonbayes = []

        for i in range(window, len(df_bayes)):
            y_window = df_bayes['y'].values[i - window:i]
            mu_bayes_window = df_bayes['mu'].values[i - window:i]
            mu_nonbayes_window = df_nonbayes['mu'].values[i - window:i]

            # IC
            ic_b = np.corrcoef(y_window, mu_bayes_window)[0, 1]
            ic_nb = np.corrcoef(y_window, mu_nonbayes_window)[0, 1]

            ic_bayes.append(ic_b if not np.isnan(ic_b) else 0)
            ic_nonbayes.append(ic_nb if not np.isnan(ic_nb) else 0)

        ax.plot(ic_nonbayes, label='Non-Bayesian', alpha=0.7)
        ax.plot(ic_bayes, label='Bayesian', alpha=0.7)

        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Rolling IC (window={window})')
        ax.set_title(f'{dataset} IC Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ic_time_series.png'), dpi=150)
    plt.close()


# ============================================================================
# Summary Report
# ============================================================================

def generate_comparison_report(all_results: Dict, output_file: str):
    """Generate text summary report"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BAYESIAN VS NON-BAYESIAN MAMBA-BGNN COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        for dataset, results in all_results.items():
            f.write(f"\n{'-'*60}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"{'-'*60}\n\n")

            # Find models
            bayes_key = [k for k in results.keys() if 'Bayesian' in k and 'Non' not in k]
            nonbayes_key = [k for k in results.keys() if 'NonBayesian' in k]

            if bayes_key and nonbayes_key:
                bayes_metrics = results[bayes_key[0]]
                nonbayes_metrics = results[nonbayes_key[0]]

                f.write("Performance Metrics:\n")
                f.write(f"{'Metric':<25} {'Non-Bayesian':>15} {'Bayesian':>15} {'Improvement':>12}\n")
                f.write(f"{'-'*70}\n")

                metrics_to_show = ['ic', 'ric', 'rmse', 'directional_accuracy', 'sharpe', 'max_drawdown']

                for metric in metrics_to_show:
                    if metric in bayes_metrics and metric in nonbayes_metrics:
                        nb_val = nonbayes_metrics[metric]
                        b_val = bayes_metrics[metric]

                        if abs(nb_val) > 1e-8:
                            improvement = (b_val - nb_val) / abs(nb_val) * 100
                        else:
                            improvement = 0.0

                        f.write(f"{metric:<25} {nb_val:>15.6f} {b_val:>15.6f} {improvement:>11.2f}%\n")

            # Regime analysis
            if 'regime_analysis' in results:
                f.write("\n\nMarket Regime Analysis:\n")

                for regime, regime_data in results['regime_analysis'].items():
                    f.write(f"\n  {regime} Market:\n")
                    f.write(f"    {'Model':<15} {'IC':>10} {'RIC':>10} {'RMSE':>10} {'DA':>10}\n")
                    f.write(f"    {'-'*55}\n")

                    if 'Bayesian' in regime_data:
                        b = regime_data['Bayesian']
                        f.write(f"    {'Bayesian':<15} {b['ic']:>10.4f} {b['ric']:>10.4f} "
                               f"{b['rmse']:>10.4f} {b['da']:>10.4f}\n")

                    if 'NonBayesian' in regime_data:
                        nb = regime_data['NonBayesian']
                        f.write(f"    {'Non-Bayesian':<15} {nb['ic']:>10.4f} {nb['ric']:>10.4f} "
                               f"{nb['rmse']:>10.4f} {nb['da']:>10.4f}\n")

                    f.write(f"    Samples: {regime_data['n_samples']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"✓ Comparison report saved to: {output_file}")
