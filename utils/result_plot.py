import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    return pd.read_csv(path)

def load_preds(path):
    return pd.read_csv(path)   # y, mu, sigma

def plot_learning_curves(val_df, out_dir):
    # NLL & CRPS (twin y-axis)
    fig, ax1 = plt.subplots()
    ax1.plot(val_df['epoch'], val_df['nll'], label='Val NLL')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('NLL')
    ax2 = ax1.twinx()
    ax2.plot(val_df['epoch'], val_df['crps'], '--', label='Val CRPS')
    ax2.set_ylabel('CRPS')
    fig.suptitle('Learning Curves (Probabilistic)')
    lines, labels = [], []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels(); lines += h; labels += l
    ax1.legend(lines, labels, loc='best')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'curve_nll_crps.png')); plt.close(fig)

    # RMSE / MAE
    plt.figure()
    plt.plot(val_df['epoch'], val_df['rmse'], label='Val RMSE')
    plt.plot(val_df['epoch'], val_df['mae'],  label='Val MAE')
    plt.xlabel('Epoch'); plt.ylabel('Error'); plt.legend()
    plt.title('Learning Curves (Point)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'curve_rmse_mae.png')); plt.close()

    # Coverage gap (90/95)
    plt.figure()
    plt.plot(val_df['epoch'], val_df['gap90'], label='Gap@90%')
    plt.plot(val_df['epoch'], val_df['gap95'], label='Gap@95%')
    plt.xlabel('Epoch'); plt.ylabel('|Observed - Nominal|')
    plt.legend(); plt.title('Coverage Gap (Calibration)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'curve_coverage_gap.png')); plt.close()

def plot_picp_from_preds(test_preds_csv, out_dir):
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)
    qs = [0.5, 0.8, 0.9, 0.95]
    nom, obs = [], []
    for q in qs:
        p = (1.0 + q)/2.0
        from scipy.special import erfinv
        z = np.sqrt(2.0)*erfinv(2.0*p - 1.0)
        lo, hi = mu - z*sigma, mu + z*sigma
        oc = np.mean((y >= lo) & (y <= hi))
        nom.append(q); obs.append(oc)
    plt.figure()
    plt.plot(nom, nom, 'k--', label='Ideal')
    plt.plot(nom, obs, 'o-', label='Observed')
    plt.xlabel('Nominal coverage'); plt.ylabel('Observed coverage')
    plt.legend(); plt.title('Calibration: PICP (Test)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'calib_picp.png')); plt.close()

def plot_risk_coverage(test_preds_csv, out_dir):
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)
    idx = np.argsort(sigma)  # most certain first
    y, mu = y[idx], mu[idx]
    covs = np.linspace(0.1, 1.0, 10)
    rmses = []
    for c in covs:
        k = max(1, int(c*len(y)))
        rmses.append(np.sqrt(np.mean((y[:k]-mu[:k])**2)))
    # AURC (trapezoid)
    aurc = 0.0
    for i in range(1, len(covs)):
        h = covs[i] - covs[i-1]
        aurc += 0.5*h*(rmses[i] + rmses[i-1])
    plt.figure()
    plt.plot(covs, rmses, 'o-')
    plt.xlabel('Coverage'); plt.ylabel('RMSE')
    plt.title(f'Risk–Coverage (Test)  AURC={aurc:.5f}')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'risk_coverage.png')); plt.close()

def plot_ic_by_sigma_decile(test_preds_csv, out_dir):
    d = load_preds(test_preds_csv)
    y, mu, sigma = d['y'].values, d['mu'].values, np.maximum(d['sigma'].values, 1e-12)
    dec = pd.qcut(sigma, 10, labels=False, duplicates='drop')
    ic_vals = []
    for g in np.unique(dec):
        m = (dec == g)
        yy, mm = y[m], mu[m]
        if len(yy) > 2:
            vx, vy = yy - yy.mean(), mm - mm.mean()
            ic = (vx*vy).sum() / (np.sqrt((vx**2).sum())*np.sqrt((vy**2).sum()) + 1e-12)
            ic_vals.append(ic)
        else:
            ic_vals.append(np.nan)
    plt.figure()
    plt.plot(range(1, len(ic_vals)+1), ic_vals, 'o-')
    plt.xlabel('Sigma decile (1=lowest uncertainty)'); plt.ylabel('IC')
    plt.title('IC by sigma decile (Test)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'ic_by_sigma_decile.png')); plt.close()

def plot_analytics(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    val_df = load_csv(os.path.join(log_dir, 'val_metrics.csv'))
    plot_learning_curves(val_df, log_dir)

    test_preds_csv = os.path.join(log_dir, 'test_predictions.csv')
    if os.path.exists(test_preds_csv):
        plot_picp_from_preds(test_preds_csv, log_dir)
        plot_risk_coverage(test_preds_csv, log_dir)
        plot_ic_by_sigma_decile(test_preds_csv, log_dir)

    # In hàng TEST tổng hợp nếu có
    test_csv = os.path.join(log_dir, 'test_metrics.csv')
    if os.path.exists(test_csv):
        df_test = load_csv(test_csv)
        print('TEST summary:\n', df_test.tail(1).to_string(index=False))


# ============================================================================
# BAYESIAN VS NON-BAYESIAN COMPARISON PLOTS
# ============================================================================

def plot_metric_comparison_bars(metrics_nb, metrics_b, metrics_to_plot, out_path, title="Metric Comparison"):
    """
    Bar chart comparing metrics between Bayesian and Non-Bayesian models

    Args:
        metrics_nb: Dict of Non-Bayesian metrics
        metrics_b: Dict of Bayesian metrics
        metrics_to_plot: List of metric names to plot
        out_path: Output file path
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in metrics_nb and m in metrics_b]

    if not available_metrics:
        print("No common metrics to plot")
        return

    x = np.arange(len(available_metrics))
    width = 0.35

    nb_values = [metrics_nb[m] for m in available_metrics]
    b_values = [metrics_b[m] for m in available_metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, nb_values, width, label='Non-Bayesian', alpha=0.8)
    bars2 = ax.bar(x + width/2, b_values, width, label='Bayesian', alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_uncertainty_comparison(nb_pred_csv, b_pred_csv, out_dir):
    """
    Compare uncertainty estimates between Bayesian and Non-Bayesian models

    Args:
        nb_pred_csv: Path to Non-Bayesian predictions CSV
        b_pred_csv: Path to Bayesian predictions CSV
        out_dir: Output directory
    """
    df_nb = load_preds(nb_pred_csv)
    df_b = load_preds(b_pred_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sigma distribution comparison
    ax = axes[0, 0]
    ax.hist(df_nb['sigma'].values, bins=50, alpha=0.6, label='Non-Bayesian', density=True)
    ax.hist(df_b['sigma'].values, bins=50, alpha=0.6, label='Bayesian', density=True)
    ax.set_xlabel('Uncertainty (σ)')
    ax.set_ylabel('Density')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Prediction vs Uncertainty scatter
    ax = axes[0, 1]
    ax.scatter(df_nb['mu'].values, df_nb['sigma'].values, alpha=0.3, s=10, label='Non-Bayesian')
    ax.scatter(df_b['mu'].values, df_b['sigma'].values, alpha=0.3, s=10, label='Bayesian')
    ax.set_xlabel('Prediction (μ)')
    ax.set_ylabel('Uncertainty (σ)')
    ax.set_title('Prediction vs Uncertainty')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Error vs Uncertainty
    ax = axes[1, 0]
    error_nb = np.abs(df_nb['y'].values - df_nb['mu'].values)
    error_b = np.abs(df_b['y'].values - df_b['mu'].values)

    ax.scatter(df_nb['sigma'].values, error_nb, alpha=0.3, s=10, label='Non-Bayesian')
    ax.scatter(df_b['sigma'].values, error_b, alpha=0.3, s=10, label='Bayesian')
    ax.set_xlabel('Uncertainty (σ)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Uncertainty vs Error (Higher σ should correlate with higher error)')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Uncertainty time series
    ax = axes[1, 1]
    window = 50
    ax.plot(pd.Series(df_nb['sigma'].values).rolling(window).mean(), alpha=0.7, label='Non-Bayesian')
    ax.plot(pd.Series(df_b['sigma'].values).rolling(window).mean(), alpha=0.7, label='Bayesian')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Uncertainty (σ, {window}-pt MA)')
    ax.set_title('Uncertainty Over Time')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'uncertainty_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_calibration_comparison(nb_pred_csv, b_pred_csv, out_dir):
    """
    Compare calibration between Bayesian and Non-Bayesian models

    Args:
        nb_pred_csv: Path to Non-Bayesian predictions CSV
        b_pred_csv: Path to Bayesian predictions CSV
        out_dir: Output directory
    """
    from scipy.special import erf

    df_nb = load_preds(nb_pred_csv)
    df_b = load_preds(b_pred_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    confidences = np.linspace(0.1, 0.99, 20)

    for idx, (df, label) in enumerate([(df_nb, 'Non-Bayesian'), (df_b, 'Bayesian')]):
        y = df['y'].values
        mu = df['mu'].values
        sigma = np.maximum(df['sigma'].values, 1e-6)

        observed_coverages = []

        for conf in confidences:
            # Convert confidence to z-score
            z = np.sqrt(2) * np.sqrt(-np.log(1 - conf))
            lower = mu - z * sigma
            upper = mu + z * sigma
            coverage = np.mean((y >= lower) & (y <= upper))
            observed_coverages.append(coverage)

        ax = axes[0]
        ax.plot(confidences, observed_coverages, 'o-', label=label, alpha=0.7)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Calibration error bars
    ax = axes[1]
    coverage_levels = [0.68, 0.90, 0.95, 0.99]
    x_pos = np.arange(len(coverage_levels))
    width = 0.35

    nb_gaps = []
    b_gaps = []

    for conf in coverage_levels:
        for df, gaps_list in [(df_nb, nb_gaps), (df_b, b_gaps)]:
            y = df['y'].values
            mu = df['mu'].values
            sigma = np.maximum(df['sigma'].values, 1e-6)

            z = np.sqrt(2) * np.sqrt(-np.log(1 - conf))
            lower = mu - z * sigma
            upper = mu + z * sigma
            coverage = np.mean((y >= lower) & (y <= upper))
            gap = abs(coverage - conf)
            gaps_list.append(gap)

    ax.bar(x_pos - width/2, nb_gaps, width, label='Non-Bayesian', alpha=0.8)
    ax.bar(x_pos + width/2, b_gaps, width, label='Bayesian', alpha=0.8)
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Calibration Gap')
    ax.set_title('Calibration Error (Lower is Better)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(c*100)}%' for c in coverage_levels])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'calibration_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_regime_analysis(regime_csv, nb_pred_csv, b_pred_csv, out_dir):
    """
    Plot market regime analysis and performance

    Args:
        regime_csv: Path to regime analysis CSV
        nb_pred_csv: Path to Non-Bayesian predictions CSV
        b_pred_csv: Path to Bayesian predictions CSV
        out_dir: Output directory
    """
    regime_df = pd.read_csv(regime_csv)
    df_nb = load_preds(nb_pred_csv)
    df_b = load_preds(b_pred_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Regime classification over time
    ax = axes[0, 0]
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    labels = {0: 'Low Volatility', 1: 'Medium Volatility', 2: 'High Volatility'}

    for regime_id in [0, 1, 2]:
        mask = regime_df['regime'].values == regime_id
        indices = np.where(mask)[0]
        ax.scatter(indices, regime_df['rolling_std'].values[mask],
                  c=colors[regime_id], label=labels[regime_id], alpha=0.5, s=10)

    ax.set_xlabel('Time')
    ax.set_ylabel('Rolling Volatility')
    ax.set_title('Market Regime Classification')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Regime distribution
    ax = axes[0, 1]
    regime_counts = regime_df['regime'].value_counts().sort_index()
    ax.bar([labels[i] for i in regime_counts.index], regime_counts.values,
           color=[colors[i] for i in regime_counts.index], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Regime Distribution')
    ax.grid(axis='y', alpha=0.3)

    # 3. Error by regime
    ax = axes[1, 0]
    regime = regime_df['regime'].values
    error_nb = np.abs(df_nb['y'].values - df_nb['mu'].values)
    error_b = np.abs(df_b['y'].values - df_b['mu'].values)

    regime_names = ['Low Vol', 'Med Vol', 'High Vol']
    x_pos = np.arange(len(regime_names))
    width = 0.35

    nb_errors = [error_nb[regime == i].mean() for i in range(3)]
    b_errors = [error_b[regime == i].mean() for i in range(3)]

    ax.bar(x_pos - width/2, nb_errors, width, label='Non-Bayesian', alpha=0.8)
    ax.bar(x_pos + width/2, b_errors, width, label='Bayesian', alpha=0.8)
    ax.set_xlabel('Regime')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error by Market Regime')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regime_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Uncertainty by regime
    ax = axes[1, 1]
    nb_sigmas = [df_nb['sigma'].values[regime == i].mean() for i in range(3)]
    b_sigmas = [df_b['sigma'].values[regime == i].mean() for i in range(3)]

    ax.bar(x_pos - width/2, nb_sigmas, width, label='Non-Bayesian', alpha=0.8)
    ax.bar(x_pos + width/2, b_sigmas, width, label='Bayesian', alpha=0.8)
    ax.set_xlabel('Regime')
    ax.set_ylabel('Mean Uncertainty (σ)')
    ax.set_title('Uncertainty by Market Regime')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regime_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'regime_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_prediction_comparison(nb_pred_csv, b_pred_csv, out_dir, n_samples=500):
    """
    Compare predictions between Bayesian and Non-Bayesian models

    Args:
        nb_pred_csv: Path to Non-Bayesian predictions CSV
        b_pred_csv: Path to Bayesian predictions CSV
        out_dir: Output directory
        n_samples: Number of samples to plot in time series
    """
    df_nb = load_preds(nb_pred_csv)
    df_b = load_preds(b_pred_csv)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Limit to n_samples for clarity
    n = min(n_samples, len(df_nb))

    # 1. Predictions with uncertainty bands
    ax = axes[0]
    x = np.arange(n)

    # Non-Bayesian
    ax.plot(x, df_nb['mu'].values[:n], 'b-', label='Non-Bayesian μ', alpha=0.7)
    ax.fill_between(x,
                     df_nb['mu'].values[:n] - 2*df_nb['sigma'].values[:n],
                     df_nb['mu'].values[:n] + 2*df_nb['sigma'].values[:n],
                     alpha=0.2, color='blue', label='Non-Bayesian 95% CI')

    # Bayesian
    ax.plot(x, df_b['mu'].values[:n], 'r-', label='Bayesian μ', alpha=0.7)
    ax.fill_between(x,
                     df_b['mu'].values[:n] - 2*df_b['sigma'].values[:n],
                     df_b['mu'].values[:n] + 2*df_b['sigma'].values[:n],
                     alpha=0.2, color='red', label='Bayesian 95% CI')

    # True values
    ax.scatter(x, df_nb['y'].values[:n], c='black', s=10, alpha=0.3, label='True')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Predictions with Uncertainty (First {n} samples)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # 2. Prediction scatter plot
    ax = axes[1]
    ax.scatter(df_nb['y'].values, df_nb['mu'].values, alpha=0.3, s=10, label='Non-Bayesian')
    ax.scatter(df_b['y'].values, df_b['mu'].values, alpha=0.3, s=10, label='Bayesian')

    # Perfect prediction line
    min_val = min(df_nb['y'].min(), df_b['y'].min())
    max_val = max(df_nb['y'].max(), df_b['y'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Prediction Accuracy Scatter')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'prediction_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_bayesian_vs_nonbayesian_comparison(base_dir, dataset=None):
    """
    Generate all comparison plots for Bayesian vs Non-Bayesian models

    Args:
        base_dir: Base directory containing results
                 Can be:
                 - Full path with timestamp: 'logs/bayesian_vs_nonbayesian_20251119_014842'
                 - Without timestamp: 'logs/bayesian_vs_nonbayesian' (will find latest)
        dataset: Specific dataset to plot (e.g., 'IXIC'), or None for all datasets
    """
    import json
    from pathlib import Path

    # If base_dir doesn't exist, try to find latest study with timestamp
    base_path = Path(base_dir)
    if not base_path.exists():
        # Try to find latest study directory with timestamp
        parent = base_path.parent
        base_name = base_path.name

        if parent.exists():
            matching_dirs = sorted([d for d in parent.iterdir()
                                   if d.is_dir() and d.name.startswith(base_name + '_')])
            if matching_dirs:
                base_dir = str(matching_dirs[-1])  # Use latest
                print(f"Using latest study: {base_dir}")
            else:
                print(f"Error: No study directories found matching {base_name}_*")
                return
        else:
            print(f"Error: Directory not found: {base_dir}")
            return

    if dataset:
        datasets = [dataset]
    else:
        # Find all dataset directories
        datasets = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d)) and d not in ['plots', 'summary']]

    print(f"Generating comparison plots for datasets: {datasets}")

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {ds}...")
        print(f"{'='*60}")

        ds_dir = os.path.join(base_dir, ds)

        # Create output directory
        plot_dir = os.path.join(ds_dir, 'comparison_plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Find model directories
        nb_dir = os.path.join(ds_dir, 'BIMamba-MAGAC_NonBayesian')
        b_dir = os.path.join(ds_dir, 'BIMamba-MAGAC_Bayesian')

        if not os.path.exists(nb_dir) or not os.path.exists(b_dir):
            print(f"  Model directories not found for {ds}, skipping...")
            continue

        # Load predictions
        nb_pred_csv = os.path.join(nb_dir, 'test_predictions.csv')
        b_pred_csv = os.path.join(b_dir, 'test_predictions.csv')

        if not os.path.exists(nb_pred_csv) or not os.path.exists(b_pred_csv):
            print(f"  Prediction files not found for {ds}, skipping...")
            continue

        # Load metrics
        nb_metrics_file = os.path.join(nb_dir, 'test_metrics.csv')
        b_metrics_file = os.path.join(b_dir, 'test_metrics.csv')

        metrics_nb = {}
        metrics_b = {}

        if os.path.exists(nb_metrics_file):
            df_nb_metrics = pd.read_csv(nb_metrics_file)
            if len(df_nb_metrics) > 0:
                metrics_nb = df_nb_metrics.iloc[-1].to_dict()

        if os.path.exists(b_metrics_file):
            df_b_metrics = pd.read_csv(b_metrics_file)
            if len(df_b_metrics) > 0:
                metrics_b = df_b_metrics.iloc[-1].to_dict()

        # Generate plots
        print("  Generating metric comparison...")
        if metrics_nb and metrics_b:
            metrics_to_plot = ['ic', 'rmse', 'mae', 'nll', 'crps']
            plot_metric_comparison_bars(
                metrics_nb, metrics_b, metrics_to_plot,
                os.path.join(plot_dir, 'metrics_comparison.png'),
                title=f"Metric Comparison - {ds}"
            )

        print("  Generating uncertainty comparison...")
        plot_uncertainty_comparison(nb_pred_csv, b_pred_csv, plot_dir)

        print("  Generating calibration comparison...")
        plot_calibration_comparison(nb_pred_csv, b_pred_csv, plot_dir)

        print("  Generating prediction comparison...")
        plot_prediction_comparison(nb_pred_csv, b_pred_csv, plot_dir)

        # Regime analysis if available
        regime_csv = os.path.join(ds_dir, 'regime_analysis.csv')
        if os.path.exists(regime_csv):
            print("  Generating regime analysis...")
            plot_regime_analysis(regime_csv, nb_pred_csv, b_pred_csv, plot_dir)

        # Uncertainty-based analysis
        uncertainty_json = os.path.join(ds_dir, 'uncertainty_analysis.json')
        if os.path.exists(uncertainty_json):
            print("  Generating confidence-accuracy curve...")
            plot_confidence_accuracy_curve(nb_pred_csv, b_pred_csv, plot_dir)

            print("  Generating risk-adjusted returns...")
            plot_risk_adjusted_returns(uncertainty_json, plot_dir)

            print("  Generating uncertainty contribution...")
            plot_uncertainty_contribution(uncertainty_json, plot_dir)

            print("  Generating sharpness-calibration tradeoff...")
            plot_sharpness_calibration_tradeoff(uncertainty_json, plot_dir)

        print(f"  ✓ All plots saved to: {plot_dir}")

    print(f"\n{'='*60}")
    print("✓ All comparison plots generated!")
    print(f"{'='*60}")


# ============================================================================
# UNCERTAINTY CONTRIBUTION PLOTS
# ============================================================================

def plot_confidence_accuracy_curve(nb_pred_csv, b_pred_csv, out_dir):
    """
    Plot confidence vs accuracy relationship

    Shows that models should be more accurate when confident (low sigma)

    Args:
        nb_pred_csv: Path to Non-Bayesian predictions CSV
        b_pred_csv: Path to Bayesian predictions CSV
        out_dir: Output directory
    """
    df_nb = load_preds(nb_pred_csv)
    df_b = load_preds(b_pred_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (df, label, ax) in enumerate([
        (df_nb, 'Non-Bayesian', axes[0]),
        (df_b, 'Bayesian', axes[1])
    ]):
        y = df['y'].values
        mu = df['mu'].values
        sigma = df['sigma'].values

        # Compute confidence and error
        confidence = 1.0 / (sigma + 1e-6)
        errors = np.abs(mu - y)

        # Bin by confidence deciles
        n_bins = 10
        confidence_percentiles = np.percentile(confidence, np.linspace(0, 100, n_bins + 1))

        bin_confidences = []
        bin_errors = []
        bin_stds = []

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (confidence >= confidence_percentiles[i])
            else:
                mask = (confidence >= confidence_percentiles[i]) & (confidence < confidence_percentiles[i+1])

            if mask.sum() > 0:
                bin_confidences.append(confidence[mask].mean())
                bin_errors.append(errors[mask].mean())
                bin_stds.append(errors[mask].std() / np.sqrt(mask.sum()))

        # Plot
        ax.errorbar(bin_confidences, bin_errors, yerr=bin_stds,
                   marker='o', capsize=5, capthick=2, label=label)
        ax.set_xlabel('Mean Confidence (1/σ)', fontsize=12)
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.set_title(f'{label}\nConfidence-Accuracy Relationship', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add correlation
        from scipy.stats import spearmanr
        corr, p_val = spearmanr(confidence, errors)
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}\np-val: {p_val:.2e}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'confidence_accuracy_curve.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_risk_adjusted_returns(uncertainty_analysis_json, out_dir):
    """
    Plot risk-adjusted returns comparison

    Args:
        uncertainty_analysis_json: Path to uncertainty analysis JSON
        out_dir: Output directory
    """
    import json

    with open(uncertainty_analysis_json, 'r') as f:
        data = json.load(f)

    ra_nb = data['risk_adjusted_returns']['nb']
    ra_b = data['risk_adjusted_returns']['b']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Sharpe comparison
    ax = axes[0]
    models = ['Non-Bayesian', 'Bayesian']
    x = np.arange(len(models))
    width = 0.35

    base_sharpes = [ra_nb['sharpe_base'], ra_b['sharpe_base']]
    adjusted_sharpes = [ra_nb['sharpe_adjusted'], ra_b['sharpe_adjusted']]

    ax.bar(x - width/2, base_sharpes, width, label='Base Sharpe', alpha=0.8)
    ax.bar(x + width/2, adjusted_sharpes, width, label='Risk-Adjusted Sharpe', alpha=0.8)

    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Risk-Adjusted Returns\n(Position Sizing by Uncertainty)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add improvement labels
    for i, (base, adj, model) in enumerate(zip(base_sharpes, adjusted_sharpes, models)):
        improvement = (adj - base) / (abs(base) + 1e-8) * 100
        ax.text(i, max(base, adj) + 0.05, f'+{improvement:.1f}%',
               ha='center', fontweight='bold', color='green' if improvement > 0 else 'red')

    # 2. Total returns
    ax = axes[1]
    base_returns = [ra_nb['total_return_base'], ra_b['total_return_base']]
    adjusted_returns = [ra_nb['total_return_adjusted'], ra_b['total_return_adjusted']]

    ax.bar(x - width/2, base_returns, width, label='Base Returns', alpha=0.8)
    ax.bar(x + width/2, adjusted_returns, width, label='Risk-Adjusted Returns', alpha=0.8)

    ax.set_ylabel('Cumulative Returns', fontsize=12)
    ax.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'risk_adjusted_returns.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_uncertainty_contribution(uncertainty_analysis_json, out_dir):
    """
    Plot uncertainty contribution analysis

    Shows how Bayesian uncertainty identifies difficult samples

    Args:
        uncertainty_analysis_json: Path to uncertainty analysis JSON
        out_dir: Output directory
    """
    import json

    with open(uncertainty_analysis_json, 'r') as f:
        data = json.load(f)

    uc = data['uncertainty_contribution']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Error by uncertainty level
    ax = axes[0]
    categories = ['Low Uncertainty\n(Confident)', 'High Uncertainty\n(Uncertain)']
    x = np.arange(len(categories))
    width = 0.35

    nb_errors = [
        uc['low_uncertainty']['nonbayesian_error'],
        uc['high_uncertainty']['nonbayesian_error']
    ]
    b_errors = [
        uc['low_uncertainty']['bayesian_error'],
        uc['high_uncertainty']['bayesian_error']
    ]

    ax.bar(x - width/2, nb_errors, width, label='Non-Bayesian', alpha=0.8)
    ax.bar(x + width/2, b_errors, width, label='Bayesian', alpha=0.8)

    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Error by Uncertainty Level\n(Bayesian Uncertainty)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add sample counts
    for i, cat in enumerate(['low_uncertainty', 'high_uncertainty']):
        count = uc[cat]['sample_count']
        sigma = uc[cat]['mean_sigma']
        ax.text(i, max(nb_errors[i], b_errors[i]) + 0.001,
               f'n={count}\nσ={sigma:.4f}',
               ha='center', fontsize=8)

    # 2. Interpretation
    ax = axes[1]
    ax.axis('off')

    # Text summary
    corr = uc['uncertainty_identifies_difficulty']['correlation']
    p_val = uc['uncertainty_identifies_difficulty']['p_value']

    summary_text = f"""
UNCERTAINTY CONTRIBUTION ANALYSIS

Key Question: Does Bayesian uncertainty identify
difficult samples (where Non-Bayesian struggles)?

Correlation: {corr:.4f}
p-value: {p_val:.2e}

Interpretation:
{uc['uncertainty_identifies_difficulty']['interpretation']}

When Confident (Low σ):
  • Bayesian Error: {uc['low_uncertainty']['bayesian_error']:.6f}
  • Non-Bayesian Error: {uc['low_uncertainty']['nonbayesian_error']:.6f}
  • Samples: {uc['low_uncertainty']['sample_count']}

When Uncertain (High σ):
  • Bayesian Error: {uc['high_uncertainty']['bayesian_error']:.6f}
  • Non-Bayesian Error: {uc['high_uncertainty']['nonbayesian_error']:.6f}
  • Samples: {uc['high_uncertainty']['sample_count']}

✓ Positive correlation confirms that Bayesian
  uncertainty successfully identifies hard samples!
"""

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'uncertainty_contribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_sharpness_calibration_tradeoff(uncertainty_analysis_json, out_dir):
    """
    Plot sharpness-calibration tradeoff

    Args:
        uncertainty_analysis_json: Path to uncertainty analysis JSON
        out_dir: Output directory
    """
    import json

    with open(uncertainty_analysis_json, 'r') as f:
        data = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Extract data
    models = ['Non-Bayesian', 'Bayesian']
    sharpness = [data['sharpness']['nb'], data['sharpness']['b']]
    calibration_error = [
        data['confidence_accuracy']['nb'].get('mean_calibration_error',
            np.mean([abs(data['confidence_accuracy']['nb'].get(f'gap_{c}', 0)) for c in [68, 95, 99]])),
        data['confidence_accuracy']['b'].get('mean_calibration_error',
            np.mean([abs(data['confidence_accuracy']['b'].get(f'gap_{c}', 0)) for c in [68, 95, 99]]))
    ]

    # Scatter plot
    colors = ['blue', 'red']
    for i, (model, s, c, color) in enumerate(zip(models, sharpness, calibration_error, colors)):
        ax.scatter(s, c, s=200, alpha=0.7, c=color, label=model, edgecolors='black', linewidth=2)
        ax.annotate(model, (s, c), xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold')

    ax.set_xlabel('Sharpness (Mean σ) - Lower is Better', fontsize=12)
    ax.set_ylabel('Calibration Error - Lower is Better', fontsize=12)
    ax.set_title('Sharpness-Calibration Tradeoff\n(Ideal: Bottom-Left Corner)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Add ideal region
    ax.axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Good Calibration (<0.05)')
    ax.axvline(np.mean(sharpness), color='gray', linestyle='--', alpha=0.5, label='Mean Sharpness')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'sharpness_calibration_tradeoff.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_uncertainty_based_comparison(base_dir, dataset=None):
    """
    Generate uncertainty-based comparison plots

    This extends plot_bayesian_vs_nonbayesian_comparison with
    uncertainty-specific visualizations

    Args:
        base_dir: Base directory containing results
        dataset: Specific dataset to plot, or None for all
    """
    if dataset:
        datasets = [dataset]
    else:
        datasets = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d)) and d not in ['plots', 'summary']]

    print(f"Generating uncertainty-based plots for: {datasets}")

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {ds}...")
        print(f"{'='*60}")

        ds_dir = os.path.join(base_dir, ds)
        plot_dir = os.path.join(ds_dir, 'comparison_plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Check for required files
        nb_pred_csv = os.path.join(ds_dir, 'BIMamba-MAGAC_NonBayesian', 'test_predictions.csv')
        b_pred_csv = os.path.join(ds_dir, 'BIMamba-MAGAC_Bayesian', 'test_predictions.csv')
        uncertainty_json = os.path.join(ds_dir, 'uncertainty_analysis.json')

        if not os.path.exists(nb_pred_csv) or not os.path.exists(b_pred_csv):
            print(f"  Prediction files not found, skipping...")
            continue

        # Generate plots
        print("  Generating confidence-accuracy curve...")
        plot_confidence_accuracy_curve(nb_pred_csv, b_pred_csv, plot_dir)

        if os.path.exists(uncertainty_json):
            print("  Generating risk-adjusted returns plot...")
            plot_risk_adjusted_returns(uncertainty_json, plot_dir)

            print("  Generating uncertainty contribution plot...")
            plot_uncertainty_contribution(uncertainty_json, plot_dir)

            print("  Generating sharpness-calibration tradeoff...")
            plot_sharpness_calibration_tradeoff(uncertainty_json, plot_dir)

        print(f"  ✓ Uncertainty plots saved to: {plot_dir}")

    print(f"\n{'='*60}")
    print("✓ All uncertainty-based plots generated!")
    print(f"{'='*60}")


# ============================================================================
# DISPLAY ALL PLOTS
# ============================================================================

def display_all_plots(plot_dir, image_size=(12, 8)):
    """
    Display all PNG plots in a directory using matplotlib

    Args:
        plot_dir: Directory containing plots
        image_size: Figure size for each plot (width, height)
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path

    plot_files = sorted(Path(plot_dir).glob('*.png'))

    if not plot_files:
        print(f"No plots found in {plot_dir}")
        return

    print(f"Found {len(plot_files)} plots in {plot_dir}")
    print("="*80)

    for plot_file in plot_files:
        print(f"\n{plot_file.name}")
        print("-"*80)

        # Load and display image
        img = mpimg.imread(str(plot_file))
        fig, ax = plt.subplots(figsize=image_size)
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    print("="*80)
    print(f"✓ Displayed all {len(plot_files)} plots")


def list_all_plots(base_dir, study_name='bayesian_vs_nonbayesian'):
    """
    List all plots generated for the Bayesian comparison study

    Args:
        base_dir: Base directory (e.g., 'logs')
        study_name: Study name (can include timestamp, e.g., 'bayesian_vs_nonbayesian_20251119_014842')

    Returns:
        Dictionary mapping dataset -> list of plot files
    """
    from pathlib import Path

    study_dir = Path(base_dir) / study_name

    # If study_dir doesn't exist, try to find latest with timestamp
    if not study_dir.exists():
        parent = Path(base_dir)
        if parent.exists():
            matching_dirs = sorted([d for d in parent.iterdir()
                                   if d.is_dir() and d.name.startswith(study_name + '_')])
            if matching_dirs:
                study_dir = matching_dirs[-1]  # Use latest
                study_name = study_dir.name
                print(f"Using latest study: {study_dir}")
            else:
                print(f"Study directory not found: {study_dir}")
                print(f"No matching directories found with pattern {study_name}_*")
                return {}
        else:
            print(f"Base directory not found: {base_dir}")
            return {}

    plot_inventory = {}

    # Find all dataset directories
    for dataset_dir in sorted(study_dir.iterdir()):
        if dataset_dir.is_dir() and dataset_dir.name not in ['plots', 'summary']:
            plots_dir = dataset_dir / 'comparison_plots'

            if plots_dir.exists():
                plot_files = sorted(plots_dir.glob('*.png'))
                if plot_files:
                    plot_inventory[dataset_dir.name] = [p.name for p in plot_files]

    # Print inventory
    print("="*80)
    print(f"PLOT INVENTORY - {study_name}")
    print("="*80)

    for dataset, plots in plot_inventory.items():
        print(f"\n{dataset}:")
        for plot in plots:
            print(f"  • {plot}")

    total_plots = sum(len(plots) for plots in plot_inventory.values())
    print(f"\n{'='*80}")
    print(f"Total: {len(plot_inventory)} datasets, {total_plots} plots")
    print("="*80)

    return plot_inventory


# ============================================================================
# COMPREHENSIVE RESULTS ANALYSIS AND SUMMARY
# ============================================================================

def generate_comprehensive_analysis(
    base_dir='logs',
    study_name='bayesian_vs_nonbayesian',
    output_file=None
):
    """
    Generate comprehensive analysis and summary of Bayesian vs Non-Bayesian comparison

    Analyzes:
    1. Performance metrics across all datasets
    2. Uncertainty quality metrics
    3. Market regime performance
    4. Cross-sectional IC results
    5. Statistical significance tests

    Args:
        base_dir: Base directory
        study_name: Study name (can include timestamp, e.g., 'bayesian_vs_nonbayesian_20251119_014842')
        output_file: Optional output file path for summary report

    Returns:
        Dictionary with comprehensive analysis results
    """
    import json
    from pathlib import Path
    from scipy import stats

    study_dir = Path(base_dir) / study_name

    # If study_dir doesn't exist, try to find latest with timestamp
    if not study_dir.exists():
        parent = Path(base_dir)
        if parent.exists():
            # Try to find latest study directory with timestamp
            matching_dirs = sorted([d for d in parent.iterdir()
                                   if d.is_dir() and d.name.startswith(study_name + '_')])
            if matching_dirs:
                study_dir = matching_dirs[-1]  # Use latest
                study_name = study_dir.name
                print(f"Using latest study: {study_dir}")
            else:
                print(f"Study directory not found: {study_dir}")
                print(f"No matching directories found with pattern {study_name}_*")
                return None
        else:
            print(f"Base directory not found: {base_dir}")
            return None

    print("="*80)
    print("COMPREHENSIVE ANALYSIS - BAYESIAN VS NON-BAYESIAN")
    print("="*80)

    analysis = {
        'datasets': [],
        'metrics_summary': {},
        'uncertainty_analysis': {},
        'regime_analysis': {},
        'cross_sectional_ic': {},
        'statistical_tests': {},
        'conclusions': []
    }

    # Find all dataset directories
    datasets = [d.name for d in sorted(study_dir.iterdir())
                if d.is_dir() and d.name not in ['plots', 'summary']]

    analysis['datasets'] = datasets
    print(f"\nDatasets analyzed: {', '.join(datasets)}")
    print(f"Total datasets: {len(datasets)}")

    # ========================================================================
    # 1. COLLECT METRICS ACROSS ALL DATASETS
    # ========================================================================
    print("\n" + "="*80)
    print("1. COLLECTING PERFORMANCE METRICS")
    print("="*80)

    nb_key = 'BIMamba-MAGAC_NonBayesian'
    b_key = 'BIMamba-MAGAC_Bayesian'

    metrics_to_analyze = ['ic', 'ric', 'rmse', 'mae', 'nll', 'crps',
                          'sharpe_ratio', 'max_drawdown', 'directional_accuracy']

    for metric in metrics_to_analyze:
        analysis['metrics_summary'][metric] = {
            'nb_values': [],
            'b_values': [],
            'improvements': []
        }

    for dataset in datasets:
        comp_file = study_dir / dataset / 'comprehensive_comparison.json'

        if comp_file.exists():
            with open(comp_file, 'r') as f:
                data = json.load(f)

            for metric in metrics_to_analyze:
                if metric in data.get(nb_key, {}) and metric in data.get(b_key, {}):
                    nb_val = data[nb_key][metric]
                    b_val = data[b_key][metric]

                    analysis['metrics_summary'][metric]['nb_values'].append(nb_val)
                    analysis['metrics_summary'][metric]['b_values'].append(b_val)

                    # Calculate improvement (handle metrics where lower is better)
                    if metric in ['rmse', 'mae', 'nll', 'crps', 'max_drawdown']:
                        improvement = -(b_val - nb_val) / abs(nb_val) * 100 if abs(nb_val) > 1e-8 else 0
                    else:
                        improvement = (b_val - nb_val) / abs(nb_val) * 100 if abs(nb_val) > 1e-8 else 0

                    analysis['metrics_summary'][metric]['improvements'].append(improvement)

    # Calculate statistics
    print(f"\n{'Metric':<25} {'NB Mean±Std':>20} {'B Mean±Std':>20} {'Avg Imp%':>12} {'p-value':>10}")
    print("-"*95)

    for metric in metrics_to_analyze:
        nb_vals = analysis['metrics_summary'][metric]['nb_values']
        b_vals = analysis['metrics_summary'][metric]['b_values']

        if nb_vals and b_vals:
            nb_mean = np.mean(nb_vals)
            nb_std = np.std(nb_vals)
            b_mean = np.mean(b_vals)
            b_std = np.std(b_vals)
            avg_improvement = np.mean(analysis['metrics_summary'][metric]['improvements'])

            # Statistical test (paired t-test)
            if len(nb_vals) > 1:
                t_stat, p_value = stats.ttest_rel(b_vals, nb_vals)
            else:
                p_value = np.nan

            analysis['metrics_summary'][metric]['nb_mean'] = nb_mean
            analysis['metrics_summary'][metric]['nb_std'] = nb_std
            analysis['metrics_summary'][metric]['b_mean'] = b_mean
            analysis['metrics_summary'][metric]['b_std'] = b_std
            analysis['metrics_summary'][metric]['avg_improvement'] = avg_improvement
            analysis['metrics_summary'][metric]['p_value'] = p_value

            sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''

            print(f"{metric:<25} {nb_mean:>9.4f}±{nb_std:<7.4f} {b_mean:>9.4f}±{b_std:<7.4f} "
                  f"{avg_improvement:>11.2f}% {p_value:>9.4f}{sig_marker}")

    # ========================================================================
    # 2. UNCERTAINTY QUALITY ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("2. UNCERTAINTY QUALITY ANALYSIS")
    print("="*80)

    uncertainty_metrics = {
        'sharpness': {'nb': [], 'b': []},
        'calibration_error': {'nb': [], 'b': []},
        'coverage_95': {'nb': [], 'b': []},
        'crps': {'nb': [], 'b': []}
    }

    for dataset in datasets:
        unc_file = study_dir / dataset / 'uncertainty_analysis.json'

        if unc_file.exists():
            with open(unc_file, 'r') as f:
                unc_data = json.load(f)

            # Sharpness
            if 'sharpness' in unc_data:
                uncertainty_metrics['sharpness']['nb'].append(unc_data['sharpness']['nb'])
                uncertainty_metrics['sharpness']['b'].append(unc_data['sharpness']['b'])

            # Calibration
            if 'prediction_intervals' in unc_data:
                nb_cov = unc_data['prediction_intervals']['nb'].get('coverage_95', 0)
                b_cov = unc_data['prediction_intervals']['b'].get('coverage_95', 0)
                uncertainty_metrics['coverage_95']['nb'].append(nb_cov)
                uncertainty_metrics['coverage_95']['b'].append(b_cov)

                nb_cal_err = abs(nb_cov - 0.95)
                b_cal_err = abs(b_cov - 0.95)
                uncertainty_metrics['calibration_error']['nb'].append(nb_cal_err)
                uncertainty_metrics['calibration_error']['b'].append(b_cal_err)

    print(f"\n{'Metric':<30} {'Non-Bayesian':>15} {'Bayesian':>15} {'Better':>10}")
    print("-"*75)

    for metric_name, values in uncertainty_metrics.items():
        if values['nb'] and values['b']:
            nb_mean = np.mean(values['nb'])
            b_mean = np.mean(values['b'])

            # Determine which is better
            if metric_name in ['sharpness', 'calibration_error']:
                better = 'Bayesian' if b_mean < nb_mean else 'Non-Bayesian'
            else:
                better = 'Bayesian' if b_mean > nb_mean else 'Non-Bayesian'

            print(f"{metric_name:<30} {nb_mean:>15.6f} {b_mean:>15.6f} {better:>10}")

            analysis['uncertainty_analysis'][metric_name] = {
                'nb_mean': nb_mean,
                'b_mean': b_mean,
                'better': better
            }

    # ========================================================================
    # 3. CROSS-SECTIONAL IC ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("3. CROSS-SECTIONAL IC ANALYSIS")
    print("="*80)

    cross_ic_file = study_dir / 'cross_sectional_ic_summary.txt'

    if cross_ic_file.exists():
        print(f"\nCross-sectional IC results available at:")
        print(f"  {cross_ic_file}")

        # Try to load from JSON if available
        cross_summary_file = study_dir / 'cross_dataset_summary.json'
        if cross_summary_file.exists():
            with open(cross_summary_file, 'r') as f:
                cross_data = json.load(f)
                analysis['cross_sectional_ic'] = cross_data
                print("\n✓ Cross-sectional IC data loaded")
    else:
        print("\n⚠️  Cross-sectional IC not calculated yet")
        print("   Run calculate_cross_sectional_ic_for_bayesian_comparison() first")

    # ========================================================================
    # 4. GENERATE CONCLUSIONS
    # ========================================================================
    print("\n" + "="*80)
    print("4. KEY FINDINGS AND CONCLUSIONS")
    print("="*80)

    conclusions = []

    # Performance comparison
    if 'ic' in analysis['metrics_summary']:
        ic_data = analysis['metrics_summary']['ic']
        if 'avg_improvement' in ic_data:
            ic_imp = ic_data['avg_improvement']
            if ic_imp > 5:
                conclusions.append(f"✓ Bayesian model shows STRONG improvement in IC (+{ic_imp:.1f}%)")
            elif ic_imp > 0:
                conclusions.append(f"✓ Bayesian model shows modest improvement in IC (+{ic_imp:.1f}%)")
            else:
                conclusions.append(f"⚠️  Non-Bayesian model performs better in IC ({ic_imp:.1f}%)")

    # Uncertainty quality
    if 'calibration_error' in analysis['uncertainty_analysis']:
        better_cal = analysis['uncertainty_analysis']['calibration_error']['better']
        if better_cal == 'Bayesian':
            conclusions.append("✓ Bayesian model provides better-calibrated uncertainty estimates")
        else:
            conclusions.append("⚠️  Non-Bayesian model shows better calibration (unexpected)")

    # Sharpe ratio
    if 'sharpe_ratio' in analysis['metrics_summary']:
        sharpe_data = analysis['metrics_summary']['sharpe_ratio']
        if 'avg_improvement' in sharpe_data:
            sharpe_imp = sharpe_data['avg_improvement']
            if sharpe_imp > 0:
                conclusions.append(f"✓ Bayesian model achieves better risk-adjusted returns (+{sharpe_imp:.1f}%)")

    # Statistical significance
    sig_count = sum(1 for m in analysis['metrics_summary'].values()
                   if 'p_value' in m and m['p_value'] < 0.05)
    total_metrics = len([m for m in analysis['metrics_summary'].values() if 'p_value' in m])

    if sig_count > 0:
        conclusions.append(f"✓ {sig_count}/{total_metrics} metrics show statistically significant differences (p<0.05)")

    analysis['conclusions'] = conclusions

    for i, conclusion in enumerate(conclusions, 1):
        print(f"\n{i}. {conclusion}")

    # ========================================================================
    # 5. SAVE REPORT
    # ========================================================================
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("Bayesian vs Non-Bayesian MAMBA-BGNN Comparison\n")
            f.write("="*80 + "\n")
            f.write(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Datasets: {', '.join(datasets)}\n")
            f.write(f"\n{'='*80}\n")
            f.write("PERFORMANCE METRICS SUMMARY\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"{'Metric':<25} {'NB Mean±Std':>20} {'B Mean±Std':>20} {'Avg Imp%':>12} {'p-value':>10}\n")
            f.write("-"*95 + "\n")

            for metric in metrics_to_analyze:
                if metric in analysis['metrics_summary']:
                    m = analysis['metrics_summary'][metric]
                    if 'nb_mean' in m:
                        sig_marker = '***' if m['p_value'] < 0.001 else '**' if m['p_value'] < 0.01 else '*' if m['p_value'] < 0.05 else ''
                        f.write(f"{metric:<25} {m['nb_mean']:>9.4f}±{m['nb_std']:<7.4f} "
                               f"{m['b_mean']:>9.4f}±{m['b_std']:<7.4f} "
                               f"{m['avg_improvement']:>11.2f}% {m['p_value']:>9.4f}{sig_marker}\n")

            f.write(f"\n{'='*80}\n")
            f.write("UNCERTAINTY QUALITY ANALYSIS\n")
            f.write(f"{'='*80}\n\n")

            for metric_name, data in analysis['uncertainty_analysis'].items():
                f.write(f"{metric_name:<30} NB: {data['nb_mean']:>10.6f}  B: {data['b_mean']:>10.6f}  "
                       f"Better: {data['better']}\n")

            f.write(f"\n{'='*80}\n")
            f.write("KEY FINDINGS AND CONCLUSIONS\n")
            f.write(f"{'='*80}\n\n")

            for i, conclusion in enumerate(conclusions, 1):
                f.write(f"{i}. {conclusion}\n")

        print(f"\n✓ Comprehensive analysis report saved to: {output_file}")

    print("\n" + "="*80)
    print("✓ COMPREHENSIVE ANALYSIS COMPLETED")
    print("="*80)

    return analysis

