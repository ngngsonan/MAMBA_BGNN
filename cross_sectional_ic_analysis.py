#!/usr/bin/env python3
"""
Cross-sectional IC Analysis for MAMBA_BGNN Results

This script calculates true cross-sectional Information Coefficient (IC) and Rank IC (RIC)
across multiple assets (IXIC, DJI, NYSE) using the test predictions from FDSE25 experiments.

True cross-sectional IC measures the correlation between predictions and actual returns
across different assets at each time point, which is the standard definition in quantitative finance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

def load_predictions(base_path="logs"):
    """Load test predictions from all three assets"""
    assets = ['IXIC', 'DJI', 'NYSE']
    data = {}

    for asset in assets:
        # Find the FDSE25 log directory for this asset
        asset_dirs = list(Path(base_path).glob(f"FDSE25_{asset}_log*"))
        if not asset_dirs:
            raise FileNotFoundError(f"No FDSE25 log directory found for {asset}")

        log_dir = asset_dirs[0]  # Take the first (should be only one)
        pred_file = log_dir / "test_predictions.csv"

        if not pred_file.exists():
            raise FileNotFoundError(f"test_predictions.csv not found in {log_dir}")

        df = pd.read_csv(pred_file)
        data[asset] = df
        print(f"Loaded {asset}: {len(df)} samples")

    return data

def calculate_cross_sectional_ic(data):
    """Calculate true cross-sectional IC and RIC"""
    assets = list(data.keys())
    n_samples = len(data[assets[0]])

    # Verify all assets have same number of samples
    for asset in assets:
        assert len(data[asset]) == n_samples, f"Sample count mismatch for {asset}"

    daily_ic = []
    daily_ric = []

    print(f"Calculating cross-sectional IC across {len(assets)} assets for {n_samples} time points...")

    for t in range(n_samples):
        # Get predictions and actual returns at time t across all assets
        pred_t = [data[asset]['mu'].iloc[t] for asset in assets]
        actual_t = [data[asset]['y'].iloc[t] for asset in assets]

        # Calculate Pearson correlation (IC)
        if len(set(pred_t)) > 1 and len(set(actual_t)) > 1:  # Avoid division by zero
            ic_t, _ = pearsonr(pred_t, actual_t)
            ric_t, _ = spearmanr(pred_t, actual_t)
        else:
            ic_t, ric_t = 0, 0

        daily_ic.append(ic_t)
        daily_ric.append(ric_t)

    return np.array(daily_ic), np.array(daily_ric)

def analyze_results(daily_ic, daily_ric):
    """Analyze and summarize the cross-sectional IC results"""
    # Remove NaN values
    daily_ic_clean = daily_ic[~np.isnan(daily_ic)]
    daily_ric_clean = daily_ric[~np.isnan(daily_ric)]

    results = {
        'IC_mean': np.mean(daily_ic_clean),
        'IC_std': np.std(daily_ic_clean),
        'IC_median': np.median(daily_ic_clean),
        'RIC_mean': np.mean(daily_ric_clean),
        'RIC_std': np.std(daily_ric_clean),
        'RIC_median': np.median(daily_ric_clean),
        'IC_positive_ratio': np.sum(daily_ic_clean > 0) / len(daily_ic_clean),
        'RIC_positive_ratio': np.sum(daily_ric_clean > 0) / len(daily_ric_clean),
        'valid_days': len(daily_ic_clean)
    }

    print("\n" + "="*60)
    print("CROSS-SECTIONAL IC ANALYSIS RESULTS")
    print("="*60)
    print(f"Valid trading days: {results['valid_days']}")
    print(f"\nInformation Coefficient (IC):")
    print(f"  Mean:   {results['IC_mean']:.6f}")
    print(f"  Std:    {results['IC_std']:.6f}")
    print(f"  Median: {results['IC_median']:.6f}")
    print(f"  % Positive: {results['IC_positive_ratio']*100:.1f}%")

    print(f"\nRank IC (RIC):")
    print(f"  Mean:   {results['RIC_mean']:.6f}")
    print(f"  Std:    {results['RIC_std']:.6f}")
    print(f"  Median: {results['RIC_median']:.6f}")
    print(f"  % Positive: {results['RIC_positive_ratio']*100:.1f}%")

    print(f"\n📊 INTERPRETATION:")
    if results['IC_mean'] > 0.05:
        print("🔥 EXCEPTIONAL: IC > 0.05 is extremely rare in real markets!")
    elif results['IC_mean'] > 0.02:
        print("🚀 EXCELLENT: IC > 0.02 indicates very strong predictive power")
    elif results['IC_mean'] > 0.01:
        print("✅ GOOD: IC > 0.01 shows solid predictive ability")
    elif results['IC_mean'] > 0.005:
        print("📈 DECENT: IC > 0.005 has commercial value")
    else:
        print("📉 WEAK: IC ≤ 0.005 may not be practically useful")

    return results, daily_ic_clean, daily_ric_clean

def create_visualizations(daily_ic, daily_ric, output_dir="cross_sectional_analysis"):
    """Create visualizations for cross-sectional IC analysis"""
    Path(output_dir).mkdir(exist_ok=True)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Sectional IC Analysis: MAMBA_BGNN (3 Assets)', fontsize=16, fontweight='bold')

    # Time series plot
    axes[0,0].plot(daily_ic, alpha=0.7, linewidth=1, label='IC')
    axes[0,0].plot(daily_ric, alpha=0.7, linewidth=1, label='RIC')
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Daily Cross-Sectional IC Time Series')
    axes[0,0].set_xlabel('Trading Day')
    axes[0,0].set_ylabel('IC Value')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # IC histogram
    axes[0,1].hist(daily_ic[~np.isnan(daily_ic)], bins=50, alpha=0.7, color='blue', density=True)
    axes[0,1].axvline(x=np.mean(daily_ic[~np.isnan(daily_ic)]), color='red', linestyle='--',
                      label=f'Mean: {np.mean(daily_ic[~np.isnan(daily_ic)]):.4f}')
    axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[0,1].set_title('IC Distribution')
    axes[0,1].set_xlabel('IC Value')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # RIC histogram
    axes[1,0].hist(daily_ric[~np.isnan(daily_ric)], bins=50, alpha=0.7, color='green', density=True)
    axes[1,0].axvline(x=np.mean(daily_ric[~np.isnan(daily_ric)]), color='red', linestyle='--',
                      label=f'Mean: {np.mean(daily_ric[~np.isnan(daily_ric)]):.4f}')
    axes[1,0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[1,0].set_title('RIC Distribution')
    axes[1,0].set_xlabel('RIC Value')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Box plot
    box_data = [daily_ic[~np.isnan(daily_ic)], daily_ric[~np.isnan(daily_ric)]]
    axes[1,1].boxplot(box_data, labels=['IC', 'RIC'], patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1,1].set_title('IC vs RIC Distribution')
    axes[1,1].set_ylabel('Value')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cross_sectional_ic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save detailed time series data
    ic_df = pd.DataFrame({
        'trading_day': range(len(daily_ic)),
        'IC': daily_ic,
        'RIC': daily_ric
    })
    ic_df.to_csv(f'{output_dir}/daily_cross_sectional_ic.csv', index=False)
    print(f"\n💾 Results saved to {output_dir}/")

def compare_with_time_series_ic(data):
    """Compare cross-sectional IC with time-series correlation for context"""
    print("\n" + "="*60)
    print("COMPARISON: Cross-sectional IC vs Time-series Correlation")
    print("="*60)

    for asset, df in data.items():
        # Time-series correlation (what the original code calculated)
        ts_ic = np.corrcoef(df['mu'], df['y'])[0,1]
        ts_ric_data = pd.DataFrame({'pred': df['mu'], 'actual': df['y']}).corr('spearman')
        ts_ric = ts_ric_data.loc['pred', 'actual']

        print(f"\n{asset}:")
        print(f"  Time-series IC:  {ts_ic:.6f}")
        print(f"  Time-series RIC: {ts_ric:.6f}")

def main():
    """Main analysis pipeline"""
    print("🔍 Cross-Sectional IC Analysis for MAMBA_BGNN")
    print("Loading prediction data from FDSE25 experiments...")

    try:
        # Load data
        data = load_predictions()

        # Calculate cross-sectional IC
        daily_ic, daily_ric = calculate_cross_sectional_ic(data)

        # Analyze results
        results, daily_ic_clean, daily_ric_clean = analyze_results(daily_ic, daily_ric)

        # Create visualizations
        create_visualizations(daily_ic_clean, daily_ric_clean)

        # Compare with time-series correlations
        compare_with_time_series_ic(data)

        print("\n✅ Analysis completed successfully!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()