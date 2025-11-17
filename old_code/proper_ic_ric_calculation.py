#!/usr/bin/env python3
"""
Proper IC/RIC Calculation for Financial Time Series Data

This script defines the correct way to calculate and interpret IC/RIC metrics
for single-asset time series financial prediction, aligned with academic standards.
"""

import numpy as np
import csv
from pathlib import Path

def calculate_time_series_ic_ric(predictions, actuals, method='time_series'):
    """
    Calculate Information Coefficient (IC) and Rank IC (RIC) properly

    Args:
        predictions: Array of predicted returns
        actuals: Array of actual returns
        method: 'time_series' or 'cross_sectional'

    Returns:
        dict with IC, RIC, and interpretation
    """

    # Clean data
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    pred_clean = predictions[mask]
    actual_clean = actuals[mask]

    if len(pred_clean) < 2:
        return {"ic": 0, "ric": 0, "n_samples": 0, "method": method}

    # IC: Pearson correlation
    if np.std(pred_clean) > 1e-8 and np.std(actual_clean) > 1e-8:
        ic = np.corrcoef(pred_clean, actual_clean)[0,1]
    else:
        ic = 0

    # RIC: Spearman rank correlation
    pred_ranks = np.argsort(np.argsort(pred_clean))
    actual_ranks = np.argsort(np.argsort(actual_clean))

    if np.std(pred_ranks) > 1e-8 and np.std(actual_ranks) > 1e-8:
        ric = np.corrcoef(pred_ranks, actual_ranks)[0,1]
    else:
        ric = 0

    return {
        "ic": ic,
        "ric": ric,
        "n_samples": len(pred_clean),
        "method": method
    }

def interpret_ic_ric_results(ic, ric, method='time_series'):
    """Provide interpretation of IC/RIC results based on method"""

    if method == 'time_series':
        # For single-asset time series prediction
        ic_interpretation = ""
        if abs(ic) > 0.15:
            ic_interpretation = "🚀 EXCELLENT - Very strong predictive power"
        elif abs(ic) > 0.08:
            ic_interpretation = "✅ GOOD - Solid predictive ability"
        elif abs(ic) > 0.04:
            ic_interpretation = "📈 DECENT - Moderate predictive power"
        elif abs(ic) > 0.02:
            ic_interpretation = "📊 WEAK - Limited but potentially useful"
        else:
            ic_interpretation = "📉 VERY WEAK - May not be practically useful"

    elif method == 'cross_sectional':
        # For cross-sectional (multi-asset) prediction
        ic_interpretation = ""
        if abs(ic) > 0.05:
            ic_interpretation = "🔥 EXCEPTIONAL - Extremely rare in real markets"
        elif abs(ic) > 0.02:
            ic_interpretation = "🚀 EXCELLENT - Industry-leading performance"
        elif abs(ic) > 0.01:
            ic_interpretation = "✅ GOOD - Strong commercial value"
        elif abs(ic) > 0.005:
            ic_interpretation = "📈 DECENT - Has practical value"
        else:
            ic_interpretation = "📉 WEAK - Limited practical use"

    return {
        "ic_interpretation": ic_interpretation,
        "ric_note": "RIC should be similar to IC but more robust to outliers"
    }

def analyze_current_model_predictions():
    """Analyze predictions from the current MAMBA_BGNN model"""

    print("🔍 ANALYZING CURRENT MODEL PREDICTIONS")
    print("="*60)

    assets = ['IXIC', 'DJI', 'NYSE']
    results = {}

    for asset in assets:
        print(f"\n📊 {asset} Analysis:")
        print("-" * 30)

        # Load predictions
        asset_dirs = list(Path('logs').glob(f'FDSE25_{asset}_log*'))
        if not asset_dirs:
            print(f"   ❌ No prediction files found for {asset}")
            continue

        pred_file = asset_dirs[0] / 'test_predictions.csv'
        if not pred_file.exists():
            print(f"   ❌ Prediction file not found: {pred_file}")
            continue

        # Read predictions
        with open(pred_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        y_true = np.array([float(row['y']) for row in rows])
        y_pred = np.array([float(row['mu']) for row in rows])

        # Calculate time-series IC/RIC (appropriate for single asset)
        ts_results = calculate_time_series_ic_ric(y_pred, y_true, 'time_series')
        interpretation = interpret_ic_ric_results(ts_results['ic'], ts_results['ric'], 'time_series')

        print(f"   Time-Series IC:  {ts_results['ic']:.4f}")
        print(f"   Time-Series RIC: {ts_results['ric']:.4f}")
        print(f"   Samples:         {ts_results['n_samples']}")
        print(f"   Assessment:      {interpretation['ic_interpretation']}")

        results[asset] = ts_results

    return results

def calculate_cross_sectional_ic():
    """Calculate true cross-sectional IC across the 3 assets"""

    print(f"\n🔗 CROSS-SECTIONAL IC CALCULATION")
    print("="*50)

    # Load all three asset predictions
    assets = ['IXIC', 'DJI', 'NYSE']
    asset_data = {}

    for asset in assets:
        asset_dirs = list(Path('logs').glob(f'FDSE25_{asset}_log*'))
        if asset_dirs:
            pred_file = asset_dirs[0] / 'test_predictions.csv'
            with open(pred_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            asset_data[asset] = {
                'y_true': np.array([float(row['y']) for row in rows]),
                'y_pred': np.array([float(row['mu']) for row in rows])
            }

    if len(asset_data) != 3:
        print("❌ Need predictions from all 3 assets for cross-sectional analysis")
        return None

    # Verify same number of samples
    n_samples = len(asset_data['IXIC']['y_true'])
    for asset in assets:
        assert len(asset_data[asset]['y_true']) == n_samples, f"Sample count mismatch for {asset}"

    print(f"Calculating cross-sectional IC across {len(assets)} assets for {n_samples} time points...")

    daily_ic = []
    daily_ric = []

    for t in range(n_samples):
        # Get predictions and actuals at time t across assets
        pred_t = [asset_data[asset]['y_pred'][t] for asset in assets]
        actual_t = [asset_data[asset]['y_true'][t] for asset in assets]

        # Calculate IC and RIC for this time point
        if len(set(pred_t)) > 1 and len(set(actual_t)) > 1:
            ic_t = np.corrcoef(pred_t, actual_t)[0,1]

            # Rank correlation
            pred_ranks = np.argsort(np.argsort(pred_t))
            actual_ranks = np.argsort(np.argsort(actual_t))
            ric_t = np.corrcoef(pred_ranks, actual_ranks)[0,1]
        else:
            ic_t, ric_t = 0, 0

        daily_ic.append(ic_t)
        daily_ric.append(ric_t)

    # Remove NaN values
    daily_ic = np.array([x for x in daily_ic if not np.isnan(x)])
    daily_ric = np.array([x for x in daily_ric if not np.isnan(x)])

    cs_ic = np.mean(daily_ic)
    cs_ric = np.mean(daily_ric)

    print(f"Cross-Sectional IC:  {cs_ic:.4f}")
    print(f"Cross-Sectional RIC: {cs_ric:.4f}")
    print(f"Valid days:          {len(daily_ic)}")

    interpretation = interpret_ic_ric_results(cs_ic, cs_ric, 'cross_sectional')
    print(f"Assessment:          {interpretation['ic_interpretation']}")

    return {
        'ic': cs_ic,
        'ric': cs_ric,
        'daily_ic': daily_ic,
        'daily_ric': daily_ric,
        'n_days': len(daily_ic)
    }

def provide_final_recommendations():
    """Provide final recommendations based on analysis"""

    print(f"\n📋 FINAL RECOMMENDATIONS")
    print("="*50)

    print("🎯 CORRECT APPROACH FOR YOUR DATA:")
    print("   ✅ Use TIME-SERIES IC/RIC (not cross-sectional)")
    print("   ✅ Each asset (IXIC, DJI, NYSE) analyzed separately")
    print("   ✅ IC measures correlation between predictions and actuals over time")

    print(f"\n📊 REALISTIC EXPECTATIONS:")
    print("   • Time-Series IC: 0.02-0.15 is good for financial prediction")
    print("   • Cross-Sectional IC: 0.005-0.05 would be excellent (but not applicable here)")
    print("   • Your current results should be much lower than before (~0.9+)")

    print(f"\n✅ SUCCESS CRITERIA:")
    print("   1. No individual features with >10% correlation to returns")
    print("   2. Time-series IC in range 0.02-0.20 (realistic)")
    print("   3. Model outperforms simple baselines (lag, mean reversion)")
    print("   4. Stable performance across different time periods")

    print(f"\n🔄 IF IC IS STILL TOO HIGH (>0.3):")
    print("   1. Investigate EMA features - ensure they use only past prices")
    print("   2. Check Volume feature - use previous day's volume")
    print("   3. Verify external market data timing alignment")
    print("   4. Consider additional feature lagging")

def main():
    """Main analysis pipeline"""

    # 1. Analyze current model predictions
    ts_results = analyze_current_model_predictions()

    # 2. Calculate cross-sectional IC for comparison
    cs_results = calculate_cross_sectional_ic()

    # 3. Provide recommendations
    provide_final_recommendations()

    print(f"\n🏁 SUMMARY:")
    if ts_results:
        avg_ts_ic = np.mean([r['ic'] for r in ts_results.values()])
        print(f"✅ Average Time-Series IC: {avg_ts_ic:.4f}")

        if abs(avg_ts_ic) < 0.3:
            print("🎉 GREAT IMPROVEMENT! IC in reasonable range")
        elif abs(avg_ts_ic) < 0.5:
            print("✅ GOOD PROGRESS! Significant improvement from ~0.9+")
        else:
            print("⚠️  STILL HIGH: More investigation needed")

    if cs_results:
        print(f"✅ Cross-Sectional IC: {cs_results['ic']:.4f}")
        print("   (For reference only - not the primary metric for your data)")

    print(f"\n🎯 Your model uses single-asset time series → Use TIME-SERIES IC/RIC as primary metric!")

if __name__ == "__main__":
    main()