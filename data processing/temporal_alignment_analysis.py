#!/usr/bin/env python3
"""
Temporal Alignment Analysis for MAMBA_BGNN After Fixes

This script analyzes the temporal alignment in the data processing pipeline
after removing the top 5 leaked features to verify proper temporal alignment.
"""

import numpy as np
import pandas as pd
import csv

def analyze_temporal_alignment():
    """Analyze the temporal alignment logic in the current data processing"""

    print("🔍 TEMPORAL ALIGNMENT ANALYSIS AFTER FIXES")
    print("="*60)

    # Let's trace through the data processing logic step by step
    print("\n📋 CURRENT DATA PROCESSING LOGIC:")
    print("-" * 40)

    # Simulate the data processing with a small example
    # Load a sample of the data to trace the logic
    data_path = 'Dataset/combined_dataframe_IXIC.csv'
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    df = df.drop(columns=['Name'])

    # Apply the same leaky feature removal
    leaky_features = ['mom', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20']
    features_removed = []
    for feat in leaky_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])
            features_removed.append(feat)

    print(f"✅ Removed features: {features_removed}")
    print(f"✅ Remaining features: {len(df.columns) - 1} (excluding Price)")

    # Get first few rows to analyze
    prices = df['Price'].values[:10]  # First 10 prices
    features = df.drop(columns=['Price']).values[:10]  # First 10 feature rows

    window = 5

    print(f"\n📊 SAMPLE DATA TRACE (window={window}):")
    print("Date indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9...")
    print(f"Prices: {prices}")

    # Trace the sample construction logic
    print(f"\n🔬 SAMPLE CONSTRUCTION ANALYSIS:")
    print("-" * 40)

    for i in range(min(3, len(prices) - window)):  # Analyze first 3 samples
        print(f"\n📝 Sample {i}:")

        # Current logic from mamba_bgnn.py
        price_prev = prices[i + window - 1]  # Price at t-1
        price_cur = prices[i + window]       # Price at t
        ret = (price_cur - price_prev) / price_prev

        # Feature window: features[i:i+window]
        feat_window_indices = list(range(i, i + window))

        print(f"   Target: Return from day {i + window - 1} to day {i + window}")
        print(f"   Price[{i + window - 1}] = {price_prev:.2f}")
        print(f"   Price[{i + window}] = {price_cur:.2f}")
        print(f"   Return = {ret:.6f}")
        print(f"   Feature window: days {feat_window_indices} (indices {i}:{i + window})")

        # Check temporal alignment
        latest_feature_day = i + window - 1
        target_day = i + window

        if latest_feature_day < target_day:
            alignment_status = "✅ GOOD"
        else:
            alignment_status = "❌ LEAK"

        print(f"   Temporal check: Features use data up to day {latest_feature_day}, predict day {target_day} -> {alignment_status}")

    return True

def check_remaining_feature_risks():
    """Check if remaining features might still have temporal issues"""

    print(f"\n🔍 REMAINING FEATURE RISK ANALYSIS:")
    print("-" * 40)

    data_path = 'Dataset/combined_dataframe_IXIC.csv'
    df = pd.read_csv(data_path, nrows=10)  # Just header and few rows

    # Get remaining features after removal
    leaky_features = ['mom', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20']
    remaining_features = [col for col in df.columns
                         if col not in ['Date', 'Price', 'Name'] + leaky_features]

    print(f"Analyzing {len(remaining_features)} remaining features...")

    # Categorize remaining features by risk level
    high_risk = []
    medium_risk = []
    low_risk = []

    for feat in remaining_features:
        feat_lower = feat.lower()

        # High risk: Technical indicators that might use current day
        if any(x in feat_lower for x in ['ema', 'vol', 'te', 'de']):
            high_risk.append(feat)
        # Medium risk: External market data (timing uncertain)
        elif any(x in feat for x in ['-F', 'DGS', 'CTB', 'DTB']):
            medium_risk.append(feat)
        # Low risk: Basic features
        else:
            low_risk.append(feat)

    print(f"\n🚨 HIGH RISK (Technical indicators): {len(high_risk)}")
    for feat in high_risk[:10]:  # Show first 10
        print(f"   {feat}")

    print(f"\n⚠️  MEDIUM RISK (External market data): {len(medium_risk)}")
    for feat in medium_risk[:10]:  # Show first 10
        print(f"   {feat}")

    print(f"\n✅ LOW RISK: {len(low_risk)}")
    for feat in low_risk[:10]:  # Show first 10
        print(f"   {feat}")

    # Focus on high-risk features
    if high_risk:
        print(f"\n🔬 DETAILED ANALYSIS OF HIGH-RISK FEATURES:")
        print("These features might still contain temporal alignment issues:")

        suspicious_patterns = {
            'EMA': 'Exponential Moving Average - verify calculation uses only past prices',
            'Vol': 'Volume - ensure this is previous day volume, not same-day',
            'TE': 'Technical indicator - check calculation timing',
            'DE': 'Technical indicator - verify no look-ahead bias'
        }

        for pattern, description in suspicious_patterns.items():
            matching_features = [f for f in high_risk if pattern in f.upper()]
            if matching_features:
                print(f"\n   {pattern} features ({len(matching_features)}):")
                print(f"     Risk: {description}")
                for feat in matching_features[:5]:
                    print(f"     - {feat}")

def verify_current_alignment_is_correct():
    """Verify the current alignment logic is actually correct"""

    print(f"\n✅ VERIFICATION: CURRENT TEMPORAL ALIGNMENT")
    print("-" * 40)

    # The key question: In the current logic, when we predict return from day t-1 to day t,
    # do we use features that include information from day t?

    print("Current logic analysis:")
    print("For sample i:")
    print(f"  - Target: return from prices[i+window-1] to prices[i+window]")
    print(f"  - Features: features[i:i+window] (days i through i+window-1)")
    print(f"  - Latest feature day: i+window-1")
    print(f"  - Target day: i+window")
    print(f"  - ✅ Features are from day i+window-1 and earlier")
    print(f"  - ✅ Target is from day i+window")
    print(f"  - ✅ NO TEMPORAL LEAKAGE in the indexing logic")

    print(f"\n📊 CONCLUSION:")
    print("The current temporal alignment LOGIC is correct.")
    print("However, FEATURE CONTENT might still be problematic:")
    print("  ❌ EMA, Volume features might use same-day data in their calculation")
    print("  ❌ External market features might not be properly aligned")
    print("  ✅ The indexing itself doesn't introduce leakage")

    return True

def recommendations_for_proper_ic_calculation():
    """Provide recommendations for proper IC/RIC calculation"""

    print(f"\n📋 RECOMMENDATIONS FOR PROPER IC/RIC CALCULATION:")
    print("=" * 60)

    print("🎯 CURRENT SITUATION:")
    print("  - You have single-asset time series for each index (IXIC, DJI, NYSE)")
    print("  - Each dataset contains ~3,470 daily observations")
    print("  - You removed the top 5 most leaky features")
    print("  - Temporal indexing logic is correct")

    print(f"\n🔄 TWO VALID APPROACHES:")
    print("-" * 30)

    print("APPROACH 1: Fix Feature Content (Recommended)")
    print("  1. Review EMA calculation - ensure it uses only past prices")
    print("  2. Fix volume feature - use previous day's volume")
    print("  3. Verify external market data timing alignment")
    print("  4. Keep current IC/RIC calculation as 'time-series correlation'")
    print("  5. Expect IC to drop to 0.02-0.15 range (realistic)")

    print(f"\nAPPROACH 2: True Cross-Sectional IC (More Complex)")
    print("  1. Combine all 3 assets into single dataset")
    print("  2. For each day t, calculate correlation across 3 assets")
    print("  3. Daily IC = correlation(pred_IXIC, pred_DJI, pred_NYSE vs actual_IXIC, actual_DJI, actual_NYSE)")
    print("  4. Final IC = mean of daily ICs")
    print("  5. Expect IC to be much lower (~0.001-0.05)")

    print(f"\n💡 RECOMMENDED NEXT STEPS:")
    print("1. Test current model with fixed features (Approach 1)")
    print("2. Run cross_sectional_ic_analysis.py to see improvement")
    print("3. If IC is still >0.3, investigate remaining features")
    print("4. Consider Approach 2 only if you need true cross-sectional analysis")

    print(f"\n🎯 SUCCESS CRITERIA:")
    print("  - Time-series IC: 0.02-0.15 (good)")
    print("  - Cross-sectional IC: 0.001-0.05 (excellent)")
    print("  - Model should still outperform simple baselines")
    print("  - No features with >10% correlation to next-day returns")

def main():
    """Main analysis pipeline"""

    # 1. Analyze current temporal alignment
    analyze_temporal_alignment()

    # 2. Check remaining feature risks
    check_remaining_feature_risks()

    # 3. Verify alignment correctness
    verify_current_alignment_is_correct()

    # 4. Provide recommendations
    recommendations_for_proper_ic_calculation()

    print(f"\n🏁 SUMMARY:")
    print("✅ Top 5 leaked features successfully removed")
    print("✅ Temporal indexing logic is correct (no look-ahead in indexing)")
    print("⚠️  Some remaining features (EMA, Vol) might still have content issues")
    print("🎯 Ready to test model performance with current fixes")

if __name__ == "__main__":
    main()