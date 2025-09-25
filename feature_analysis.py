#!/usr/bin/env python3
"""
Feature Analysis for Look-ahead Bias Detection

Analyzing the 81 features in the dataset to identify potential look-ahead bias
in technical indicators and other derived features.
"""

import numpy as np
import csv
from pathlib import Path

def analyze_feature_headers():
    """Analyze feature column headers for suspicious patterns"""
    data_path = 'Dataset/combined_dataframe_IXIC.csv'

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

    print("🔍 FEATURE ANALYSIS FOR LOOK-AHEAD BIAS")
    print("="*60)

    # Categorize features
    price_features = []
    tech_indicators = []
    external_market = []
    other_features = []

    suspicious_features = []

    for i, col in enumerate(headers):
        if col in ['Date', 'Name']:
            continue

        col_lower = col.lower()

        # Price-based features
        if 'price' in col_lower or col == 'Price':
            price_features.append(col)

        # Technical indicators (potentially problematic)
        elif any(x in col_lower for x in ['ema', 'roc', 'mom', 'vol', 'te', 'de']):
            tech_indicators.append(col)
            # Check for suspicious naming
            if any(x in col_lower for x in ['future', 'next', 'forward', 'lead']):
                suspicious_features.append(col)

        # External market data
        elif any(x in col for x in ['-F', 'DGS', 'CTB', 'DTB', 'WTI', 'Gold', 'EUR', 'USD', 'JPY']):
            external_market.append(col)

        else:
            other_features.append(col)

    print(f"📊 FEATURE CATEGORIZATION:")
    print(f"  Price features: {len(price_features)}")
    print(f"  Technical indicators: {len(tech_indicators)}")
    print(f"  External market data: {len(external_market)}")
    print(f"  Other features: {len(other_features)}")
    print(f"  TOTAL: {len(headers)-2} features")  # Exclude Date, Name

    if suspicious_features:
        print(f"\n🚨 SUSPICIOUS FEATURE NAMES:")
        for feat in suspicious_features:
            print(f"    {feat}")

    print(f"\n📋 TECHNICAL INDICATORS (Potential Look-ahead Risk):")
    for feat in tech_indicators[:20]:  # Show first 20
        print(f"  {feat}")
    if len(tech_indicators) > 20:
        print(f"  ... and {len(tech_indicators)-20} more")

    return headers, tech_indicators, external_market

def check_data_freshness_bias(data_path='Dataset/combined_dataframe_IXIC.csv'):
    """Check if features contain future information relative to target"""
    print(f"\n🔬 DATA FRESHNESS ANALYSIS:")
    print("-"*40)

    # Load sample data
    data = []
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Get first few rows to check alignment
    print("Checking temporal alignment...")
    print("Sample data points (Date, Price, first few features):")

    for i in range(min(5, len(data))):
        row = data[i]
        date = row['Date']
        price = row['Price']
        vol = row.get('Vol.', 'N/A')
        mom = row.get('mom', 'N/A')
        ema10 = row.get('EMA_10', 'N/A')

        print(f"  {date}: Price={price}, Vol={vol}, mom={mom}, EMA_10={ema10}")

    # Check if there are any obvious future leakage patterns
    print(f"\nDataset spans: {data[0]['Date']} to {data[-1]['Date']}")
    print(f"Total records: {len(data)}")

def analyze_feature_correlations():
    """Analyze correlations between features and next-day returns"""
    print(f"\n📈 FEATURE-TARGET CORRELATION ANALYSIS:")
    print("-"*40)

    data_path = 'Dataset/combined_dataframe_IXIC.csv'

    # Load data
    prices = []
    features_dict = {}

    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        feature_cols = [h for h in headers if h not in ['Date', 'Price', 'Name']]

        for col in feature_cols:
            features_dict[col] = []

        for row in reader:
            prices.append(float(row['Price']))
            for col in feature_cols:
                try:
                    features_dict[col].append(float(row[col]))
                except (ValueError, TypeError):
                    features_dict[col].append(0.0)

    # Calculate next-day returns
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)

    # Align features with next-day returns
    # Feature at time t should predict return from t to t+1
    aligned_features = {}
    for col in feature_cols:
        aligned_features[col] = features_dict[col][:-1]  # Remove last day

    returns = np.array(returns)
    print(f"Analyzing {len(returns)} return observations...")

    # Calculate correlations and rank by strength
    correlations = []
    for col in feature_cols:
        feat_values = np.array(aligned_features[col])

        # Skip if constant
        if np.std(feat_values) < 1e-8 or np.std(returns) < 1e-8:
            continue

        # Calculate correlation
        corr = np.corrcoef(feat_values, returns)[0,1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr), corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔝 TOP 15 FEATURES BY CORRELATION WITH NEXT-DAY RETURNS:")
    print("   Feature Name                    |Abs Corr| Raw Corr")
    print("   " + "-"*55)

    for i, (feat, abs_corr, raw_corr) in enumerate(correlations[:15]):
        print(f"   {feat:30s} | {abs_corr:6.4f} | {raw_corr:7.4f}")

    # Flag suspiciously high correlations
    high_corr_threshold = 0.1  # Anything above 10% is suspicious for daily returns
    suspicious_high = [c for c in correlations if c[1] > high_corr_threshold]

    if suspicious_high:
        print(f"\n🚨 SUSPICIOUSLY HIGH CORRELATIONS (>{high_corr_threshold:.1%}):")
        for feat, abs_corr, raw_corr in suspicious_high:
            print(f"    {feat}: {raw_corr:.4f}")
        print("   → These features may contain look-ahead bias!")
    else:
        print(f"\n✅ No suspiciously high individual feature correlations found")

    return correlations

def check_feature_timing_consistency():
    """Check if feature construction timing is consistent"""
    print(f"\n⏰ FEATURE TIMING CONSISTENCY CHECK:")
    print("-"*40)

    # This is a conceptual check - in practice would require
    # access to the original feature construction code

    suspicious_timing_patterns = [
        "Features computed using data from time t to predict return at time t",
        "Technical indicators using future prices in calculation window",
        "Moving averages calculated with look-ahead bias",
        "Volume/momentum features misaligned temporally"
    ]

    print("POTENTIAL TIMING ISSUES TO INVESTIGATE:")
    for i, pattern in enumerate(suspicious_timing_patterns, 1):
        print(f"  {i}. {pattern}")

    print(f"\n💡 RECOMMENDATIONS:")
    print("  • Verify that ALL features at time t use only data ≤ t")
    print("  • Check technical indicator calculations for off-by-one errors")
    print("  • Ensure price-based features don't leak current day's price")
    print("  • Validate that external market data timing is aligned")

def main():
    """Main feature analysis pipeline"""

    # 1. Analyze feature headers
    headers, tech_indicators, external_market = analyze_feature_headers()

    # 2. Check data freshness
    check_data_freshness_bias()

    # 3. Analyze feature-target correlations
    correlations = analyze_feature_correlations()

    # 4. Check timing consistency
    check_feature_timing_consistency()

    # 5. Summary recommendations
    print(f"\n📋 SUMMARY & ACTION ITEMS:")
    print("="*60)
    print("1. 🔍 IMMEDIATE CHECKS NEEDED:")
    print("   • Inspect original feature construction code")
    print("   • Verify technical indicator timing (EMA, ROC, momentum)")
    print("   • Check if 'Vol.' is same-day volume (potential leakage)")

    print(f"\n2. 🧪 SUGGESTED EXPERIMENTS:")
    print("   • Remove top 10 most correlated features and retrain")
    print("   • Test with only basic lagged price features")
    print("   • Compare performance with 1-day delayed features")

    print(f"\n3. 🔧 PROPER FEATURE ENGINEERING:")
    print("   • Use only t-1 price data for features at time t")
    print("   • Implement proper lag alignment for all indicators")
    print("   • Add explicit temporal validation checks")

    print(f"\n✅ Feature analysis completed!")

if __name__ == "__main__":
    main()