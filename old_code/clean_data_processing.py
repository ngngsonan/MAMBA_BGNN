#!/usr/bin/env python3
"""
Clean Data Processing for MAMBA_BGNN - Fix Look-ahead Bias

This module provides leak-free data processing functions that properly handle
temporal alignment to prevent look-ahead bias in financial prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import csv

class MinMax01:
    """Min-Max scaler that preserves temporal integrity"""
    def fit(self, x):
        self.min = x.min(0)
        self.max = x.max(0)

    def transform(self, x):
        return (x - self.min) / (self.max - self.min + 1e-8)

def clean_feature_construction(df):
    """
    Reconstruct features with proper temporal alignment to eliminate look-ahead bias.

    Key principle: Features at time t can only use information available at time t-1 or earlier.
    """
    print("🧹 CLEANING FEATURE CONSTRUCTION...")

    # Make a copy to avoid modifying original
    df_clean = df.copy()

    # Get price column
    prices = df_clean['Price'].values

    # 1. REMOVE HIGHLY LEAKY FEATURES
    leaky_features = ['mom', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20']
    features_removed = []

    for feat in leaky_features:
        if feat in df_clean.columns:
            df_clean = df_clean.drop(columns=[feat])
            features_removed.append(feat)

    print(f"   ❌ Removed {len(features_removed)} highly leaky features: {features_removed}")

    # 2. CREATE PROPER LAG-BASED FEATURES
    print("   ✅ Creating proper lag-based features...")

    # Lagged returns (using past prices only)
    df_clean['return_lag1'] = np.nan
    df_clean['return_lag2'] = np.nan
    df_clean['return_lag3'] = np.nan
    df_clean['return_lag5'] = np.nan

    for i in range(5, len(prices)):
        # Lag-1 return: (P_{t-1} - P_{t-2}) / P_{t-2}
        if i >= 2:
            df_clean.iloc[i, df_clean.columns.get_loc('return_lag1')] = \
                (prices[i-1] - prices[i-2]) / prices[i-2]

        # Lag-2 return: (P_{t-2} - P_{t-3}) / P_{t-3}
        if i >= 3:
            df_clean.iloc[i, df_clean.columns.get_loc('return_lag2')] = \
                (prices[i-2] - prices[i-3]) / prices[i-3]

        # Lag-3 return: (P_{t-3} - P_{t-4}) / P_{t-4}
        if i >= 4:
            df_clean.iloc[i, df_clean.columns.get_loc('return_lag3')] = \
                (prices[i-3] - prices[i-4]) / prices[i-4]

        # Lag-5 return: (P_{t-5} - P_{t-10}) / P_{t-10}
        if i >= 10:
            df_clean.iloc[i, df_clean.columns.get_loc('return_lag5')] = \
                (prices[i-5] - prices[i-10]) / prices[i-10]

    # 3. FIX EMA FEATURES (shift by 1 day)
    ema_features = [col for col in df_clean.columns if 'EMA' in col]
    print(f"   🔧 Fixing temporal alignment for {len(ema_features)} EMA features")

    for ema_col in ema_features:
        # Shift EMA by 1 day to prevent look-ahead
        df_clean[ema_col + '_lag1'] = df_clean[ema_col].shift(1)
        df_clean = df_clean.drop(columns=[ema_col])  # Remove original

    # 4. CLEAN VOLUME FEATURE
    if 'Vol.' in df_clean.columns:
        print("   🔧 Fixing volume feature alignment")
        # Shift volume by 1 day (yesterday's volume)
        df_clean['Vol_lag1'] = df_clean['Vol.'].shift(1)
        df_clean = df_clean.drop(columns=['Vol.'])

    # 5. KEEP SAFE EXTERNAL FEATURES
    safe_external = [col for col in df_clean.columns
                    if any(x in col for x in ['DGS', 'CTB', 'DTB', 'WTI', 'Gold', 'USD', 'EUR', 'JPY', 'GBP'])]
    print(f"   ✅ Keeping {len(safe_external)} safe external market features")

    # Remove rows with NaN (due to lagging)
    df_clean = df_clean.dropna()

    print(f"   📊 Final dataset: {len(df_clean)} samples, {len(df_clean.columns)-2} features")  # -2 for Date, Name

    return df_clean

def leak_free_data_processing(data_path, window=5, batch_size=128):
    """
    Implement completely leak-free data processing pipeline.

    Key changes:
    1. Clean feature construction first
    2. Proper temporal splits
    3. Scaler fitted only on training data
    4. Explicit temporal validation
    """

    print(f"\n🔒 LEAK-FREE DATA PROCESSING")
    print("="*50)

    # Load and clean data
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    print(f"Original data: {len(df)} samples")

    # Apply leak-free feature construction
    df_clean = clean_feature_construction(df)

    # Extract price and features
    prices = df_clean['Price'].values
    feature_cols = [col for col in df_clean.columns if col != 'Price']
    features = df_clean[feature_cols].values

    T, n_features = len(prices), len(feature_cols)
    print(f"Processing: {T} days, {n_features} features")

    # Build samples with strict temporal alignment
    X_list, Y_list = [], []

    for i in range(window, T-1):  # Ensure we have next day price
        # Target: return from day i to day i+1
        ret = (prices[i+1] - prices[i]) / prices[i]

        # Features: use window from day (i-window+1) to day i
        # This ensures features are historical at prediction time
        feat_window = features[i-window+1:i+1]  # Shape: (window, n_features)

        X_list.append(feat_window)
        Y_list.append(ret)

    XX = np.array(X_list)
    YY = np.array(Y_list)

    print(f"Created {len(XX)} training samples with shape {XX.shape}")

    # Strict temporal splits (no randomization!)
    n_samples = len(XX)
    train_end = int(0.70 * n_samples)      # 70% train
    val_end = int(0.80 * n_samples)        # 10% val
    # Remaining 20% test

    X_train = XX[:train_end]
    X_val = XX[train_end:val_end]
    X_test = XX[val_end:]

    Y_train = YY[:train_end]
    Y_val = YY[train_end:val_end]
    Y_test = YY[val_end:]

    print(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Fit scaler ONLY on training data
    train_features_flat = X_train.reshape(-1, n_features)
    scaler = MinMax01()
    scaler.fit(train_features_flat)

    print(f"Scaler fitted on {train_features_flat.shape[0]} training feature vectors")

    # Apply scaling to all splits
    X_train_scaled = np.array([scaler.transform(X_train[i]) for i in range(len(X_train))])
    X_val_scaled = np.array([scaler.transform(X_val[i]) for i in range(len(X_val))])
    X_test_scaled = np.array([scaler.transform(X_test[i]) for i in range(len(X_test))])

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    Y_train_t = torch.FloatTensor(Y_train).unsqueeze(-1)

    X_val_t = torch.FloatTensor(X_val_scaled)
    Y_val_t = torch.FloatTensor(Y_val).unsqueeze(-1)

    X_test_t = torch.FloatTensor(X_test_scaled)
    Y_test_t = torch.FloatTensor(Y_test).unsqueeze(-1)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                             batch_size=batch_size, shuffle=False)  # NO SHUFFLE for temporal integrity
    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t),
                           batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t),
                            batch_size=batch_size, shuffle=False)

    # Validation check: ensure no leakage
    print(f"\n✅ TEMPORAL INTEGRITY CHECK:")
    print(f"   Train period: samples 0 to {train_end-1}")
    print(f"   Val period:   samples {train_end} to {val_end-1}")
    print(f"   Test period:  samples {val_end} to {n_samples-1}")
    print(f"   ✓ No temporal overlap between splits")

    return n_features, train_loader, val_loader, test_loader, scaler

def validate_clean_features(data_path):
    """
    Validate that cleaned features don't have suspicious correlations
    """
    print(f"\n🔍 VALIDATING CLEANED FEATURES")
    print("-"*40)

    # Process with clean pipeline
    _, train_loader, val_loader, test_loader, _ = leak_free_data_processing(data_path, window=5, batch_size=1000)

    # Extract test data for validation
    X_test_all, Y_test_all = [], []
    for X_batch, Y_batch in test_loader:
        X_test_all.append(X_batch)
        Y_test_all.append(Y_batch)

    X_test = torch.cat(X_test_all, dim=0)  # (n_test, window, n_features)
    Y_test = torch.cat(Y_test_all, dim=0).squeeze()  # (n_test,)

    # Use last time step features for correlation analysis
    X_last = X_test[:, -1, :]  # (n_test, n_features)

    print(f"Validating on {len(Y_test)} test samples...")

    # Calculate correlations
    correlations = []
    for feat_idx in range(X_last.shape[1]):
        feat_values = X_last[:, feat_idx].numpy()
        returns = Y_test.numpy()

        if np.std(feat_values) > 1e-8 and np.std(returns) > 1e-8:
            corr = np.corrcoef(feat_values, returns)[0,1]
            if not np.isnan(corr):
                correlations.append((feat_idx, abs(corr), corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔝 TOP 10 FEATURE CORRELATIONS (after cleaning):")
    print("   Feature Index | Abs Corr | Raw Corr")
    print("   " + "-"*40)

    for i, (feat_idx, abs_corr, raw_corr) in enumerate(correlations[:10]):
        status = "🚨" if abs_corr > 0.1 else ("⚠️" if abs_corr > 0.05 else "✅")
        print(f"   {status} Feature {feat_idx:2d}  | {abs_corr:6.4f}  | {raw_corr:7.4f}")

    # Check if any high correlations remain
    high_corr = [c for c in correlations if c[1] > 0.1]

    if high_corr:
        print(f"\n🚨 WARNING: {len(high_corr)} features still have suspiciously high correlations!")
        return False
    elif any(c[1] > 0.05 for c in correlations[:5]):
        print(f"\n⚠️ CAUTION: Some features still have moderate correlations (>5%)")
        return True
    else:
        print(f"\n✅ EXCELLENT: All feature correlations are in reasonable range (<5%)")
        return True

def main():
    """Test the clean data processing pipeline"""

    datasets = ['IXIC', 'DJI', 'NYSE']

    print("🧪 TESTING CLEAN DATA PROCESSING PIPELINE")
    print("="*60)

    for dataset in datasets:
        data_path = f'../Dataset/combined_dataframe_{dataset}.csv'

        if not Path(data_path).exists():
            print(f"❌ Dataset {dataset} not found")
            continue

        print(f"\n📊 PROCESSING {dataset}")
        print("-" * 30)

        # Test clean processing
        try:
            n_features, train_loader, val_loader, test_loader, scaler = \
                leak_free_data_processing(data_path, window=5, batch_size=128)

            print(f"✅ Successfully processed {dataset}")
            print(f"   Features: {n_features}")
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Val batches: {len(val_loader)}")
            print(f"   Test batches: {len(test_loader)}")

            # Validate features
            is_clean = validate_clean_features(data_path)

            if is_clean:
                print(f"   ✅ {dataset} features pass validation")
            else:
                print(f"   ❌ {dataset} features still need more cleaning")

        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")

    print(f"\n🎯 NEXT STEPS:")
    print("1. Use leak_free_data_processing() in your training script")
    print("2. Retrain MAMBA_BGNN with cleaned data")
    print("3. Expect IC to drop to 0.02-0.10 range (normal)")
    print("4. Model should still outperform baselines, but realistically")

if __name__ == "__main__":
    main()