#!/usr/bin/env python3
"""
Data Leakage Analysis for MAMBA_BGNN

This script investigates potential data leakage issues that could explain
the abnormally high IC/RIC values (~0.90+).
"""

import numpy as np
import csv
from pathlib import Path

class MinMax01:
    def fit(self, x):
        self.min = x.min(0)
        self.max = x.max(0)
    def transform(self, x):
        return (x - self.min) / (self.max - self.min + 1e-8)

def load_raw_data(data_path):
    """Load raw CSV data"""
    print(f"Loading data from: {data_path}")

    data = []
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Extract price and features (skip Date and Name)
    prices = [float(row['Price']) for row in data]

    # Get all feature columns (exclude Date, Price, Name)
    feature_cols = [k for k in data[0].keys() if k not in ['Date', 'Price', 'Name']]

    features = []
    for row in data:
        feat_row = []
        for col in feature_cols:
            try:
                feat_row.append(float(row[col]))
            except (ValueError, TypeError):
                feat_row.append(0.0)  # Handle NaN/missing
        features.append(feat_row)

    return np.array(prices), np.array(features), feature_cols

def original_data_processing(prices, features, window=5):
    """Replicate the original data processing logic"""
    T, Fdim = len(prices), features.shape[1]

    # Original split calculation (SUSPICIOUS!)
    train_len = int(0.80 * (T - window))  # Based on raw time series length

    # Fit scaler on overlapping windows from training period
    train_feat_matrix = []
    for i in range(train_len):  # Only first train_len samples
        train_feat_matrix.append(features[i:i+window])
    train_feat_matrix = np.concatenate(train_feat_matrix, axis=0)

    scaler = MinMax01()
    scaler.fit(train_feat_matrix)
    print(f"Scaler fitted on: {train_feat_matrix.shape} samples from overlapping windows")

    # Build samples
    X_list, Y_list = [], []
    for i in range(T - window):
        # Target: return from t-1 to t
        price_prev = prices[i+window-1]  # t-1
        price_cur = prices[i+window]     # t
        ret = (price_cur - price_prev) / price_prev

        # Features: t-window to t-1 (should be clean)
        feat_block = scaler.transform(features[i:i+window])

        X_list.append(feat_block)
        Y_list.append(ret)

    XX = np.array(X_list)  # (num_samples, window, n_features)
    YY = np.array(Y_list)  # (num_samples,)

    # Final split
    num_samples = len(XX)
    train_len_final = int(0.80 * num_samples)
    val_len_final = int(0.05 * num_samples)

    X_test = XX[-int(0.15 * num_samples):]  # Last 15%
    Y_test = YY[-int(0.15 * num_samples):]

    return X_test, Y_test, scaler

def correct_data_processing(prices, features, window=5):
    """Implement leak-free data processing"""
    T = len(prices)

    # Build samples first
    X_list, Y_list = [], []
    for i in range(T - window):
        price_prev = prices[i+window-1]  # t-1
        price_cur = prices[i+window]     # t
        ret = (price_cur - price_prev) / price_prev

        feat_block = features[i:i+window]  # Raw features, unscaled

        X_list.append(feat_block)
        Y_list.append(ret)

    XX = np.array(X_list)  # (num_samples, window, n_features)
    YY = np.array(Y_list)  # (num_samples,)

    # Split BEFORE scaling
    num_samples = len(XX)
    train_len = int(0.80 * num_samples)

    X_train = XX[:train_len]
    X_test = XX[-int(0.15 * num_samples):]
    Y_test = YY[-int(0.15 * num_samples):]

    # Fit scaler ONLY on training data
    train_feat_matrix = X_train.reshape(-1, XX.shape[-1])
    scaler = MinMax01()
    scaler.fit(train_feat_matrix)
    print(f"CORRECTED: Scaler fitted on {train_feat_matrix.shape} training samples only")

    # Apply scaling
    X_test_scaled = []
    for i in range(len(X_test)):
        X_test_scaled.append(scaler.transform(X_test[i]))
    X_test_scaled = np.array(X_test_scaled)

    return X_test_scaled, Y_test, scaler

def create_baseline_predictions(X_test, Y_test):
    """Create simple baseline predictions"""
    n_test = len(Y_test)

    baselines = {}

    # 1. Zero prediction
    baselines['zero'] = np.zeros(n_test)

    # 2. Lag-1 prediction (previous return)
    lag1_pred = np.zeros(n_test)
    # Approximate lag-1 return from features (if available)
    # For simplicity, use zero for first prediction
    for i in range(1, n_test):
        lag1_pred[i] = Y_test[i-1]  # Use previous actual return as prediction
    baselines['lag1'] = lag1_pred

    # 3. Random noise
    np.random.seed(42)
    baselines['random'] = np.random.normal(0, np.std(Y_test), n_test)

    # 4. Mean reversion
    mean_return = np.mean(Y_test[:int(0.5*n_test)])  # Use first half mean
    baselines['mean_revert'] = np.full(n_test, mean_return)

    return baselines

def evaluate_predictions(y_true, y_pred, name="Model"):
    """Evaluate prediction performance"""
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) < 2:
        return {"name": name, "ic": 0, "ric": 0, "rmse": float('inf')}

    # IC (Pearson correlation)
    if np.std(y_true_clean) > 1e-8 and np.std(y_pred_clean) > 1e-8:
        ic = np.corrcoef(y_true_clean, y_pred_clean)[0,1]
    else:
        ic = 0

    # RIC (Rank correlation)
    true_ranks = np.argsort(np.argsort(y_true_clean))
    pred_ranks = np.argsort(np.argsort(y_pred_clean))

    if np.std(true_ranks) > 1e-8 and np.std(pred_ranks) > 1e-8:
        ric = np.corrcoef(true_ranks, pred_ranks)[0,1]
    else:
        ric = 0

    # RMSE
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))

    return {"name": name, "ic": ic, "ric": ric, "rmse": rmse, "n_samples": len(y_true_clean)}

def load_model_predictions():
    """Load the actual model predictions for comparison"""
    assets = ['IXIC', 'DJI', 'NYSE']
    predictions = {}

    for asset in assets:
        asset_dirs = list(Path('logs').glob(f'FDSE25_{asset}_log*'))
        if asset_dirs:
            pred_file = asset_dirs[0] / 'test_predictions.csv'
            with open(pred_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                predictions[asset] = {
                    'y_true': np.array([float(row['y']) for row in rows]),
                    'y_pred': np.array([float(row['mu']) for row in rows])
                }

    return predictions

def main():
    """Main analysis pipeline"""
    print("🔍 DATA LEAKAGE ANALYSIS FOR MAMBA_BGNN")
    print("="*60)

    # Analyze each dataset
    datasets = ['IXIC', 'DJI', 'NYSE']

    all_results = []

    for dataset in datasets:
        print(f"\n📊 ANALYZING {dataset}")
        print("-" * 40)

        # Load raw data
        data_path = f'Dataset/combined_dataframe_{dataset}.csv'
        if not Path(data_path).exists():
            print(f"❌ Data file not found: {data_path}")
            continue

        prices, features, feature_cols = load_raw_data(data_path)
        print(f"Loaded: {len(prices)} days, {len(feature_cols)} features")

        # Original processing (potentially leaky)
        print("\n1️⃣ ORIGINAL PROCESSING:")
        X_test_orig, Y_test_orig, scaler_orig = original_data_processing(prices, features)

        # Corrected processing
        print("\n2️⃣ CORRECTED PROCESSING:")
        X_test_corr, Y_test_corr, scaler_corr = correct_data_processing(prices, features)

        print(f"Test samples: Original={len(Y_test_orig)}, Corrected={len(Y_test_corr)}")

        # Create baseline predictions for corrected data
        baselines = create_baseline_predictions(X_test_corr, Y_test_corr)

        # Evaluate baselines
        print(f"\n3️⃣ BASELINE PERFORMANCE ON {dataset}:")
        baseline_results = []
        for name, pred in baselines.items():
            result = evaluate_predictions(Y_test_corr, pred, name)
            baseline_results.append(result)
            print(f"  {name:12s}: IC={result['ic']:7.4f}, RIC={result['ric']:7.4f}, RMSE={result['rmse']:.6f}")

        all_results.extend([(dataset, r) for r in baseline_results])

    # Load and compare model predictions
    print(f"\n4️⃣ MODEL PERFORMANCE COMPARISON:")
    print("-" * 40)

    model_preds = load_model_predictions()
    for asset, preds in model_preds.items():
        result = evaluate_predictions(preds['y_true'], preds['y_pred'], f"MAMBA_{asset}")
        print(f"  {result['name']:15s}: IC={result['ic']:7.4f}, RIC={result['ric']:7.4f}, RMSE={result['rmse']:.6f}")
        all_results.append((asset, result))

    # Summary analysis
    print(f"\n📋 SUMMARY & DIAGNOSIS:")
    print("="*60)

    # Check if any baseline achieves high IC
    baseline_ics = [r[1]['ic'] for r in all_results if not r[1]['name'].startswith('MAMBA')]
    model_ics = [r[1]['ic'] for r in all_results if r[1]['name'].startswith('MAMBA')]

    max_baseline_ic = max(baseline_ics) if baseline_ics else 0
    min_model_ic = min(model_ics) if model_ics else 0

    print(f"Highest Baseline IC: {max_baseline_ic:.4f}")
    print(f"Lowest Model IC:     {min_model_ic:.4f}")

    if min_model_ic > 0.5:
        print("\n🚨 CRITICAL ISSUE: Model IC still extremely high (>0.5)")
        print("   Likely causes:")
        print("   • Fundamental data leakage in feature construction")
        print("   • Look-ahead bias in technical indicators")
        print("   • Target calculation errors")
    elif min_model_ic > max_baseline_ic + 0.2:
        print(f"\n⚠️  SUSPICIOUS: Model IC much higher than baselines (+{min_model_ic - max_baseline_ic:.3f})")
        print("   • May indicate sophisticated leakage or overfitting")
    else:
        print(f"\n✅ REASONABLE: Model IC close to baseline range")

    if max_baseline_ic > 0.1:
        print(f"\n🔥 BASELINE LEAKAGE: Even simple baselines achieve IC > 0.1")
        print("   • Data preprocessing issues")
        print("   • Feature construction problems")

    print(f"\n✅ Analysis completed!")

if __name__ == "__main__":
    main()