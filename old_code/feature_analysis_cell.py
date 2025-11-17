# ============================================================================
# FEATURE ANALYSIS & REMOVAL - Copy vào cell mới trong notebook
# ============================================================================

def analyze_and_remove_bad_features(dataset='IXIC', window=5, corr_threshold=0.15):
    """
    Phân tích chi tiết features và tự động loại bỏ các features có vấn đề

    Args:
        dataset: dataset name ('IXIC', 'DJI', 'NYSE')
        window: lookback window
        corr_threshold: ngưỡng correlation để coi là leakage (default: 0.15)

    Returns:
        dict với danh sách features tốt và xấu
    """
    import numpy as np
    import pandas as pd

    print(f"\n{'='*70}")
    print(f"FEATURE ANALYSIS: {dataset}")
    print(f"{'='*70}")

    # Scaler
    class MinMax01:
        def fit(self, x):
            self.min, self.max = x.min(0), x.max(0)
        def transform(self, x):
            return (x - self.min) / (self.max - self.min + 1e-8)

    # Load data
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    _ = df.pop('Name')

    # Remove known leaky features
    initial_leaky = ['mom', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20']
    for feat in initial_leaky:
        if feat in df.columns:
            df = df.drop(columns=[feat])

    print(f"✅ Initial cleanup: {df.shape[1]} columns (Price + {df.shape[1]-1} features)")

    # Get feature names
    feature_names = [col for col in df.columns if col != 'Price']
    print(f"✅ Analyzing {len(feature_names)} features...\n")

    # Fill NaN
    df = df.fillna(df.median()).dropna()

    # Process
    raw_np = df.values.astype('float32')
    prices = raw_np[:, 0]
    features = raw_np[:, 1:]
    T = len(prices)

    # Fit scaler on training data
    train_len = int(0.80 * (T - window))
    train_feat_matrix = np.concatenate([features[i:i+window] for i in range(train_len)])

    scaler = MinMax01()
    scaler.fit(train_feat_matrix)

    # Build samples
    X = np.array([scaler.transform(features[i:i+window]) for i in range(T-window)])
    Y = np.array([(prices[i+window] - prices[i+window-1])/prices[i+window-1] for i in range(T-window)])

    # Split
    n = len(X)
    test_start = n - int(0.15 * n)
    X_test, Y_test = X[test_start:], Y[test_start:]

    # ========== ANALYZE EACH FEATURE ==========
    print(f"{'='*70}")
    print("FEATURE-TARGET CORRELATION ANALYSIS")
    print(f"{'='*70}")

    # Flatten features: (samples, window, features) -> analyze each feature across all timesteps
    feature_correlations = []

    for feat_idx in range(features.shape[1]):
        # Get this feature across all windows in test set
        feat_values_all_timesteps = X_test[:, :, feat_idx].flatten()  # All timesteps

        # Also check just the last timestep (most recent, most likely to leak)
        feat_values_last = X_test[:, -1, feat_idx]

        # Replicate Y to match flattened feature length
        Y_replicated = np.repeat(Y_test, window)

        # Calculate correlations
        if np.std(feat_values_all_timesteps) > 1e-8:
            corr_all = abs(np.corrcoef(feat_values_all_timesteps, Y_replicated)[0, 1])
        else:
            corr_all = 0.0

        if np.std(feat_values_last) > 1e-8:
            corr_last = abs(np.corrcoef(feat_values_last, Y_test)[0, 1])
        else:
            corr_last = 0.0

        # Use max correlation
        max_corr = max(corr_all, corr_last)

        feature_correlations.append({
            'feature_name': feature_names[feat_idx],
            'feature_idx': feat_idx,
            'corr_all_timesteps': corr_all,
            'corr_last_timestep': corr_last,
            'max_corr': max_corr,
            'is_bad': max_corr > corr_threshold
        })

    # Sort by correlation
    feature_correlations.sort(key=lambda x: x['max_corr'], reverse=True)

    # Print top problematic features
    print(f"\n🔍 TOP 20 FEATURES BY CORRELATION:")
    print(f"{'Rank':<6}{'Feature':<25}{'Corr(All)':<12}{'Corr(Last)':<12}{'Max':<12}{'Status'}")
    print(f"{'-'*70}")

    for i, fc in enumerate(feature_correlations[:20], 1):
        status = "❌ BAD" if fc['is_bad'] else "✅ OK"
        print(f"{i:<6}{fc['feature_name']:<25}{fc['corr_all_timesteps']:<12.4f}"
              f"{fc['corr_last_timestep']:<12.4f}{fc['max_corr']:<12.4f}{status}")

    # Separate good and bad features
    bad_features = [fc for fc in feature_correlations if fc['is_bad']]
    good_features = [fc for fc in feature_correlations if not fc['is_bad']]

    print(f"\n{'='*70}")
    print(f"📊 SUMMARY:")
    print(f"   Total features: {len(feature_correlations)}")
    print(f"   ❌ Bad features (|corr| > {corr_threshold}): {len(bad_features)}")
    print(f"   ✅ Good features: {len(good_features)}")
    print(f"{'='*70}")

    if bad_features:
        print(f"\n🚨 FEATURES TO REMOVE ({len(bad_features)}):")
        for fc in bad_features:
            print(f"   ❌ {fc['feature_name']:<30} (max corr: {fc['max_corr']:.4f})")

    return {
        'dataset': dataset,
        'all_features': feature_correlations,
        'bad_features': [fc['feature_name'] for fc in bad_features],
        'good_features': [fc['feature_name'] for fc in good_features],
        'bad_count': len(bad_features),
        'good_count': len(good_features)
    }


# ============================================================================
# TEST WITH CLEANED DATA
# ============================================================================

def test_with_cleaned_features(dataset='IXIC', window=5, features_to_remove=None):
    """
    Test validation sau khi loại bỏ bad features

    Args:
        dataset: dataset name
        window: lookback window
        features_to_remove: list of feature names to remove

    Returns:
        validation result
    """
    import numpy as np
    import pandas as pd

    if features_to_remove is None:
        features_to_remove = []

    print(f"\n{'='*70}")
    print(f"TESTING WITH CLEANED DATA: {dataset}")
    print(f"{'='*70}")
    print(f"Removing {len(features_to_remove)} features...")

    # Scaler
    class MinMax01:
        def fit(self, x):
            self.min, self.max = x.min(0), x.max(0)
        def transform(self, x):
            return (x - self.min) / (self.max - self.min + 1e-8)

    # Load data
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    _ = df.pop('Name')

    # Remove ALL bad features (initial + newly discovered)
    initial_leaky = ['mom', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20']
    all_to_remove = list(set(initial_leaky + features_to_remove))

    for feat in all_to_remove:
        if feat in df.columns:
            df = df.drop(columns=[feat])

    print(f"✅ Removed {len(all_to_remove)} features")
    print(f"✅ Remaining: {df.shape[1]-1} features\n")

    # Fill NaN
    df = df.fillna(df.median()).dropna()

    # Process
    raw_np = df.values.astype('float32')
    prices = raw_np[:, 0]
    features = raw_np[:, 1:]
    T = len(prices)

    # Fit scaler
    train_len = int(0.80 * (T - window))
    train_feat_matrix = np.concatenate([features[i:i+window] for i in range(train_len)])
    scaler = MinMax01()
    scaler.fit(train_feat_matrix)

    # Build samples
    X = np.array([scaler.transform(features[i:i+window]) for i in range(T-window)])
    Y = np.array([(prices[i+window] - prices[i+window-1])/prices[i+window-1] for i in range(T-window)])

    # Split
    n = len(X)
    test_start = n - int(0.15 * n)
    X_test, Y_test = X[test_start:], Y[test_start:]

    # Validation checks
    print(f"{'='*70}")
    print("VALIDATION RESULTS AFTER CLEANUP")
    print(f"{'='*70}")

    # Check scaling
    x_min, x_max = X.min(), X.max()
    scaling_ok = -0.05 <= x_min and x_max <= 1.05
    print(f"✓ Feature range: [{x_min:.3f}, {x_max:.3f}] - {'✅ OK' if scaling_ok else '❌ BAD'}")

    # Check leakage
    X_flat = X_test.reshape(len(X_test), -1)
    max_corr = 0
    high_corr_count = 0

    for i in range(X_flat.shape[1]):
        if np.std(X_flat[:, i]) > 1e-8:
            corr = abs(np.corrcoef(X_flat[:, i], Y_test)[0, 1])
            if not np.isnan(corr):
                max_corr = max(max_corr, corr)
                if corr > 0.15:
                    high_corr_count += 1

    leakage_ok = max_corr < 0.15
    print(f"✓ Max correlation: {max_corr:.4f} - {'✅ OK' if leakage_ok else '❌ STILL BAD'}")
    print(f"✓ High corr features: {high_corr_count}")

    # Returns
    print(f"✓ Test returns: mean={Y_test.mean():.6f}, std={Y_test.std():.6f}")

    print(f"{'='*70}")
    if scaling_ok and leakage_ok:
        print("🎉 SUCCESS: Data is now clean!")
    else:
        print("⚠️  WARNING: Still have issues, may need more cleanup")
    print(f"{'='*70}\n")

    return {
        'scaling_ok': scaling_ok,
        'leakage_ok': leakage_ok,
        'max_corr': max_corr,
        'remaining_features': features.shape[1],
        'high_corr_count': high_corr_count
    }


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def clean_dataset_pipeline(dataset='IXIC', window=5, corr_threshold=0.15):
    """
    Complete pipeline: analyze -> remove -> test

    Args:
        dataset: dataset name
        window: lookback window
        corr_threshold: correlation threshold

    Returns:
        dict with analysis and test results
    """
    # Step 1: Analyze
    analysis = analyze_and_remove_bad_features(dataset, window, corr_threshold)

    # Step 2: Test with cleaned data
    test_result = test_with_cleaned_features(dataset, window, analysis['bad_features'])

    return {
        'analysis': analysis,
        'test_result': test_result,
        'features_to_remove': analysis['bad_features']
    }


# ============================================================================
# USAGE
# ============================================================================

# Cách 1: Phân tích chi tiết
# analysis = analyze_and_remove_bad_features('IXIC', window=5, corr_threshold=0.15)

# Cách 2: Test với cleaned data
# test_result = test_with_cleaned_features('IXIC', window=5,
#                                          features_to_remove=['feature1', 'feature2'])

# Cách 3: Complete pipeline (RECOMMENDED)
# result = clean_dataset_pipeline('IXIC', window=5, corr_threshold=0.15)
# print(f"\nFeatures to remove: {result['features_to_remove']}")
