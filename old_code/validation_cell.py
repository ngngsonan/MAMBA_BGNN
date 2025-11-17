# ============================================================================
# COPY THIS CELL TO YOUR JUPYTER NOTEBOOK - Data Validation for MAMBA_BGNN
# ============================================================================

def validate_dataset(dataset_name='IXIC', window=5, verbose=True):
    """
    Validate a single dataset for data quality and potential issues

    Args:
        dataset_name: 'IXIC', 'DJI', or 'NYSE'
        window: lookback window size (default: 5)
        verbose: print detailed results

    Returns:
        dict with validation results
    """
    import numpy as np
    import pandas as pd
    import torch

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATING: {dataset_name}")
        print(f"{'='*60}")

    # Scaler class
    class MinMax01:
        def fit(self, x):
            self.min = x.min(0)
            self.max = x.max(0)
        def transform(self, x):
            return (x - self.min) / (self.max - self.min + 1e-8)

    # Load data
    data_path = f'Dataset/combined_dataframe_{dataset_name}.csv'
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    _ = df.pop('Name')

    # Remove leaky features
    leaky_features = ['mom', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20']
    for feat in leaky_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])

    if verbose:
        print(f"✅ Loaded: {df.shape[0]} days, {df.shape[1]} columns (Price + {df.shape[1]-1} features)")

    # Handle NaN
    df.fillna(df.median(), inplace=True)
    df.dropna(inplace=True)

    # Process data
    raw_np = df.values.astype('float32')
    prices = raw_np[:, 0]
    features = raw_np[:, 1:]
    T = len(prices)

    # Fit scaler on training data
    train_len = int(0.80 * (T - window))
    train_feat_matrix = []
    for i in range(train_len):
        train_feat_matrix.append(features[i:i+window])
    train_feat_matrix = np.concatenate(train_feat_matrix, axis=0)

    scaler = MinMax01()
    scaler.fit(train_feat_matrix)

    # Build samples
    X_list, Y_list = [], []
    for i in range(T - window):
        price_prev = prices[i+window-1]
        price_cur = prices[i+window]
        ret = (price_cur - price_prev) / price_prev

        feat_block = scaler.transform(features[i:i+window])
        X_list.append(feat_block)
        Y_list.append(ret)

    X = np.array(X_list)  # (samples, window, features)
    Y = np.array(Y_list)  # (samples,)

    # Split data
    num_samples = len(X)
    train_len = int(0.80 * num_samples)
    val_len = int(0.05 * num_samples)
    test_len = num_samples - train_len - val_len

    X_train, Y_train = X[:train_len], Y[:train_len]
    X_val, Y_val = X[train_len:train_len+val_len], Y[train_len:train_len+val_len]
    X_test, Y_test = X[-test_len:], Y[-test_len:]

    # ========== VALIDATION CHECKS ==========
    results = {'dataset': dataset_name, 'all_passed': True, 'checks': {}}

    if verbose:
        print(f"\n{'='*60}")
        print("VALIDATION CHECKS")
        print(f"{'='*60}")

    # Check 1: Data splits
    if verbose:
        print(f"\n✓ CHECK 1: Data Splits")
        print(f"   Train: {len(X_train)} ({len(X_train)/num_samples:.1%})")
        print(f"   Val:   {len(X_val)} ({len(X_val)/num_samples:.1%})")
        print(f"   Test:  {len(X_test)} ({len(X_test)/num_samples:.1%})")

    split_ok = (len(X_train) >= 500 and len(X_val) >= 50 and len(X_test) >= 100)
    results['checks']['sufficient_samples'] = split_ok

    # Check 2: Feature quality
    if verbose:
        print(f"\n✓ CHECK 2: Feature Quality")

    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()
    scaling_ok = -0.1 <= X.min() and X.max() <= 1.1

    if verbose:
        status = "✅" if not (has_nan or has_inf) else "❌"
        print(f"   {status} No NaN/Inf: {not (has_nan or has_inf)}")
        status = "✅" if scaling_ok else "❌"
        print(f"   {status} Scaled properly: [{X.min():.3f}, {X.max():.3f}]")

    results['checks']['no_nan_inf'] = not (has_nan or has_inf)
    results['checks']['proper_scaling'] = scaling_ok

    # Check 3: Target quality
    if verbose:
        print(f"\n✓ CHECK 3: Return Quality")
        print(f"   Train returns: mean={Y_train.mean():.6f}, std={Y_train.std():.6f}")
        print(f"   Test returns:  mean={Y_test.mean():.6f}, std={Y_test.std():.6f}")

    reasonable_range = (abs(Y_train.min()) <= 0.20 and Y_train.max() <= 0.20)
    results['checks']['reasonable_returns'] = reasonable_range

    # Check 4: Data Leakage Detection
    if verbose:
        print(f"\n✓ CHECK 4: Data Leakage Detection")

    # Feature-target correlations
    X_test_flat = X_test.reshape(len(X_test), -1)
    max_corr = 0
    high_corr_count = 0

    for feat_idx in range(X_test_flat.shape[1]):
        feat_values = X_test_flat[:, feat_idx]
        if np.std(feat_values) > 1e-8:
            corr = np.corrcoef(feat_values, Y_test)[0, 1]
            if not np.isnan(corr):
                abs_corr = abs(corr)
                max_corr = max(max_corr, abs_corr)
                if abs_corr > 0.15:
                    high_corr_count += 1

    leakage_ok = max_corr < 0.15

    if verbose:
        status = "✅" if leakage_ok else "❌"
        print(f"   {status} Max feature-target correlation: {max_corr:.4f} (threshold: 0.15)")
        if high_corr_count > 0:
            print(f"   ⚠️  Warning: {high_corr_count} features with |corr| > 0.15")

    results['checks']['no_leakage'] = leakage_ok
    results['stats'] = {
        'max_corr': float(max_corr),
        'high_corr_features': int(high_corr_count),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test)
    }

    # Overall result
    results['all_passed'] = all(results['checks'].values())

    if verbose:
        print(f"\n{'='*60}")
        if results['all_passed']:
            print(f"✅ {dataset_name}: ALL CHECKS PASSED")
        else:
            print(f"❌ {dataset_name}: SOME CHECKS FAILED")
            for check, passed in results['checks'].items():
                if not passed:
                    print(f"   ❌ {check}")
        print(f"{'='*60}")

    return results


# ============================================================================
# RUN VALIDATION ON MULTIPLE DATASETS
# ============================================================================

def validate_all_datasets(datasets=['IXIC', 'DJI', 'NYSE'], window=5):
    """
    Validate multiple datasets and print summary

    Args:
        datasets: list of dataset names to validate
        window: lookback window size

    Returns:
        list of validation results
    """
    print("="*60)
    print("MAMBA_BGNN - Data Validation")
    print("="*60)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Window: {window}")

    all_results = []

    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"Dataset {i}/{len(datasets)}: {dataset}")
        print(f"{'#'*60}")

        try:
            result = validate_dataset(dataset, window=window, verbose=True)
            all_results.append(result)
        except FileNotFoundError:
            print(f"❌ Dataset file not found for {dataset}")
            all_results.append({'dataset': dataset, 'all_passed': False, 'error': 'File not found'})
        except Exception as e:
            print(f"❌ Error: {e}")
            all_results.append({'dataset': dataset, 'all_passed': False, 'error': str(e)})

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for result in all_results:
        dataset = result['dataset']
        if 'error' in result:
            print(f"❌ {dataset:10s}: ERROR - {result['error']}")
        elif result['all_passed']:
            print(f"✅ {dataset:10s}: PASSED")
        else:
            print(f"❌ {dataset:10s}: FAILED")
            if 'checks' in result:
                for check, passed in result['checks'].items():
                    if not passed:
                        print(f"      ❌ {check}")

    passed_count = sum(1 for r in all_results if r.get('all_passed', False))
    print(f"\n{'='*60}")
    print(f"Overall: {passed_count}/{len(all_results)} datasets passed")
    print(f"{'='*60}")

    return all_results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Example 1: Validate single dataset
# result = validate_dataset('IXIC', window=5)

# Example 2: Validate all 3 datasets
# results = validate_all_datasets(['IXIC', 'DJI', 'NYSE'], window=5)

# Example 3: Quick check without verbose output
# result = validate_dataset('IXIC', window=5, verbose=False)
# print(f"Passed: {result['all_passed']}")
