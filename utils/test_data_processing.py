# ============================================================================
# TEST DATA PROCESSING - Copy vào Jupyter Notebook Cell
# ============================================================================
# File này test data processing từ utils/data_processing.py chính
# Chấp nhận feature range ngoài [0,1] như một alert (không phải error)
# ============================================================================

def test_data_processing(dataset='IXIC', window=5, batch_size=32, verbose=True):
    """
    Test data processing pipeline từ file chính utils/data_processing.py

    Args:
        dataset: dataset name ('IXIC', 'DJI', 'NYSE')
        window: lookback window size
        batch_size: batch size for dataloaders
        verbose: print detailed information

    Returns:
        dict với thông tin về processed data
    """
    import sys
    import os
    import torch
    import numpy as np

    # Import from main data processing file
    sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
    from utils.data_processing import data_processing

    if verbose:
        print(f"\n{'='*70}")
        print(f"TESTING DATA PROCESSING: {dataset}")
        print(f"{'='*70}")
        print(f"Dataset: {dataset}")
        print(f"Window: {window}")
        print(f"Batch size: {batch_size}\n")

    # Call main data processing function
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'

    try:
        num_features, train_loader, val_loader, test_loader = data_processing(
            data_path=data_path,
            window=window,
            batch_size=batch_size
        )

        if verbose:
            print(f"\n{'='*70}")
            print("DATA PROCESSING SUCCESSFUL")
            print(f"{'='*70}")
            print(f"✅ Number of features: {num_features}")

        # Extract all data for analysis
        X_train_list, Y_train_list = [], []
        for X, Y in train_loader:
            X_train_list.append(X)
            Y_train_list.append(Y)
        X_train = torch.cat(X_train_list, dim=0)
        Y_train = torch.cat(Y_train_list, dim=0)

        X_val_list, Y_val_list = [], []
        for X, Y in val_loader:
            X_val_list.append(X)
            Y_val_list.append(Y)
        X_val = torch.cat(X_val_list, dim=0)
        Y_val = torch.cat(Y_val_list, dim=0)

        X_test_list, Y_test_list = [], []
        for X, Y in test_loader:
            X_test_list.append(X)
            Y_test_list.append(Y)
        X_test = torch.cat(X_test_list, dim=0)
        Y_test = torch.cat(Y_test_list, dim=0)

        # Analyze data
        if verbose:
            print(f"\n{'='*70}")
            print("DATA ANALYSIS")
            print(f"{'='*70}")

            # Sample counts
            total_samples = len(X_train) + len(X_val) + len(X_test)
            print(f"\n📊 SAMPLE COUNTS:")
            print(f"   Train: {len(X_train):5d} ({len(X_train)/total_samples:6.1%})")
            print(f"   Val:   {len(X_val):5d} ({len(X_val)/total_samples:6.1%})")
            print(f"   Test:  {len(X_test):5d} ({len(X_test)/total_samples:6.1%})")
            print(f"   Total: {total_samples:5d}")

            # Feature dimensions
            print(f"\n📐 FEATURE DIMENSIONS:")
            print(f"   Shape: (batch, window, features) = {X_train.shape}")
            print(f"   Window size (L): {X_train.shape[1]}")
            print(f"   Num features (N): {X_train.shape[2]}")

            # Feature range (ALERT only, not error)
            X_all = torch.cat([X_train, X_val, X_test], dim=0)
            feat_min = X_all.min().item()
            feat_max = X_all.max().item()

            print(f"\n📏 FEATURE RANGE:")
            print(f"   Min: {feat_min:7.4f}")
            print(f"   Max: {feat_max:7.4f}")

            if feat_min < -0.05 or feat_max > 1.05:
                print(f"   ⚠️  ALERT: Features outside [0,1] range")
                print(f"   ℹ️  This is acceptable - may be due to test set having values outside training range")
            else:
                print(f"   ✅ Features well-scaled")

            # Data quality
            print(f"\n🔍 DATA QUALITY:")
            has_nan = torch.isnan(X_all).any().item() or torch.isnan(Y_train).any().item()
            has_inf = torch.isinf(X_all).any().item() or torch.isinf(Y_train).any().item()

            print(f"   NaN values: {'❌ Found' if has_nan else '✅ None'}")
            print(f"   Inf values: {'❌ Found' if has_inf else '✅ None'}")

            # Target statistics
            Y_all = torch.cat([Y_train, Y_val, Y_test], dim=0).squeeze()

            print(f"\n📈 TARGET STATISTICS (Returns):")
            print(f"   Train:")
            print(f"      Mean:  {Y_train.mean().item():9.6f}")
            print(f"      Std:   {Y_train.std().item():9.6f}")
            print(f"      Range: [{Y_train.min().item():8.4f}, {Y_train.max().item():8.4f}]")

            print(f"   Test:")
            print(f"      Mean:  {Y_test.mean().item():9.6f}")
            print(f"      Std:   {Y_test.std().item():9.6f}")
            print(f"      Range: [{Y_test.min().item():8.4f}, {Y_test.max().item():8.4f}]")

            # Feature-target correlation check
            print(f"\n🔬 LEAKAGE CHECK:")
            X_test_flat = X_test.reshape(len(X_test), -1).numpy()
            Y_test_np = Y_test.squeeze().numpy()

            max_corr = 0
            high_corr_count = 0

            for i in range(X_test_flat.shape[1]):
                feat_vals = X_test_flat[:, i]
                if np.std(feat_vals) > 1e-8:
                    corr = abs(np.corrcoef(feat_vals, Y_test_np)[0, 1])
                    if not np.isnan(corr):
                        max_corr = max(max_corr, corr)
                        if corr > 0.15:
                            high_corr_count += 1

            if max_corr < 0.15:
                status = "✅ PASS"
                emoji = "🎉"
            elif max_corr < 0.30:
                status = "⚠️  WARNING"
                emoji = "⚠️"
            else:
                status = "❌ FAIL"
                emoji = "🚨"

            print(f"   Max feature-target correlation: {max_corr:.4f} {status}")
            print(f"   Features with |corr| > 0.15: {high_corr_count}")

            # Overall summary
            print(f"\n{'='*70}")
            if max_corr < 0.15 and not has_nan and not has_inf:
                print(f"✅ DATA PROCESSING SUCCESSFUL - READY FOR TRAINING")
            elif max_corr < 0.30:
                print(f"⚠️  DATA PROCESSED - Minor issues detected")
            else:
                print(f"❌ DATA PROCESSED - Major issues detected")
            print(f"{'='*70}\n")

        # Return summary
        return {
            'success': True,
            'dataset': dataset,
            'num_features': num_features,
            'samples': {
                'train': len(X_train),
                'val': len(X_val),
                'test': len(X_test),
                'total': len(X_train) + len(X_val) + len(X_test)
            },
            'feature_range': {
                'min': feat_min,
                'max': feat_max
            },
            'data_quality': {
                'has_nan': has_nan,
                'has_inf': has_inf
            },
            'leakage_check': {
                'max_correlation': max_corr,
                'high_corr_count': high_corr_count,
                'passed': max_corr < 0.15
            },
            'loaders': {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader
            }
        }

    except Exception as e:
        if verbose:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

        return {
            'success': False,
            'dataset': dataset,
            'error': str(e)
        }


# ============================================================================
# TEST MULTIPLE DATASETS
# ============================================================================

def test_all_datasets(datasets=['IXIC', 'DJI', 'NYSE'], window=5, batch_size=32):
    """
    Test data processing cho nhiều datasets

    Args:
        datasets: list of dataset names
        window: lookback window size
        batch_size: batch size

    Returns:
        dict với kết quả cho tất cả datasets
    """
    print("="*70)
    print("TESTING DATA PROCESSING - ALL DATASETS")
    print("="*70)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Window: {window}, Batch size: {batch_size}\n")

    results = {}

    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'#'*70}")
        print(f"Dataset {i}/{len(datasets)}: {dataset}")
        print(f"{'#'*70}")

        result = test_data_processing(dataset, window, batch_size, verbose=True)
        results[dataset] = result

    # Print summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for dataset, result in results.items():
        if result['success']:
            leakage_status = "✅" if result['leakage_check']['passed'] else "❌"
            print(f"\n{dataset}:")
            print(f"   Samples: {result['samples']['total']}")
            print(f"   Features: {result['num_features']}")
            print(f"   Max correlation: {result['leakage_check']['max_correlation']:.4f} {leakage_status}")
        else:
            print(f"\n{dataset}: ❌ ERROR - {result['error']}")

    print(f"\n{'='*70}\n")

    return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Example 1: Test single dataset
# result = test_data_processing('IXIC', window=5, batch_size=32)

# Example 2: Test all datasets
# results = test_all_datasets(['IXIC', 'DJI', 'NYSE'], window=5, batch_size=32)

# Example 3: Quick test without verbose
# result = test_data_processing('IXIC', window=5, batch_size=32, verbose=False)
# print(f"Success: {result['success']}, Max corr: {result['leakage_check']['max_correlation']:.4f}")

# Example 4: Get loaders for training
# result = test_data_processing('IXIC', window=5, batch_size=32)
# if result['success']:
#     train_loader = result['loaders']['train_loader']
#     val_loader = result['loaders']['val_loader']
#     test_loader = result['loaders']['test_loader']
#     print("Ready to train!")
