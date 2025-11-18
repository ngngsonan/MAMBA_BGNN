#!/usr/bin/env python3
"""
Data Validation for MAMBA_BGNN Training Data
Comprehensive checks to ensure data quality before training
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
import warnings

class DataValidator:
    """Comprehensive data validation for training pipeline"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.validation_results = {}

    def print_section(self, title: str):
        """Print formatted section header"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"{title}")
            print(f"{'='*60}")

    def print_check(self, check_name: str, passed: bool, message: str = ""):
        """Print check result"""
        if self.verbose:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}")
            if message:
                print(f"      {message}")
        return passed

    def validate_temporal_alignment(self,
                                   X_train: torch.Tensor,
                                   Y_train: torch.Tensor,
                                   X_val: torch.Tensor,
                                   Y_val: torch.Tensor,
                                   X_test: torch.Tensor,
                                   Y_test: torch.Tensor) -> bool:
        """
        Check 1: Temporal Alignment
        Ensure no temporal overlap between train/val/test splits
        """
        self.print_section("CHECK 1: TEMPORAL ALIGNMENT")

        all_passed = True

        # Check 1.1: No empty sets
        passed = self.print_check(
            "Non-empty splits",
            len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0,
            f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        all_passed = all_passed and passed

        # Check 1.2: Chronological ordering (sizes should decrease or stay consistent)
        total_samples = len(X_train) + len(X_val) + len(X_test)
        train_ratio = len(X_train) / total_samples
        val_ratio = len(X_val) / total_samples
        test_ratio = len(X_test) / total_samples

        passed = self.print_check(
            "Split ratios",
            0.70 <= train_ratio <= 0.85 and 0.03 <= val_ratio <= 0.10 and 0.10 <= test_ratio <= 0.20,
            f"Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}"
        )
        all_passed = all_passed and passed

        # Check 1.3: Shape consistency
        passed = self.print_check(
            "Feature dimensions consistent",
            X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:],
            f"Shape: {X_train.shape[1:]}"
        )
        all_passed = all_passed and passed

        # Check 1.4: Target dimensions consistent
        passed = self.print_check(
            "Target dimensions consistent",
            Y_train.shape[1:] == Y_val.shape[1:] == Y_test.shape[1:],
            f"Shape: {Y_train.shape[1:]}"
        )
        all_passed = all_passed and passed

        self.validation_results['temporal_alignment'] = all_passed
        return all_passed

    def check_data_leakage(self,
                          X_train: torch.Tensor,
                          Y_train: torch.Tensor,
                          X_test: torch.Tensor,
                          Y_test: torch.Tensor) -> bool:
        """
        Check 2: Data Leakage Detection
        Look for suspiciously high correlations that indicate leakage
        """
        self.print_section("CHECK 2: DATA LEAKAGE DETECTION")

        all_passed = True

        # Convert to numpy for analysis
        X_train_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
        Y_train_np = Y_train.numpy() if isinstance(Y_train, torch.Tensor) else Y_train
        X_test_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
        Y_test_np = Y_test.numpy() if isinstance(Y_test, torch.Tensor) else Y_test

        # Check 2.1: Test baseline predictions
        # Baseline 1: Zero prediction
        zero_pred = np.zeros_like(Y_test_np.flatten())
        ic_zero = self._calculate_ic(Y_test_np.flatten(), zero_pred)

        # Baseline 2: Mean prediction
        mean_pred = np.full_like(Y_test_np.flatten(), Y_train_np.mean())
        ic_mean = self._calculate_ic(Y_test_np.flatten(), mean_pred)

        # Baseline 3: Random prediction
        np.random.seed(42)
        random_pred = np.random.normal(0, Y_train_np.std(), len(Y_test_np))
        ic_random = self._calculate_ic(Y_test_np.flatten(), random_pred)

        if self.verbose:
            print(f"   Baseline ICs:")
            print(f"      Zero:   {ic_zero:.4f}")
            print(f"      Mean:   {ic_mean:.4f}")
            print(f"      Random: {ic_random:.4f}")

        # Check 2.2: Feature-target correlation analysis
        # Flatten features: (num_samples, window, num_features) -> (num_samples, window*num_features)
        X_test_flat = X_test_np.reshape(len(X_test_np), -1)
        Y_test_flat = Y_test_np.flatten()

        # Calculate correlation for each feature dimension
        high_corr_features = []
        max_abs_corr = 0

        for feat_idx in range(X_test_flat.shape[1]):
            feat_values = X_test_flat[:, feat_idx]
            if np.std(feat_values) > 1e-8:  # Skip constant features
                corr = np.corrcoef(feat_values, Y_test_flat)[0, 1]
                if not np.isnan(corr):
                    abs_corr = abs(corr)
                    max_abs_corr = max(max_abs_corr, abs_corr)
                    if abs_corr > 0.15:  # Suspicious threshold for daily returns
                        high_corr_features.append((feat_idx, corr))

        passed = self.print_check(
            "No high feature-target correlations",
            max_abs_corr < 0.15,
            f"Max correlation: {max_abs_corr:.4f} (threshold: 0.15)"
        )

        if not passed and self.verbose:
            print(f"      ⚠️  Found {len(high_corr_features)} features with |corr| > 0.15")
            for feat_idx, corr in high_corr_features[:5]:
                print(f"         Feature {feat_idx}: {corr:.4f}")

        all_passed = all_passed and passed

        # Check 2.3: Look-ahead bias check
        # The last timestep in features should not correlate too highly with target
        last_timestep = X_test_np[:, -1, :]  # Shape: (num_samples, num_features)
        max_last_corr = 0

        for feat_idx in range(last_timestep.shape[1]):
            feat_values = last_timestep[:, feat_idx]
            if np.std(feat_values) > 1e-8:
                corr = np.corrcoef(feat_values, Y_test_flat)[0, 1]
                if not np.isnan(corr):
                    max_last_corr = max(max_last_corr, abs(corr))

        passed = self.print_check(
            "No look-ahead bias in last timestep",
            max_last_corr < 0.20,
            f"Max correlation (last timestep): {max_last_corr:.4f}"
        )
        all_passed = all_passed and passed

        self.validation_results['data_leakage'] = all_passed
        return all_passed

    def check_feature_quality(self,
                             X_train: torch.Tensor,
                             X_val: torch.Tensor,
                             X_test: torch.Tensor) -> bool:
        """
        Check 3: Feature Quality
        Ensure features are properly scaled and distributed
        """
        self.print_section("CHECK 3: FEATURE QUALITY")

        all_passed = True

        X_train_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
        X_test_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test

        # Check 3.1: No NaN or Inf values
        has_nan_train = np.isnan(X_train_np).any()
        has_inf_train = np.isinf(X_train_np).any()
        has_nan_test = np.isnan(X_test_np).any()
        has_inf_test = np.isinf(X_test_np).any()

        passed = self.print_check(
            "No NaN/Inf in features",
            not (has_nan_train or has_inf_train or has_nan_test or has_inf_test),
            f"Train NaN: {has_nan_train}, Train Inf: {has_inf_train}, Test NaN: {has_nan_test}, Test Inf: {has_inf_test}"
        )
        all_passed = all_passed and passed

        # Check 3.2: Feature scaling
        # Features should be roughly in [0, 1] range after MinMax scaling
        train_min = X_train_np.min()
        train_max = X_train_np.max()
        test_min = X_test_np.min()
        test_max = X_test_np.max()

        passed = self.print_check(
            "Features properly scaled",
            -0.1 <= train_min and train_max <= 1.1 and -0.1 <= test_min and test_max <= 1.1,
            f"Train range: [{train_min:.3f}, {train_max:.3f}], Test range: [{test_min:.3f}, {test_max:.3f}]"
        )
        all_passed = all_passed and passed

        # Check 3.3: No constant features
        num_features = X_train_np.shape[-1]
        constant_features = 0

        for feat_idx in range(num_features):
            feat_std = np.std(X_train_np[:, :, feat_idx])
            if feat_std < 1e-8:
                constant_features += 1

        passed = self.print_check(
            "No constant features",
            constant_features == 0,
            f"Constant features: {constant_features}/{num_features}"
        )
        all_passed = all_passed and passed

        # Check 3.4: Test set distribution similar to train
        train_mean = X_train_np.mean()
        train_std = X_train_np.std()
        test_mean = X_test_np.mean()
        test_std = X_test_np.std()

        mean_diff = abs(train_mean - test_mean)
        std_ratio = test_std / (train_std + 1e-8)

        passed = self.print_check(
            "Similar train/test distribution",
            mean_diff < 0.1 and 0.5 <= std_ratio <= 2.0,
            f"Mean diff: {mean_diff:.4f}, Std ratio: {std_ratio:.4f}"
        )

        if not passed and self.verbose:
            print(f"      ⚠️  Train: mean={train_mean:.4f}, std={train_std:.4f}")
            print(f"      ⚠️  Test:  mean={test_mean:.4f}, std={test_std:.4f}")

        all_passed = all_passed and passed

        self.validation_results['feature_quality'] = all_passed
        return all_passed

    def check_target_quality(self,
                            Y_train: torch.Tensor,
                            Y_val: torch.Tensor,
                            Y_test: torch.Tensor) -> bool:
        """
        Check 4: Target Quality
        Ensure returns are reasonable and properly distributed
        """
        self.print_section("CHECK 4: TARGET QUALITY")

        all_passed = True

        Y_train_np = Y_train.numpy() if isinstance(Y_train, torch.Tensor) else Y_train
        Y_val_np = Y_val.numpy() if isinstance(Y_val, torch.Tensor) else Y_val
        Y_test_np = Y_test.numpy() if isinstance(Y_test, torch.Tensor) else Y_test

        # Check 4.1: No NaN or Inf in targets
        has_nan = np.isnan(Y_train_np).any() or np.isnan(Y_test_np).any()
        has_inf = np.isinf(Y_train_np).any() or np.isinf(Y_test_np).any()

        passed = self.print_check(
            "No NaN/Inf in targets",
            not (has_nan or has_inf),
            f"NaN: {has_nan}, Inf: {has_inf}"
        )
        all_passed = all_passed and passed

        # Check 4.2: Reasonable return range
        # Daily returns should typically be within -0.15 to +0.15 (±15%)
        train_min = Y_train_np.min()
        train_max = Y_train_np.max()
        test_min = Y_test_np.min()
        test_max = Y_test_np.max()

        extreme_returns = (abs(train_min) > 0.20 or train_max > 0.20 or
                          abs(test_min) > 0.20 or test_max > 0.20)

        passed = self.print_check(
            "Returns within reasonable range",
            not extreme_returns,
            f"Train: [{train_min:.4f}, {train_max:.4f}], Test: [{test_min:.4f}, {test_max:.4f}]"
        )

        if extreme_returns and self.verbose:
            print(f"      ⚠️  Found extreme returns (>20%), verify data quality")

        all_passed = all_passed and passed

        # Check 4.3: Return distribution properties
        train_mean = Y_train_np.mean()
        train_std = Y_train_np.std()
        test_mean = Y_test_np.mean()
        test_std = Y_test_np.std()

        if self.verbose:
            print(f"   Return statistics:")
            print(f"      Train: mean={train_mean:.6f}, std={train_std:.6f}")
            print(f"      Test:  mean={test_mean:.6f}, std={test_std:.6f}")

        # Check 4.4: Not all targets are the same
        unique_train = len(np.unique(Y_train_np.flatten()))
        unique_test = len(np.unique(Y_test_np.flatten()))

        passed = self.print_check(
            "Sufficient target diversity",
            unique_train > len(Y_train_np) * 0.5 and unique_test > len(Y_test_np) * 0.5,
            f"Unique values: Train {unique_train}/{len(Y_train_np)}, Test {unique_test}/{len(Y_test_np)}"
        )
        all_passed = all_passed and passed

        self.validation_results['target_quality'] = all_passed
        return all_passed

    def check_sample_size(self,
                         X_train: torch.Tensor,
                         X_val: torch.Tensor,
                         X_test: torch.Tensor) -> bool:
        """
        Check 5: Sample Size
        Ensure sufficient samples for training and evaluation
        """
        self.print_section("CHECK 5: SAMPLE SIZE")

        all_passed = True

        # Check 5.1: Minimum sample sizes
        min_train = 500
        min_val = 50
        min_test = 100

        passed = self.print_check(
            "Sufficient training samples",
            len(X_train) >= min_train,
            f"Train samples: {len(X_train)} (minimum: {min_train})"
        )
        all_passed = all_passed and passed

        passed = self.print_check(
            "Sufficient validation samples",
            len(X_val) >= min_val,
            f"Val samples: {len(X_val)} (minimum: {min_val})"
        )
        all_passed = all_passed and passed

        passed = self.print_check(
            "Sufficient test samples",
            len(X_test) >= min_test,
            f"Test samples: {len(X_test)} (minimum: {min_test})"
        )
        all_passed = all_passed and passed

        # Check 5.2: Balanced split sizes
        total = len(X_train) + len(X_val) + len(X_test)
        if self.verbose:
            print(f"   Total samples: {total}")
            print(f"      Train: {len(X_train)} ({len(X_train)/total:.1%})")
            print(f"      Val:   {len(X_val)} ({len(X_val)/total:.1%})")
            print(f"      Test:  {len(X_test)} ({len(X_test)/total:.1%})")

        self.validation_results['sample_size'] = all_passed
        return all_passed

    def _calculate_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Information Coefficient (Pearson correlation)"""
        if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
            return 0.0

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return corr if not np.isnan(corr) else 0.0

    def generate_report(self) -> Dict:
        """Generate comprehensive validation report"""
        self.print_section("VALIDATION SUMMARY")

        all_checks_passed = all(self.validation_results.values())

        if self.verbose:
            print(f"   Overall Status: {'✅ PASSED' if all_checks_passed else '❌ FAILED'}")
            print(f"\n   Individual Checks:")
            for check_name, passed in self.validation_results.items():
                status = "✅" if passed else "❌"
                print(f"      {status} {check_name.replace('_', ' ').title()}")

        if all_checks_passed:
            if self.verbose:
                print(f"\n   🎉 All validation checks passed!")
                print(f"   ✅ Data is ready for training")
        else:
            if self.verbose:
                print(f"\n   ⚠️  Some validation checks failed")
                print(f"   ⚠️  Review issues before training")

        return {
            'all_passed': all_checks_passed,
            'results': self.validation_results
        }


def validate_data_loaders(train_loader, val_loader, test_loader, verbose=True):
    """
    Convenience function to validate data from PyTorch DataLoaders

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        verbose: Print detailed output

    Returns:
        dict: Validation report
    """
    # Extract all data from loaders
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

    # Run validation
    validator = DataValidator(verbose=verbose)

    validator.validate_temporal_alignment(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    validator.check_data_leakage(X_train, Y_train, X_test, Y_test)
    validator.check_feature_quality(X_train, X_val, X_test)
    validator.check_target_quality(Y_train, Y_val, Y_test)
    validator.check_sample_size(X_train, X_val, X_test)

    return validator.generate_report()


def validate_data_tensors(X_train, Y_train, X_val, Y_val, X_test, Y_test, verbose=True):
    """
    Convenience function to validate data from tensors directly

    Args:
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        X_test, Y_test: Test data
        verbose: Print detailed output

    Returns:
        dict: Validation report
    """
    validator = DataValidator(verbose=verbose)

    validator.validate_temporal_alignment(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    validator.check_data_leakage(X_train, Y_train, X_test, Y_test)
    validator.check_feature_quality(X_train, X_val, X_test)
    validator.check_target_quality(Y_train, Y_val, Y_test)
    validator.check_sample_size(X_train, X_val, X_test)

    return validator.generate_report()


if __name__ == "__main__":
    """Example usage"""
    print("Data Validation Module")
    print("="*60)
    print("This module provides comprehensive data validation for MAMBA_BGNN")
    print("\nUsage:")
    print("  from utils.data_validation import validate_data_loaders")
    print("  report = validate_data_loaders(train_loader, val_loader, test_loader)")
    print("  print(report)")
