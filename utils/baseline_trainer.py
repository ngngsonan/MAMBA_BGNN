"""
Baseline Trainer Module - Unified Training Pipeline for Baseline Models

This module provides a generic training pipeline that can be used for:
- Baseline models (Linear, LSTM, Transformer, AGCRN, TemporalGN)
- MAMBA ablation models (MAMBA, BIMAMBA, MAMBA+)
- Any other probabilistic models that output (mean, log_var)

Key Features:
- Comprehensive metrics (probabilistic, financial, regime analysis)
- Early stopping with patience
- CSV logging for validation and test metrics
- Model checkpointing
- Summary report generation
- Cross-sectional IC/RIC calculation across multiple assets
- Automatic prediction saving for cross-asset analysis
- Batch processing for multiple models and datasets

IMPORTANT: IC/RIC Metrics Clarification
- Single-Asset IC/RIC: Correlation across TIME for a single asset (usually high ~0.3-0.9)
- Cross-Sectional IC/RIC: Correlation across ASSETS at each time point (usually low ~0.01-0.05)
- The metrics shown during training are SINGLE-ASSET time-series correlations
- For true cross-sectional IC, use calculate_cross_sectional_metrics()

Usage:

    1. Train models on a single dataset:
    ====================================
    from utils.baseline_trainer import train_models

    results = train_models(
        models_dict={'Linear': linear_model, 'LSTM': lstm_model},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={'epochs': 50, 'lr': 0.001, ...}
    )


    2. Calculate cross-sectional IC for all models (RECOMMENDED):
    =============================================================
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # After training on multiple datasets
    cross_results = calculate_cross_sectional_for_all_models(
        models=['Linear', 'LSTM', 'Transformer'],
        datasets=['IXIC', 'DJI', 'NYSE'],
        output_file='logs/cross_sectional_summary.txt'
    )


    3. Calculate cross-sectional IC for a single model:
    ===================================================
    from utils.baseline_trainer import calculate_cross_sectional_metrics

    prediction_files = {
        'IXIC': 'logs/.../IXIC_.../test_predictions.csv',
        'DJI': 'logs/.../DJI_.../test_predictions.csv',
        'NYSE': 'logs/.../NYSE_.../test_predictions.csv'
    }
    cross_ic_results = calculate_cross_sectional_metrics(prediction_files)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import csv
import time
import copy
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, mu, y, var):
        var = var.clamp(min=1e-6)
        nll = 0.5 * (torch.log(var) + ((y - mu) ** 2) / var)
        return nll.mean()


class DeterministicLoss(nn.Module):
    """Wrapper for deterministic losses"""
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, mu, y, var=None):
        return self.loss_fn(mu, y)


def get_loss_function(loss_type: str, model_name: str = None):
    """
    Get appropriate loss function

    Auto mode selects:
    - Linear/AGCRN/TemporalGN → MSE (simple, stable)
    - LSTM/Transformer → Huber (robust to outliers in financial data)
    - MAMBA variants → Huber (robust to outliers)
    """
    if loss_type == 'auto':
        if model_name and 'Linear' in model_name:
            return DeterministicLoss('mse')
        elif model_name and ('LSTM' in model_name or 'Transformer' in model_name):
            return DeterministicLoss('huber')
        elif model_name and ('AGCRN' in model_name or 'TemporalGN' in model_name):
            return DeterministicLoss('mse')
        elif model_name and 'MAMBA' in model_name.upper():
            return DeterministicLoss('huber')
        else:
            return GaussianNLLLoss()
    elif loss_type == 'nll':
        return GaussianNLLLoss()
    else:
        return DeterministicLoss(loss_type)


# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

def pearson(x, y):
    """
    Pearson correlation coefficient (Single-Asset Time-Series)

    NOTE: This calculates correlation across TIME for a SINGLE asset.
    This is NOT the cross-sectional IC used in quantitative finance.
    For cross-sectional IC, use calculate_cross_sectional_metrics().
    """
    vx, vy = x - x.mean(), y - y.mean()
    return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-12)


def rank_tensor(x):
    """Convert tensor to ranks"""
    flat = x.view(-1)
    idx = torch.argsort(flat)
    ranks = torch.empty_like(flat, dtype=torch.float32)
    ranks[idx] = torch.arange(1, flat.numel() + 1, dtype=torch.float32, device=x.device)
    return ranks.view_as(x)


def ric(x, y):
    """
    Rank Information Coefficient (Single-Asset Time-Series Spearman)

    NOTE: This calculates Spearman correlation across TIME for a SINGLE asset.
    This is NOT the cross-sectional RIC used in quantitative finance.
    For cross-sectional RIC, use calculate_cross_sectional_metrics().
    """
    rx, ry = rank_tensor(x), rank_tensor(y)
    return pearson(rx, ry)


def directional_accuracy(y_pred, y_true):
    """Directional accuracy"""
    pred_dir = torch.sign(y_pred)
    true_dir = torch.sign(y_true)
    return float((pred_dir == true_dir).float().mean())


def calculate_portfolio_metrics(preds, returns, transaction_cost=0.001):
    """Calculate comprehensive portfolio metrics"""
    preds_np = preds.cpu().numpy()
    returns_np = returns.cpu().numpy()

    positions = np.sign(preds_np)
    turnover = np.abs(np.diff(positions, axis=0)).mean() if len(positions) > 1 else 0
    costs = turnover * transaction_cost

    portfolio_returns = positions[1:] * returns_np[1:] - costs if len(positions) > 1 else np.array([0])

    annual_factor = np.sqrt(252)
    std_ret = portfolio_returns.std()
    sharpe = portfolio_returns.mean() / std_ret * annual_factor if std_ret != 0 else 0.0

    cum_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = 1 - cum_returns / peak
    max_drawdown = drawdown.max()

    calmar = (portfolio_returns.mean() * 252) / max_drawdown if max_drawdown > 0 else 0.0
    hit_rate = float(np.mean(np.sign(portfolio_returns) == np.sign(returns_np[1:]))) if len(returns_np) > 1 else 0.5

    return {
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'calmar': float(calmar),
        'hit_rate': float(hit_rate),
        'turnover': float(turnover)
    }


def crps_gaussian(mu, sigma, y):
    """CRPS for Gaussian distribution"""
    import math
    sigma = sigma.clamp_min(1e-8)
    z = (y - mu) / sigma
    try:
        Phi = torch.special.ndtr(z)
    except AttributeError:
        Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    phi = torch.exp(-0.5 * z**2) / math.sqrt(2*math.pi)
    crps = sigma * (z * (2*Phi - 1) + 2*phi - 1.0/math.sqrt(math.pi))
    return torch.clamp(crps, min=0.0)


def picp_and_gap(mu, sigma, y, q=0.90):
    """Prediction Interval Coverage Probability and Gap"""
    import math
    p = (1.0 + q)/2.0
    try:
        z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0*p - 1.0, device=mu.device, dtype=mu.dtype))
    except AttributeError:
        z = math.sqrt(2.0) * torch.special.erfinv(torch.tensor(2.0*p - 1.0, device=mu.device, dtype=mu.dtype))
    lo, hi = mu - z*sigma, mu + z*sigma
    obs = ((y >= lo) & (y <= hi)).float().mean().item()
    gap = abs(obs - q)
    return obs, gap


# ============================================================================
# CROSS-SECTIONAL METRICS (Multi-Asset Analysis)
# ============================================================================

def calculate_cross_sectional_metrics(
    prediction_files: Dict[str, str],
    verbose: bool = True
) -> Dict:
    """
    Calculate CROSS-SECTIONAL IC and RIC across multiple assets.

    This function computes the true cross-sectional Information Coefficient (IC)
    and Rank Information Coefficient (RIC) as used in quantitative finance.
    At each time point t, it calculates the correlation between predictions
    and actual returns ACROSS DIFFERENT ASSETS (not across time).

    Args:
        prediction_files: Dictionary mapping asset names to prediction CSV file paths
                         Example: {'IXIC': 'logs/.../test_predictions.csv',
                                  'DJI': 'logs/.../test_predictions.csv',
                                  'NYSE': 'logs/.../test_predictions.csv'}
        verbose: Print detailed results

    Returns:
        Dictionary containing:
            - daily_ic: Array of cross-sectional IC at each time point
            - daily_ric: Array of cross-sectional RIC at each time point
            - ic_mean: Mean cross-sectional IC
            - ic_std: Std of cross-sectional IC
            - ric_mean: Mean cross-sectional RIC
            - ric_std: Std of cross-sectional RIC
            - ic_positive_ratio: % of days with positive IC
            - ric_positive_ratio: % of days with positive RIC

    Usage (in notebook):
        from utils.baseline_trainer import calculate_cross_sectional_metrics

        prediction_files = {
            'IXIC': 'logs/baseline/IXIC_Linear_20250118_120000/test_predictions.csv',
            'DJI': 'logs/baseline/DJI_Linear_20250118_120000/test_predictions.csv',
            'NYSE': 'logs/baseline/NYSE_Linear_20250118_120000/test_predictions.csv'
        }

        results = calculate_cross_sectional_metrics(prediction_files)
        print(f"Cross-sectional IC: {results['ic_mean']:.6f}")
    """
    from scipy.stats import pearsonr, spearmanr

    # Load predictions
    data = {}
    for asset_name, file_path in prediction_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prediction file not found: {file_path}")
        df = pd.read_csv(file_path)
        data[asset_name] = df
        if verbose:
            print(f"Loaded {asset_name}: {len(df)} samples")

    # Verify all assets have same number of samples
    assets = list(data.keys())
    n_samples = len(data[assets[0]])
    for asset in assets:
        if len(data[asset]) != n_samples:
            raise ValueError(f"Sample count mismatch for {asset}: {len(data[asset])} vs {n_samples}")

    # Calculate cross-sectional IC and RIC for each time point
    daily_ic = []
    daily_ric = []

    if verbose:
        print(f"\nCalculating cross-sectional IC across {len(assets)} assets for {n_samples} time points...")

    for t in range(n_samples):
        # Get predictions and actual returns at time t across all assets
        pred_t = [data[asset]['mu'].iloc[t] for asset in assets]
        actual_t = [data[asset]['y'].iloc[t] for asset in assets]

        # Calculate Pearson correlation (IC) and Spearman correlation (RIC)
        if len(set(pred_t)) > 1 and len(set(actual_t)) > 1:
            ic_t, _ = pearsonr(pred_t, actual_t)
            ric_t, _ = spearmanr(pred_t, actual_t)
        else:
            ic_t, ric_t = 0, 0

        daily_ic.append(ic_t)
        daily_ric.append(ric_t)

    # Convert to numpy arrays
    daily_ic = np.array(daily_ic)
    daily_ric = np.array(daily_ric)

    # Remove NaN values for statistics
    daily_ic_clean = daily_ic[~np.isnan(daily_ic)]
    daily_ric_clean = daily_ric[~np.isnan(daily_ric)]

    # Calculate statistics
    results = {
        'daily_ic': daily_ic,
        'daily_ric': daily_ric,
        'ic_mean': float(np.mean(daily_ic_clean)),
        'ic_std': float(np.std(daily_ic_clean)),
        'ic_median': float(np.median(daily_ic_clean)),
        'ric_mean': float(np.mean(daily_ric_clean)),
        'ric_std': float(np.std(daily_ric_clean)),
        'ric_median': float(np.median(daily_ric_clean)),
        'ic_positive_ratio': float(np.sum(daily_ic_clean > 0) / len(daily_ic_clean)),
        'ric_positive_ratio': float(np.sum(daily_ric_clean > 0) / len(daily_ric_clean)),
        'valid_days': len(daily_ic_clean),
        'num_time_points': len(daily_ic_clean),  # Alias for consistency
        'assets': assets
    }

    if verbose:
        print("\n" + "="*70)
        print("CROSS-SECTIONAL IC ANALYSIS (Multi-Asset)")
        print("="*70)
        print(f"Assets analyzed: {', '.join(assets)}")
        print(f"Valid trading days: {results['valid_days']}")
        print(f"\nCross-Sectional Information Coefficient (IC):")
        print(f"  Mean:       {results['ic_mean']:.6f}")
        print(f"  Std:        {results['ic_std']:.6f}")
        print(f"  Median:     {results['ic_median']:.6f}")
        print(f"  % Positive: {results['ic_positive_ratio']*100:.1f}%")

        print(f"\nCross-Sectional Rank IC (RIC):")
        print(f"  Mean:       {results['ric_mean']:.6f}")
        print(f"  Std:        {results['ric_std']:.6f}")
        print(f"  Median:     {results['ric_median']:.6f}")
        print(f"  % Positive: {results['ric_positive_ratio']*100:.1f}%")

        print(f"\n📊 INTERPRETATION:")
        if results['ic_mean'] > 0.05:
            print("   🔥 EXCEPTIONAL: IC > 0.05 is extremely rare in real markets!")
        elif results['ic_mean'] > 0.02:
            print("   🚀 EXCELLENT: IC > 0.02 indicates very strong predictive power")
        elif results['ic_mean'] > 0.01:
            print("   ✅ GOOD: IC > 0.01 shows solid predictive ability")
        elif results['ic_mean'] > 0.005:
            print("   📈 DECENT: IC > 0.005 has commercial value")
        else:
            print("   📉 WEAK: IC ≤ 0.005 may not be practically useful")

        print("\nNOTE: Cross-sectional IC measures correlation ACROSS ASSETS at each")
        print("      time point, unlike single-asset IC which measures correlation")
        print("      across TIME for a single asset.")
        print("="*70)

    return results


def compare_single_vs_cross_sectional_ic(
    prediction_files: Dict[str, str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare single-asset time-series IC vs cross-sectional IC.

    This function helps understand the difference between:
    - Single-asset IC: Correlation across TIME for each asset individually
    - Cross-sectional IC: Correlation across ASSETS at each time point

    Args:
        prediction_files: Dictionary mapping asset names to prediction CSV paths
        verbose: Print comparison table

    Returns:
        DataFrame with comparison results

    Usage (in notebook):
        from utils.baseline_trainer import compare_single_vs_cross_sectional_ic

        prediction_files = {
            'IXIC': 'logs/.../IXIC_.../test_predictions.csv',
            'DJI': 'logs/.../DJI_.../test_predictions.csv',
            'NYSE': 'logs/.../NYSE_.../test_predictions.csv'
        }

        comparison_df = compare_single_vs_cross_sectional_ic(prediction_files)
    """
    # Calculate cross-sectional IC
    cross_results = calculate_cross_sectional_metrics(prediction_files, verbose=False)

    # Calculate single-asset time-series IC for each asset
    single_asset_results = []
    for asset_name, file_path in prediction_files.items():
        df = pd.read_csv(file_path)

        # Time-series correlation (across time for single asset)
        ts_ic = np.corrcoef(df['mu'], df['y'])[0, 1]
        ts_ric = pd.DataFrame({'pred': df['mu'], 'actual': df['y']}).corr('spearman').loc['pred', 'actual']

        single_asset_results.append({
            'Asset': asset_name,
            'Single-Asset IC (Time-Series)': ts_ic,
            'Single-Asset RIC (Time-Series)': ts_ric
        })

    # Create comparison table
    comparison_df = pd.DataFrame(single_asset_results)

    # Add cross-sectional metrics
    summary = {
        'Asset': 'CROSS-SECTIONAL',
        'Single-Asset IC (Time-Series)': cross_results['ic_mean'],
        'Single-Asset RIC (Time-Series)': cross_results['ric_mean']
    }
    comparison_df = pd.concat([comparison_df, pd.DataFrame([summary])], ignore_index=True)

    # Rename columns for clarity
    comparison_df.columns = ['Asset', 'IC', 'RIC']

    if verbose:
        print("\n" + "="*70)
        print("COMPARISON: Single-Asset (Time-Series) vs Cross-Sectional IC")
        print("="*70)
        print("\nSingle-Asset IC: Correlation of predictions vs actuals ACROSS TIME")
        print("                 for each individual asset (usually high ~0.3-0.9)")
        print("\nCross-Sectional IC: Correlation of predictions vs actuals ACROSS ASSETS")
        print("                    at each time point (usually low ~0.01-0.05)")
        print("\n" + "-"*70)
        print(comparison_df.to_string(index=False))
        print("="*70)

    return comparison_df


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_model(
    model_name: str,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    val_loader,
    test_loader,
    epochs: int,
    patience: int,
    log_dir: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """
    Train a single model with comprehensive metrics tracking

    Args:
        model_name: Name of the model
        model: PyTorch model that outputs (mean, log_var)
        loss_fn: Loss function that accepts (mu, y, var)
        optimizer: PyTorch optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        patience: Early stopping patience
        log_dir: Directory to save logs
        scheduler: Optional learning rate scheduler
        device: Device to train on ('cpu' or 'cuda')
        verbose: Print progress

    Returns:
        Dictionary with training history and test metrics
    """
    os.makedirs(log_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)

    # CSV paths
    val_csv = os.path.join(log_dir, 'val_metrics.csv')
    test_csv = os.path.join(log_dir, 'test_metrics.csv')

    # Initialize CSVs
    # NOTE: IC and RIC in these CSVs are SINGLE-ASSET time-series metrics
    val_header = ['epoch', 'nll', 'rmse', 'mae', 'ic_single_asset', 'ric_single_asset']
    test_header = ['nll', 'rmse', 'mae', 'ic_single_asset', 'ric_single_asset', 'dir_acc', 'sharpe',
                  'max_drawdown', 'calmar', 'hit_rate', 'crps', 'picp90', 'gap90']

    if not os.path.exists(val_csv):
        with open(val_csv, 'w', newline='') as f:
            csv.writer(f).writerow(val_header)
    if not os.path.exists(test_csv):
        with open(test_csv, 'w', newline='') as f:
            csv.writer(f).writerow(test_header)

    best_loss = float('inf')
    best_state = None
    not_improved = 0
    history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

    if verbose:
        print(f"\n{'='*80}")
        print(f"Training: {model_name}")
        print(f"{'='*80}")

    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mu, log_var = model(x)
            loss = loss_fn(mu, y.squeeze(), log_var.exp())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        preds, trues, logvars = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mu, log_var = model(x)
                val_loss += loss_fn(mu, y.squeeze(), log_var.exp()).item()
                preds.append(mu)
                trues.append(y.squeeze())
                logvars.append(log_var)

        val_loss /= len(val_loader)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        logvars = torch.cat(logvars, 0)

        # Calculate validation metrics
        # NOTE: IC and RIC here are SINGLE-ASSET time-series correlations
        rmse = torch.sqrt(torch.mean((trues - preds)**2)).item()
        mae = torch.mean(torch.abs(trues - preds)).item()
        ic = pearson(trues, preds).item()  # Single-asset time-series IC
        ric_val = ric(trues, preds).item()  # Single-asset time-series RIC

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_times'].append(epoch_time)

        # Log to CSV
        with open(val_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, val_loss, rmse, mae, ic, ric_val])

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
                  f"RMSE: {rmse:.6f}, IC: {ic:.4f}, Time: {epoch_time:.2f}s")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            not_improved = 0
            if verbose:
                print(f"✓ New best model (Val Loss: {val_loss:.6f})")
        else:
            not_improved += 1
            if not_improved >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    total_training_time = time.time() - training_start_time
    avg_epoch_time = np.mean(history['epoch_times']) if history['epoch_times'] else 0.0

    if verbose:
        print(f"\n⏱️  Training completed in {total_training_time:.2f}s")
        print(f"   Average: {avg_epoch_time:.2f}s/epoch")

    # Load best model and test
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test with comprehensive metrics
    model.eval()
    test_preds, test_trues, test_logvars = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mu, log_var = model(x)
            test_preds.append(mu)
            test_trues.append(y.squeeze())
            test_logvars.append(log_var)

    test_preds = torch.cat(test_preds, 0)
    test_trues = torch.cat(test_trues, 0)
    test_logvars = torch.cat(test_logvars, 0)
    test_sigmas = torch.exp(0.5 * test_logvars).clamp_min(1e-8)

    # Calculate test metrics
    # NOTE: IC and RIC here are SINGLE-ASSET time-series correlations
    test_nll = loss_fn(test_preds, test_trues, test_sigmas.pow(2)).item()
    test_rmse = torch.sqrt(torch.mean((test_trues - test_preds)**2)).item()
    test_mae = torch.mean(torch.abs(test_trues - test_preds)).item()
    test_ic = pearson(test_trues, test_preds).item()  # Single-asset time-series IC
    test_ric = ric(test_trues, test_preds).item()  # Single-asset time-series RIC
    test_dir_acc = directional_accuracy(test_preds, test_trues)

    # Portfolio metrics
    port_metrics = calculate_portfolio_metrics(test_preds, test_trues)

    # Probabilistic metrics
    test_crps = crps_gaussian(test_preds, test_sigmas, test_trues).mean().item()
    test_picp90, test_gap90 = picp_and_gap(test_preds, test_sigmas, test_trues, q=0.90)

    # Save test metrics
    test_metrics = {
        'model': model_name,
        'nll': test_nll,
        'rmse': test_rmse,
        'mae': test_mae,
        'ic': test_ic,
        'ric': test_ric,
        'dir_acc': test_dir_acc,
        'sharpe': port_metrics['sharpe'],
        'max_drawdown': port_metrics['max_drawdown'],
        'calmar': port_metrics['calmar'],
        'hit_rate': port_metrics['hit_rate'],
        'crps': test_crps,
        'picp90': test_picp90,
        'gap90': test_gap90,
        'total_time': total_training_time,
        'avg_epoch_time': avg_epoch_time
    }

    with open(test_csv, 'a', newline='') as f:
        csv.writer(f).writerow([
            test_nll, test_rmse, test_mae, test_ic, test_ric, test_dir_acc,
            port_metrics['sharpe'], port_metrics['max_drawdown'],
            port_metrics['calmar'], port_metrics['hit_rate'],
            test_crps, test_picp90, test_gap90
        ])

    if verbose:
        print(f"\n[{model_name}] Test Results:")
        print(f"  RMSE: {test_rmse:.6f}")
        print(f"  MAE:  {test_mae:.6f}")
        print(f"  IC (single-asset):  {test_ic:.6f}")
        print(f"  RIC (single-asset): {test_ric:.6f}")
        print(f"  Dir Acc: {test_dir_acc:.6f}")
        print(f"  Sharpe: {port_metrics['sharpe']:.4f}")
        print(f"  Max DD: {port_metrics['max_drawdown']:.4f}")
        print(f"  ⏱️  Avg time: {avg_epoch_time:.2f}s/epoch")
        print(f"\n  NOTE: IC/RIC are single-asset time-series correlations.")
        print(f"        For cross-sectional IC, use calculate_cross_sectional_metrics()")

    # Save model
    torch.save(best_state, os.path.join(log_dir, 'best_model.pth'))

    # Save test predictions for cross-sectional analysis
    predictions_df = pd.DataFrame({
        'sample_idx': range(len(test_preds)),
        'mu': test_preds.cpu().numpy(),
        'log_var': test_logvars.cpu().numpy(),
        'sigma': test_sigmas.cpu().numpy(),
        'y': test_trues.cpu().numpy()
    })
    predictions_csv = os.path.join(log_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_csv, index=False)

    if verbose:
        print(f"  💾 Predictions saved to: {predictions_csv}")

    test_metrics['history'] = history
    return test_metrics


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_models(
    models_dict: Dict[str, nn.Module],
    train_loader,
    val_loader,
    test_loader,
    dataset: str,
    config: Optional[Dict] = None,
    log_base_dir: str = 'logs',
    study_name: str = 'baseline',
    verbose: bool = True,
    device: str = 'cpu'
) -> Dict:
    """
    Train multiple models with unified pipeline

    Args:
        models_dict: Dictionary of {model_name: model_instance}
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        dataset: Dataset name (e.g., 'IXIC', 'DJI', 'NYSE')
        config: Configuration dictionary with keys:
            - epochs (int): Number of epochs (default: 50)
            - lr (float): Learning rate (default: 0.001)
            - loss_type (str): Loss type 'auto', 'nll', 'mse', 'mae', 'huber' (default: 'auto')
            - patience (int): Early stopping patience (default: 10)
            - optimizer_fn (Callable): Optimizer factory (default: Adam)
            - scheduler_fn (Callable): Scheduler factory (default: None)
        log_base_dir: Base directory for logs
        study_name: Name of study (e.g., 'baseline', 'ablation')
        verbose: Print progress
        device: Device to train on

    Returns:
        Dictionary with results for each model
    """
    # Default config
    default_config = {
        'epochs': 50,
        'lr': 0.001,
        'loss_type': 'auto',
        'patience': 10,
        'optimizer_fn': lambda params, lr: torch.optim.Adam(params, lr=lr),
        'scheduler_fn': None
    }

    if config is None:
        config = {}
    config = {**default_config, **config}

    print("="*80)
    print(f"{study_name.upper()} TRAINING - Unified Pipeline")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Models: {', '.join(models_dict.keys())}")
    print(f"Config: {config}")
    print(f"Device: {device}")
    print("="*80)

    # Train each model
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, model in models_dict.items():
        # Setup
        loss_fn = get_loss_function(config['loss_type'], model_name)
        optimizer = config['optimizer_fn'](model.parameters(), config['lr'])
        scheduler = config['scheduler_fn'](optimizer) if config['scheduler_fn'] is not None else None

        # Create log directory
        log_dir = os.path.join(log_base_dir, study_name, f'{dataset}_{model_name}_{timestamp}')

        # Train
        metrics = train_one_model(
            model_name=model_name,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=config['epochs'],
            patience=config['patience'],
            log_dir=log_dir,
            scheduler=scheduler,
            device=device,
            verbose=verbose
        )

        results[model_name] = metrics

    # Create summary report
    summary_dir = os.path.join(log_base_dir, study_name, f'{dataset}_summary_{timestamp}')
    os.makedirs(summary_dir, exist_ok=True)

    # Save comparison table
    comparison_df = pd.DataFrame([
        {
            'Model': name,
            'RMSE': f"{data['rmse']:.6f}",
            'MAE': f"{data['mae']:.6f}",
            'IC': f"{data['ic']:.6f}",
            'RIC': f"{data['ric']:.6f}",
            'Dir Acc': f"{data['dir_acc']:.6f}",
            'Sharpe': f"{data['sharpe']:.4f}",
            'Max DD': f"{data['max_drawdown']:.4f}",
            'Calmar': f"{data['calmar']:.4f}",
            'CRPS': f"{data['crps']:.6f}",
            'Avg Time (s/epoch)': f"{data['avg_epoch_time']:.2f}",
            'Total Time (s)': f"{data['total_time']:.1f}"
        }
        for name, data in results.items()
    ])

    comparison_csv = os.path.join(summary_dir, f'{study_name}_comparison.csv')
    comparison_df.to_csv(comparison_csv, index=False)

    # Create text summary
    summary_txt = os.path.join(summary_dir, f'{study_name}_summary.txt')
    with open(summary_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{study_name.upper()} MODELS COMPARISON SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Models trained: {', '.join(results.keys())}\n")
        f.write("="*80 + "\n\n")
        f.write("METRICS COMPARISON:\n")
        f.write("-"*80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")

        # Best model by each metric
        f.write("\nBEST MODELS BY METRIC:\n")
        f.write("-"*80 + "\n")

        metrics_to_check = ['rmse', 'mae', 'ic', 'ric', 'dir_acc', 'sharpe', 'calmar', 'crps']
        for metric in metrics_to_check:
            values = {name: data[metric] for name, data in results.items()}
            if metric in ['rmse', 'mae', 'crps']:
                best_model = min(values, key=values.get)
            else:
                best_model = max(values, key=values.get)
            f.write(f"{metric.upper():12s}: {best_model:15s} ({values[best_model]:.6f})\n")

        # Training time summary
        f.write("\nTRAINING TIME SUMMARY:\n")
        f.write("-"*80 + "\n")
        for name, data in results.items():
            f.write(f"{name:15s}: {data['total_time']:6.1f}s total, {data['avg_epoch_time']:5.2f}s/epoch\n")
        total_all = sum(data['total_time'] for data in results.values())
        f.write(f"{'TOTAL':15s}: {total_all:6.1f}s ({total_all/60:.1f} minutes)\n")

    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("\n" + "="*80)
    print(f"✓ Results saved to: {summary_dir}")
    print(f"  - Comparison table: {comparison_csv}")
    print(f"  - Summary report: {summary_txt}")
    print("="*80)
    print("\n📊 NOTE: IC/RIC shown are SINGLE-ASSET time-series correlations.")
    print("   For CROSS-SECTIONAL IC across multiple assets, use:")
    print("   >>> from utils.baseline_trainer import calculate_cross_sectional_metrics")
    print("   >>> prediction_files = {")
    print(f"   ...     'IXIC': 'logs/.../{dataset}_IXIC_.../test_predictions.csv',")
    print(f"   ...     'DJI': 'logs/.../{dataset}_DJI_.../test_predictions.csv',")
    print(f"   ...     'NYSE': 'logs/.../{dataset}_NYSE_.../test_predictions.csv'")
    print("   ... }")
    print("   >>> results = calculate_cross_sectional_metrics(prediction_files)")
    print("="*80)

    return results


# ============================================================================
# BATCH CROSS-SECTIONAL IC CALCULATION
# ============================================================================

def calculate_cross_sectional_for_all_models(
    models: List[str],
    datasets: List[str],
    log_base_dir: str = 'logs',
    study_name: str = 'baseline',
    output_file: str = None,
    verbose: bool = True
) -> Dict:
    """
    Calculate cross-sectional IC for all models across multiple datasets

    This is a simple loop function that:
    1. For each model, calculates cross-sectional IC across all datasets
    2. Compares single-asset vs cross-sectional IC
    3. Collects all results into one dictionary
    4. Optionally saves to a summary file

    Args:
        models: List of model names (e.g., ['Linear', 'LSTM', 'Transformer'])
        datasets: List of dataset names (e.g., ['IXIC', 'DJI', 'NYSE'])
        log_base_dir: Base directory for logs
        study_name: Study name (e.g., 'baseline', 'ablation')
        output_file: Path to save summary (None = don't save)
        verbose: Print progress

    Returns:
        Dictionary with structure:
        {
            'Linear': {
                'cross_sectional': {...},  # Results from calculate_cross_sectional_metrics
                'comparison': DataFrame    # Results from compare_single_vs_cross_sectional_ic
            },
            'LSTM': {...},
            ...
        }

    Example Usage (Copy to notebook cell):
        ```python
        from utils.baseline_trainer import calculate_cross_sectional_for_all_models

        # After training all models on all datasets
        results = calculate_cross_sectional_for_all_models(
            models=['Linear', 'LSTM', 'Transformer'],
            datasets=['IXIC', 'DJI', 'NYSE'],
            output_file='logs/cross_sectional_summary.txt',
            verbose=True
        )

        # Access results
        print(f"Linear IC: {results['Linear']['cross_sectional']['ic_mean']:.6f}")
        print(results['Linear']['comparison'])
        ```
    """
    import glob

    if len(datasets) < 2:
        raise ValueError(f"Need at least 2 datasets for cross-sectional IC. Got {len(datasets)}: {datasets}")

    if verbose:
        print("\n" + "="*80)
        print("CROSS-SECTIONAL IC ANALYSIS FOR ALL MODELS")
        print("="*80)
        print(f"Models: {', '.join(models)}")
        print(f"Datasets: {', '.join(datasets)}")
        print("="*80)

    all_results = {}

    for model_name in models:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing model: {model_name}")
            print(f"{'='*70}")

        # Find prediction files for this model across all datasets
        prediction_files = {}
        for dataset in datasets:
            pattern = f"{log_base_dir}/{study_name}/{dataset}_{model_name}_*/test_predictions.csv"
            matches = glob.glob(pattern)

            if not matches:
                if verbose:
                    print(f"  ⚠️  No predictions found for {dataset} (pattern: {pattern})")
                continue

            # Use the most recent
            pred_file = sorted(matches)[-1]
            prediction_files[dataset] = pred_file
            if verbose:
                print(f"  ✓ {dataset}: {pred_file}")

        if len(prediction_files) < 2:
            if verbose:
                print(f"  ⚠️  Skipping {model_name}: need >=2 datasets, found {len(prediction_files)}")
            all_results[model_name] = {
                'error': f'Insufficient datasets (need >=2, found {len(prediction_files)})',
                'cross_sectional': None,
                'comparison': None
            }
            continue

        try:
            # Calculate cross-sectional IC
            if verbose:
                print(f"\n  Calculating cross-sectional IC...")
            cross_results = calculate_cross_sectional_metrics(
                prediction_files,
                verbose=verbose
            )

            # Compare single-asset vs cross-sectional
            if verbose:
                print(f"\n  Comparing single-asset vs cross-sectional IC...")
            comparison_df = compare_single_vs_cross_sectional_ic(
                prediction_files,
                verbose=verbose
            )

            all_results[model_name] = {
                'cross_sectional': cross_results,
                'comparison': comparison_df,
                'datasets_used': list(prediction_files.keys())
            }

            if verbose:
                print(f"\n  ✓ {model_name} completed")
                print(f"    Cross-Sectional IC:  {cross_results['ic_mean']:.6f}")
                print(f"    Cross-Sectional RIC: {cross_results['ric_mean']:.6f}")

        except Exception as e:
            if verbose:
                print(f"  ❌ Error processing {model_name}: {str(e)}")
            all_results[model_name] = {
                'error': str(e),
                'cross_sectional': None,
                'comparison': None
            }

    # Save summary if requested
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CROSS-SECTIONAL IC SUMMARY FOR ALL MODELS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models: {', '.join(models)}\n")
            f.write(f"Datasets: {', '.join(datasets)}\n")
            f.write("="*80 + "\n\n")

            for model_name, result in all_results.items():
                f.write(f"\n{'-'*70}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"{'-'*70}\n")

                if result.get('error'):
                    f.write(f"Error: {result['error']}\n")
                    continue

                cross = result['cross_sectional']
                f.write(f"Datasets used: {', '.join(result['datasets_used'])}\n")
                f.write(f"Valid time points: {cross['valid_days']}\n\n")

                f.write(f"Cross-Sectional IC:\n")
                f.write(f"  Mean:       {cross['ic_mean']:>8.6f}\n")
                f.write(f"  Std:        {cross['ic_std']:>8.6f}\n")
                f.write(f"  Median:     {cross['ic_median']:>8.6f}\n")
                f.write(f"  % Positive: {cross['ic_positive_ratio']*100:>7.1f}%\n\n")

                f.write(f"Cross-Sectional RIC:\n")
                f.write(f"  Mean:       {cross['ric_mean']:>8.6f}\n")
                f.write(f"  Std:        {cross['ric_std']:>8.6f}\n")
                f.write(f"  Median:     {cross['ric_median']:>8.6f}\n")
                f.write(f"  % Positive: {cross['ric_positive_ratio']*100:>7.1f}%\n\n")

                # Comparison table
                f.write("Comparison (Single-Asset vs Cross-Sectional):\n")
                f.write(result['comparison'].to_string(index=False) + "\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*80 + "\n")

        if verbose:
            print(f"\n{'='*80}")
            print(f"✓ Summary saved to: {output_file}")
            print(f"{'='*80}")

    return all_results
