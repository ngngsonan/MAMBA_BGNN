"""
Baseline Trainer with Advanced Financial Analysis

This module provides:
    1. Training utilities for models
    2. Financial metrics computation
    3. Market regime analysis
    4. Bayesian vs Non-Bayesian comparison
    5. Cross-sectional IC analysis

Key analyses commonly used in quantitative finance papers:
    - IC (Information Coefficient)
    - RIC (Rank IC)
    - Sharpe Ratio
    - Maximum Drawdown
    - Directional Accuracy (Hit Rate)
    - Market Regime Performance (Bull/Bear, High/Low Volatility)
    - Uncertainty Calibration (for Bayesian models)

Usage:
    from utils.baseline_trainer import train_models, regime_analysis

    results = train_models(models_dict, train_loader, val_loader, test_loader)
    regime_results = regime_analysis(model, test_loader)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import os
import json


# ============================================================================
# Loss Functions
# ============================================================================

def gaussian_nll_loss(mu, log_var, y):
    """
    Gaussian Negative Log-Likelihood Loss

    Args:
        mu: (B,) predicted mean
        log_var: (B,) predicted log variance
        y: (B,) true values

    Returns:
        loss: scalar
    """
    var = torch.exp(log_var)
    loss = 0.5 * (log_var + ((y - mu) ** 2) / var)
    return loss.mean()


def mse_loss(mu, log_var, y):
    """MSE loss (ignores variance)"""
    return F.mse_loss(mu, y)


def mae_loss(mu, log_var, y):
    """MAE loss"""
    return F.l1_loss(mu, y)


def huber_loss(mu, log_var, y, delta=1.0):
    """Smooth L1 / Huber loss"""
    return F.smooth_l1_loss(mu, y)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_ic(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Information Coefficient (Pearson correlation)

    Key metric for stock prediction - measures linear relationship
    """
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    # Demean
    y_pred_centered = y_pred_np - y_pred_np.mean()
    y_true_centered = y_true_np - y_true_np.mean()

    # Pearson correlation
    numerator = (y_pred_centered * y_true_centered).sum()
    denominator = np.sqrt((y_pred_centered ** 2).sum()) * np.sqrt((y_true_centered ** 2).sum())

    if denominator < 1e-12:
        return 0.0

    return float(numerator / denominator)


def compute_ric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Rank Information Coefficient (Spearman correlation)

    Robust version of IC using ranks instead of values
    """
    from scipy.stats import spearmanr

    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    if len(y_pred_np) < 3:
        return 0.0

    corr, _ = spearmanr(y_pred_np, y_true_np)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Root Mean Squared Error"""
    return float(torch.sqrt(F.mse_loss(y_pred, y_true)).item())


def compute_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Mean Absolute Error"""
    return float(F.l1_loss(y_pred, y_true).item())


def compute_directional_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Directional Accuracy (Hit Rate)

    Percentage of predictions with correct sign
    Critical for trading applications
    """
    pred_direction = torch.sign(y_pred)
    true_direction = torch.sign(y_true)

    mask = (true_direction != 0) & (pred_direction != 0)
    if mask.sum() == 0:
        return 0.5  # Random baseline

    correct = (pred_direction[mask] == true_direction[mask]).float()
    return float(correct.mean().item())


def compute_sharpe_ratio(returns: torch.Tensor, risk_free_rate: float = 0.02) -> float:
    """
    Sharpe Ratio - risk-adjusted return

    Higher is better
    """
    returns_np = returns.detach().cpu().numpy()

    if len(returns_np) == 0:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = returns_np - daily_rf

    if np.std(excess_returns) < 1e-12:
        return 0.0

    return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


def compute_max_drawdown(returns: torch.Tensor) -> float:
    """
    Maximum Drawdown

    Worst peak-to-trough decline
    More negative is worse
    """
    returns_np = returns.detach().cpu().numpy()
    cumulative = np.cumprod(1 + returns_np)

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max

    return float(np.min(drawdown))


# ============================================================================
# Training Utilities
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        mu, log_var = model(batch_x)
        loss = loss_fn(mu, log_var, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        all_preds.append(mu.detach())
        all_targets.append(batch_y.detach())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = {
        'loss': total_loss / len(all_targets),
        'ic': compute_ic(all_preds, all_targets),
        'ric': compute_ric(all_preds, all_targets)
    }

    return metrics


def evaluate(
    model: nn.Module,
    data_loader,
    loss_fn,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_log_vars = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            mu, log_var = model(batch_x)
            loss = loss_fn(mu, log_var, batch_y)

            total_loss += loss.item() * len(batch_y)
            all_preds.append(mu)
            all_targets.append(batch_y)
            all_log_vars.append(log_var)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_log_vars = torch.cat(all_log_vars)

    metrics = {
        'loss': total_loss / len(all_targets),
        'ic': compute_ic(all_preds, all_targets),
        'ric': compute_ric(all_preds, all_targets),
        'rmse': compute_rmse(all_preds, all_targets),
        'mae': compute_mae(all_preds, all_targets),
        'directional_accuracy': compute_directional_accuracy(all_preds, all_targets),
        'sharpe': compute_sharpe_ratio(all_targets),
        'max_drawdown': compute_max_drawdown(all_targets)
    }

    return metrics, all_preds, all_targets, all_log_vars


def train_models(
    models_dict: Dict[str, nn.Module],
    train_loader,
    val_loader,
    test_loader,
    dataset: str,
    config: Dict,
    log_base_dir: str = 'logs',
    study_name: str = 'experiment',
    verbose: bool = True,
    device: str = 'cpu'
) -> Dict:
    """
    Train multiple models with unified pipeline

    Args:
        models_dict: {model_name: model}
        config: Training configuration dict with keys:
            - epochs: int
            - lr: float
            - loss_type: str ('nll', 'mse', 'mae', 'huber')
            - patience: int (early stopping)
            - optimizer_fn: Callable
            - scheduler_fn: Optional[Callable]

    Returns:
        results: Dict with metrics for each model
    """

    # Loss function mapping
    loss_fns = {
        'nll': gaussian_nll_loss,
        'mse': mse_loss,
        'mae': mae_loss,
        'huber': huber_loss,
        'auto': gaussian_nll_loss  # Default
    }

    loss_fn = loss_fns.get(config['loss_type'], gaussian_nll_loss)

    results = {}

    for model_name, model in models_dict.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")

        model = model.to(device)

        # Optimizer
        optimizer = config['optimizer_fn'](model.parameters(), config['lr'])

        # Scheduler (optional)
        scheduler = None
        if config.get('scheduler_fn'):
            scheduler = config['scheduler_fn'](optimizer)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(config['epochs']):
            # Train
            train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

            # Validate
            val_metrics, _, _, _ = evaluate(model, val_loader, loss_fn, device)

            if scheduler:
                scheduler.step()

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                # Save best model
                save_dir = os.path.join(log_base_dir, study_name, dataset)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(),
                          os.path.join(save_dir, f'{model_name}_best.pth'))
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']}: "
                      f"Train Loss={train_metrics['loss']:.6f}, "
                      f"Val Loss={val_metrics['loss']:.6f}, "
                      f"Val IC={val_metrics['ic']:.4f}")

            if patience_counter >= config['patience']:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(torch.load(
            os.path.join(save_dir, f'{model_name}_best.pth'),
            map_location=device
        ))

        # Test
        test_metrics, test_preds, test_targets, test_log_vars = evaluate(
            model, test_loader, loss_fn, device
        )

        if verbose:
            print(f"\nTest Results:")
            print(f"  IC: {test_metrics['ic']:.6f}")
            print(f"  RIC: {test_metrics['ric']:.6f}")
            print(f"  RMSE: {test_metrics['rmse']:.6f}")
            print(f"  DA: {test_metrics['directional_accuracy']:.4f}")
            print(f"  Sharpe: {test_metrics['sharpe']:.4f}")

        results[model_name] = test_metrics

        # Save predictions
        pred_df = pd.DataFrame({
            'y': test_targets.cpu().numpy(),
            'mu': test_preds.cpu().numpy(),
            'log_var': test_log_vars.cpu().numpy(),
            'sigma': np.exp(0.5 * test_log_vars.cpu().numpy())
        })
        pred_df.to_csv(
            os.path.join(save_dir, f'{model_name}_predictions.csv'),
            index=False
        )

    return results


# ============================================================================
# Market Regime Analysis
# ============================================================================

def classify_market_regime(returns: torch.Tensor, window: int = 20) -> torch.Tensor:
    """
    Classify market regime based on rolling volatility

    Returns:
        regimes: (N,) tensor with values:
            0 = Low volatility (stable market)
            1 = High volatility (volatile market)
    """
    returns_np = returns.detach().cpu().numpy()

    # Calculate rolling volatility
    rolling_vol = pd.Series(returns_np).rolling(window=window, min_periods=window//2).std()
    vol_median = rolling_vol.median()

    # Classify
    regime = (rolling_vol > vol_median).astype(int).values

    return torch.tensor(regime, dtype=torch.int)


def regime_analysis(
    model_bayesian: nn.Module,
    model_nonbayesian: nn.Module,
    test_loader,
    device: str = 'cpu'
) -> Dict:
    """
    Analyze model performance across market regimes

    Returns performance in:
        - Stable markets (low volatility)
        - Volatile markets (high volatility)
    """
    model_bayesian.eval()
    model_nonbayesian.eval()

    all_preds_bayes = []
    all_preds_nonbayes = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            mu_bayes, _ = model_bayesian(batch_x)
            mu_nonbayes, _ = model_nonbayesian(batch_x)

            all_preds_bayes.append(mu_bayes.cpu())
            all_preds_nonbayes.append(mu_nonbayes.cpu())
            all_targets.append(batch_y)

    all_preds_bayes = torch.cat(all_preds_bayes)
    all_preds_nonbayes = torch.cat(all_preds_nonbayes)
    all_targets = torch.cat(all_targets)

    # Classify regimes
    regimes = classify_market_regime(all_targets)

    results = {}

    for regime_id, regime_name in [(0, 'Stable'), (1, 'Volatile')]:
        mask = (regimes == regime_id)

        if mask.sum() < 10:  # Need enough samples
            continue

        preds_bayes = all_preds_bayes[mask]
        preds_nonbayes = all_preds_nonbayes[mask]
        targets = all_targets[mask]

        results[regime_name] = {
            'Bayesian': {
                'ic': compute_ic(preds_bayes, targets),
                'ric': compute_ric(preds_bayes, targets),
                'rmse': compute_rmse(preds_bayes, targets),
                'da': compute_directional_accuracy(preds_bayes, targets)
            },
            'NonBayesian': {
                'ic': compute_ic(preds_nonbayes, targets),
                'ric': compute_ric(preds_nonbayes, targets),
                'rmse': compute_rmse(preds_nonbayes, targets),
                'da': compute_directional_accuracy(preds_nonbayes, targets)
            },
            'n_samples': int(mask.sum())
        }

    return results


# ============================================================================
# Financial Metrics Computation
# ============================================================================

def compute_financial_metrics(
    models_dict: Dict[str, nn.Module],
    test_loader,
    device: str = 'cpu'
) -> Dict:
    """
    Compute comprehensive financial metrics

    Returns:
        metrics: Dict with:
            - Sharpe ratio
            - Maximum drawdown
            - Calmar ratio
            - Hit rate
            - etc.
    """
    results = {}

    for model_name, model in models_dict.items():
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)

                mu, _ = model(batch_x)

                all_preds.append(mu.cpu())
                all_targets.append(batch_y)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Strategy returns (follow predictions)
        positions = torch.sign(all_preds)
        strategy_returns = positions * all_targets

        results[model_name] = {
            'sharpe_ratio': compute_sharpe_ratio(strategy_returns),
            'max_drawdown': compute_max_drawdown(strategy_returns),
            'hit_rate': compute_directional_accuracy(all_preds, all_targets),
            'mean_return': float(strategy_returns.mean().item()),
            'return_std': float(strategy_returns.std().item()),
            'total_return': float((torch.cumprod(1 + strategy_returns, dim=0)[-1] - 1).item())
        }

    return results


# ============================================================================
# Bayesian vs Non-Bayesian Comparison
# ============================================================================

def compare_bayesian_vs_nonbayesian(all_results: Dict) -> Dict:
    """
    Cross-dataset comparison of Bayesian vs Non-Bayesian models

    Args:
        all_results: {dataset: {model_name: metrics}}

    Returns:
        comparison: Summary statistics and improvements
    """
    comparison = {
        'per_dataset': {},
        'average': {},
        'improvement': {}
    }

    metrics_to_compare = ['ic', 'ric', 'rmse', 'directional_accuracy', 'sharpe', 'max_drawdown']

    for dataset, results in all_results.items():
        if 'Bayesian' not in str(results) or 'NonBayesian' not in str(results):
            continue

        # Find bayesian and non-bayesian results
        bayes_key = [k for k in results.keys() if 'Bayesian' in k and 'Non' not in k]
        nonbayes_key = [k for k in results.keys() if 'NonBayesian' in k]

        if not bayes_key or not nonbayes_key:
            continue

        bayes_metrics = results[bayes_key[0]]
        nonbayes_metrics = results[nonbayes_key[0]]

        dataset_comp = {}
        for metric in metrics_to_compare:
            if metric in bayes_metrics and metric in nonbayes_metrics:
                bayes_val = bayes_metrics[metric]
                nonbayes_val = nonbayes_metrics[metric]

                # Compute improvement
                if abs(nonbayes_val) > 1e-8:
                    improvement = (bayes_val - nonbayes_val) / abs(nonbayes_val) * 100
                else:
                    improvement = 0.0

                dataset_comp[metric] = {
                    'Bayesian': bayes_val,
                    'NonBayesian': nonbayes_val,
                    'improvement_%': improvement
                }

        comparison['per_dataset'][dataset] = dataset_comp

    # Compute averages
    if comparison['per_dataset']:
        avg_metrics = {}
        for metric in metrics_to_compare:
            bayes_vals = []
            nonbayes_vals = []
            improvements = []

            for dataset_comp in comparison['per_dataset'].values():
                if metric in dataset_comp:
                    bayes_vals.append(dataset_comp[metric]['Bayesian'])
                    nonbayes_vals.append(dataset_comp[metric]['NonBayesian'])
                    improvements.append(dataset_comp[metric]['improvement_%'])

            if bayes_vals:
                avg_metrics[metric] = {
                    'Bayesian_avg': np.mean(bayes_vals),
                    'NonBayesian_avg': np.mean(nonbayes_vals),
                    'avg_improvement_%': np.mean(improvements)
                }

        comparison['average'] = avg_metrics

    return comparison


# ============================================================================
# Cross-Sectional IC Analysis
# ============================================================================

def calculate_cross_sectional_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    timestamps: Optional[np.ndarray] = None
) -> Dict:
    """
    Calculate cross-sectional IC (across stocks at each time point)

    Args:
        predictions: (T, N) - predictions for N stocks over T time steps
        targets: (T, N) - actual returns
        timestamps: (T,) - optional timestamps

    Returns:
        ic_metrics: Dict with IC time series and statistics
    """
    T, N = predictions.shape

    ic_series = []
    for t in range(T):
        pred_t = predictions[t, :]
        target_t = targets[t, :]

        # Remove NaN
        valid = ~(np.isnan(pred_t) | np.isnan(target_t))
        if valid.sum() < 3:
            ic_series.append(np.nan)
            continue

        # Pearson correlation
        pred_valid = pred_t[valid]
        target_valid = target_t[valid]

        ic = np.corrcoef(pred_valid, target_valid)[0, 1]
        ic_series.append(ic if not np.isnan(ic) else 0.0)

    ic_series = np.array(ic_series)

    return {
        'ic_series': ic_series,
        'ic_mean': np.nanmean(ic_series),
        'ic_std': np.nanstd(ic_series),
        'ic_sharpe': np.nanmean(ic_series) / (np.nanstd(ic_series) + 1e-8),
        'positive_ic_ratio': np.mean(ic_series > 0)
    }


def calculate_cross_sectional_for_all_models(
    models: List[str],
    datasets: List[str],
    study_name: str,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Calculate cross-sectional IC for all models across datasets

    Args:
        models: List of model names
        datasets: List of dataset names
        study_name: Study name (for log directory)
        output_file: Optional output file path

    Returns:
        results: Dict with cross-sectional IC for each model
    """
    results = {}

    for model_name in models:
        model_results = {}

        for dataset in datasets:
            pred_file = f'logs/{study_name}/{dataset}/{model_name}_predictions.csv'

            if not os.path.exists(pred_file):
                if verbose:
                    print(f"Warning: {pred_file} not found")
                continue

            # Load predictions
            df = pd.read_csv(pred_file)

            # Placeholder: Assume single-stock for now
            # In multi-stock scenario, reshape to (T, N)
            predictions = df['mu'].values.reshape(-1, 1)
            targets = df['y'].values.reshape(-1, 1)

            ic_metrics = calculate_cross_sectional_ic(predictions, targets)
            model_results[dataset] = ic_metrics

        results[model_name] = model_results

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if verbose:
            print(f"✓ Cross-sectional IC results saved to: {output_file}")

    return results
