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

Usage:
    from utils.baseline_trainer import train_models

    results = train_models(
        models_dict={'Linear': linear_model, 'LSTM': lstm_model},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={'epochs': 50, 'lr': 0.001, ...}
    )
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
    """Pearson correlation coefficient"""
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
    """Rank Information Coefficient (Spearman correlation)"""
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
    val_header = ['epoch', 'nll', 'rmse', 'mae', 'ic', 'ric']
    test_header = ['nll', 'rmse', 'mae', 'ic', 'ric', 'dir_acc', 'sharpe',
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
        rmse = torch.sqrt(torch.mean((trues - preds)**2)).item()
        mae = torch.mean(torch.abs(trues - preds)).item()
        ic = pearson(trues, preds).item()
        ric_val = ric(trues, preds).item()

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
    test_nll = loss_fn(test_preds, test_trues, test_sigmas.pow(2)).item()
    test_rmse = torch.sqrt(torch.mean((test_trues - test_preds)**2)).item()
    test_mae = torch.mean(torch.abs(test_trues - test_preds)).item()
    test_ic = pearson(test_trues, test_preds).item()
    test_ric = ric(test_trues, test_preds).item()
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
        print(f"  IC:   {test_ic:.6f}")
        print(f"  RIC:  {test_ric:.6f}")
        print(f"  Dir Acc: {test_dir_acc:.6f}")
        print(f"  Sharpe: {port_metrics['sharpe']:.4f}")
        print(f"  Max DD: {port_metrics['max_drawdown']:.4f}")
        print(f"  ⏱️  Avg time: {avg_epoch_time:.2f}s/epoch")

    # Save model
    torch.save(best_state, os.path.join(log_dir, 'best_model.pth'))

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
            model_name, model, loss_fn, optimizer,
            train_loader, val_loader, test_loader,
            config['epochs'], config['patience'], log_dir,
            scheduler, device, verbose
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

    return results
