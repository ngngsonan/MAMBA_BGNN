"""
Baseline Models - Model Definitions and Training Interface

This module provides baseline model definitions for comparison with MAMBA_BGNN.
Uses utils.baseline_trainer for the unified training pipeline.

Key Features:
- 5 baseline model architectures (Linear, LSTM, Transformer, AGCRN, TemporalGN)
- Automatic prediction saving for cross-sectional IC analysis
- Clear distinction between single-asset IC and cross-sectional IC
- Uses utils.data_processing for consistent train/val/test splits (80/5/15)
- Comprehensive metrics (probabilistic, financial, regime analysis)

Available Models:
    1. Linear - Simple feedforward baseline
    2. LSTM - Long Short-Term Memory network
    3. Transformer - Transformer encoder
    4. AGCRN - Adaptive Graph Convolution RNN
    5. TemporalGN - Temporal Graph Network

Usage:

    Train baseline models on a dataset:
    ===================================
    from baseline.baseline_notebook import train_all_baselines

    results = train_all_baselines(
        dataset='IXIC',
        models=['Linear', 'LSTM', 'Transformer'],
        epochs=50,
        loss_type='auto'
    )


    Calculate cross-sectional IC for all models:
    ============================================
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # After training on multiple datasets
    cross_results = calculate_cross_sectional_for_all_models(
        models=['Linear', 'LSTM', 'Transformer'],
        datasets=['IXIC', 'DJI', 'NYSE'],
        output_file='logs/cross_sectional_summary.txt'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Dict, List
import math
from datetime import datetime
import copy
import csv
import json
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data processing and training utilities from utils
from utils.data_processing import data_processing
from utils.baseline_trainer import (
    train_models,
    calculate_cross_sectional_metrics,
    compare_single_vs_cross_sectional_ic
)

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
    """
    if loss_type == 'auto':
        if model_name and 'Linear' in model_name:
            return DeterministicLoss('mse')
        elif model_name and ('LSTM' in model_name or 'Transformer' in model_name):
            return DeterministicLoss('huber')
        elif model_name and ('AGCRN' in model_name or 'TemporalGN' in model_name):
            return DeterministicLoss('mse')
        else:
            return GaussianNLLLoss()
    elif loss_type == 'nll':
        return GaussianNLLLoss()
    else:
        return DeterministicLoss(loss_type)


# ============================================================================
# BASELINE MODELS
# ============================================================================

class BaselineModel(nn.Module):
    """Base class for all baseline models"""
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class LinearBaseline(BaselineModel):
    """Simple linear baseline"""
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64):
        super().__init__(input_dim, seq_len, hidden_dim)
        self.flatten_dim = seq_len * input_dim
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.reshape(x.size(0), -1)
        h = F.relu(self.fc1(x_flat))
        h = F.relu(self.fc2(h))
        mean = self.mean_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1)
        return mean, log_var


class LSTMBaseline(BaselineModel):
    """LSTM baseline"""
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__(input_dim, seq_len, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        mean = self.mean_head(last_hidden).squeeze(-1)
        log_var = self.logvar_head(last_hidden).squeeze(-1)
        return mean, log_var


class TransformerBaseline(BaselineModel):
    """Transformer baseline"""
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 64, nhead: int = 8, num_layers: int = 3):
        super().__init__(input_dim, seq_len, d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = self._create_pos_encoding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mean_head = nn.Linear(d_model, 1)
        self.logvar_head = nn.Linear(d_model, 1)

    def _create_pos_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = self.transformer(x)
        x = x.mean(dim=1)
        mean = self.mean_head(x).squeeze(-1)
        log_var = self.logvar_head(x).squeeze(-1)
        return mean, log_var


class AGCRNBaseline(BaselineModel):
    """Adaptive Graph Convolution RNN baseline"""
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64):
        super().__init__(input_dim, seq_len, hidden_dim)
        self.adaptive_adj = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
        self.gcn = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, N = x.shape
        gcn_out = []
        adj = torch.softmax(self.adaptive_adj, dim=1)
        for t in range(L):
            x_t = x[:, t, :]
            h_t = torch.matmul(x_t, adj)
            h_t = self.gcn(h_t)
            gcn_out.append(h_t)
        temporal_features = torch.stack(gcn_out, dim=1)
        rnn_out, _ = self.rnn(temporal_features)
        last_hidden = rnn_out[:, -1, :]
        mean = self.mean_head(last_hidden).squeeze(-1)
        log_var = self.logvar_head(last_hidden).squeeze(-1)
        return mean, log_var


class TemporalGNBaseline(BaselineModel):
    """Temporal Graph Network baseline"""
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64):
        super().__init__(input_dim, seq_len, hidden_dim)
        self.node_embedding = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        self.graph_conv1 = nn.Linear(input_dim, hidden_dim)
        self.graph_conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, N = x.shape
        graph_features = []
        for t in range(L):
            x_t = x[:, t, :]
            h1 = F.relu(self.graph_conv1(x_t))
            h2 = self.graph_conv2(h1)
            graph_features.append(h2)
        temporal_graph = torch.stack(graph_features, dim=1)
        attn_out, _ = self.temporal_attn(
            temporal_graph, temporal_graph, temporal_graph
        )
        pooled = torch.mean(attn_out, dim=1)
        mean = self.mean_head(pooled).squeeze(-1)
        log_var = self.logvar_head(pooled).squeeze(-1)
        return mean, log_var


# ============================================================================
# METRICS HELPER FUNCTIONS (From trainer.py)
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
# NOTE: train_one_model() is now imported from utils.baseline_trainer
# The unified version supports scheduler, device, and cross-sectional IC
# ============================================================================


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_all_baselines(
    dataset: str = 'IXIC',
    models: List[str] = None,
    window: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    loss_type: str = 'auto',
    early_stop_patience: int = 10,
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu'
) -> Dict:
    """
    Train all baseline models using unified baseline_trainer pipeline

    This function uses the unified baseline_trainer.py which:
    - Automatically saves predictions for cross-sectional IC analysis
    - Clearly labels IC/RIC as single-asset time-series metrics
    - Provides cross-sectional IC calculation functions

    Args:
        dataset: Dataset name (IXIC, DJI, NYSE)
        models: List of models to train (None = all 5 models)
        window: Lookback window size
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        loss_type: 'auto', 'nll', 'mse', 'mae', 'huber'
        early_stop_patience: Early stopping patience
        verbose: Print progress
        log_base_dir: Base directory for logs
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Dictionary with results for each model

    Note:
        IC/RIC shown in results are SINGLE-ASSET time-series correlations.
        For cross-sectional IC, use calculate_cross_sectional_ic_example()
    """
    if models is None:
        models = ['Linear', 'LSTM', 'Transformer', 'AGCRN', 'TemporalGN']

    # Load data using utils.data_processing
    print("\nLoading data with utils.data_processing...")
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'
    num_features, train_loader, val_loader, test_loader = data_processing(
        data_path, window, batch_size
    )
    print(f"✓ Data loaded: {num_features} features")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples (out-of-time)")

    # Create model instances
    models_dict = {}
    for model_name in models:
        if model_name == 'Linear':
            models_dict[model_name] = LinearBaseline(num_features, window, hidden_dim)
        elif model_name == 'LSTM':
            models_dict[model_name] = LSTMBaseline(num_features, window, hidden_dim)
        elif model_name == 'Transformer':
            models_dict[model_name] = TransformerBaseline(num_features, window, hidden_dim)
        elif model_name == 'AGCRN':
            models_dict[model_name] = AGCRNBaseline(num_features, window, hidden_dim)
        elif model_name == 'TemporalGN':
            models_dict[model_name] = TemporalGNBaseline(num_features, window, hidden_dim)
        else:
            print(f"Unknown model: {model_name}, skipping...")

    # Train using unified baseline_trainer
    config = {
        'epochs': epochs,
        'lr': learning_rate,
        'loss_type': loss_type,
        'patience': early_stop_patience,
        'optimizer_fn': lambda params, lr: torch.optim.Adam(params, lr=lr),
        'scheduler_fn': None
    }

    results = train_models(
        models_dict=models_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=dataset,
        config=config,
        log_base_dir=log_base_dir,
        study_name='baseline',
        verbose=verbose,
        device=device
    )

    return results


# ============================================================================
# CROSS-SECTIONAL IC HELPER FUNCTIONS
# ============================================================================

def calculate_cross_sectional_ic_example(
    model_name: str = 'Linear',
    assets: List[str] = None,
    log_base_dir: str = 'logs',
    study_name: str = 'baseline',
    timestamp: str = None
) -> Dict:
    """
    Calculate cross-sectional IC across multiple assets (helper function)

    This function demonstrates how to use calculate_cross_sectional_metrics()
    to compute true cross-sectional IC after training models on multiple datasets.

    Args:
        model_name: Name of the model (e.g., 'Linear', 'LSTM')
        assets: List of asset names (default: ['IXIC', 'DJI', 'NYSE'])
        log_base_dir: Base log directory
        study_name: Study name (e.g., 'baseline', 'ablation')
        timestamp: Specific timestamp to use (None = find latest)

    Returns:
        Dictionary with cross-sectional IC results

    Example Usage (Copy to notebook cell):
        ```python
        from baseline.baseline_notebook import calculate_cross_sectional_ic_example

        # After training models on IXIC, DJI, NYSE
        results = calculate_cross_sectional_ic_example(
            model_name='Linear',
            assets=['IXIC', 'DJI', 'NYSE']
        )

        print(f"Cross-Sectional IC: {results['ic_mean']:.6f}")
        print(f"Cross-Sectional RIC: {results['ric_mean']:.6f}")
        ```
    """
    import glob

    if assets is None:
        assets = ['IXIC', 'DJI', 'NYSE']

    print(f"\n{'='*70}")
    print(f"CROSS-SECTIONAL IC ANALYSIS: {model_name}")
    print(f"{'='*70}")
    print(f"Assets: {', '.join(assets)}")

    # Find prediction files
    prediction_files = {}
    for asset in assets:
        # Search pattern: logs/baseline/{asset}_{model_name}_{timestamp}/test_predictions.csv
        if timestamp:
            pattern = f"{log_base_dir}/{study_name}/{asset}_{model_name}_{timestamp}/test_predictions.csv"
        else:
            pattern = f"{log_base_dir}/{study_name}/{asset}_{model_name}_*/test_predictions.csv"

        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(
                f"No predictions found for {asset} {model_name}\n"
                f"Pattern: {pattern}\n"
                f"Make sure you've trained the model on all assets first!"
            )

        # Use the most recent if multiple matches
        pred_file = sorted(matches)[-1]
        prediction_files[asset] = pred_file
        print(f"  {asset}: {pred_file}")

    # Calculate cross-sectional IC
    print(f"\n{'='*70}")
    results = calculate_cross_sectional_metrics(prediction_files, verbose=True)

    return results


def compare_single_vs_cross_ic_example(
    model_name: str = 'Linear',
    assets: List[str] = None,
    log_base_dir: str = 'logs',
    study_name: str = 'baseline'
):
    """
    Compare single-asset IC vs cross-sectional IC (helper function)

    Example Usage (Copy to notebook cell):
        ```python
        from baseline.baseline_notebook import compare_single_vs_cross_ic_example

        comparison_df = compare_single_vs_cross_ic_example(
            model_name='Linear',
            assets=['IXIC', 'DJI', 'NYSE']
        )

        print(comparison_df)
        ```
    """
    import glob

    if assets is None:
        assets = ['IXIC', 'DJI', 'NYSE']

    # Find prediction files
    prediction_files = {}
    for asset in assets:
        pattern = f"{log_base_dir}/{study_name}/{asset}_{model_name}_*/test_predictions.csv"
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No predictions found for {asset} {model_name}")
        prediction_files[asset] = sorted(matches)[-1]

    # Compare
    comparison_df = compare_single_vs_cross_sectional_ic(prediction_files, verbose=True)

    return comparison_df




# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Train baselines and calculate cross-sectional IC

    Step 1: Train models on multiple datasets
    Step 2: Calculate cross-sectional IC for all models
    """

    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # Step 1: Train models on multiple datasets
    print("\n" + "="*80)
    print("STEP 1: Training baseline models on multiple datasets")
    print("="*80)

    datasets = ['IXIC', 'DJI', 'NYSE']
    models = ['Linear', 'LSTM', 'Transformer']

    for dataset in datasets:
        print(f"\n>>> Training on {dataset}...")
        results = train_all_baselines(
            dataset=dataset,
            models=models,
            epochs=50,
            loss_type='auto',
            verbose=True
        )

    # Step 2: Calculate cross-sectional IC for all models
    print("\n" + "="*80)
    print("STEP 2: Calculating cross-sectional IC for all models")
    print("="*80)

    cross_results = calculate_cross_sectional_for_all_models(
        models=models,
        datasets=datasets,
        output_file='logs/cross_sectional_summary.txt',
        verbose=True
    )

    print("\n" + "="*80)
    print("✓ All steps completed!")
    print("="*80)
    print("\nResults summary:")
    for model_name, result in cross_results.items():
        if result.get('error'):
            print(f"  {model_name}: Error - {result['error']}")
        else:
            print(f"  {model_name}: IC={result['cross_sectional']['ic_mean']:.6f}, "
                  f"RIC={result['cross_sectional']['ric_mean']:.6f}")
    print("\nDetailed results saved to: logs/cross_sectional_summary.txt")
