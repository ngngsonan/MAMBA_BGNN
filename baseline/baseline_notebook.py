"""
Baseline Models Training - Integrated with Main Project Pipeline

This script trains baseline models using the same data processing and evaluation
pipeline as the main MAMBA_BGNN model for fair comparison.

Key Features:
- Uses utils.data_processing for consistent train/val/test splits (80/5/15)
- Out-of-time testing (no data leakage)
- Comprehensive metrics (probabilistic, financial, regime analysis)
- Logs saved to logs/baselines/{model_name}/
- Summary report with model comparison

Available Models:
    1. Linear - Simple feedforward baseline
    2. LSTM - Long Short-Term Memory network
    3. Transformer - Transformer encoder
    4. AGCRN - Adaptive Graph Convolution RNN
    5. TemporalGN - Temporal Graph Network

Usage:
    # In Python script or Jupyter notebook
    from baseline.baseline_notebook import train_all_baselines

    results = train_all_baselines(
        dataset='IXIC',
        models=['Linear', 'LSTM', 'Transformer', 'AGCRN', 'TemporalGN'],
        epochs=50,
        loss_type='auto'
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

# Import data processing from utils
from utils.data_processing import data_processing

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
# TRAINER
# ============================================================================

def train_one_model(model_name: str, model: nn.Module, loss_fn, optimizer,
                   train_loader, val_loader, test_loader,
                   epochs: int, patience: int, log_dir: str,
                   verbose: bool = True):
    """
    Train a single baseline model with comprehensive metrics tracking

    Returns:
        Dictionary with training history and test metrics
    """
    os.makedirs(log_dir, exist_ok=True)

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

    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")

    # Track total training time
    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
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

        # Calculate epoch time
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

    # Calculate total training time
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
    log_base_dir: str = 'logs/baselines'
) -> Dict:
    """
    Train all baseline models using project's data processing pipeline

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

    Returns:
        Dictionary with results for each model
    """
    if models is None:
        models = ['Linear', 'LSTM', 'Transformer', 'AGCRN', 'TemporalGN']

    print("="*80)
    print("BASELINE MODELS TRAINING - Integrated Pipeline")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Window: {window}, Batch: {batch_size}, Epochs: {epochs}")
    print(f"Loss: {loss_type}, LR: {learning_rate}")
    print("="*80)

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

    # Train each model
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in models:
        # Create model
        if model_name == 'Linear':
            model = LinearBaseline(num_features, window, hidden_dim)
        elif model_name == 'LSTM':
            model = LSTMBaseline(num_features, window, hidden_dim)
        elif model_name == 'Transformer':
            model = TransformerBaseline(num_features, window, hidden_dim)
        elif model_name == 'AGCRN':
            model = AGCRNBaseline(num_features, window, hidden_dim)
        elif model_name == 'TemporalGN':
            model = TemporalGNBaseline(num_features, window, hidden_dim)
        else:
            print(f"Unknown model: {model_name}, skipping...")
            continue

        # Setup loss and optimizer
        loss_fn = get_loss_function(loss_type, model_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Create log directory
        log_dir = os.path.join(log_base_dir, f'{dataset}_{model_name}_{timestamp}')

        # Train
        metrics = train_one_model(
            model_name, model, loss_fn, optimizer,
            train_loader, val_loader, test_loader,
            epochs, early_stop_patience, log_dir, verbose
        )

        results[model_name] = metrics

    # Create summary report
    summary_dir = os.path.join(log_base_dir, f'{dataset}_summary_{timestamp}')
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
            'Avg Time (s/epoch)': f"{data['avg_epoch_time']:.2f}",
            'Total Time (s)': f"{data['total_time']:.1f}"
        }
        for name, data in results.items()
    ])

    comparison_csv = os.path.join(summary_dir, 'baseline_comparison.csv')
    comparison_df.to_csv(comparison_csv, index=False)

    # Create text summary
    summary_txt = os.path.join(summary_dir, 'baseline_summary.txt')
    with open(summary_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE MODELS COMPARISON SUMMARY\n")
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

        metrics_to_check = ['rmse', 'mae', 'ic', 'ric', 'dir_acc', 'sharpe']
        for metric in metrics_to_check:
            values = {name: data[metric] for name, data in results.items()}
            if metric in ['rmse', 'mae']:
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Train all baseline models
    """
    results = train_all_baselines(
        dataset='IXIC',
        models=['Linear', 'LSTM', 'Transformer', 'AGCRN', 'TemporalGN'],
        epochs=50,
        loss_type='auto',
        verbose=True
    )

    print("\nTraining completed!")
    print("Check logs/baselines/ for detailed results and comparison.")
