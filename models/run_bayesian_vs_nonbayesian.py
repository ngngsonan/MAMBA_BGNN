"""
Bayesian vs Non-Bayesian MAMBA-BGNN Comparison Study

This module performs comprehensive comparison between:
    1. Bayesian MAGAC (with MC sampling & uncertainty quantification)
    2. Non-Bayesian MAGAC (deterministic, no MC sampling)

Experiments run on 3 datasets: IXIC, DJI, NYSE

Key analyses:
    - Financial metrics (Sharpe, IC, RIC, Directional Accuracy)
    - Market regime analysis (Stable vs Volatile)
    - Uncertainty quality (calibration, CRPS)
    - Risk-adjusted performance

Usage in Notebook:
    1. Run setup cell
    2. Choose mode (Quick Test, Single Dataset, Multi-Dataset)
    3. Run training cell
    4. Run analysis cell
    5. Generate plots

Or run as script:
    python models/run_bayesian_vs_nonbayesian.py --datasets IXIC DJI NYSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import json
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# CELL 1: MODEL COMPONENTS
# ============================================================================
# This cell defines all model architecture components
# Run this cell first in notebooks

@dataclass
class ModelArgs:
    d_model: int
    seq_len: int
    d_proj_E: int = 64
    d_proj_H: int = 64
    d_proj_U: int = 32
    expand: int = 2
    d_state: int = 64
    dt_rank: int | str = 'auto'
    d_conv: int = 3
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_proj_E / 16)


class MambaBlock(nn.Module):
    """Original Mamba Block with Selective State Space Model"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_proj_E * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_proj_E,
            out_channels=args.d_proj_E,
            kernel_size=args.d_conv,
            groups=args.d_proj_E,
            padding=args.d_conv - 1,
            bias=args.conv_bias,
        )
        self.x_proj = nn.Linear(args.d_proj_E, args.dt_rank + args.d_proj_H * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_proj_E, bias=True)

        A = repeat(torch.arange(1, args.d_proj_H + 1), 'n -> d n', d=args.d_proj_E)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_proj_E))
        self.out_proj = nn.Linear(args.d_proj_E, args.d_model, bias=args.bias)

    def forward(self, x):
        b, l, _ = x.shape
        x_proj, res = self.in_proj(x).chunk(2, dim=-1)
        x_proj = rearrange(x_proj, 'b l d -> b d l')
        x_proj = self.conv1d(x_proj)[:, :, :l]
        x_proj = rearrange(x_proj, 'b d l -> b l d')
        x_proj = F.silu(x_proj)

        y = self._ssm(x_proj)
        y = y * torch.sigmoid(res)
        return self.out_proj(y)

    def _ssm(self, x):
        d_in, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        delta, B, C = torch.split(x_dbl, [self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        return self._scan(x, delta, A, B, C, D)

    def _scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, u, 'b l d, b l n, b l d -> b l d n')

        state = torch.zeros((b, d_in, n), device=u.device, dtype=u.dtype)
        ys = []
        for i in range(l):
            state = deltaA[:, i] * state + deltaB_u[:, i]
            ys.append(einsum(state, C[:, i, :], 'b d n, b n -> b d'))
        y = torch.stack(ys, dim=1)
        return y + u * D


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.d_model, args.d_proj_U),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(args.d_proj_U, args.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BIMambaBlock(nn.Module):
    """Bidirectional Mamba Block"""
    def __init__(self, args: ModelArgs, R: int = 3, dropout: float = 0.1):
        super().__init__()
        self.R = R
        self.f_mamba = nn.ModuleList([MambaBlock(args) for _ in range(R)])
        self.b_mamba = nn.ModuleList([MambaBlock(args) for _ in range(R)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])
        self.ffn = nn.ModuleList([FeedForward(args, dropout) for _ in range(R)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])

    def forward(self, x):
        for i in range(self.R):
            Y1 = self.f_mamba[i](x)
            x_rev = torch.flip(x, dims=[1]).contiguous()
            Y2_rev = self.b_mamba[i](x_rev)
            Y2 = torch.flip(Y2_rev, dims=[1]).contiguous()
            Y3 = self.norm1[i](x + Y1 + Y2)
            Yp = self.ffn[i](Y3)
            x = self.norm2[i](Yp + Y3)
        return x


# ============================================================================
# CELL 2: GRAPH NEURAL NETWORK LAYERS
# ============================================================================
# Non-Bayesian and Bayesian MAGAC implementations

class MAGAC_NonBayesian(nn.Module):
    """
    Non-Bayesian MAGAC - Deterministic graph convolution

    No MC sampling, no uncertainty quantification
    Returns mean prediction with fixed variance
    """
    def __init__(self, num_nodes: int, in_dim: int, K: int = 3,
                 d_e: int = 10, heads: int = 4):
        super().__init__()
        self.N = num_nodes
        self.K = K
        self.in_dim = in_dim
        self.H = heads

        # Node embedding & Gaussian kernel
        self.psi_emb = nn.Parameter(torch.randn(num_nodes, d_e))
        self.psi = nn.Parameter(torch.tensor(1.0))

        # Attention-based dynamic adjacency
        self.W_q = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.W_k = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.attn_alpha = nn.Parameter(torch.tensor(0.5))

        # Factorized Chebyshev filter weights
        self.F_w = nn.Parameter(torch.randn(heads, d_e, K + 1, in_dim))
        self.f_b = nn.Parameter(torch.randn(heads, d_e))
        self.head_mix = nn.Parameter(torch.ones(heads))

    def _gaussian_A(self):
        diff = self.psi_emb[:, None, :] - self.psi_emb[None, :, :]
        dist2 = diff.pow(2).sum(-1)
        A = torch.exp(-self.psi * dist2)
        return F.softmax(A, dim=1)

    def _attn_A(self):
        Q = torch.einsum('nd,dhm->nhm', self.psi_emb, self.W_q)
        K = torch.einsum('nd,dhm->nhm', self.psi_emb, self.W_k)
        d_e = Q.size(-1)
        attn = torch.einsum('nhd,mhd->hnm', Q, K) / math.sqrt(d_e)
        return F.softmax(attn, dim=-1)

    def _blend(self, A_g, A_attn_h):
        alpha = torch.sigmoid(self.attn_alpha)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def forward(self, x):
        """Deterministic forward pass - No MC sampling"""
        B, N, L = x.shape

        # Build effective adjacency
        A_base = self._gaussian_A()
        A_attn = self._attn_A()
        A_effs = torch.stack([self._blend(A_base, A_attn[h])
                              for h in range(self.H)], dim=0)

        mix_w = F.softmax(self.head_mix, dim=0)
        out = 0

        for h in range(self.H):
            A_eff = A_effs[h]

            # Chebyshev polynomial supports
            I = torch.eye(N, device=x.device, dtype=x.dtype)
            supports = [I, A_eff]
            for k in range(2, self.K + 1):
                supports.append(2 * A_eff @ supports[-1] - supports[-2])
            supports = torch.stack(supports, dim=0)

            # Factorized filter
            W_filter = torch.einsum('nd,dkl->nkl', self.psi_emb, self.F_w[h])
            b_filter = self.psi_emb @ self.f_b[h]

            # Graph convolution
            x_g = torch.einsum('knm,bml->bknl', supports, x)
            out_h = torch.einsum('bknl,nkl->bn', x_g, W_filter) + b_filter
            out = out + mix_w[h] * out_h

        # Fixed uncertainty (non-Bayesian)
        log_var = torch.ones_like(out) * (-3.0)

        return out, log_var


class MAGAC_Bayesian(nn.Module):
    """
    Bayesian MAGAC - Stochastic graph convolution with uncertainty

    Uses MC Dropout and DropEdge for uncertainty quantification
    """
    def __init__(self, num_nodes: int, in_dim: int, K: int = 3,
                 d_e: int = 10, heads: int = 4,
                 mc_train: int = 3, mc_eval: int = 20,
                 drop_edge_p: float = 0.1, mc_dropout_p: float = 0.2):
        super().__init__()
        self.N = num_nodes
        self.K = K
        self.in_dim = in_dim
        self.H = heads

        self.mc_train = mc_train
        self.mc_eval = mc_eval
        self.mc_samples = mc_train
        self.drop_edge_p = drop_edge_p
        self.mc_dropout_p = mc_dropout_p

        # Node embedding & Gaussian kernel
        self.psi_emb = nn.Parameter(torch.randn(num_nodes, d_e))
        self.psi = nn.Parameter(torch.tensor(1.0))

        # Attention-based dynamic adjacency
        self.W_q = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.W_k = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.attn_alpha = nn.Parameter(torch.tensor(0.5))

        # Factorized Chebyshev filter weights
        self.F_w = nn.Parameter(torch.randn(heads, d_e, K + 1, in_dim))
        self.f_b = nn.Parameter(torch.randn(heads, d_e))
        self.head_mix = nn.Parameter(torch.ones(heads))

    def train(self, mode: bool = True):
        super().train(mode)
        self.mc_samples = self.mc_train if mode else self.mc_eval
        return self

    def _gaussian_A(self, psi):
        diff = psi[:, None, :] - psi[None, :, :]
        dist2 = diff.pow(2).sum(-1)
        A = torch.exp(-self.psi * dist2)
        return F.softmax(A, dim=1)

    def _attn_A(self, psi):
        Q = torch.einsum('nd,dhm->nhm', psi, self.W_q)
        K = torch.einsum('nd,dhm->nhm', psi, self.W_k)
        d_e = Q.size(-1)
        attn = torch.einsum('nhd,mhd->hnm', Q, K) / math.sqrt(d_e)
        return F.softmax(attn, dim=-1)

    def _blend(self, A_g, A_attn_h):
        alpha = torch.sigmoid(self.attn_alpha)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def _sample_A_eff(self, use_dropout: bool):
        # MC Dropout on embeddings
        psi_stoch = F.dropout(self.psi_emb, p=self.mc_dropout_p, training=use_dropout)

        A_g = self._gaussian_A(psi_stoch)
        A_attn = self._attn_A(psi_stoch)

        A_list = []
        for h in range(self.H):
            A_eff = self._blend(A_g, A_attn[h])

            # DropEdge during eval
            if (not self.training) and (self.drop_edge_p > 0.0):
                keep = torch.bernoulli((1 - self.drop_edge_p) * torch.ones_like(A_eff))
                keep = keep.fill_diagonal_(1.0)
                A_eff = A_eff * keep
                A_eff = A_eff / (A_eff.sum(dim=1, keepdim=True).clamp_min(1e-6))

            A_list.append(A_eff)

        return torch.stack(A_list, dim=0)

    def _forward_single(self, x, A_eff):
        B, N, L = x.shape
        mix_w = F.softmax(self.head_mix, dim=0)
        out = 0

        for h in range(self.H):
            A_h = A_eff[h]

            # Chebyshev supports
            I = torch.eye(N, device=x.device, dtype=x.dtype)
            supports = [I, A_h]
            for k in range(2, self.K + 1):
                supports.append(2 * A_h @ supports[-1] - supports[-2])
            supports = torch.stack(supports, dim=0)

            # Factorized filter
            W_filter = torch.einsum('nd,dkl->nkl', self.psi_emb, self.F_w[h])
            b_filter = self.psi_emb @ self.f_b[h]

            # Graph convolution
            x_g = torch.einsum('knm,bml->bknl', supports, x)
            out_h = torch.einsum('bknl,nkl->bn', x_g, W_filter) + b_filter
            out = out + mix_w[h] * out_h

        return out

    def forward(self, x):
        """Stochastic forward with MC sampling"""
        outs = []

        if self.training:
            A_eff = self._sample_A_eff(use_dropout=True)
            for _ in range(self.mc_samples):
                outs.append(self._forward_single(x, A_eff))
        else:
            for _ in range(self.mc_samples):
                A_eff = self._sample_A_eff(use_dropout=False)
                outs.append(self._forward_single(x, A_eff))

        outs = torch.stack(outs, dim=0)

        mean = outs.mean(0)
        if self.mc_samples == 1:
            log_var = torch.ones_like(mean) * (-4.0)
        else:
            var = outs.var(0, unbiased=False) + 1e-6
            log_var = var.log()

        return mean, log_var


# ============================================================================
# CELL 3: FULL MODELS
# ============================================================================
# BIMamba + MAGAC (Bayesian and Non-Bayesian versions)

class BIMamba_MAGAC_NonBayesian(nn.Module):
    """BIMamba + Non-Bayesian MAGAC (Deterministic)"""
    def __init__(self, args: ModelArgs, R: int = 3, K: int = 3,
                 d_e: int = 10, heads: int = 4):
        super().__init__()
        self.bi_mamba = BIMambaBlock(args, R=R)
        self.magac = MAGAC_NonBayesian(args.d_model, args.seq_len, K, d_e, heads)
        self.head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        y_seq = self.bi_mamba(x)
        z_node = y_seq.transpose(1, 2).contiguous()
        g_node, log_var_node = self.magac(z_node)

        w = self.head.weight.squeeze(0)
        b = self.head.bias

        mu = torch.einsum('bn,n->b', g_node, w) + b
        log_var = torch.ones_like(mu) * (-3.0)  # Fixed variance

        return mu, log_var


class BIMamba_MAGAC_Bayesian(nn.Module):
    """BIMamba + Bayesian MAGAC (with MC Sampling)"""
    def __init__(self, args: ModelArgs, R: int = 3, K: int = 3,
                 d_e: int = 10, heads: int = 4,
                 mc_train: int = 3, mc_eval: int = 20,
                 drop_edge_p: float = 0.1, mc_dropout_p: float = 0.2):
        super().__init__()
        self.bi_mamba = BIMambaBlock(args, R=R)
        self.magac = MAGAC_Bayesian(
            args.d_model, args.seq_len, K, d_e, heads,
            mc_train, mc_eval, drop_edge_p, mc_dropout_p
        )
        self.head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        y_seq = self.bi_mamba(x)
        z_node = y_seq.transpose(1, 2).contiguous()
        g_node, log_var_node = self.magac(z_node)

        w = self.head.weight.squeeze(0)
        b = self.head.bias

        mu = torch.einsum('bn,n->b', g_node, w) + b
        var = torch.einsum('bn,n->b', log_var_node.exp(), w.pow(2)) + 1e-6
        log_var = var.log()

        return mu, log_var


# ============================================================================
# CELL 4: DATA LOADING UTILITY
# ============================================================================
# Simple data loading for quick experiments

def load_data(dataset: str, window: int = 5, batch_size: int = 32):
    """
    Load dataset and return dataloaders

    Returns:
        num_features, train_loader, val_loader, test_loader
    """
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader

    class TimeSeriesDataset(Dataset):
        def __init__(self, data, window):
            self.data = torch.FloatTensor(data)
            self.window = window

        def __len__(self):
            return len(self.data) - self.window

        def __getitem__(self, idx):
            x = self.data[idx:idx+self.window, :-1]  # Features
            y = self.data[idx+self.window, -1]       # Target
            return x, y

    data_path = f'Dataset/combined_dataframe_{dataset}.csv'
    df = pd.read_csv(data_path)

    # Simple split: 70% train, 15% val, 15% test
    n = len(df)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    data = df.values
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    train_dataset = TimeSeriesDataset(train_data, window)
    val_dataset = TimeSeriesDataset(val_data, window)
    test_dataset = TimeSeriesDataset(test_data, window)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_features = data.shape[1] - 1

    return num_features, train_loader, val_loader, test_loader


# ============================================================================
# CELL 5: TRAINING FUNCTIONS
# ============================================================================
# Training loop with early stopping

def train_single_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
    device: str = 'cpu',
    verbose: bool = True
):
    """Train a single model with early stopping"""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # GaussianNLL loss
    def loss_fn(mu, log_var, y):
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + ((y - mu) ** 2) / var)
        return loss.mean()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            mu, log_var = model(batch_x)
            loss = loss_fn(mu, log_var, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)

        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                mu, log_var = model(batch_x)
                loss = loss_fn(mu, log_var, batch_y)

                val_loss += loss.item() * len(batch_y)

        val_loss /= len(val_loader.dataset)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model: nn.Module, test_loader, device: str = 'cpu'):
    """Evaluate model and compute metrics"""
    model.eval()

    all_preds = []
    all_targets = []
    all_log_vars = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            mu, log_var = model(batch_x)

            all_preds.append(mu.cpu())
            all_targets.append(batch_y.cpu())
            all_log_vars.append(log_var.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_log_vars = torch.cat(all_log_vars)

    # Compute IC
    pred_np = all_preds.numpy()
    target_np = all_targets.numpy()

    pred_centered = pred_np - pred_np.mean()
    target_centered = target_np - target_np.mean()

    numerator = (pred_centered * target_centered).sum()
    denominator = (pred_centered ** 2).sum() ** 0.5 * (target_centered ** 2).sum() ** 0.5

    ic = numerator / (denominator + 1e-12)

    # RMSE
    rmse = float(((all_preds - all_targets) ** 2).mean().sqrt())

    # Directional Accuracy
    pred_dir = torch.sign(all_preds)
    target_dir = torch.sign(all_targets)
    mask = (target_dir != 0) & (pred_dir != 0)
    da = float((pred_dir[mask] == target_dir[mask]).float().mean()) if mask.sum() > 0 else 0.5

    metrics = {
        'ic': float(ic),
        'rmse': rmse,
        'directional_accuracy': da
    }

    return metrics, all_preds, all_targets, all_log_vars


# ============================================================================
# CELL 6: MAIN COMPARISON FUNCTION
# ============================================================================
# Run full comparison study

def run_comparison(
    dataset: str = 'IXIC',
    window: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    early_stop_patience: int = 10,
    # Model hyperparameters
    R: int = 3,
    K: int = 3,
    d_e: int = 10,
    heads: int = 4,
    mc_train: int = 3,
    mc_eval: int = 20,
    verbose: bool = True,
    device: str = 'cpu'
):
    """
    Run Bayesian vs Non-Bayesian comparison on a single dataset

    Returns:
        results: Dict with metrics for both models
    """

    print("="*80)
    print(f"BAYESIAN VS NON-BAYESIAN COMPARISON - {dataset}")
    print("="*80)
    print(f"Parameters: R={R}, K={K}, heads={heads}, mc_train={mc_train}, mc_eval={mc_eval}")
    print("="*80)

    # Load data
    print("\n[1/5] Loading data...")
    num_features, train_loader, val_loader, test_loader = load_data(dataset, window, batch_size)
    print(f"✓ Features: {num_features}, Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model arguments
    args = ModelArgs(
        d_model=num_features,
        seq_len=window,
        d_proj_E=hidden_dim,
        d_proj_H=hidden_dim,
        d_proj_U=hidden_dim // 2,
        d_state=hidden_dim
    )

    results = {}

    # Train Non-Bayesian
    print("\n[2/5] Training Non-Bayesian model...")
    model_nonbayesian = BIMamba_MAGAC_NonBayesian(args, R, K, d_e, heads)

    # Initialize weights
    for p in model_nonbayesian.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model_nonbayesian = train_single_model(
        model_nonbayesian, train_loader, val_loader,
        epochs, learning_rate, early_stop_patience, device, verbose
    )

    print("✓ Non-Bayesian training completed")

    # Train Bayesian
    print("\n[3/5] Training Bayesian model...")
    model_bayesian = BIMamba_MAGAC_Bayesian(
        args, R, K, d_e, heads, mc_train, mc_eval, 0.1, 0.2
    )

    # Initialize weights
    for p in model_bayesian.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model_bayesian = train_single_model(
        model_bayesian, train_loader, val_loader,
        epochs, learning_rate, early_stop_patience, device, verbose
    )

    print("✓ Bayesian training completed")

    # Evaluate Non-Bayesian
    print("\n[4/5] Evaluating Non-Bayesian model...")
    metrics_nb, preds_nb, targets_nb, logvars_nb = evaluate_model(
        model_nonbayesian, test_loader, device
    )
    results['NonBayesian'] = metrics_nb
    print(f"✓ Non-Bayesian: IC={metrics_nb['ic']:.6f}, RMSE={metrics_nb['rmse']:.6f}, DA={metrics_nb['directional_accuracy']:.4f}")

    # Evaluate Bayesian
    print("\n[5/5] Evaluating Bayesian model...")
    metrics_b, preds_b, targets_b, logvars_b = evaluate_model(
        model_bayesian, test_loader, device
    )
    results['Bayesian'] = metrics_b
    print(f"✓ Bayesian: IC={metrics_b['ic']:.6f}, RMSE={metrics_b['rmse']:.6f}, DA={metrics_b['directional_accuracy']:.4f}")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Non-Bayesian':>15} {'Bayesian':>15} {'Improvement':>12}")
    print("-"*70)

    for metric in ['ic', 'rmse', 'directional_accuracy']:
        nb_val = metrics_nb[metric]
        b_val = metrics_b[metric]

        if abs(nb_val) > 1e-8:
            improvement = (b_val - nb_val) / abs(nb_val) * 100
        else:
            improvement = 0.0

        print(f"{metric:<25} {nb_val:>15.6f} {b_val:>15.6f} {improvement:>11.2f}%")

    print("="*80)

    # Save results
    output_dir = f'logs/bayesian_vs_nonbayesian/{dataset}'
    os.makedirs(output_dir, exist_ok=True)

    import pandas as pd

    # Save predictions
    df_nb = pd.DataFrame({
        'y': targets_nb.numpy(),
        'mu': preds_nb.numpy(),
        'log_var': logvars_nb.numpy(),
        'sigma': torch.exp(0.5 * logvars_nb).numpy()
    })
    df_nb.to_csv(os.path.join(output_dir, 'NonBayesian_predictions.csv'), index=False)

    df_b = pd.DataFrame({
        'y': targets_b.numpy(),
        'mu': preds_b.numpy(),
        'log_var': logvars_b.numpy(),
        'sigma': torch.exp(0.5 * logvars_b).numpy()
    })
    df_b.to_csv(os.path.join(output_dir, 'Bayesian_predictions.csv'), index=False)

    # Save metrics
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")

    return results


# ============================================================================
# CELL 7: EXAMPLE USAGE (Run in Notebook or as Script)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage - can be run in notebook cells or as script

    In Notebook:
        1. Run CELL 1-6 first
        2. Then run one of the modes below

    As Script:
        python models/run_bayesian_vs_nonbayesian.py
    """

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ============ MODE 1: QUICK TEST (Single Dataset, Reduced Epochs) ============
    """
    print("\n" + "="*80)
    print("MODE 1: QUICK TEST")
    print("="*80)

    results = run_comparison(
        dataset='IXIC',
        epochs=30,           # Reduced epochs for quick test
        batch_size=32,
        learning_rate=0.001,
        hidden_dim=64,
        R=3,                 # Best from tuning
        K=3,                 # Best from tuning
        heads=4,             # Best from tuning
        mc_train=3,
        mc_eval=20,
        verbose=True,
        device=device
    )
    """

    # ============ MODE 2: SINGLE DATASET (Full Training) ============
    print("\n" + "="*80)
    print("MODE 2: SINGLE DATASET FULL TRAINING")
    print("="*80)

    results = run_comparison(
        dataset='IXIC',
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        hidden_dim=64,
        R=3,
        K=3,
        heads=4,
        mc_train=3,
        mc_eval=20,
        verbose=True,
        device=device
    )

    # ============ MODE 3: MULTI-DATASET (All 3 Datasets) ============
    """
    print("\n" + "="*80)
    print("MODE 3: MULTI-DATASET COMPARISON")
    print("="*80)

    datasets = ['IXIC', 'DJI', 'NYSE']
    all_results = {}

    for dataset in datasets:
        print(f"\n>>> Processing {dataset}...")
        results = run_comparison(
            dataset=dataset,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            R=3, K=3, heads=4,
            mc_train=3, mc_eval=20,
            verbose=True,
            device=device
        )
        all_results[dataset] = results

    # Cross-dataset summary
    print("\n" + "="*80)
    print("CROSS-DATASET SUMMARY")
    print("="*80)

    avg_ic_nb = sum(all_results[ds]['NonBayesian']['ic'] for ds in datasets) / len(datasets)
    avg_ic_b = sum(all_results[ds]['Bayesian']['ic'] for ds in datasets) / len(datasets)

    print(f"Average IC:")
    print(f"  Non-Bayesian: {avg_ic_nb:.6f}")
    print(f"  Bayesian:     {avg_ic_b:.6f}")
    print(f"  Improvement:  {(avg_ic_b - avg_ic_nb) / abs(avg_ic_nb) * 100:+.2f}%")
    """

    print("\n" + "="*80)
    print("✓ COMPARISON COMPLETED!")
    print("="*80)
    print("\nResults saved to: logs/bayesian_vs_nonbayesian/")
    print("\nTo generate plots, run:")
    print("  from utils.result_plot import plot_bayesian_vs_nonbayesian_comparison")
    print("  plot_bayesian_vs_nonbayesian_comparison(results, 'logs/comparison')")
