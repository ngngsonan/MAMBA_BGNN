"""
Bayesian vs Non-Bayesian MAMBA-BGNN Comparison Study

This module performs comprehensive comparison between:
    1. Bayesian MAGAC (with MC sampling & uncertainty quantification)
    2. Non-Bayesian MAGAC (deterministic, no MC sampling)

Experiments run on 3 datasets: IXIC, DJI, NYSE

CRITICAL TRAINING OPTIMIZATIONS:
================================

1. Loss Function Selection:
   - Non-Bayesian: Deterministic loss (Huber) - fixed variance
   - Bayesian: GaussianNLL - optimizes both mean & variance via MC sampling

2. Gradient Clipping (ESSENTIAL for Bayesian):
   - Enabled for BOTH models: max_grad_norm=5.0
   - CRITICAL for Bayesian models due to:
     * High-variance gradients from MC sampling (3 samples train, 20 eval)
     * Stochastic graph structures (DropEdge, MC Dropout)
     * Variance term explosion in GaussianNLL loss
   - Prevents training instability and NaN losses

3. Learning Rate Scheduler:
   - MultiStepLR with milestones=[40, 60, 80], gamma=0.1
   - Matches mamba_bgnn.py configuration
   - Enables better convergence for long training runs

This configuration ensures fair comparison where each model is optimized
with its appropriate loss AND training strategy, matching ablation studies:
    - bimamba_bgnn_tuneparams.py: Bayesian with GaussianNLL
    - mamba_gnn_study.py: Non-Bayesian with deterministic loss
    - mamba_bgnn.py: Training infrastructure with grad clipping

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

# Import utilities
from utils.data_processing import data_processing
from utils.baseline_trainer import train_models, calculate_cross_sectional_metrics, compare_single_vs_cross_sectional_ic
import pandas as pd
import numpy as np


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
# CELL 4: ADVANCED ANALYSIS FUNCTIONS
# ============================================================================
# Financial metrics and market regime analysis

def compute_sharpe_ratio(predictions: np.ndarray, targets: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio from predictions

    Args:
        predictions: Model predictions
        targets: True targets
        risk_free_rate: Risk-free rate (default 0)

    Returns:
        Sharpe ratio (annualized)
    """
    # Simulate returns based on predictions
    returns = predictions * targets  # Assuming predictions are signals

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    # Annualized Sharpe (assuming daily data, 252 trading days)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = (mean_return - risk_free_rate) / (std_return + 1e-8) * np.sqrt(252)

    return float(sharpe)


def compute_max_drawdown(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute maximum drawdown from cumulative returns

    Args:
        predictions: Model predictions (signals)
        targets: True targets

    Returns:
        Maximum drawdown (%)
    """
    returns = predictions * targets
    cumulative = np.cumsum(returns)

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max)
    max_dd = np.min(drawdown)

    return float(max_dd)


def compute_ric(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Rank IC (Spearman correlation)

    Args:
        predictions: Model predictions
        targets: True targets

    Returns:
        Rank IC
    """
    from scipy.stats import spearmanr

    if len(predictions) < 2:
        return 0.0

    ric, _ = spearmanr(predictions, targets)
    return float(ric) if not np.isnan(ric) else 0.0


def compute_crps(predictions: np.ndarray, sigmas: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Continuous Ranked Probability Score

    CRPS measures the quality of probabilistic predictions
    Lower is better

    Args:
        predictions: Mean predictions
        sigmas: Standard deviations
        targets: True targets

    Returns:
        CRPS score
    """
    from scipy.special import erf

    sigmas = np.maximum(sigmas, 1e-6)
    z = (targets - predictions) / sigmas

    # CRPS for Gaussian distribution
    crps = sigmas * (z * (2 * 0.5 * (1 + erf(z / np.sqrt(2)))) - 1 / np.sqrt(np.pi) - 1 / np.sqrt(np.pi) * np.exp(-0.5 * z**2))

    return float(np.mean(crps))


def compute_calibration_metrics(predictions: np.ndarray, sigmas: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Compute calibration metrics for uncertainty quantification

    Args:
        predictions: Mean predictions
        sigmas: Standard deviations
        targets: True targets

    Returns:
        Dict with calibration metrics
    """
    from scipy.special import erf

    sigmas = np.maximum(sigmas, 1e-6)

    # Coverage at different confidence levels
    coverages = {}
    for confidence in [0.68, 0.95, 0.99]:  # 1σ, 2σ, 3σ
        z_score = np.sqrt(2) * np.sqrt(-np.log(1 - confidence))
        lower = predictions - z_score * sigmas
        upper = predictions + z_score * sigmas

        coverage = np.mean((targets >= lower) & (targets <= upper))
        coverages[f'coverage_{int(confidence*100)}'] = float(coverage)
        coverages[f'gap_{int(confidence*100)}'] = float(abs(coverage - confidence))

    # Mean calibration error
    mce = np.mean([coverages[f'gap_{int(c*100)}'] for c in [0.68, 0.95, 0.99]])
    coverages['mean_calibration_error'] = float(mce)

    return coverages


def analyze_market_regime(targets: np.ndarray, window: int = 20) -> Dict:
    """
    Analyze market regime based on volatility

    Classifies periods into:
    - Low volatility (stable)
    - Medium volatility
    - High volatility (turbulent)

    Args:
        targets: True target values
        window: Rolling window for volatility calculation

    Returns:
        Dict with regime indices and statistics
    """
    # Compute rolling volatility
    rolling_std = pd.Series(targets).rolling(window=window, min_periods=1).std().values

    # Define regime thresholds (using percentiles)
    low_threshold = np.percentile(rolling_std, 33)
    high_threshold = np.percentile(rolling_std, 67)

    # Classify regimes
    regime = np.zeros(len(targets), dtype=int)
    regime[rolling_std <= low_threshold] = 0  # Low volatility
    regime[(rolling_std > low_threshold) & (rolling_std <= high_threshold)] = 1  # Medium
    regime[rolling_std > high_threshold] = 2  # High volatility

    return {
        'regime': regime,
        'rolling_std': rolling_std,
        'low_vol_pct': float(np.mean(regime == 0) * 100),
        'med_vol_pct': float(np.mean(regime == 1) * 100),
        'high_vol_pct': float(np.mean(regime == 2) * 100),
        'low_threshold': float(low_threshold),
        'high_threshold': float(high_threshold)
    }


def compute_regime_performance(predictions: np.ndarray, targets: np.ndarray, regime: np.ndarray) -> Dict:
    """
    Compute performance metrics for each market regime

    Args:
        predictions: Model predictions
        targets: True targets
        regime: Regime classification (0=low vol, 1=med vol, 2=high vol)

    Returns:
        Dict with metrics per regime
    """
    results = {}
    regime_names = {0: 'low_volatility', 1: 'medium_volatility', 2: 'high_volatility'}

    for regime_id, regime_name in regime_names.items():
        mask = (regime == regime_id)
        if mask.sum() < 5:  # Need minimum samples
            continue

        pred_regime = predictions[mask]
        target_regime = targets[mask]

        # IC
        pred_c = pred_regime - pred_regime.mean()
        target_c = target_regime - target_regime.mean()
        ic = (pred_c * target_c).sum() / (np.sqrt((pred_c**2).sum()) * np.sqrt((target_c**2).sum()) + 1e-12)

        # RMSE
        rmse = np.sqrt(np.mean((pred_regime - target_regime)**2))

        # Directional Accuracy
        pred_dir = np.sign(pred_regime)
        target_dir = np.sign(target_regime)
        dir_mask = (target_dir != 0) & (pred_dir != 0)
        da = np.mean(pred_dir[dir_mask] == target_dir[dir_mask]) if dir_mask.sum() > 0 else 0.5

        results[regime_name] = {
            'ic': float(ic),
            'rmse': float(rmse),
            'directional_accuracy': float(da),
            'sample_count': int(mask.sum())
        }

    return results


def compute_uncertainty_weighted_ic(predictions: np.ndarray, sigmas: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute uncertainty-weighted IC

    Predictions with higher confidence (lower sigma) get higher weight
    This rewards the model for being confident when correct

    Args:
        predictions: Model predictions
        sigmas: Standard deviations
        targets: True targets

    Returns:
        Weighted IC
    """
    # Weight by inverse uncertainty (confidence)
    weights = 1.0 / (sigmas + 1e-6)
    weights = weights / weights.sum()  # Normalize

    # Weighted correlation
    pred_weighted = predictions * weights
    target_weighted = targets * weights

    pred_c = pred_weighted - pred_weighted.mean()
    target_c = target_weighted - target_weighted.mean()

    ic = (pred_c * target_c).sum() / (np.sqrt((pred_c**2).sum()) * np.sqrt((target_c**2).sum()) + 1e-12)

    return float(ic)


def compute_risk_adjusted_returns(predictions: np.ndarray, sigmas: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Compute risk-adjusted returns using uncertainty for position sizing

    Strategy: Position size inversely proportional to uncertainty
    Lower uncertainty = larger position

    Args:
        predictions: Model predictions (signals)
        sigmas: Standard deviations
        targets: True targets

    Returns:
        Dict with risk-adjusted metrics
    """
    # Position sizing based on uncertainty
    # High confidence (low sigma) -> larger position
    max_position = 1.0
    positions = max_position / (1 + sigmas)  # Simple inverse relationship
    positions = positions / positions.max()  # Normalize to [0, 1]

    # Returns with position sizing
    base_returns = predictions * targets
    risk_adjusted_returns = positions * base_returns

    # Compute metrics
    total_return_base = base_returns.sum()
    total_return_adjusted = risk_adjusted_returns.sum()

    sharpe_base = (base_returns.mean() / (base_returns.std() + 1e-8)) * np.sqrt(252)
    sharpe_adjusted = (risk_adjusted_returns.mean() / (risk_adjusted_returns.std() + 1e-8)) * np.sqrt(252)

    return {
        'total_return_base': float(total_return_base),
        'total_return_adjusted': float(total_return_adjusted),
        'sharpe_base': float(sharpe_base),
        'sharpe_adjusted': float(sharpe_adjusted),
        'improvement': float((sharpe_adjusted - sharpe_base) / (abs(sharpe_base) + 1e-8) * 100),
        'avg_position_size': float(positions.mean())
    }


def compute_sharpness(sigmas: np.ndarray) -> float:
    """
    Compute sharpness of uncertainty estimates

    Sharpness measures how "confident" the model is on average
    Lower sharpness (lower sigma) is better IF well-calibrated

    Args:
        sigmas: Standard deviations

    Returns:
        Mean sharpness (average uncertainty)
    """
    return float(np.mean(sigmas))


def compute_prediction_interval_metrics(predictions: np.ndarray, sigmas: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Compute prediction interval metrics

    Args:
        predictions: Mean predictions
        sigmas: Standard deviations
        targets: True targets

    Returns:
        Dict with interval metrics
    """
    # 95% prediction interval
    z_95 = 1.96
    lower_95 = predictions - z_95 * sigmas
    upper_95 = predictions + z_95 * sigmas
    width_95 = upper_95 - lower_95

    # 68% prediction interval (1 sigma)
    lower_68 = predictions - sigmas
    upper_68 = predictions + sigmas
    width_68 = upper_68 - lower_68

    # Coverage
    coverage_95 = np.mean((targets >= lower_95) & (targets <= upper_95))
    coverage_68 = np.mean((targets >= lower_68) & (targets <= upper_68))

    return {
        'mean_width_95': float(np.mean(width_95)),
        'mean_width_68': float(np.mean(width_68)),
        'median_width_95': float(np.median(width_95)),
        'median_width_68': float(np.median(width_68)),
        'coverage_95': float(coverage_95),
        'coverage_68': float(coverage_68),
        'width_std_95': float(np.std(width_95)),
        'width_std_68': float(np.std(width_68))
    }


def compute_confidence_accuracy_relationship(predictions: np.ndarray, sigmas: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Analyze relationship between confidence (1/sigma) and accuracy

    Models should be more accurate when confident (low sigma)

    Args:
        predictions: Model predictions
        sigmas: Standard deviations
        targets: True targets
        n_bins: Number of bins for analysis

    Returns:
        Dict with binned analysis
    """
    from scipy.stats import spearmanr

    # Compute confidence and error
    confidence = 1.0 / (sigmas + 1e-6)
    errors = np.abs(predictions - targets)

    # Bin by confidence
    confidence_percentiles = np.percentile(confidence, np.linspace(0, 100, n_bins + 1))

    bin_results = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidence >= confidence_percentiles[i])
        else:
            mask = (confidence >= confidence_percentiles[i]) & (confidence < confidence_percentiles[i+1])

        if mask.sum() > 0:
            bin_results.append({
                'bin': i + 1,
                'confidence_min': float(confidence_percentiles[i]),
                'confidence_max': float(confidence_percentiles[i+1]) if i < n_bins - 1 else float(confidence.max()),
                'mean_confidence': float(confidence[mask].mean()),
                'mean_sigma': float(sigmas[mask].mean()),
                'mean_error': float(errors[mask].mean()),
                'std_error': float(errors[mask].std()),
                'sample_count': int(mask.sum())
            })

    # Overall correlation
    corr, p_value = spearmanr(confidence, errors)

    return {
        'bins': bin_results,
        'correlation_confidence_vs_error': float(corr),
        'correlation_p_value': float(p_value),
        'expected_trend': 'negative (higher confidence -> lower error)'
    }


def analyze_uncertainty_contribution(
    predictions_nb: np.ndarray,
    predictions_b: np.ndarray,
    sigmas_b: np.ndarray,
    targets: np.ndarray
) -> Dict:
    """
    Analyze the contribution of Bayesian uncertainty to performance improvement

    Compares scenarios:
    1. Non-Bayesian (no uncertainty)
    2. Bayesian mean only (ignore uncertainty)
    3. Bayesian with uncertainty (full model)

    Args:
        predictions_nb: Non-Bayesian predictions
        predictions_b: Bayesian mean predictions
        sigmas_b: Bayesian uncertainties
        targets: True targets

    Returns:
        Dict with contribution analysis
    """
    # Compute errors
    error_nb = np.abs(predictions_nb - targets)
    error_b = np.abs(predictions_b - targets)

    # Find samples where Bayesian is more uncertain
    high_uncertainty_mask = sigmas_b > np.percentile(sigmas_b, 75)
    low_uncertainty_mask = sigmas_b <= np.percentile(sigmas_b, 25)

    # Performance when uncertain vs confident
    results = {
        'high_uncertainty': {
            'bayesian_error': float(error_b[high_uncertainty_mask].mean()),
            'nonbayesian_error': float(error_nb[high_uncertainty_mask].mean()),
            'sample_count': int(high_uncertainty_mask.sum()),
            'mean_sigma': float(sigmas_b[high_uncertainty_mask].mean())
        },
        'low_uncertainty': {
            'bayesian_error': float(error_b[low_uncertainty_mask].mean()),
            'nonbayesian_error': float(error_nb[low_uncertainty_mask].mean()),
            'sample_count': int(low_uncertainty_mask.sum()),
            'mean_sigma': float(sigmas_b[low_uncertainty_mask].mean())
        }
    }

    # Uncertainty helps identify difficult samples
    # Correlation between Bayesian uncertainty and Non-Bayesian error
    from scipy.stats import spearmanr
    corr, p_val = spearmanr(sigmas_b, error_nb)

    results['uncertainty_identifies_difficulty'] = {
        'correlation': float(corr),
        'p_value': float(p_val),
        'interpretation': 'Positive correlation means Bayesian uncertainty identifies samples where Non-Bayesian struggles'
    }

    return results


# ============================================================================
# CELL 5: DATA LOADING UTILITY
# ============================================================================
# Simple data loading for quick experiments

def load_data(dataset: str, window: int = 5, batch_size: int = 32):
    """
    Load dataset and return dataloaders

    Returns:
        num_features, train_loader, val_loader, test_loader
    """
    import pandas as pd
    import numpy as np
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

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Convert to float
    df = df.astype(np.float32)

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
# CELL 6: MAIN COMPARISON FUNCTION (Unified Pipeline)
# ============================================================================
# Run full comparison study using unified training pipeline

def run_comparison(
    dataset: str = 'IXIC',
    window: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    loss_type: str = 'auto',
    early_stop_patience: int = 10,
    # Model hyperparameters
    R: int = 3,
    K: int = 3,
    d_e: int = 10,
    heads: int = 4,
    mc_train: int = 3,
    mc_eval: int = 20,
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu',
    study_timestamp: str = None  # Optional timestamp for study directory
):
    """
    Run Bayesian vs Non-Bayesian comparison on a single dataset
    Uses unified training pipeline from utils.baseline_trainer

    IMPORTANT: Training Configuration Strategy
    ------------------------------------------
    Models are trained with DIFFERENT configurations for optimal performance:

    1. Loss Functions:
       - Non-Bayesian: 'auto' → Huber loss (deterministic, robust to outliers)
       - Bayesian: 'nll' → GaussianNLL (optimizes mean + variance from MC sampling)

    2. Gradient Clipping (CRITICAL for Bayesian models):
       - Enabled for BOTH models with max_grad_norm=5.0
       - Essential for Bayesian due to high-variance gradients from MC sampling
       - Prevents gradient explosion from stochastic graph structures (DropEdge, MC Dropout)
       - Stabilizes variance term in GaussianNLL loss

    3. Learning Rate Scheduler:
       - MultiStepLR with milestones=[40, 60, 80], gamma=0.1
       - Matches mamba_bgnn.py configuration for consistency
       - Enables better convergence for long training runs

    This matches the ablation studies in:
    - bimamba_bgnn_tuneparams.py: Bayesian with GaussianNLL
    - mamba_gnn_study.py: Non-Bayesian with deterministic loss
    - mamba_bgnn.py: Training infrastructure with grad clipping support

    Args:
        dataset: Dataset name (IXIC, DJI, NYSE)
        window: Sequence length
        batch_size: Batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        loss_type: Legacy parameter (kept for compatibility, but ignored)
                   Models use their optimal loss automatically
        early_stop_patience: Early stopping patience
        R: Number of BIMamba layers
        K: Chebyshev polynomial order
        d_e: Node embedding dimension
        heads: Number of attention heads
        mc_train: MC samples during training (Bayesian only)
        mc_eval: MC samples during evaluation (Bayesian only)
        verbose: Print progress
        log_base_dir: Base directory for logs
        device: Device to train on
        study_timestamp: Optional timestamp string. If provided, will use existing study directory.
                        If None, creates new timestamp.

    Returns:
        Dict with:
            - results: Comprehensive metrics for both models
            - study_name: Study directory name
            - study_timestamp: Timestamp used
    """

    # Create study name with timestamp
    if study_timestamp is None:
        study_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    study_name = f'bayesian_vs_nonbayesian_{study_timestamp}'

    print("="*80)
    print(f"BAYESIAN VS NON-BAYESIAN COMPARISON - {dataset}")
    print("="*80)
    print(f"Study: {study_name}")
    print(f"Architecture: R={R}, K={K}, heads={heads}, mc_train={mc_train}, mc_eval={mc_eval}")
    print(f"Loss: Non-Bayesian=Huber, Bayesian=GaussianNLL")
    print(f"Training: LR={learning_rate}, Epochs={epochs}, Grad Clip=True (max_norm=5.0)")
    print(f"Scheduler: MultiStepLR (milestones=[40, 60, 80], gamma=0.1)")
    print("="*80)

    # Load data using unified pipeline
    print("\nLoading data...")
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'
    num_features, train_loader, val_loader, test_loader = data_processing(
        data_path, window, batch_size
    )
    print(f"✓ Data loaded: {num_features} features")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    # Model arguments
    args = ModelArgs(
        d_model=num_features,
        seq_len=window,
        d_proj_E=hidden_dim,
        d_proj_H=hidden_dim,
        d_proj_U=hidden_dim // 2,
        d_state=hidden_dim
    )

    # Create models
    print("\nCreating models...")

    # Non-Bayesian model
    model_nb = BIMamba_MAGAC_NonBayesian(args, R, K, d_e, heads)
    for p in model_nb.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Bayesian model
    model_b = BIMamba_MAGAC_Bayesian(args, R, K, d_e, heads, mc_train, mc_eval, 0.1, 0.2)
    for p in model_b.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # IMPORTANT: Train models separately with appropriate loss functions
    # Non-Bayesian: Use deterministic loss (MSE/Huber) - fixed variance
    # Bayesian: Use GaussianNLL - optimizes both mean and variance via MC sampling

    print("\n" + "="*80)
    print("TRAINING NON-BAYESIAN MODEL")
    print("="*80)
    print("Loss function: Deterministic (auto → Huber for MAMBA variants)")
    print("Reason: Fixed variance, no MC sampling")
    print("Gradient clipping: ENABLED (max_norm=5.0) - prevents gradient instability")
    print("LR scheduler: MultiStepLR - improves convergence")

    config_nb = {
        'epochs': epochs,
        'lr': learning_rate,
        'loss_type': 'auto',  # Non-Bayesian uses deterministic loss
        'patience': early_stop_patience,
        'optimizer_fn': lambda params, lr: torch.optim.Adam(params, lr=lr),
        'scheduler_fn': None, #lambda opt: torch.optim.lr_scheduler.MultiStepLR(
            # opt, milestones=[30, 60, 90], gamma=0.1
        # ),
        'grad_clip': True,  # Enable gradient clipping
        'max_grad_norm': 5.0  # Match mamba_bgnn.py configuration
    }

    results_nb = train_models(
        models_dict={'BIMamba-MAGAC_NonBayesian': model_nb},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=dataset,
        config=config_nb,
        log_base_dir=log_base_dir,
        study_name=study_name,
        verbose=verbose,
        device=device,
        use_nested_structure=True
    )

    print("\n" + "="*80)
    print("TRAINING BAYESIAN MODEL")
    print("="*80)
    print("Loss function: GaussianNLL")
    print("Reason: Optimizes both mean and variance from MC sampling")
    print("Gradient clipping: ENABLED (max_norm=5.0) - CRITICAL for Bayesian!")
    print("  → Stabilizes high-variance gradients from MC sampling")
    print("  → Prevents explosion in variance term of GaussianNLL")
    print("  → Controls gradients from stochastic graph (DropEdge, MC Dropout)")
    print("LR scheduler: MultiStepLR - ensures stable convergence")

    config_b = {
        'epochs': epochs,
        'lr': learning_rate,
        'loss_type': 'nll',  # Bayesian uses GaussianNLL for uncertainty
        'patience': early_stop_patience,
        'optimizer_fn': lambda params, lr: torch.optim.Adam(params, lr=lr),
        'scheduler_fn': None, #lambda opt: torch.optim.lr_scheduler.MultiStepLR(
        #     opt, milestones=[40, 60, 80], gamma=0.1
        # ),
        'grad_clip': True,  # CRITICAL for Bayesian models!
        'max_grad_norm': 5.0  # Match mamba_bgnn.py configuration
    }

    results_b = train_models(
        models_dict={'BIMamba-MAGAC_Bayesian': model_b},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=dataset,
        config=config_b,
        log_base_dir=log_base_dir,
        study_name=study_name,
        verbose=verbose,
        device=device,
        use_nested_structure=True
    )

    # Merge results
    results = {**results_nb, **results_b}

    # Load predictions for advanced analysis
    nb_key = 'BIMamba-MAGAC_NonBayesian'
    b_key = 'BIMamba-MAGAC_Bayesian'

    nb_pred_path = f'{log_base_dir}/{study_name}/{dataset}/{nb_key}/test_predictions.csv'
    b_pred_path = f'{log_base_dir}/{study_name}/{dataset}/{b_key}/test_predictions.csv'

    if os.path.exists(nb_pred_path) and os.path.exists(b_pred_path):
        print("\n" + "="*80)
        print("ADVANCED ANALYSIS")
        print("="*80)

        df_nb = pd.read_csv(nb_pred_path)
        df_b = pd.read_csv(b_pred_path)

        y_true = df_nb['y'].values
        mu_nb = df_nb['mu'].values
        sigma_nb = df_nb['sigma'].values
        mu_b = df_b['mu'].values
        sigma_b = df_b['sigma'].values

        # Compute advanced metrics
        print("\n[1/4] Computing financial metrics...")

        # Sharpe ratio
        sharpe_nb = compute_sharpe_ratio(mu_nb, y_true)
        sharpe_b = compute_sharpe_ratio(mu_b, y_true)
        results[nb_key]['sharpe_ratio'] = sharpe_nb
        results[b_key]['sharpe_ratio'] = sharpe_b
        print(f"  Sharpe Ratio - NB: {sharpe_nb:.4f}, B: {sharpe_b:.4f}")

        # Max drawdown
        mdd_nb = compute_max_drawdown(mu_nb, y_true)
        mdd_b = compute_max_drawdown(mu_b, y_true)
        results[nb_key]['max_drawdown'] = mdd_nb
        results[b_key]['max_drawdown'] = mdd_b
        print(f"  Max Drawdown - NB: {mdd_nb:.4f}, B: {mdd_b:.4f}")

        # RIC
        ric_nb = compute_ric(mu_nb, y_true)
        ric_b = compute_ric(mu_b, y_true)
        results[nb_key]['ric'] = ric_nb
        results[b_key]['ric'] = ric_b
        print(f"  RIC - NB: {ric_nb:.4f}, B: {ric_b:.4f}")

        # CRPS (only for Bayesian with uncertainty)
        print("\n[2/4] Computing probabilistic metrics...")
        crps_nb = compute_crps(mu_nb, sigma_nb, y_true)
        crps_b = compute_crps(mu_b, sigma_b, y_true)
        results[nb_key]['crps'] = crps_nb
        results[b_key]['crps'] = crps_b
        print(f"  CRPS - NB: {crps_nb:.6f}, B: {crps_b:.6f}")

        # Calibration metrics
        print("\n[3/4] Computing calibration metrics...")
        calib_nb = compute_calibration_metrics(mu_nb, sigma_nb, y_true)
        calib_b = compute_calibration_metrics(mu_b, sigma_b, y_true)
        results[nb_key]['calibration'] = calib_nb
        results[b_key]['calibration'] = calib_b

        print(f"  Mean Calibration Error - NB: {calib_nb['mean_calibration_error']:.4f}, B: {calib_b['mean_calibration_error']:.4f}")
        print(f"  Coverage@95% - NB: {calib_nb['coverage_95']:.4f}, B: {calib_b['coverage_95']:.4f}")

        # Market regime analysis
        print("\n[4/4] Analyzing market regimes...")
        regime_info = analyze_market_regime(y_true, window=20)
        print(f"  Low volatility:    {regime_info['low_vol_pct']:.1f}%")
        print(f"  Medium volatility: {regime_info['med_vol_pct']:.1f}%")
        print(f"  High volatility:   {regime_info['high_vol_pct']:.1f}%")

        # Performance by regime
        regime_perf_nb = compute_regime_performance(mu_nb, y_true, regime_info['regime'])
        regime_perf_b = compute_regime_performance(mu_b, y_true, regime_info['regime'])
        results[nb_key]['regime_performance'] = regime_perf_nb
        results[b_key]['regime_performance'] = regime_perf_b

        print("\n  Regime Performance:")
        for regime_name in ['low_volatility', 'medium_volatility', 'high_volatility']:
            if regime_name in regime_perf_nb and regime_name in regime_perf_b:
                ic_nb = regime_perf_nb[regime_name]['ic']
                ic_b = regime_perf_b[regime_name]['ic']
                print(f"    {regime_name:20s} - IC: NB={ic_nb:.4f}, B={ic_b:.4f}")

        # Save regime data
        regime_df = pd.DataFrame({
            'regime': regime_info['regime'],
            'rolling_std': regime_info['rolling_std']
        })
        regime_df.to_csv(f'{log_base_dir}/{study_name}/{dataset}/regime_analysis.csv', index=False)

        # ========== NEW: Uncertainty-based metrics ==========
        print("\n" + "="*80)
        print("UNCERTAINTY-BASED ANALYSIS (Exploiting Bayesian Mean + Variance)")
        print("="*80)

        print("\n[1/5] Uncertainty-weighted IC...")
        uw_ic_nb = compute_uncertainty_weighted_ic(mu_nb, sigma_nb, y_true)
        uw_ic_b = compute_uncertainty_weighted_ic(mu_b, sigma_b, y_true)
        results[nb_key]['uncertainty_weighted_ic'] = uw_ic_nb
        results[b_key]['uncertainty_weighted_ic'] = uw_ic_b
        print(f"  UW-IC - NB: {uw_ic_nb:.4f}, B: {uw_ic_b:.4f}")
        print(f"  → Rewards confidence when correct")

        print("\n[2/5] Risk-adjusted returns (position sizing by uncertainty)...")
        ra_returns_nb = compute_risk_adjusted_returns(mu_nb, sigma_nb, y_true)
        ra_returns_b = compute_risk_adjusted_returns(mu_b, sigma_b, y_true)
        results[nb_key]['risk_adjusted_returns'] = ra_returns_nb
        results[b_key]['risk_adjusted_returns'] = ra_returns_b
        print(f"  Base Sharpe - NB: {ra_returns_nb['sharpe_base']:.4f}, B: {ra_returns_b['sharpe_base']:.4f}")
        print(f"  Adjusted Sharpe - NB: {ra_returns_nb['sharpe_adjusted']:.4f}, B: {ra_returns_b['sharpe_adjusted']:.4f}")
        print(f"  Improvement - NB: {ra_returns_nb['improvement']:.2f}%, B: {ra_returns_b['improvement']:.2f}%")
        print(f"  → Lower uncertainty = larger position size")

        print("\n[3/5] Sharpness (average uncertainty)...")
        sharpness_nb = compute_sharpness(sigma_nb)
        sharpness_b = compute_sharpness(sigma_b)
        results[nb_key]['sharpness'] = sharpness_nb
        results[b_key]['sharpness'] = sharpness_b
        print(f"  Sharpness - NB: {sharpness_nb:.6f}, B: {sharpness_b:.6f}")
        print(f"  → Lower is better (more confident) IF calibrated")

        print("\n[4/5] Prediction interval metrics...")
        pi_metrics_nb = compute_prediction_interval_metrics(mu_nb, sigma_nb, y_true)
        pi_metrics_b = compute_prediction_interval_metrics(mu_b, sigma_b, y_true)
        results[nb_key]['prediction_intervals'] = pi_metrics_nb
        results[b_key]['prediction_intervals'] = pi_metrics_b
        print(f"  95% PI Width - NB: {pi_metrics_nb['mean_width_95']:.4f}, B: {pi_metrics_b['mean_width_95']:.4f}")
        print(f"  95% Coverage - NB: {pi_metrics_nb['coverage_95']:.4f}, B: {pi_metrics_b['coverage_95']:.4f}")

        print("\n[5/5] Confidence-accuracy relationship...")
        conf_acc_nb = compute_confidence_accuracy_relationship(mu_nb, sigma_nb, y_true)
        conf_acc_b = compute_confidence_accuracy_relationship(mu_b, sigma_b, y_true)
        results[nb_key]['confidence_accuracy'] = conf_acc_nb
        results[b_key]['confidence_accuracy'] = conf_acc_b
        print(f"  Correlation (Confidence vs Error):")
        print(f"    NB: {conf_acc_nb['correlation_confidence_vs_error']:.4f} (p={conf_acc_nb['correlation_p_value']:.4f})")
        print(f"    B:  {conf_acc_b['correlation_confidence_vs_error']:.4f} (p={conf_acc_b['correlation_p_value']:.4f})")
        print(f"  → Expect negative correlation (high confidence = low error)")

        # Uncertainty contribution analysis
        print("\n" + "-"*80)
        print("UNCERTAINTY CONTRIBUTION ANALYSIS")
        print("-"*80)
        uc_analysis = analyze_uncertainty_contribution(mu_nb, mu_b, sigma_b, y_true)
        results['uncertainty_contribution'] = uc_analysis

        print("\nWhen Bayesian is HIGHLY uncertain (top 25%):")
        print(f"  Bayesian error:     {uc_analysis['high_uncertainty']['bayesian_error']:.6f}")
        print(f"  Non-Bayesian error: {uc_analysis['high_uncertainty']['nonbayesian_error']:.6f}")
        print(f"  Mean sigma:         {uc_analysis['high_uncertainty']['mean_sigma']:.6f}")

        print("\nWhen Bayesian is CONFIDENT (bottom 25%):")
        print(f"  Bayesian error:     {uc_analysis['low_uncertainty']['bayesian_error']:.6f}")
        print(f"  Non-Bayesian error: {uc_analysis['low_uncertainty']['nonbayesian_error']:.6f}")
        print(f"  Mean sigma:         {uc_analysis['low_uncertainty']['mean_sigma']:.6f}")

        print("\nDoes uncertainty identify difficult samples?")
        print(f"  Correlation (Bayesian σ vs Non-Bayesian error): {uc_analysis['uncertainty_identifies_difficulty']['correlation']:.4f}")
        print(f"  p-value: {uc_analysis['uncertainty_identifies_difficulty']['p_value']:.4e}")
        print(f"  → {uc_analysis['uncertainty_identifies_difficulty']['interpretation']}")

        # Save detailed uncertainty analysis
        uncertainty_analysis_path = f'{log_base_dir}/{study_name}/{dataset}/uncertainty_analysis.json'
        with open(uncertainty_analysis_path, 'w') as f:
            json.dump({
                'uncertainty_weighted_ic': {'nb': uw_ic_nb, 'b': uw_ic_b},
                'risk_adjusted_returns': {'nb': ra_returns_nb, 'b': ra_returns_b},
                'sharpness': {'nb': sharpness_nb, 'b': sharpness_b},
                'prediction_intervals': {'nb': pi_metrics_nb, 'b': pi_metrics_b},
                'confidence_accuracy': {'nb': conf_acc_nb, 'b': conf_acc_b},
                'uncertainty_contribution': uc_analysis
            }, f, indent=2)
        print(f"\n✓ Uncertainty analysis saved to: {uncertainty_analysis_path}")

    # Comprehensive comparison summary
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)

    metrics_to_compare = ['ic', 'ric', 'rmse', 'directional_accuracy', 'sharpe_ratio', 'max_drawdown', 'crps']
    print(f"{'Metric':<25} {'Non-Bayesian':>15} {'Bayesian':>15} {'Improvement':>12}")
    print("-"*80)

    for metric in metrics_to_compare:
        if metric in results[nb_key] and metric in results[b_key]:
            nb_val = results[nb_key][metric]
            b_val = results[b_key][metric]

            # For metrics where lower is better (RMSE, drawdown, CRPS)
            if metric in ['rmse', 'max_drawdown', 'crps']:
                if abs(nb_val) > 1e-8:
                    improvement = -(b_val - nb_val) / abs(nb_val) * 100
                else:
                    improvement = 0.0
            else:
                if abs(nb_val) > 1e-8:
                    improvement = (b_val - nb_val) / abs(nb_val) * 100
                else:
                    improvement = 0.0

            print(f"{metric:<25} {nb_val:>15.6f} {b_val:>15.6f} {improvement:>11.2f}%")

    print("="*80)

    # Save comprehensive results
    output_dir = f'{log_base_dir}/{study_name}/{dataset}'
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    with open(os.path.join(output_dir, 'comprehensive_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Comprehensive results saved to: {output_dir}")

    # Return results with study_name for downstream use
    return {
        'results': results,
        'study_name': study_name,
        'study_timestamp': study_timestamp
    }


# ============================================================================
# CELL 7: CROSS-SECTIONAL IC CALCULATION
# ============================================================================

def calculate_cross_sectional_ic_for_bayesian_comparison(
    datasets: List[str] = ['IXIC', 'DJI', 'NYSE'],
    log_base_dir: str = 'logs',
    study_name: str = 'bayesian_vs_nonbayesian',
    output_file: str = None,
    verbose: bool = True
) -> Dict:
    """
    Calculate cross-sectional IC for Bayesian vs Non-Bayesian models

    Similar to calculate_cross_sectional_for_all_models but specifically for
    the Bayesian comparison study.

    Args:
        datasets: List of dataset names
        log_base_dir: Base log directory
        study_name: Study name (can include timestamp, e.g., 'bayesian_vs_nonbayesian_20251119_014842')
        output_file: Optional output file path
        verbose: Print detailed output

    Returns:
        Dict with cross-sectional IC results for both models
    """
    import glob

    if len(datasets) < 2:
        raise ValueError(f"Need at least 2 datasets for cross-sectional IC. Got {len(datasets)}")

    if verbose:
        print("\n" + "="*80)
        print("CROSS-SECTIONAL IC ANALYSIS - BAYESIAN VS NON-BAYESIAN")
        print("="*80)
        print(f"Datasets: {', '.join(datasets)}")
        print("="*80)

    models = ['BIMamba-MAGAC_NonBayesian', 'BIMamba-MAGAC_Bayesian']
    all_results = {}

    for model_name in models:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {model_name}")
            print(f"{'='*70}")

        # Find prediction files
        prediction_files = {}
        for dataset in datasets:
            # Pattern: logs/bayesian_vs_nonbayesian/DATASET/MODEL_NAME/test_predictions.csv
            pred_path = f"{log_base_dir}/{study_name}/{dataset}/{model_name}/test_predictions.csv"

            if os.path.exists(pred_path):
                prediction_files[dataset] = pred_path
                if verbose:
                    print(f"  ✓ {dataset}: {pred_path}")
            else:
                if verbose:
                    print(f"  ⚠️  Not found: {pred_path}")

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
            f.write("CROSS-SECTIONAL IC COMPARISON - BAYESIAN VS NON-BAYESIAN\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Datasets: {', '.join(datasets)}\n")
            f.write("="*80 + "\n\n")

            for model_name in models:
                f.write(f"\n{'='*70}\n")
                f.write(f"{model_name}\n")
                f.write(f"{'='*70}\n")

                if 'error' in all_results[model_name]:
                    f.write(f"Error: {all_results[model_name]['error']}\n")
                else:
                    res = all_results[model_name]['cross_sectional']
                    f.write(f"Datasets used: {', '.join(all_results[model_name]['datasets_used'])}\n")
                    f.write(f"\nCross-Sectional IC:\n")
                    f.write(f"  Mean:       {res['ic_mean']:.6f}\n")
                    f.write(f"  Std:        {res['ic_std']:.6f}\n")
                    f.write(f"  Median:     {res['ic_median']:.6f}\n")
                    f.write(f"  % Positive: {res['ic_positive_ratio']*100:.1f}%\n")
                    f.write(f"\nCross-Sectional RIC:\n")
                    f.write(f"  Mean:       {res['ric_mean']:.6f}\n")
                    f.write(f"  Std:        {res['ric_std']:.6f}\n")
                    f.write(f"  Median:     {res['ric_median']:.6f}\n")
                    f.write(f"  % Positive: {res['ric_positive_ratio']*100:.1f}%\n")

            # Comparison summary
            f.write(f"\n{'='*80}\n")
            f.write("COMPARISON SUMMARY\n")
            f.write(f"{'='*80}\n")

            nb_key = 'BIMamba-MAGAC_NonBayesian'
            b_key = 'BIMamba-MAGAC_Bayesian'

            if nb_key in all_results and 'cross_sectional' in all_results[nb_key] and all_results[nb_key]['cross_sectional']:
                if b_key in all_results and 'cross_sectional' in all_results[b_key] and all_results[b_key]['cross_sectional']:
                    nb_ic = all_results[nb_key]['cross_sectional']['ic_mean']
                    b_ic = all_results[b_key]['cross_sectional']['ic_mean']
                    nb_ric = all_results[nb_key]['cross_sectional']['ric_mean']
                    b_ric = all_results[b_key]['cross_sectional']['ric_mean']

                    ic_improvement = ((b_ic - nb_ic) / abs(nb_ic)) * 100 if abs(nb_ic) > 1e-8 else 0
                    ric_improvement = ((b_ric - nb_ric) / abs(nb_ric)) * 100 if abs(nb_ric) > 1e-8 else 0

                    f.write(f"Cross-Sectional IC:\n")
                    f.write(f"  Non-Bayesian: {nb_ic:.6f}\n")
                    f.write(f"  Bayesian:     {b_ic:.6f}\n")
                    f.write(f"  Improvement:  {ic_improvement:+.2f}%\n\n")

                    f.write(f"Cross-Sectional RIC:\n")
                    f.write(f"  Non-Bayesian: {nb_ric:.6f}\n")
                    f.write(f"  Bayesian:     {b_ric:.6f}\n")
                    f.write(f"  Improvement:  {ric_improvement:+.2f}%\n")

        if verbose:
            print(f"\n✓ Summary saved to: {output_file}")

    return all_results


# ============================================================================
# CELL 8: MULTI-DATASET RUNNER
# ============================================================================

def run_multi_dataset_comparison(
    datasets: List[str] = ['IXIC', 'DJI', 'NYSE'],
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    R: int = 3,
    K: int = 3,
    d_e: int = 10,
    heads: int = 4,
    mc_train: int = 3,
    mc_eval: int = 20,
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu',
    early_stop_patience: int = 10
) -> Dict:
    """
    Run Bayesian vs Non-Bayesian comparison on multiple datasets

    Returns:
        Dict with results for each dataset and cross-dataset summary
    """
    # Create single timestamp for entire study
    study_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    study_name = f'bayesian_vs_nonbayesian_{study_timestamp}'

    print("="*80)
    print("MULTI-DATASET BAYESIAN VS NON-BAYESIAN COMPARISON")
    print("="*80)
    print(f"Study: {study_name}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: BIMamba-MAGAC (Bayesian vs Non-Bayesian)")
    print("="*80)

    all_results = {}

    # Run comparison on each dataset
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset}...")
        print(f"{'='*80}")

        result_dict = run_comparison(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            # Note: loss_type not specified - each model uses its optimal loss automatically
            early_stop_patience=early_stop_patience,
            R=R, K=K, d_e=d_e, heads=heads,
            mc_train=mc_train, mc_eval=mc_eval,
            verbose=verbose,
            log_base_dir=log_base_dir,
            device=device,
            study_timestamp=study_timestamp  # Use same timestamp for all datasets
        )

        all_results[dataset] = result_dict['results']  # Extract results from dict

    # Cross-dataset summary
    print("\n" + "="*80)
    print("CROSS-DATASET SUMMARY")
    print("="*80)

    nb_key = 'BIMamba-MAGAC_NonBayesian'
    b_key = 'BIMamba-MAGAC_Bayesian'

    # Aggregate metrics across datasets
    metrics_to_aggregate = ['ic', 'ric', 'rmse', 'directional_accuracy', 'sharpe_ratio', 'max_drawdown', 'crps']

    summary = {
        'NonBayesian': {},
        'Bayesian': {},
        'Improvement': {}
    }

    for metric in metrics_to_aggregate:
        nb_values = []
        b_values = []

        for dataset in datasets:
            if metric in all_results[dataset].get(nb_key, {}):
                nb_values.append(all_results[dataset][nb_key][metric])
            if metric in all_results[dataset].get(b_key, {}):
                b_values.append(all_results[dataset][b_key][metric])

        if nb_values and b_values:
            avg_nb = np.mean(nb_values)
            avg_b = np.mean(b_values)
            std_nb = np.std(nb_values)
            std_b = np.std(b_values)

            summary['NonBayesian'][metric] = {'mean': float(avg_nb), 'std': float(std_nb)}
            summary['Bayesian'][metric] = {'mean': float(avg_b), 'std': float(std_b)}

            # Compute improvement
            if metric in ['rmse', 'max_drawdown', 'crps']:
                improvement = -(avg_b - avg_nb) / abs(avg_nb) * 100 if abs(avg_nb) > 1e-8 else 0.0
            else:
                improvement = (avg_b - avg_nb) / abs(avg_nb) * 100 if abs(avg_nb) > 1e-8 else 0.0

            summary['Improvement'][metric] = float(improvement)

    # Print summary
    print(f"\n{'Metric':<25} {'Non-Bayesian':>20} {'Bayesian':>20} {'Improvement':>12}")
    print("-"*80)

    for metric in metrics_to_aggregate:
        if metric in summary['NonBayesian']:
            nb_mean = summary['NonBayesian'][metric]['mean']
            nb_std = summary['NonBayesian'][metric]['std']
            b_mean = summary['Bayesian'][metric]['mean']
            b_std = summary['Bayesian'][metric]['std']
            improvement = summary['Improvement'][metric]

            print(f"{metric:<25} {nb_mean:>9.4f}±{nb_std:<7.4f} {b_mean:>9.4f}±{b_std:<7.4f} {improvement:>11.2f}%")

    # Regime analysis summary
    print("\n" + "="*80)
    print("REGIME PERFORMANCE SUMMARY (Average across datasets)")
    print("="*80)

    regime_names = ['low_volatility', 'medium_volatility', 'high_volatility']
    print(f"\n{'Regime':<25} {'Non-Bayesian IC':>20} {'Bayesian IC':>20}")
    print("-"*70)

    for regime_name in regime_names:
        nb_ics = []
        b_ics = []

        for dataset in datasets:
            nb_regime_perf = all_results[dataset].get(nb_key, {}).get('regime_performance', {})
            b_regime_perf = all_results[dataset].get(b_key, {}).get('regime_performance', {})

            if regime_name in nb_regime_perf:
                nb_ics.append(nb_regime_perf[regime_name]['ic'])
            if regime_name in b_regime_perf:
                b_ics.append(b_regime_perf[regime_name]['ic'])

        if nb_ics and b_ics:
            avg_nb_ic = np.mean(nb_ics)
            avg_b_ic = np.mean(b_ics)
            std_nb_ic = np.std(nb_ics)
            std_b_ic = np.std(b_ics)

            print(f"{regime_name:<25} {avg_nb_ic:>9.4f}±{std_nb_ic:<7.4f} {avg_b_ic:>9.4f}±{std_b_ic:<7.4f}")

    print("="*80)

    # Save cross-dataset summary
    summary_path = f'{log_base_dir}/{study_name}/cross_dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'datasets': datasets,
            'summary': summary,
            'detailed_results': {ds: {
                'NonBayesian': all_results[ds].get(nb_key, {}),
                'Bayesian': all_results[ds].get(b_key, {})
            } for ds in datasets}
        }, f, indent=2)

    print(f"\n✓ Cross-dataset summary saved to: {summary_path}")

    # ============================================================================
    # STEP 2: Calculate Cross-Sectional IC
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2: CALCULATING CROSS-SECTIONAL IC")
    print("="*80)

    cross_sectional_results = calculate_cross_sectional_ic_for_bayesian_comparison(
        datasets=datasets,
        log_base_dir=log_base_dir,
        study_name=study_name,
        output_file=f'{log_base_dir}/{study_name}/cross_sectional_ic_summary.txt',
        verbose=verbose
    )

    # Print summary comparison
    print("\n" + "="*80)
    print("CROSS-SECTIONAL IC COMPARISON SUMMARY")
    print("="*80)

    if nb_key in cross_sectional_results and 'cross_sectional' in cross_sectional_results[nb_key]:
        if cross_sectional_results[nb_key]['cross_sectional'] is not None:
            nb_cs = cross_sectional_results[nb_key]['cross_sectional']
            b_cs = cross_sectional_results[b_key]['cross_sectional']

            print(f"\n{'Metric':<30} {'Non-Bayesian':>15} {'Bayesian':>15} {'Improvement':>12}")
            print("-"*75)

            # IC
            nb_ic = nb_cs['ic_mean']
            b_ic = b_cs['ic_mean']
            ic_imp = ((b_ic - nb_ic) / abs(nb_ic)) * 100 if abs(nb_ic) > 1e-8 else 0
            print(f"{'Cross-Sectional IC Mean':<30} {nb_ic:>15.6f} {b_ic:>15.6f} {ic_imp:>11.2f}%")

            # RIC
            nb_ric = nb_cs['ric_mean']
            b_ric = b_cs['ric_mean']
            ric_imp = ((b_ric - nb_ric) / abs(nb_ric)) * 100 if abs(nb_ric) > 1e-8 else 0
            print(f"{'Cross-Sectional RIC Mean':<30} {nb_ric:>15.6f} {b_ric:>15.6f} {ric_imp:>11.2f}%")

            # Positive ratios
            print(f"{'IC Positive Ratio':<30} {nb_cs['ic_positive_ratio']:>15.2%} {b_cs['ic_positive_ratio']:>15.2%}")
            print(f"{'RIC Positive Ratio':<30} {nb_cs['ric_positive_ratio']:>15.2%} {b_cs['ric_positive_ratio']:>15.2%}")

            print("="*80)

            # Interpretation
            print("\n📊 CROSS-SECTIONAL IC INTERPRETATION:")
            max_ic = max(nb_ic, b_ic)
            if max_ic > 0.05:
                print("   🔥 EXCEPTIONAL: IC > 0.05 is extremely rare in real markets!")
            elif max_ic > 0.02:
                print("   🚀 EXCELLENT: IC > 0.02 indicates very strong predictive power")
            elif max_ic > 0.01:
                print("   ✅ GOOD: IC > 0.01 shows solid predictive ability")
            elif max_ic > 0.005:
                print("   📈 DECENT: IC > 0.005 has commercial value")
            else:
                print("   📉 WEAK: IC ≤ 0.005 may not be practically useful")

            print("\nNOTE: Cross-sectional IC measures correlation ACROSS ASSETS at each")
            print("      time point (typical range: 0.01-0.05), unlike single-asset IC")
            print("      which measures correlation across TIME (typical range: 0.3-0.9)")
            print("="*80)

    return {
        'all_results': all_results,
        'summary': summary,
        'cross_sectional': cross_sectional_results,
        'study_name': study_name,
        'study_timestamp': study_timestamp
    }


# ============================================================================
# CELL 8: EXAMPLE USAGE (Run in Notebook or as Script)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage - can be run in notebook cells or as script

    In Notebook:
        1. Run CELL 1-7 first
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
        loss_type='auto',
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
    """
    print("\n" + "="*80)
    print("MODE 2: SINGLE DATASET FULL TRAINING")
    print("="*80)

    results = run_comparison(
        dataset='IXIC',
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        hidden_dim=64,
        loss_type='auto',
        R=3,
        K=3,
        heads=4,
        mc_train=3,
        mc_eval=20,
        verbose=True,
        device=device
    )
    """

    # ============ MODE 3: MULTI-DATASET (All 3 Datasets) ============
    print("\n" + "="*80)
    print("MODE 3: MULTI-DATASET COMPARISON")
    print("="*80)

    all_results = run_multi_dataset_comparison(
        datasets=['IXIC', 'DJI', 'NYSE'],
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        hidden_dim=64,
        R=3, K=3, d_e=10, heads=4,
        mc_train=3, mc_eval=20,
        verbose=True,
        log_base_dir='logs',
        device=device
    )

    print("\n" + "="*80)
    print("✓ MULTI-DATASET COMPARISON COMPLETED!")
    print("="*80)
    print(f"\nStudy: {all_results['study_name']}")
    print(f"Results saved to: logs/{all_results['study_name']}/")
    print("\nTo generate comparison plots, run:")
    print("  from utils.result_plot import plot_bayesian_vs_nonbayesian_comparison")
    print(f"  plot_bayesian_vs_nonbayesian_comparison('logs/{all_results['study_name']}')")