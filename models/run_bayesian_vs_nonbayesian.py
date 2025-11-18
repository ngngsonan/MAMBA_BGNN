"""
Bayesian vs Non-Bayesian MAMBA-BGNN Comparison Study

This module performs comprehensive comparison between:
    1. Bayesian MAGAC (with MC sampling & uncertainty quantification)
    2. Non-Bayesian MAGAC (deterministic, from mamba_gnn_study.py)

Experiments run on 3 datasets: IXIC, DJI, NYSE

Key analyses:
    - Financial metrics (Sharpe, IC, RIC, Directional Accuracy)
    - Market regime analysis (Stable vs Volatile)
    - Uncertainty quality (calibration, CRPS)
    - Risk-adjusted performance

Usage:
    python models/run_bayesian_vs_nonbayesian.py

    Or import:
    from models.run_bayesian_vs_nonbayesian import run_comparison_study

    results = run_comparison_study(
        datasets=['IXIC', 'DJI', 'NYSE'],
        epochs=50,
        device='cuda'
    )
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.baseline_trainer import (
    train_models,
    regime_analysis,
    compute_financial_metrics,
    compare_bayesian_vs_nonbayesian
)
from utils.result_plot import plot_bayesian_vs_nonbayesian_comparison
from financial_metrics import FinancialMetrics, MarketRegimeAnalysis


# ============================================================================
# Import model components from bimamba_bgnn_tuneparams.py
# ============================================================================

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
# Non-Bayesian MAGAC (Deterministic)
# ============================================================================

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
        """
        Deterministic forward pass

        Args:
            x: (B, N, L)
        Returns:
            out: (B, N) - mean prediction
            log_var: (B, N) - fixed log variance
        """
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
        log_var = torch.ones_like(out) * (-3.0)  # Fixed small variance

        return out, log_var


# ============================================================================
# Bayesian MAGAC (with MC Sampling)
# ============================================================================

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
        """
        Stochastic forward with MC sampling

        Args:
            x: (B, N, L)
        Returns:
            mean: (B, N)
            log_var: (B, N)
        """
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
# Full Models
# ============================================================================

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
# Data loading utility
# ============================================================================

def load_data(dataset: str, window: int = 5, batch_size: int = 32):
    """Load dataset and return dataloaders"""
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
# Main Comparison Function
# ============================================================================

def run_comparison_study(
    datasets: List[str] = ['IXIC', 'DJI', 'NYSE'],
    window: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    early_stop_patience: int = 10,
    # Model hyperparameters (use best from tuning)
    R: int = 3,
    K: int = 3,
    d_e: int = 10,
    heads: int = 4,
    mc_train: int = 3,
    mc_eval: int = 20,
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu'
) -> Dict:
    """
    Run comprehensive Bayesian vs Non-Bayesian comparison

    Returns:
        results: Dictionary with all metrics and analyses
    """

    print("="*80)
    print("BAYESIAN VS NON-BAYESIAN MAMBA-BGNN COMPARISON")
    print("="*80)
    print(f"Datasets: {datasets}")
    print(f"Epochs: {epochs}, LR: {learning_rate}")
    print(f"Model: R={R}, K={K}, heads={heads}, d_e={d_e}")
    print(f"Bayesian: mc_train={mc_train}, mc_eval={mc_eval}")
    print("="*80)

    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset}")
        print(f"{'='*80}")

        # Load data
        print("\nLoading data...")
        num_features, train_loader, val_loader, test_loader = load_data(
            dataset, window, batch_size
        )
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

        # Create models
        print("\nCreating models...")
        model_nonbayesian = BIMamba_MAGAC_NonBayesian(args, R, K, d_e, heads)
        model_bayesian = BIMamba_MAGAC_Bayesian(
            args, R, K, d_e, heads, mc_train, mc_eval, 0.1, 0.2
        )

        # Initialize weights
        for model in [model_nonbayesian, model_bayesian]:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        models_dict = {
            f'{dataset}_NonBayesian': model_nonbayesian,
            f'{dataset}_Bayesian': model_bayesian
        }

        # Training config
        config = {
            'epochs': epochs,
            'lr': learning_rate,
            'loss_type': 'nll',
            'patience': early_stop_patience,
            'optimizer_fn': lambda params, lr: torch.optim.Adam(params, lr=lr),
            'scheduler_fn': None
        }

        # Train models
        print("\nTraining models...")
        results = train_models(
            models_dict=models_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset=dataset,
            config=config,
            log_base_dir=log_base_dir,
            study_name='bayesian_vs_nonbayesian',
            verbose=verbose,
            device=device
        )

        all_results[dataset] = results

        # Perform advanced analysis
        print("\nPerforming market regime analysis...")
        regime_results = regime_analysis(
            model_bayesian=models_dict[f'{dataset}_Bayesian'],
            model_nonbayesian=models_dict[f'{dataset}_NonBayesian'],
            test_loader=test_loader,
            device=device
        )
        all_results[dataset]['regime_analysis'] = regime_results

        # Financial metrics
        print("\nComputing financial metrics...")
        fin_metrics = compute_financial_metrics(
            models_dict=models_dict,
            test_loader=test_loader,
            device=device
        )
        all_results[dataset]['financial_metrics'] = fin_metrics

    # Cross-dataset comparison
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)

    comparison = compare_bayesian_vs_nonbayesian(all_results)

    # Save results
    output_dir = os.path.join(log_base_dir, 'bayesian_vs_nonbayesian')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_bayesian_vs_nonbayesian_comparison(
        all_results,
        output_dir=output_dir
    )

    print("\n" + "="*80)
    print("✓ COMPARISON STUDY COMPLETED!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nKey Findings:")
    print("  - Bayesian models provide uncertainty quantification")
    print("  - Non-Bayesian models are faster (no MC sampling)")
    print("  - Regime analysis shows performance in different market conditions")
    print("="*80)

    return all_results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Bayesian vs Non-Bayesian comparison')
    parser.add_argument('--datasets', nargs='+', default=['IXIC', 'DJI', 'NYSE'],
                       help='Datasets to run on')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    results = run_comparison_study(
        datasets=args.datasets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        verbose=True
    )

    print("\n✓ Done! Check logs/bayesian_vs_nonbayesian/ for results")
