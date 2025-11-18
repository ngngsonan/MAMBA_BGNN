"""
MAMBA Bayesian GNN - Loss Function Study with Hyperparameter Tuning

This module compares different loss functions with Bayesian MAGAC and tunes critical hyperparameters:

    Loss Functions (all with Bayesian MAGAC):
        1. GaussianNLL  - Probabilistic loss with MC-based uncertainty (optimal for Bayesian)
        2. MSE          - Mean Squared Error
        3. SmoothL1     - Robust Huber-like loss

    Critical Hyperparameters (from TUNE_PARAMS.md):
        - K (Chebyshev order):    Impact 10/10, test [2, 3, 4]
        - R (BIMamba layers):     Impact 9/10, test [2, 3]
        - heads (Attention heads): Impact 8/10, test [2, 4, 8]

    Total configurations: 3 losses × 3 K × 2 R × 3 heads = 54 experiments

Note:
    - For Non-Bayesian models, use models/mamba_gnn_study.py (BIMamba+MAGAC)
    - This module focuses on Bayesian MAGAC with MC sampling for uncertainty quantification

Usage:

    Train loss function comparison with hyperparameter tuning:
    ==========================================================
    from models.bimamba_bgnn_lossfn_study import train_lossfn_with_hparam_tuning

    results = train_lossfn_with_hparam_tuning(
        dataset='IXIC',
        loss_functions=['GaussianNLL', 'MSE', 'SmoothL1'],
        K_values=[2, 3, 4],
        R_values=[2, 3],
        heads_values=[2, 4, 8],
        epochs=50,
        verbose=True
    )


    Calculate cross-sectional IC:
    =============================
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # After training, analyze cross-sectional IC
    cross_results = calculate_cross_sectional_for_all_models(
        models=['GaussianNLL_K2_R2_H2', 'GaussianNLL_K3_R3_H4', ...],
        datasets=['IXIC', 'DJI', 'NYSE'],
        study_name='lossfn_hparam',
        output_file='logs/lossfn_hparam_cross_sectional_summary.txt'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.data_processing import data_processing
from utils.baseline_trainer import train_models


# ============================================================================
# MODEL ARGUMENTS
# ============================================================================

@dataclass
class ModelArgs:
    d_model: int          # Number of features N
    seq_len: int          # Sequence length L
    d_proj_E: int = 64    # Embedding dimension
    d_proj_H: int = 64    # Latent dimension in SSM
    d_proj_U: int = 32    # Hidden layer in FFN
    expand: int = 2
    d_state: int = 64
    dt_rank: int | str = 'auto'
    d_conv: int = 3
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_proj_E / 16)


# ============================================================================
# MAMBA CORE COMPONENTS
# ============================================================================

class MambaBlock(nn.Module):
    """Original Mamba Block"""
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
    """Feed-forward network"""
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
        # x: (B, L, N=d_model)
        for i in range(self.R):
            # Forward direction
            Y1 = self.f_mamba[i](x)

            # Backward direction
            x_rev = torch.flip(x, dims=[1]).contiguous()
            Y2_rev = self.b_mamba[i](x_rev)
            Y2 = torch.flip(Y2_rev, dims=[1]).contiguous()

            # Combine with residual and FFN
            Y3 = self.norm1[i](x + Y1 + Y2)
            Yp = self.ffn[i](Y3)
            x = self.norm2[i](Yp + Y3)
        return x


# ============================================================================
# GRAPH NEURAL NETWORK LAYERS
# ============================================================================

class MAGAC(nn.Module):
    """
    Multi-head Adaptive Graph Attention Convolution (Non-Bayesian)
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

        # Factorized Chebyshev filter weights per head
        self.F_w = nn.Parameter(torch.randn(heads, d_e, K + 1, in_dim))
        self.f_b = nn.Parameter(torch.randn(heads, d_e))
        self.head_mix = nn.Parameter(torch.ones(heads))

    def _gaussian_A(self):
        diff = self.psi_emb[:, None, :] - self.psi_emb[None, :, :]
        dist2 = diff.pow(2).sum(-1)
        A = torch.exp(-self.psi * dist2)
        return F.softmax(A, dim=1)  # (N, N)

    def _attn_A(self, psi=None):
        psi = self.psi_emb if psi is None else psi
        Q = torch.einsum('nd,dhm->nhm', psi, self.W_q)  # (N, H, d_e)
        K = torch.einsum('nd,dhm->nhm', psi, self.W_k)  # (N, H, d_e)
        d_e = Q.size(-1)
        attn = torch.einsum('nhd,mhd->hnm', Q, K) / math.sqrt(d_e)  # (H, N, N)
        return F.softmax(attn, dim=-1)

    def _blend(self, A_g, A_attn_h):
        alpha = torch.sigmoid(self.attn_alpha)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def forward(self, x, A_eff_override: torch.Tensor | None = None):
        """
        Args:
            x: (B, N, L)
        Returns:
            out: (B, N)
        """
        B, N, L = x.shape
        assert N == self.N

        if A_eff_override is not None:
            A_effs = A_eff_override
        else:
            A_base = self._gaussian_A()  # (N, N)
            A_attn = self._attn_A()  # (H, N, N)
            A_effs = torch.stack([self._blend(A_base, A_attn[h])
                                   for h in range(self.H)], dim=0)  # (H, N, N)

        mix_w = F.softmax(self.head_mix, dim=0)
        out = 0

        for h in range(self.H):
            A_eff = A_effs[h]  # (N, N)

            # Chebyshev polynomial supports
            I = torch.eye(N, device=x.device, dtype=x.dtype)
            supports = [I, A_eff]
            for k in range(2, self.K + 1):
                supports.append(2 * A_eff @ supports[-1] - supports[-2])
            supports = torch.stack(supports, dim=0)  # (K+1, N, N)

            # Factorized filter
            W_filter = torch.einsum('nd,dkl->nkl', self.psi_emb, self.F_w[h])  # (N, K+1, L)
            b_filter = self.psi_emb @ self.f_b[h]  # (N,)

            # Apply convolution
            x_g = torch.einsum('knm,bml->bknl', supports, x)  # (B, K+1, N, L)
            out_h = torch.einsum('bknl,nkl->bn', x_g, W_filter) + b_filter
            out = out + mix_w[h] * out_h

        return out


class BayesianMAGAC(MAGAC):
    """
    Bayesian Multi-head Adaptive Graph Attention Convolution
    
    Uses Monte Carlo sampling for uncertainty quantification:
    - MC Dropout on node embeddings
    - Multiple forward passes during training and evaluation
    - Returns mean and log variance
    """
    def __init__(self, num_nodes, in_dim, K=3, d_e=10, heads=4,
                 mc_train=3, mc_eval=20,
                 drop_edge_p: float = 0.1, mc_dropout_p: float = 0.2):
        super().__init__(num_nodes, in_dim, K, d_e, heads)
        self.mc_train = mc_train
        self.mc_eval = mc_eval
        self.mc_samples = mc_train
        self.drop_edge_p = drop_edge_p
        self.dropout_emb = nn.Dropout(p=mc_dropout_p)
        self.register_buffer("eye_N", torch.eye(num_nodes))

    def train(self, mode: bool = True):
        super().train(mode)
        self.mc_samples = self.mc_train if mode else self.mc_eval
        return self

    def forward(self, x):
        """
        Args:
            x: (B, N, L)
        Returns:
            mean: (B, N)
            log_var: (B, N)
        """
        outs = []
        if self.training:
            # Cache one A_eff for the mini-batch
            A_eff_one = self._sample_A_eff(use_dropout_on_psi=True)
            for _ in range(self.mc_samples):
                outs.append(super().forward(x, A_eff_override=A_eff_one))
        else:
            # Eval: each pass samples a new A_eff
            for _ in range(self.mc_samples):
                A_eff_s = self._sample_A_eff(use_dropout_on_psi=False)
                outs.append(super().forward(x, A_eff_override=A_eff_s))

        outs = torch.stack(outs, dim=0)  # (S, B, N)
        mean = outs.mean(0)
        if self.mc_samples == 1:
            log_var = torch.zeros_like(mean)
        else:
            var = outs.var(0, unbiased=False) + 1e-6
            log_var = var.log()
        return mean, log_var

    def _sample_A_eff(self, use_dropout_on_psi: bool):
        # (1) Stochastic/deterministic node embedding
        psi_stoch = F.dropout(self.psi_emb, p=self.dropout_emb.p, training=use_dropout_on_psi)

        # (2) Gaussian part
        diff = psi_stoch[:, None, :] - psi_stoch[None, :, :]
        A_g = torch.exp(-self.psi * diff.pow(2).sum(-1))
        A_g = F.softmax(A_g, dim=1)  # (N, N)

        # (3) Attention part
        A_attn = self._attn_A(psi_stoch)  # (H, N, N)

        # (4) Blend + DropEdge + row-renorm
        A_list = []
        for h in range(self.H):
            A_eff = self._blend(A_g, A_attn[h])  # (N, N)

            if (self.training is False) and (self.drop_edge_p > 0.0):
                keep = torch.bernoulli((1 - self.drop_edge_p) * torch.ones_like(A_eff))
                keep = keep.fill_diagonal_(1.0)
                A_eff = A_eff * keep
                A_eff = A_eff / (A_eff.sum(dim=1, keepdim=True).clamp_min(1e-6))

            A_list.append(A_eff)
        return torch.stack(A_list, dim=0)  # (H, N, N)


# ============================================================================
# MODEL WRAPPERS: BIMamba + Bayesian/Non-Bayesian GNN
# ============================================================================

class MAMBA_BayesMAGAC(nn.Module):
    """BIMamba + Bayesian MAGAC (with MC sampling)"""
    def __init__(self, args: ModelArgs, R: int = 3, K: int = 3,
                 d_e: int = 10, heads=4,
                 mc_train=3, mc_eval=20, drop_edge_p=0.1, mc_dropout_p=0.2):
        super().__init__()
        self.bi_mamba = BIMambaBlock(args, R=R)
        self.agc_bayes = BayesianMAGAC(
            args.d_model, args.seq_len, K, d_e, heads=heads,
            mc_train=mc_train, mc_eval=mc_eval,
            drop_edge_p=drop_edge_p, mc_dropout_p=mc_dropout_p
        )
        self.head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        """
        Args:
            x: (B, L, N)
        Returns:
            mu: (B,)
            log_var: (B,)
        """
        y_seq = self.bi_mamba(x)  # (B, L, N)
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node, log_var_node = self.agc_bayes(z_node)  # both (B, N)

        # Linear head
        w = self.head.weight.squeeze(0)  # (N,)
        b = self.head.bias  # scalar

        mu = torch.einsum('bn,n->b', g_node, w) + b
        var = torch.einsum('bn,n->b', log_var_node.exp(), w.pow(2)) + 1e-6
        log_var = var.log()
        return mu, log_var


# Non-Bayesian MAGAC removed - use models/mamba_gnn_study.py (BIMamba+MAGAC) for deterministic models


# ============================================================================
# TRAINING WRAPPER WITH HYPERPARAMETER TUNING
# ============================================================================

def train_lossfn_with_hparam_tuning(
    dataset: str = 'IXIC',
    loss_functions: Optional[List[str]] = None,
    K_values: Optional[List[int]] = None,
    R_values: Optional[List[int]] = None,
    heads_values: Optional[List[int]] = None,
    window: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    early_stop_patience: int = 10,
    d_e: int = 10,
    mc_train: int = 3,
    mc_eval: int = 20,
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu'
) -> Dict:
    """
    Train Bayesian MAGAC with different loss functions and hyperparameter combinations

    Args:
        dataset: Dataset name (IXIC, DJI, NYSE)
        loss_functions: Loss functions to test (None = ['GaussianNLL', 'MSE', 'SmoothL1'])
        K_values: Chebyshev polynomial orders to test (None = [2, 3, 4])
        R_values: Number of BIMamba layers to test (None = [2, 3])
        heads_values: Number of attention heads to test (None = [2, 4, 8])
        window: Lookback window size
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        early_stop_patience: Early stopping patience
        d_e: Node embedding dimension (fixed)
        mc_train: MC samples during training (fixed)
        mc_eval: MC samples during evaluation (fixed)
        verbose: Print progress
        log_base_dir: Base directory for logs
        device: Device to train on

    Returns:
        Dictionary with results for each configuration:
            {model_name: {'ic': ..., 'ric': ..., 'rmse': ..., 'nll': ...}}
    """
    # Default values from TUNE_PARAMS.md
    if loss_functions is None:
        loss_functions = ['GaussianNLL', 'MSE', 'SmoothL1']
    if K_values is None:
        K_values = [2, 3, 4]  # Impact 10/10
    if R_values is None:
        R_values = [2, 3]     # Impact 9/10
    if heads_values is None:
        heads_values = [2, 4, 8]  # Impact 8/10

    total_configs = len(loss_functions) * len(K_values) * len(R_values) * len(heads_values)

    print("="*80)
    print("BAYESIAN MAGAC - LOSS FUNCTION + HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Loss Functions: {loss_functions}")
    print(f"K values (Chebyshev order): {K_values}")
    print(f"R values (BIMamba layers): {R_values}")
    print(f"heads values (Attention heads): {heads_values}")
    print(f"Total configurations: {total_configs}")
    print(f"Window: {window}, Batch: {batch_size}, Epochs: {epochs}")
    print(f"LR: {learning_rate}, MC: {mc_train}/{mc_eval}")
    print("="*80)

    # Load data
    print("\nLoading data with utils.data_processing...")
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'
    num_features, train_loader, val_loader, test_loader = data_processing(
        data_path, window, batch_size
    )
    print(f"✓ Data loaded: {num_features} features")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    # Loss function mapping
    loss_mapping = {
        'GaussianNLL': 'nll',
        'MSE': 'mse',
        'SmoothL1': 'huber'
    }

    # Train all combinations of hyperparameters
    all_results = {}
    config_count = 0

    for loss_fn in loss_functions:
        for K in K_values:
            for R in R_values:
                for heads in heads_values:
                    config_count += 1

                    # Model name with hyperparameters
                    model_name = f'{loss_fn}_K{K}_R{R}_H{heads}'

                    print(f"\n{'='*80}")
                    print(f"Configuration {config_count}/{total_configs}: {model_name}")
                    print(f"{'='*80}")
                    print(f"Loss: {loss_fn}, K: {K}, R: {R}, heads: {heads}")

                    # Create model arguments
                    args = ModelArgs(
                        d_model=num_features,
                        seq_len=window,
                        d_proj_E=hidden_dim,
                        d_proj_H=hidden_dim,
                        d_proj_U=hidden_dim // 2,
                        d_state=hidden_dim
                    )

                    # Create Bayesian MAGAC model with current hyperparameters
                    model = MAMBA_BayesMAGAC(
                        args, R=R, K=K, d_e=d_e, heads=heads,
                        mc_train=mc_train, mc_eval=mc_eval,
                        drop_edge_p=0.1, mc_dropout_p=0.2
                    )

                    # Initialize weights
                    for p in model.parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)

                    # Training config
                    loss_type = loss_mapping.get(loss_fn, 'auto')
                    config = {
                        'epochs': epochs,
                        'lr': learning_rate,
                        'loss_type': loss_type,
                        'patience': early_stop_patience,
                        'optimizer_fn': lambda params, lr: torch.optim.Adam(params, lr=lr),
                        'scheduler_fn': None
                    }

                    # Train model
                    results = train_models(
                        models_dict={model_name: model},
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        dataset=dataset,
                        config=config,
                        log_base_dir=log_base_dir,
                        study_name='lossfn_hparam',
                        verbose=verbose,
                        device=device
                    )

                    all_results.update(results)

    # Analysis - Find best configurations
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*80)

    # Find best configuration for each loss function
    print("\nBest Configuration per Loss Function:")
    print(f"{'Loss Function':<15} {'Best Config':<20} {'IC':>10} {'RIC':>10} {'RMSE':>10}")
    print("-"*80)

    for loss_fn in loss_functions:
        # Find best config for this loss function
        loss_configs = {k: v for k, v in all_results.items() if k.startswith(loss_fn)}
        if loss_configs:
            best_config = max(loss_configs.keys(), key=lambda x: loss_configs[x].get('ic', -999))
            metrics = loss_configs[best_config]
            config_str = best_config.replace(f'{loss_fn}_', '')
            print(f"{loss_fn:<15} {config_str:<20} {metrics['ic']:>10.6f} "
                  f"{metrics['ric']:>10.6f} {metrics.get('rmse', 0):>10.6f}")

    # Find best configuration for each hyperparameter value
    print("\n\nBest Loss Function per Hyperparameter:")
    print("-"*80)

    # Best for each K value
    print("\nChebyshev Order (K):")
    for K in K_values:
        k_configs = {k: v for k, v in all_results.items() if f'_K{K}_' in k}
        if k_configs:
            best = max(k_configs.keys(), key=lambda x: k_configs[x].get('ic', -999))
            print(f"  K={K}: {best} (IC={k_configs[best]['ic']:.6f})")

    # Best for each R value
    print("\nBIMamba Layers (R):")
    for R in R_values:
        r_configs = {k: v for k, v in all_results.items() if f'_R{R}_' in k}
        if r_configs:
            best = max(r_configs.keys(), key=lambda x: r_configs[x].get('ic', -999))
            print(f"  R={R}: {best} (IC={r_configs[best]['ic']:.6f})")

    # Best for each heads value
    print("\nAttention Heads:")
    for heads in heads_values:
        h_configs = {k: v for k, v in all_results.items() if f'_H{heads}' in k}
        if h_configs:
            best = max(h_configs.keys(), key=lambda x: h_configs[x].get('ic', -999))
            print(f"  heads={heads}: {best} (IC={h_configs[best]['ic']:.6f})")

    # Overall best configuration
    best_overall = max(all_results.keys(), key=lambda x: all_results[x].get('ic', -999))
    best_metrics = all_results[best_overall]
    print("\n" + "-"*80)
    print(f"✓ Best Overall Configuration: {best_overall}")
    print(f"  IC:    {best_metrics['ic']:.6f}")
    print(f"  RIC:   {best_metrics['ric']:.6f}")
    print(f"  RMSE:  {best_metrics.get('rmse', 0):.6f}")
    if 'nll' in best_metrics:
        print(f"  NLL:   {best_metrics['nll']:.6f}")
    print("="*80)

    return all_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Loss function comparison with hyperparameter tuning

    Quick test: Test 1 loss function with reduced hyperparameter space
    Full study: Test all 3 losses with full hyperparameter grid
    """
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ============ QUICK TEST MODE ============
    # Uncomment for fast testing (9 configurations)
    """
    print("\n" + "="*80)
    print("QUICK TEST: GaussianNLL with reduced hyperparameter space")
    print("="*80)

    results = train_lossfn_with_hparam_tuning(
        dataset='IXIC',
        loss_functions=['GaussianNLL'],  # Test only NLL
        K_values=[2, 3, 4],               # 3 values
        R_values=[3],                     # Fix R=3
        heads_values=[4],                 # Fix heads=4
        epochs=30,                        # Reduced epochs
        early_stop_patience=10,
        verbose=True,
        device=device
    )
    """

    # ============ FULL STUDY MODE ============
    # Train on single dataset with full hyperparameter grid
    print("\n" + "="*80)
    print("FULL STUDY: Loss functions + Hyperparameter tuning on IXIC")
    print("="*80)

    results = train_lossfn_with_hparam_tuning(
        dataset='IXIC',
        loss_functions=['GaussianNLL', 'MSE', 'SmoothL1'],
        K_values=[2, 3, 4],      # Chebyshev order (Impact 10/10)
        R_values=[2, 3],         # BIMamba layers (Impact 9/10)
        heads_values=[2, 4, 8],  # Attention heads (Impact 8/10)
        epochs=50,
        learning_rate=0.001,
        early_stop_patience=10,
        hidden_dim=64,
        window=5,
        batch_size=32,
        mc_train=3,
        mc_eval=20,
        verbose=True,
        device=device
    )

    # ============ MULTI-DATASET STUDY ============
    # Uncomment to run on multiple datasets (takes longer)
    """
    print("\n" + "="*80)
    print("MULTI-DATASET STUDY: Train best configs on all datasets")
    print("="*80)

    # First, identify best configurations from IXIC study above
    # Then train only those on DJI and NYSE

    datasets = ['IXIC', 'DJI', 'NYSE']

    # Use reduced hyperparameter space or best configs from previous run
    for dataset in datasets:
        print(f"\n>>> Training on {dataset}...")
        results = train_lossfn_with_hparam_tuning(
            dataset=dataset,
            loss_functions=['GaussianNLL'],  # Best loss from IXIC
            K_values=[3, 4],                 # Best K values
            R_values=[3],                    # Best R
            heads_values=[4, 8],             # Best heads
            epochs=50,
            verbose=True,
            device=device
        )

    # Calculate cross-sectional IC
    print("\n" + "="*80)
    print("STEP 2: Calculating cross-sectional IC")
    print("="*80)

    # List model names from training
    models = ['GaussianNLL_K3_R3_H4', 'GaussianNLL_K3_R3_H8',
              'GaussianNLL_K4_R3_H4', 'GaussianNLL_K4_R3_H8']

    cross_results = calculate_cross_sectional_for_all_models(
        models=models,
        datasets=datasets,
        study_name='lossfn_hparam',
        output_file='logs/lossfn_hparam_cross_sectional_summary.txt',
        verbose=True
    )

    print("\nCross-Sectional IC Results:")
    for model_name, result in cross_results.items():
        if result.get('error'):
            print(f"  {model_name}: Error - {result['error']}")
        else:
            print(f"  {model_name}: IC={result['cross_sectional']['ic_mean']:.6f}, "
                  f"RIC={result['cross_sectional']['ric_mean']:.6f}")
    """

    print("\n" + "="*80)
    print("✓ Hyperparameter Tuning Study Completed!")
    print("="*80)
    print("\nKey Findings:")
    print("  - Tested 3 loss functions × 3 K × 2 R × 3 heads = 54 configurations")
    print("  - Identified optimal hyperparameters for each loss function")
    print("  - Results saved to: logs/lossfn_hparam/")
    print("\nNext Steps:")
    print("  1. Review best configurations from logs/")
    print("  2. Run multi-dataset study with best configs")
    print("  3. Calculate cross-sectional IC across datasets")
    print("  4. For Non-Bayesian comparison, use models/mamba_gnn_study.py")
