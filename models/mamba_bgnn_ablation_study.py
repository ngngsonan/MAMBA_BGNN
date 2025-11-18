"""
MAMBA_BGNN Ablation Study - Loss Functions & Bayesian GNN Ablation

This module contains ablation studies for:
    1. Loss Function Ablation: GaussianNLLLoss vs MSELoss vs SmoothL1Loss
    2. Bayesian GNN Ablation: With BayesianMAGAC vs Without BayesianMAGAC

All models output (mean, log_var) for probabilistic evaluation.

Key Ablation Dimensions:
    - Bayesian Graph: With BayesianMAGAC (graph neural network) vs Without (pure MAMBA)
    - Loss Function: GaussianNLLLoss (Bayesian) vs MSELoss (L2) vs SmoothL1Loss (Huber)

Model Variants:
    1. MAMBA_NoGraph_GaussianNLL    - Pure MAMBA with GaussianNLLLoss
    2. MAMBA_NoGraph_MSE             - Pure MAMBA with MSELoss
    3. MAMBA_NoGraph_SmoothL1        - Pure MAMBA with SmoothL1Loss
    4. MAMBA_BayesianGNN_GaussianNLL - MAMBA + BayesianMAGAC with GaussianNLLLoss
    5. MAMBA_BayesianGNN_MSE         - MAMBA + BayesianMAGAC with MSELoss
    6. MAMBA_BayesianGNN_SmoothL1    - MAMBA + BayesianMAGAC with SmoothL1Loss

Usage:

    Train all ablation combinations:
    ================================
    from models.mamba_bgnn_ablation_study import train_bgnn_ablation_models

    results = train_bgnn_ablation_models(
        dataset='IXIC',
        loss_functions=['GaussianNLLLoss', 'MSELoss', 'SmoothL1Loss'],
        bayesian_variants=['with', 'without'],  # with/without BayesianMAGAC
        epochs=100,
        verbose=True
    )

    Calculate cross-sectional IC for all variants:
    ==============================================
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    cross_results = calculate_cross_sectional_for_all_models(
        models=['MAMBA_NoGraph_GaussianNLL', 'MAMBA_NoGraph_MSE', 'MAMBA_NoGraph_SmoothL1',
                'MAMBA_BayesianGNN_GaussianNLL', 'MAMBA_BayesianGNN_MSE', 'MAMBA_BayesianGNN_SmoothL1'],
        datasets=['IXIC', 'DJI', 'NYSE'],
        study_name='mamba_bgnn_ablation',
        output_file='logs/mamba_bgnn_ablation_cross_sectional_summary.txt'
    )

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import Tuple, List, Dict, Optional
import math
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
# GRAPH NEURAL NETWORK COMPONENTS (MAGAC)
# ============================================================================

class MAGAC(nn.Module):
    """Multi-head Adaptive Graph Attention Convolution"""
    def __init__(self, num_nodes: int, in_dim: int, K: int = 3,
                 d_e: int = 10, heads: int = 4):
        super().__init__()
        self.N      = num_nodes
        self.K      = K
        self.in_dim = in_dim
        self.H      = heads

        self.psi_emb = nn.Parameter(torch.randn(num_nodes, d_e))
        self.psi     = nn.Parameter(torch.tensor(1.0))

        self.W_q = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.W_k = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.attn_alpha = nn.Parameter(torch.tensor(0.5))

        self.F_w = nn.Parameter(torch.randn(heads, d_e, K + 1, in_dim))
        self.f_b = nn.Parameter(torch.randn(heads, d_e))
        self.head_mix = nn.Parameter(torch.ones(heads))

    def _gaussian_A(self):
        diff = self.psi_emb[:, None, :] - self.psi_emb[None, :, :]
        dist2 = diff.pow(2).sum(-1)
        A = torch.exp(-self.psi * dist2)
        return F.softmax(A, dim=1)

    def _attn_A(self, psi=None):
        if hasattr(self, "_cache_attn") and self._cache_attn is not None and self.training is False:
            return self._cache_attn
        psi = self.psi_emb if psi is None else psi
        Q = torch.einsum('nd,dhm->nhm', psi, self.W_q)
        K = torch.einsum('nd,dhm->nhm', psi, self.W_k)
        d_e = Q.size(-1)
        attn = torch.einsum('nhd,mhd->hnm', Q, K) / math.sqrt(d_e)
        attn = F.softmax(attn, dim=-1)
        if hasattr(self, "_cache_attn") and self.training is False:
            self._cache_attn = attn.detach()
        return attn

    def _blend(self, A_g, A_attn_h):
        alpha = torch.sigmoid(self.attn_alpha)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def forward(self, x, A_eff_override: torch.Tensor | None = None):
        B, N, L = x.shape
        assert N == self.N, "num_nodes mismatch"

        if A_eff_override is not None:
            A_effs = A_eff_override
        else:
            A_base = self._gaussian_A()
            A_attn = self._attn_A()
            A_effs = torch.stack([self._blend(A_base, A_attn[h])
                                   for h in range(self.H)], dim=0)

        mix_w = F.softmax(self.head_mix, dim=0)
        out = 0
        for h in range(self.H):
            A_eff = A_effs[h]

            I = torch.eye(N, device=x.device, dtype=x.dtype)
            supports = [I, A_eff]
            for k in range(2, self.K + 1):
                supports.append(2 * A_eff @ supports[-1] - supports[-2])
            supports = torch.stack(supports, dim=0)

            W_filter = torch.einsum('nd,dkl->nkl', self.psi_emb, self.F_w[h])
            b_filter = self.psi_emb @ self.f_b[h]

            x_g = torch.einsum('knm,bml->bknl', supports, x)
            out_h = torch.einsum('bknl,nkl->bn', x_g, W_filter) + b_filter
            out = out + mix_w[h] * out_h
        return out


class BayesianMAGAC(MAGAC):
    """Bayesian version of MAGAC with MC dropout"""
    def __init__(self, num_nodes, in_dim, K=3, d_e=10, heads=4,
                 mc_train=3, mc_eval=20, drop_edge_p: float = 0.1, mc_dropout_p: float = 0.2):
        super().__init__(num_nodes, in_dim, K, d_e, heads)
        self.mc_train = mc_train
        self.mc_eval  = mc_eval
        self.mc_samples   = mc_train
        self.drop_edge_p  = drop_edge_p
        self.dropout_emb  = nn.Dropout(p=mc_dropout_p)
        self.register_buffer("eye_N", torch.eye(num_nodes))
        self._cache_attn  = None

    def _blend(self, A_g, A_attn_h):
        alpha = torch.sigmoid(self.attn_alpha)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def train(self, mode: bool = True):
        super().train(mode)
        self.mc_samples = self.mc_train if mode else self.mc_eval
        return self

    def forward(self, x):
        outs = []
        if self.training:
            A_eff_one = self._sample_A_eff(use_dropout_on_psi=True)
            for _ in range(self.mc_samples):
                outs.append(super().forward(x, A_eff_override=A_eff_one))
        else:
            for _ in range(self.mc_samples):
                A_eff_s = self._sample_A_eff(use_dropout_on_psi=False)
                outs.append(super().forward(x, A_eff_override=A_eff_s))

        outs = torch.stack(outs, dim=0)
        mean = outs.mean(0)
        if self.mc_samples == 1:
            log_var = torch.zeros_like(mean)
        else:
            var = outs.var(0, unbiased=False) + 1e-6
            log_var = var.log()
        return mean, log_var

    def _sample_A_eff(self, use_dropout_on_psi: bool):
        psi_stoch = F.dropout(self.psi_emb, p=self.dropout_emb.p, training=use_dropout_on_psi)

        diff = psi_stoch[:, None, :] - psi_stoch[None, :, :]
        A_g  = torch.exp(-self.psi * diff.pow(2).sum(-1))
        A_g  = F.softmax(A_g, dim=1)

        A_attn = self._attn_A(psi_stoch)

        A_list = []
        for h in range(self.H):
            A_eff = self._blend(A_g, A_attn[h])

            if (self.training is False) and (self.drop_edge_p > 0.0):
                keep = torch.bernoulli((1 - self.drop_edge_p) * torch.ones_like(A_eff))
                keep = keep.fill_diagonal_(1.0)
                A_eff = A_eff * keep
                A_eff = A_eff / (A_eff.sum(dim=1, keepdim=True).clamp_min(1e-6))

            A_list.append(A_eff)
        return torch.stack(A_list, dim=0)


# ============================================================================
# ABLATION MODELS
# ============================================================================

class MAMBA_NoGraph(nn.Module):
    """MAMBA without Graph Neural Network (pure sequence model)"""
    def __init__(self, args: ModelArgs, R: int = 3):
        super().__init__()
        self.bi_mamba = BIMambaBlock(args, R=R)

        # Pooling and output heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mean_head = nn.Linear(args.d_model, 1)
        self.logvar_head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        # x: (B, L, N)
        x = self.bi_mamba(x)  # (B, L, N)

        # Pool over sequence: (B, L, N) -> (B, N)
        x = x.transpose(1, 2)  # (B, N, L)
        x = self.pool(x).squeeze(-1)  # (B, N)

        mean = self.mean_head(x).squeeze(-1)  # (B,)
        log_var = self.logvar_head(x).squeeze(-1)  # (B,)
        return mean, log_var


class MAMBA_WithBayesianGNN(nn.Module):
    """MAMBA with Bayesian Graph Neural Network"""
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
        y_seq = self.bi_mamba(x)  # (B, L, N)
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node, log_var_node = self.agc_bayes(z_node)  # both (B, N)

        # Linear head
        w = self.head.weight.squeeze(0)  # (N,)
        b = self.head.bias  # (1,)

        mu = torch.einsum('bn,n->b', g_node, w) + b  # (B,)
        var = torch.einsum('bn,n->b',
                            log_var_node.exp(),
                            w.pow(2)) + 1e-6
        log_var = var.log()  # (B,)
        return mu, log_var


# ============================================================================
# TRAINING WRAPPER
# ============================================================================

def train_bgnn_ablation_models(
    dataset: str = 'IXIC',
    loss_functions: Optional[List[str]] = None,
    bayesian_variants: Optional[List[str]] = None,
    window: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    early_stop_patience: int = 10,
    R: int = 3,
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu'
) -> Dict:
    """
    Train MAMBA BGNN ablation models with different loss functions and Bayesian GNN variants

    Args:
        dataset: Dataset name (IXIC, DJI, NYSE)
        loss_functions: List of loss functions to test
                       Options: 'GaussianNLLLoss', 'MSELoss', 'SmoothL1Loss'
                       None = all three
        bayesian_variants: List of variants to test
                          Options: 'with', 'without' (BayesianMAGAC)
                          None = both
        window: Lookback window size
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        early_stop_patience: Early stopping patience
        R: Number of MAMBA layers
        verbose: Print progress
        log_base_dir: Base directory for logs
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Dictionary with results for each model combination
    """
    if loss_functions is None:
        loss_functions = ['GaussianNLLLoss', 'MSELoss', 'SmoothL1Loss']
    if bayesian_variants is None:
        bayesian_variants = ['with', 'without']

    print("="*80)
    print("MAMBA BGNN ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Loss Functions: {', '.join(loss_functions)}")
    print(f"Bayesian Variants: {', '.join([f'BayesianGNN ({v})' for v in bayesian_variants])}")
    print(f"Window: {window}, Batch: {batch_size}, Epochs: {epochs}")
    print(f"LR: {learning_rate}, Layers: {R}")
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

    # Model arguments
    args = ModelArgs(
        d_model=num_features,
        seq_len=window,
        d_proj_E=hidden_dim,
        d_proj_H=hidden_dim,
        d_proj_U=hidden_dim // 2,
        d_state=hidden_dim
    )

    # Create all model combinations
    models_dict = {}
    for bayesian_variant in bayesian_variants:
        for loss_fn_name in loss_functions:
            model_name = f"MAMBA_{'WithBayesianGNN' if bayesian_variant == 'with' else 'NoGraph'}_{loss_fn_name}"

            if bayesian_variant == 'with':
                model = MAMBA_WithBayesianGNN(args, R=R, K=3, d_e=10, mc_train=3, mc_eval=10)
            else:
                model = MAMBA_NoGraph(args, R=R)

            # Initialize weights
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            models_dict[model_name] = (model, loss_fn_name)

    # Training config
    def create_optimizer(params, lr):
        return torch.optim.Adam(params, lr=lr)

    # Train models
    results = {}
    for model_name, (model, loss_fn_name) in models_dict.items():
        print(f"\n{'='*80}")
        print(f"Training: {model_name}")
        print(f"{'='*80}")

        config = {
            'epochs': epochs,
            'lr': learning_rate,
            'loss_type': loss_fn_name.lower(),
            'patience': early_stop_patience,
            'optimizer_fn': create_optimizer,
            'scheduler_fn': None
        }

        # Create single model dict for training
        single_model_dict = {model_name: model}

        # Train using unified pipeline
        result = train_models(
            models_dict=single_model_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset=dataset,
            config=config,
            log_base_dir=log_base_dir,
            study_name='mamba_bgnn_ablation',
            verbose=verbose,
            device=device
        )

        results.update(result)

    # Analysis: Compare loss functions effect
    print("\n" + "="*80)
    print("LOSS FUNCTION COMPARISON")
    print("="*80)

    # Within NoGraph models
    print("\n[NoGraph Models - Loss Function Ablation]")
    if 'MAMBA_NoGraph_GaussianNLLLoss' in results and 'MAMBA_NoGraph_MSE' in results:
        gaussian_ic = results['MAMBA_NoGraph_GaussianNLLLoss']['ic']
        mse_ic = results['MAMBA_NoGraph_MSE']['ic']
        improvement = ((gaussian_ic - mse_ic) / abs(mse_ic)) * 100
        print(f"GaussianNLLLoss vs MSELoss (NoGraph):     {improvement:+.2f}%")

    if 'MAMBA_NoGraph_MSE' in results and 'MAMBA_NoGraph_SmoothL1Loss' in results:
        mse_ic = results['MAMBA_NoGraph_MSE']['ic']
        smoothl1_ic = results['MAMBA_NoGraph_SmoothL1Loss']['ic']
        improvement = ((mse_ic - smoothl1_ic) / abs(smoothl1_ic)) * 100
        print(f"MSELoss vs SmoothL1Loss (NoGraph):       {improvement:+.2f}%")

    # Within BayesianGNN models
    print("\n[BayesianGNN Models - Loss Function Ablation]")
    if 'MAMBA_WithBayesianGNN_GaussianNLLLoss' in results and 'MAMBA_WithBayesianGNN_MSE' in results:
        gaussian_ic = results['MAMBA_WithBayesianGNN_GaussianNLLLoss']['ic']
        mse_ic = results['MAMBA_WithBayesianGNN_MSE']['ic']
        improvement = ((gaussian_ic - mse_ic) / abs(mse_ic)) * 100
        print(f"GaussianNLLLoss vs MSELoss (BayesianGNN): {improvement:+.2f}%")

    if 'MAMBA_WithBayesianGNN_MSE' in results and 'MAMBA_WithBayesianGNN_SmoothL1Loss' in results:
        mse_ic = results['MAMBA_WithBayesianGNN_MSE']['ic']
        smoothl1_ic = results['MAMBA_WithBayesianGNN_SmoothL1Loss']['ic']
        improvement = ((mse_ic - smoothl1_ic) / abs(smoothl1_ic)) * 100
        print(f"MSELoss vs SmoothL1Loss (BayesianGNN):   {improvement:+.2f}%")

    # Analysis: Compare Bayesian GNN effect
    print("\n" + "="*80)
    print("BAYESIAN GNN ABLATION")
    print("="*80)

    for loss_fn_name in loss_functions:
        no_graph_name = f"MAMBA_NoGraph_{loss_fn_name}"
        with_graph_name = f"MAMBA_WithBayesianGNN_{loss_fn_name}"

        if no_graph_name in results and with_graph_name in results:
            no_graph_ic = results[no_graph_name]['ic']
            with_graph_ic = results[with_graph_name]['ic']
            improvement = ((with_graph_ic - no_graph_ic) / abs(no_graph_ic)) * 100
            print(f"BayesianGNN gain ({loss_fn_name:15s}): {improvement:+.2f}%  "
                  f"({with_graph_ic:.6f} vs {no_graph_ic:.6f})")

    print("="*80)

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Train all MAMBA BGNN ablation variants

    This will train 6 model combinations:
    1. MAMBA_NoGraph with GaussianNLLLoss
    2. MAMBA_NoGraph with MSELoss
    3. MAMBA_NoGraph with SmoothL1Loss
    4. MAMBA_WithBayesianGNN with GaussianNLLLoss
    5. MAMBA_WithBayesianGNN with MSELoss
    6. MAMBA_WithBayesianGNN with SmoothL1Loss
    """
    import random
    import numpy as np
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(26)
    np.random.seed(10)
    random.seed(95)

    # Step 1: Train all ablation models on a dataset
    print("\n" + "="*80)
    print("STEP 1: Training MAMBA BGNN ablation models")
    print("="*80)

    dataset = 'IXIC'
    print(f"\n>>> Training on {dataset}...")
    results = train_bgnn_ablation_models(
        dataset=dataset,
        loss_functions=['GaussianNLLLoss', 'MSELoss', 'SmoothL1Loss'],
        bayesian_variants=['with', 'without'],
        epochs=50,
        early_stop_patience=10,
        hidden_dim=64,
        window=5,
        batch_size=32,
        learning_rate=0.001,
        R=3,
        verbose=True,
        device=device
    )

    print("\n" + "="*80)
    print("✓ MAMBA BGNN Ablation Study Completed!")
    print("="*80)

    print("\nModel Architecture Summary:")
    print("  MAMBA_NoGraph:       Pure MAMBA without Graph component")
    print("  MAMBA_WithBayesianGNN: MAMBA + BayesianMAGAC (Bayesian Graph)")
    print("\nLoss Functions:")
    print("  GaussianNLLLoss: Probabilistic loss for Bayesian uncertainty")
    print("  MSELoss:         Standard L2 regression loss")
    print("  SmoothL1Loss:    Robust Huber loss")

    print("\nDetailed results:")
    print(f"  - Single-dataset results: logs/mamba_bgnn_ablation/")
