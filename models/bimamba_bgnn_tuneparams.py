"""
MAMBA Bayesian GNN - Hyperparameter Tuning for Critical Parameters

This module performs systematic hyperparameter tuning for the top 5 critical parameters
identified in TUNE_PARAMS.md, using GaussianNLL loss optimized for Bayesian uncertainty quantification.

Critical Parameters (from TUNE_PARAMS.md):
    1. K (Chebyshev order):    Impact 10/10, test [2, 3, 4]
    2. R (BIMamba layers):     Impact 9/10, test [2, 3]
    3. mc_eval (MC samples):   Impact 9/10, test [10, 20, 30]
    4. mc_train (MC samples):  Impact 8/10, test [3, 5]
    5. heads (Attention heads): Impact 8/10, test [2, 4, 8]

Total configurations: 3 K × 2 R × 3 mc_eval × 2 mc_train × 3 heads = 108 experiments

Loss Function:
    - GaussianNLL: Optimal for Bayesian models with uncertainty quantification
    - Directly optimizes mean and variance predictions
    - Provides proper probabilistic calibration

Note:
    - This module focuses on Bayesian MAGAC with MC sampling
    - For Non-Bayesian models (deterministic MAGAC), use models/mamba_gnn_study.py
    - All models output (mean, log_var) for probabilistic evaluation

Usage:

    Hyperparameter tuning on a single dataset:
    ==========================================
    from models.bimamba_bgnn_tuneparams import tune_hyperparameters

    results = tune_hyperparameters(
        dataset='IXIC',
        K_values=[2, 3, 4],
        R_values=[2, 3],
        mc_eval_values=[10, 20, 30],
        mc_train_values=[3, 5],
        heads_values=[2, 4, 8],
        epochs=50,
        verbose=True
    )


    Quick test with reduced parameter space:
    ========================================
    results = tune_hyperparameters(
        dataset='IXIC',
        K_values=[3, 4],           # Best performers
        R_values=[3],              # Default
        mc_eval_values=[20],       # Default
        mc_train_values=[3],       # Default
        heads_values=[4, 8],       # Best performers
        epochs=30
    )


    Calculate cross-sectional IC:
    =============================
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    cross_results = calculate_cross_sectional_for_all_models(
        models=['K3_R3_MC3-20_H4', 'K4_R3_MC3-20_H8', ...],
        datasets=['IXIC', 'DJI', 'NYSE'],
        study_name='hparam_tuning',
        output_file='logs/hparam_tuning_cross_sectional.txt'
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
    """Feed-forward network with dropout"""
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
    """Bidirectional Mamba Block - captures temporal patterns in both directions"""
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
# BAYESIAN GRAPH NEURAL NETWORK LAYER
# ============================================================================

class BayesianMAGAC(nn.Module):
    """
    Bayesian Multi-head Adaptive Graph Attention Convolution

    Key features:
    - MC Dropout on node embeddings for uncertainty quantification
    - DropEdge for graph structure uncertainty
    - Multiple forward passes (MC sampling) during train and eval
    - Returns both mean and log variance predictions

    Optimized for GaussianNLL loss:
    - Direct variance estimation via MC sampling
    - Proper uncertainty calibration
    - Efficient gradient flow for both mean and variance
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

        # MC sampling configuration
        self.mc_train = mc_train
        self.mc_eval = mc_eval
        self.mc_samples = mc_train
        self.drop_edge_p = drop_edge_p
        self.mc_dropout_p = mc_dropout_p

        # Node embedding & Gaussian kernel
        self.psi_emb = nn.Parameter(torch.randn(num_nodes, d_e))
        self.psi = nn.Parameter(torch.tensor(1.0))

        # Attention-based dynamic adjacency (multi-head)
        self.W_q = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.W_k = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.attn_alpha = nn.Parameter(torch.tensor(0.5))

        # Factorized Chebyshev filter weights per head
        self.F_w = nn.Parameter(torch.randn(heads, d_e, K + 1, in_dim))
        self.f_b = nn.Parameter(torch.randn(heads, d_e))
        self.head_mix = nn.Parameter(torch.ones(heads))

        self.register_buffer("eye_N", torch.eye(num_nodes))

    def train(self, mode: bool = True):
        """Switch between train and eval MC sampling"""
        super().train(mode)
        self.mc_samples = self.mc_train if mode else self.mc_eval
        return self

    def _gaussian_A(self, psi):
        """Gaussian kernel adjacency matrix"""
        diff = psi[:, None, :] - psi[None, :, :]
        dist2 = diff.pow(2).sum(-1)
        A = torch.exp(-self.psi * dist2)
        return F.softmax(A, dim=1)  # (N, N)

    def _attn_A(self, psi):
        """Multi-head attention adjacency matrix"""
        Q = torch.einsum('nd,dhm->nhm', psi, self.W_q)  # (N, H, d_e)
        K = torch.einsum('nd,dhm->nhm', psi, self.W_k)  # (N, H, d_e)
        d_e = Q.size(-1)
        attn = torch.einsum('nhd,mhd->hnm', Q, K) / math.sqrt(d_e)  # (H, N, N)
        return F.softmax(attn, dim=-1)

    def _blend(self, A_g, A_attn_h):
        """Blend Gaussian and attention adjacency"""
        alpha = torch.sigmoid(self.attn_alpha)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def _sample_A_eff(self, use_dropout: bool):
        """
        Sample effective adjacency matrix with stochastic regularization

        Args:
            use_dropout: Whether to apply MC dropout on embeddings

        Returns:
            A_eff: (H, N, N) - Effective adjacency per head
        """
        # (1) Stochastic node embedding with MC Dropout
        psi_stoch = F.dropout(self.psi_emb, p=self.mc_dropout_p, training=use_dropout)

        # (2) Gaussian adjacency
        A_g = self._gaussian_A(psi_stoch)  # (N, N)

        # (3) Attention adjacency (multi-head)
        A_attn = self._attn_A(psi_stoch)  # (H, N, N)

        # (4) Blend and apply DropEdge
        A_list = []
        for h in range(self.H):
            A_eff = self._blend(A_g, A_attn[h])  # (N, N)

            # DropEdge (only during eval for uncertainty)
            if (not self.training) and (self.drop_edge_p > 0.0):
                keep = torch.bernoulli((1 - self.drop_edge_p) * torch.ones_like(A_eff))
                keep = keep.fill_diagonal_(1.0)
                A_eff = A_eff * keep
                # Row-normalize
                A_eff = A_eff / (A_eff.sum(dim=1, keepdim=True).clamp_min(1e-6))

            A_list.append(A_eff)

        return torch.stack(A_list, dim=0)  # (H, N, N)

    def _forward_single(self, x, A_eff):
        """
        Single forward pass with given adjacency

        Args:
            x: (B, N, L)
            A_eff: (H, N, N)

        Returns:
            out: (B, N)
        """
        B, N, L = x.shape

        mix_w = F.softmax(self.head_mix, dim=0)
        out = 0

        for h in range(self.H):
            A_h = A_eff[h]  # (N, N)

            # Chebyshev polynomial supports
            I = torch.eye(N, device=x.device, dtype=x.dtype)
            supports = [I, A_h]
            for k in range(2, self.K + 1):
                supports.append(2 * A_h @ supports[-1] - supports[-2])
            supports = torch.stack(supports, dim=0)  # (K+1, N, N)

            # Factorized filter
            W_filter = torch.einsum('nd,dkl->nkl', self.psi_emb, self.F_w[h])  # (N, K+1, L)
            b_filter = self.psi_emb @ self.f_b[h]  # (N,)

            # Graph convolution: supports @ x
            x_g = torch.einsum('knm,bml->bknl', supports, x)  # (B, K+1, N, L)
            out_h = torch.einsum('bknl,nkl->bn', x_g, W_filter) + b_filter
            out = out + mix_w[h] * out_h

        return out

    def forward(self, x):
        """
        Forward pass with MC sampling for uncertainty quantification

        Optimized for GaussianNLL:
        - During training: Use single A_eff, multiple MC samples for efficiency
        - During eval: Sample new A_eff per sample for better uncertainty

        Args:
            x: (B, N, L)

        Returns:
            mean: (B, N) - Mean prediction
            log_var: (B, N) - Log variance prediction
        """
        outs = []

        if self.training:
            # Training: Cache one A_eff for efficiency
            A_eff = self._sample_A_eff(use_dropout=True)
            for _ in range(self.mc_samples):
                outs.append(self._forward_single(x, A_eff))
        else:
            # Evaluation: Sample new A_eff per pass for better uncertainty
            for _ in range(self.mc_samples):
                A_eff = self._sample_A_eff(use_dropout=False)
                outs.append(self._forward_single(x, A_eff))

        outs = torch.stack(outs, dim=0)  # (S, B, N)

        # Compute mean and variance
        mean = outs.mean(0)  # (B, N)

        if self.mc_samples == 1:
            # Single sample: use fixed small variance
            log_var = torch.ones_like(mean) * (-4.0)  # log(0.018) ≈ -4
        else:
            # Multiple samples: empirical variance
            var = outs.var(0, unbiased=False) + 1e-6
            log_var = var.log()

        return mean, log_var


# ============================================================================
# FULL MODEL: BIMamba + Bayesian MAGAC
# ============================================================================

class MAMBA_BayesMAGAC(nn.Module):
    """
    Full MAMBA-BGNN Model: BIMamba Encoder + Bayesian MAGAC

    Architecture:
        Input (B, L, N)
            ↓
        BIMamba Encoder [R layers]
            - Forward Mamba
            - Backward Mamba
            - FFN
            ↓
        Bayesian MAGAC [K-hop, multi-head]
            - MC Dropout
            - DropEdge
            - MC Sampling
            ↓
        Linear Head
            ↓
        Output: (mean, log_var)

    Optimized for GaussianNLL loss with proper variance propagation.
    """
    def __init__(self, args: ModelArgs, R: int = 3, K: int = 3,
                 d_e: int = 10, heads: int = 4,
                 mc_train: int = 3, mc_eval: int = 20,
                 drop_edge_p: float = 0.1, mc_dropout_p: float = 0.2):
        super().__init__()

        # Temporal encoder
        self.bi_mamba = BIMambaBlock(args, R=R)

        # Spatial encoder with uncertainty
        self.agc_bayes = BayesianMAGAC(
            args.d_model, args.seq_len, K, d_e, heads=heads,
            mc_train=mc_train, mc_eval=mc_eval,
            drop_edge_p=drop_edge_p, mc_dropout_p=mc_dropout_p
        )

        # Prediction head
        self.head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        """
        Forward pass with uncertainty quantification

        Args:
            x: (B, L, N) - Input sequence

        Returns:
            mu: (B,) - Mean prediction
            log_var: (B,) - Log variance prediction
        """
        # Temporal encoding
        y_seq = self.bi_mamba(x)  # (B, L, N)

        # Spatial encoding with uncertainty
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node, log_var_node = self.agc_bayes(z_node)  # both (B, N)

        # Linear aggregation with variance propagation
        w = self.head.weight.squeeze(0)  # (N,)
        b = self.head.bias  # scalar

        # Mean: E[wx + b] = w·E[x] + b
        mu = torch.einsum('bn,n->b', g_node, w) + b

        # Variance: Var[wx] = w^2·Var[x]
        var = torch.einsum('bn,n->b', log_var_node.exp(), w.pow(2)) + 1e-6
        log_var = var.log()

        return mu, log_var


# ============================================================================
# HYPERPARAMETER TUNING FUNCTION
# ============================================================================

def tune_hyperparameters(
    dataset: str = 'IXIC',
    K_values: Optional[List[int]] = None,
    R_values: Optional[List[int]] = None,
    mc_eval_values: Optional[List[int]] = None,
    mc_train_values: Optional[List[int]] = None,
    heads_values: Optional[List[int]] = None,
    window: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    early_stop_patience: int = 10,
    d_e: int = 10,
    drop_edge_p: float = 0.1,
    mc_dropout_p: float = 0.2,
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu'
) -> Dict:
    """
    Systematic hyperparameter tuning for critical MAMBA-BGNN parameters

    Tunes the top 5 parameters from TUNE_PARAMS.md:
        1. K (Chebyshev order) - Impact 10/10
        2. R (BIMamba layers) - Impact 9/10
        3. mc_eval (MC samples eval) - Impact 9/10
        4. mc_train (MC samples train) - Impact 8/10
        5. heads (Attention heads) - Impact 8/10

    Loss function: GaussianNLL (optimal for Bayesian uncertainty)

    Args:
        dataset: Dataset name (IXIC, DJI, NYSE)
        K_values: Chebyshev polynomial orders to test (None = [2, 3, 4])
        R_values: Number of BIMamba layers to test (None = [2, 3])
        mc_eval_values: MC samples during evaluation (None = [10, 20, 30])
        mc_train_values: MC samples during training (None = [3, 5])
        heads_values: Number of attention heads to test (None = [2, 4, 8])
        window: Lookback window size
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        hidden_dim: Hidden dimension (d_proj_E)
        early_stop_patience: Early stopping patience
        d_e: Node embedding dimension (fixed at 10)
        drop_edge_p: DropEdge probability (fixed at 0.1)
        mc_dropout_p: MC Dropout probability (fixed at 0.2)
        verbose: Print progress
        log_base_dir: Base directory for logs
        device: Device to train on

    Returns:
        Dictionary with results for each configuration:
            {model_name: {'ic': ..., 'ric': ..., 'rmse': ..., 'nll': ...}}
    """
    # Default values from TUNE_PARAMS.md recommendations
    if K_values is None:
        K_values = [2, 3, 4]  # Impact 10/10
    if R_values is None:
        R_values = [2, 3]     # Impact 9/10
    if mc_eval_values is None:
        mc_eval_values = [10, 20, 30]  # Impact 9/10
    if mc_train_values is None:
        mc_train_values = [3, 5]  # Impact 8/10
    if heads_values is None:
        heads_values = [2, 4, 8]  # Impact 8/10

    total_configs = (len(K_values) * len(R_values) * len(mc_eval_values) *
                     len(mc_train_values) * len(heads_values))

    print("="*80)
    print("MAMBA-BGNN: CRITICAL HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Loss Function: GaussianNLL (optimized for Bayesian uncertainty)")
    print("\nTuning Parameters:")
    print(f"  K (Chebyshev order):    {K_values}")
    print(f"  R (BIMamba layers):     {R_values}")
    print(f"  mc_eval (MC samples):   {mc_eval_values}")
    print(f"  mc_train (MC samples):  {mc_train_values}")
    print(f"  heads (Attention heads): {heads_values}")
    print(f"\nFixed Parameters:")
    print(f"  d_e: {d_e}, drop_edge_p: {drop_edge_p}, mc_dropout_p: {mc_dropout_p}")
    print(f"\nTraining Configuration:")
    print(f"  Window: {window}, Batch: {batch_size}, Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}, Hidden Dim: {hidden_dim}")
    print(f"\nTotal configurations: {total_configs}")
    print("="*80)

    # Load data
    print("\nLoading data...")
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'
    num_features, train_loader, val_loader, test_loader = data_processing(
        data_path, window, batch_size
    )
    print(f"✓ Data loaded: {num_features} features")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    # Train all combinations
    all_results = {}
    config_count = 0

    for K in K_values:
        for R in R_values:
            for mc_eval in mc_eval_values:
                for mc_train in mc_train_values:
                    for heads in heads_values:
                        config_count += 1

                        # Model name
                        model_name = f'K{K}_R{R}_MC{mc_train}-{mc_eval}_H{heads}'

                        print(f"\n{'='*80}")
                        print(f"Config {config_count}/{total_configs}: {model_name}")
                        print(f"{'='*80}")
                        print(f"K={K}, R={R}, mc_train={mc_train}, mc_eval={mc_eval}, heads={heads}")

                        # Create model
                        args = ModelArgs(
                            d_model=num_features,
                            seq_len=window,
                            d_proj_E=hidden_dim,
                            d_proj_H=hidden_dim,
                            d_proj_U=hidden_dim // 2,
                            d_state=hidden_dim
                        )

                        model = MAMBA_BayesMAGAC(
                            args, R=R, K=K, d_e=d_e, heads=heads,
                            mc_train=mc_train, mc_eval=mc_eval,
                            drop_edge_p=drop_edge_p, mc_dropout_p=mc_dropout_p
                        )

                        # Initialize weights
                        for p in model.parameters():
                            if p.dim() > 1:
                                nn.init.xavier_uniform_(p)

                        # Training config (GaussianNLL loss)
                        config = {
                            'epochs': epochs,
                            'lr': learning_rate,
                            'loss_type': 'nll',  # GaussianNLL only
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
                            study_name='hparam_tuning',
                            verbose=verbose,
                            device=device
                        )

                        all_results.update(results)

    # Analysis
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*80)

    # Find best for each parameter
    print("\n1. Best Configuration per Parameter Value:")
    print("-"*80)

    # Best for each K
    print(f"\n{'K (Chebyshev Order)'}")
    for K in K_values:
        k_configs = {k: v for k, v in all_results.items() if f'K{K}_' in k}
        if k_configs:
            best = max(k_configs.keys(), key=lambda x: k_configs[x].get('ic', -999))
            print(f"  K={K}: {best:30s} IC={k_configs[best]['ic']:.6f}, "
                  f"RIC={k_configs[best]['ric']:.6f}, NLL={k_configs[best].get('nll', 0):.6f}")

    # Best for each R
    print(f"\n{'R (BIMamba Layers)'}")
    for R in R_values:
        r_configs = {k: v for k, v in all_results.items() if f'_R{R}_' in k}
        if r_configs:
            best = max(r_configs.keys(), key=lambda x: r_configs[x].get('ic', -999))
            print(f"  R={R}: {best:30s} IC={r_configs[best]['ic']:.6f}, "
                  f"RIC={r_configs[best]['ric']:.6f}, NLL={r_configs[best].get('nll', 0):.6f}")

    # Best for each mc_eval
    print(f"\n{'mc_eval (MC Samples Evaluation)'}")
    for mc_eval in mc_eval_values:
        mc_configs = {k: v for k, v in all_results.items() if f'-{mc_eval}_' in k}
        if mc_configs:
            best = max(mc_configs.keys(), key=lambda x: mc_configs[x].get('ic', -999))
            print(f"  mc_eval={mc_eval}: {best:30s} IC={mc_configs[best]['ic']:.6f}, "
                  f"RIC={mc_configs[best]['ric']:.6f}, NLL={mc_configs[best].get('nll', 0):.6f}")

    # Best for each mc_train
    print(f"\n{'mc_train (MC Samples Training)'}")
    for mc_train in mc_train_values:
        mc_configs = {k: v for k, v in all_results.items() if f'MC{mc_train}-' in k}
        if mc_configs:
            best = max(mc_configs.keys(), key=lambda x: mc_configs[x].get('ic', -999))
            print(f"  mc_train={mc_train}: {best:30s} IC={mc_configs[best]['ic']:.6f}, "
                  f"RIC={mc_configs[best]['ric']:.6f}, NLL={mc_configs[best].get('nll', 0):.6f}")

    # Best for each heads
    print(f"\n{'heads (Attention Heads)'}")
    for heads in heads_values:
        h_configs = {k: v for k, v in all_results.items() if f'_H{heads}' in k}
        if h_configs:
            best = max(h_configs.keys(), key=lambda x: h_configs[x].get('ic', -999))
            print(f"  heads={heads}: {best:30s} IC={h_configs[best]['ic']:.6f}, "
                  f"RIC={h_configs[best]['ric']:.6f}, NLL={h_configs[best].get('nll', 0):.6f}")

    # Overall best
    best_overall = max(all_results.keys(), key=lambda x: all_results[x].get('ic', -999))
    best_metrics = all_results[best_overall]

    print("\n" + "="*80)
    print(f"✓ BEST OVERALL CONFIGURATION: {best_overall}")
    print("="*80)
    print(f"  IC:    {best_metrics['ic']:.6f}")
    print(f"  RIC:   {best_metrics['ric']:.6f}")
    print(f"  RMSE:  {best_metrics.get('rmse', 0):.6f}")
    print(f"  NLL:   {best_metrics.get('nll', 0):.6f}")
    print("="*80)

    # Top 5 configurations
    print("\n2. Top 5 Configurations by IC:")
    print("-"*80)
    sorted_configs = sorted(all_results.items(), key=lambda x: x[1].get('ic', -999), reverse=True)
    for i, (name, metrics) in enumerate(sorted_configs[:5], 1):
        print(f"{i}. {name:30s} IC={metrics['ic']:.6f}, RIC={metrics['ric']:.6f}, "
              f"NLL={metrics.get('nll', 0):.6f}")

    print("\n" + "="*80)
    print("Results saved to: logs/hparam_tuning/")
    print("="*80)

    return all_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Hyperparameter tuning for critical parameters

    Modes:
        1. Quick test: Reduced parameter space for fast testing
        2. Full tuning: Complete grid search over all parameters
        3. Multi-dataset: Apply best configs to multiple datasets
    """
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ============ MODE 1: QUICK TEST (9 configs) ============
    """
    print("\n" + "="*80)
    print("QUICK TEST: Reduced parameter space")
    print("="*80)

    results = tune_hyperparameters(
        dataset='IXIC',
        K_values=[3, 4],           # 2 values
        R_values=[3],              # 1 value (fixed)
        mc_eval_values=[20],       # 1 value (fixed)
        mc_train_values=[3],       # 1 value (fixed)
        heads_values=[4, 8],       # 2 values
        epochs=30,
        verbose=True,
        device=device
    )
    # Total: 2 K × 1 R × 1 mc_eval × 1 mc_train × 2 heads = 4 configs
    """

    # ============ MODE 2: FULL TUNING (108 configs) ============
    print("\n" + "="*80)
    print("FULL TUNING: Complete grid search")
    print("="*80)

    results = tune_hyperparameters(
        dataset='IXIC',
        K_values=[2, 3, 4],          # Impact 10/10
        R_values=[2, 3],             # Impact 9/10
        mc_eval_values=[10, 20, 30], # Impact 9/10
        mc_train_values=[3, 5],      # Impact 8/10
        heads_values=[2, 4, 8],      # Impact 8/10
        epochs=50,
        learning_rate=0.001,
        early_stop_patience=10,
        hidden_dim=64,
        window=5,
        batch_size=32,
        verbose=True,
        device=device
    )
    # Total: 3 × 2 × 3 × 2 × 3 = 108 configs

    # ============ MODE 3: MULTI-DATASET (Optional) ============
    """
    print("\n" + "="*80)
    print("MULTI-DATASET: Apply best configs to all datasets")
    print("="*80)

    datasets = ['IXIC', 'DJI', 'NYSE']

    # Use best configurations from IXIC study
    for dataset in datasets:
        print(f"\n>>> Training on {dataset}...")
        results = tune_hyperparameters(
            dataset=dataset,
            K_values=[3, 4],        # Best from IXIC
            R_values=[3],           # Best from IXIC
            mc_eval_values=[20, 30], # Best from IXIC
            mc_train_values=[3],    # Best from IXIC
            heads_values=[4, 8],    # Best from IXIC
            epochs=50,
            verbose=True,
            device=device
        )

    # Calculate cross-sectional IC
    print("\n" + "="*80)
    print("Calculating cross-sectional IC")
    print("="*80)

    models = ['K3_R3_MC3-20_H4', 'K3_R3_MC3-30_H8', 'K4_R3_MC3-20_H4', 'K4_R3_MC3-20_H8']

    cross_results = calculate_cross_sectional_for_all_models(
        models=models,
        datasets=datasets,
        study_name='hparam_tuning',
        output_file='logs/hparam_tuning_cross_sectional.txt',
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
    print("✓ Hyperparameter Tuning Completed!")
    print("="*80)
    print("\nKey Findings:")
    print("  - Tested top 5 critical parameters from TUNE_PARAMS.md")
    print("  - Optimized for GaussianNLL loss (Bayesian uncertainty)")
    print("  - Results saved to: logs/hparam_tuning/")
    print("\nNext Steps:")
    print("  1. Review best configurations from analysis above")
    print("  2. Apply best configs to other datasets (DJI, NYSE)")
    print("  3. Calculate cross-sectional IC for validation")
    print("  4. For Non-Bayesian comparison, use models/mamba_gnn_study.py")
