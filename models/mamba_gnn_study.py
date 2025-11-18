"""
MAMBA-GNN Ablation Study - Graph Layer Comparison

This module compares different graph neural network layers combined with BIMamba:
    1. BIMamba + GCN        - Graph Convolutional Network
    2. BIMamba + GAT        - Graph Attention Network
    3. BIMamba + GraphSAGE  - GraphSAGE aggregation
    4. BIMamba + MAGAC      - Multi-head Adaptive Graph Attention Convolution (baseline)

All models output (mean, log_var) for probabilistic evaluation.

Key Ablation Dimension:
    - Graph Layer: Compare different graph convolution architectures after BIMamba encoding

Usage:

    Train GNN comparison models on a dataset:
    =========================================
    from models.mamba_gnn_study import train_gnn_comparison

    results = train_gnn_comparison(
        dataset='IXIC',
        models=['GCN', 'GAT', 'GraphSAGE', 'MAGAC'],
        epochs=50,
        verbose=True
    )


    Calculate cross-sectional IC for all GNN models:
    ================================================
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # After training on multiple datasets
    cross_results = calculate_cross_sectional_for_all_models(
        models=['BIMamba+GCN', 'BIMamba+GAT', 'BIMamba+GraphSAGE', 'BIMamba+MAGAC'],
        datasets=['IXIC', 'DJI', 'NYSE'],
        study_name='mamba_gnn',
        output_file='logs/mamba_gnn_cross_sectional_summary.txt'
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
# MAMBA CORE COMPONENTS (from mamba_bgnn.py)
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

class GCNLayer(nn.Module):
    """
    Basic Graph Convolutional Network Layer

    Reference: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
    """
    def __init__(self, num_nodes: int, in_dim: int, d_e: int = 10):
        super().__init__()
        self.N = num_nodes
        self.in_dim = in_dim

        # Learnable node embeddings for constructing adjacency
        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_e))

        # GCN weight matrix
        self.W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    def _build_adjacency(self):
        """Build adjacency matrix from node embeddings using cosine similarity"""
        # Normalize embeddings
        emb_norm = F.normalize(self.node_emb, p=2, dim=1)
        # Compute similarity matrix
        A = torch.mm(emb_norm, emb_norm.t())  # (N, N)
        # Apply softmax to get row-stochastic matrix
        A = F.softmax(A, dim=1)
        return A

    def forward(self, x):
        """
        Args:
            x: (B, N, L) - batch, nodes, sequence_length
        Returns:
            out: (B, N) - aggregated node features
        """
        B, N, L = x.shape
        assert N == self.N, f"Expected {self.N} nodes, got {N}"

        # Build adjacency matrix
        A = self._build_adjacency()  # (N, N)

        # Add self-loops and normalize (symmetric normalization)
        A_hat = A + torch.eye(N, device=A.device)
        D = torch.diag(A_hat.sum(1).pow(-0.5))
        A_norm = D @ A_hat @ D  # Normalized adjacency

        # Graph convolution: A_norm @ X @ W
        # x: (B, N, L) -> transpose to (B, L, N) for weight multiplication
        x_t = x.transpose(1, 2)  # (B, L, N)

        # Apply weights: (B, L, N) @ (N, out_dim)
        # But we want output (B, N), so we aggregate over L first
        x_pooled = x.mean(dim=2)  # (B, N) - mean pooling over sequence

        # Graph convolution: (N, N) @ (B, N)^T -> (N, B) -> (B, N)
        out = torch.mm(A_norm, x_pooled.t()).t()  # (B, N)

        # Apply transformation
        out = out @ self.W.t()  # (B, N) @ (N, N) -> (B, N)
        out = out + self.bias

        return out


class GATLayer(nn.Module):
    """
    Basic Graph Attention Network Layer

    Reference: Veličković et al. (2018) - Graph Attention Networks
    """
    def __init__(self, num_nodes: int, in_dim: int, d_e: int = 10, heads: int = 4):
        super().__init__()
        self.N = num_nodes
        self.in_dim = in_dim
        self.H = heads
        self.d_e = d_e

        # Node embeddings
        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_e))

        # Attention parameters per head
        self.W_heads = nn.Parameter(torch.randn(heads, d_e, d_e))
        self.a_heads = nn.Parameter(torch.randn(heads, 2 * d_e, 1))  # attention mechanism

        # Output projection: aggregate multi-head features to node-level
        self.W_out = nn.Linear(num_nodes * heads, num_nodes)

    def forward(self, x):
        """
        Args:
            x: (B, N, L) - batch, nodes, sequence_length
        Returns:
            out: (B, N) - aggregated node features
        """
        B, N, L = x.shape
        assert N == self.N

        # Pool sequence dimension
        x_pooled = x.mean(dim=2)  # (B, N)

        # Multi-head attention
        head_outputs = []
        for h in range(self.H):
            # Transform node embeddings
            node_feat = self.node_emb @ self.W_heads[h]  # (N, d_e)

            # Compute attention scores
            # Concatenate each pair of nodes
            node_i = node_feat.unsqueeze(1).expand(-1, N, -1)  # (N, N, d_e)
            node_j = node_feat.unsqueeze(0).expand(N, -1, -1)  # (N, N, d_e)
            concat = torch.cat([node_i, node_j], dim=2)  # (N, N, 2*d_e)

            # Attention mechanism
            e = torch.matmul(concat, self.a_heads[h]).squeeze(-1)  # (N, N)
            alpha = F.softmax(F.leaky_relu(e, 0.2), dim=1)  # (N, N)

            # Aggregate: alpha @ x_pooled
            h_out = torch.mm(alpha, x_pooled.t()).t()  # (B, N)
            head_outputs.append(h_out)

        # Concatenate heads: (B, N, H) -> (B, N*H)
        multi_head = torch.cat(head_outputs, dim=1)  # (B, N*H)

        # Output projection
        out = self.W_out(multi_head)  # (B, N)

        return out


class GraphSAGELayer(nn.Module):
    """
    Basic GraphSAGE Layer with mean aggregation

    Reference: Hamilton et al. (2017) - Inductive Representation Learning on Large Graphs
    """
    def __init__(self, num_nodes: int, in_dim: int, d_e: int = 10):
        super().__init__()
        self.N = num_nodes
        self.in_dim = in_dim

        # Node embeddings for neighborhood construction
        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_e))

        # SAGE aggregation weights
        self.W_neigh = nn.Parameter(torch.randn(d_e, num_nodes))
        self.W_self = nn.Parameter(torch.randn(in_dim, num_nodes))
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    def _build_adjacency(self):
        """Build adjacency from node embeddings"""
        emb_norm = F.normalize(self.node_emb, p=2, dim=1)
        A = torch.mm(emb_norm, emb_norm.t())
        A = F.softmax(A, dim=1)
        # Make binary adjacency (top-k neighbors)
        k = max(3, self.N // 10)  # Connect to ~10% of nodes
        topk_vals, topk_idx = torch.topk(A, k, dim=1)
        A_binary = torch.zeros_like(A)
        A_binary.scatter_(1, topk_idx, 1.0)
        # Normalize
        D_inv = 1.0 / (A_binary.sum(1) + 1e-6)
        A_norm = A_binary * D_inv.unsqueeze(1)
        return A_norm

    def forward(self, x):
        """
        Args:
            x: (B, N, L)
        Returns:
            out: (B, N)
        """
        B, N, L = x.shape
        assert N == self.N

        # Pool sequence
        x_pooled = x.mean(dim=2)  # (B, N)

        # Build adjacency
        A = self._build_adjacency()  # (N, N)

        # Aggregate neighbors: A @ x
        neigh_agg = torch.mm(A, x_pooled.t()).t()  # (B, N)

        # Combine neighbor and self
        # Use node embeddings as neighbor features
        neigh_feat = neigh_agg @ self.node_emb @ self.W_neigh  # (B, N) @ (N, d_e) @ (d_e, N) -> needs fix

        # Simpler approach: concatenate and transform
        self_feat = x_pooled  # (B, N)

        # Simple mean aggregation
        out = (self_feat + neigh_agg) / 2  # (B, N)

        return out


class MAGACLayer(nn.Module):
    """
    Multi-head Adaptive Graph Attention Convolution (from mamba_bgnn.py)
    Simplified non-Bayesian version for fair comparison
    """
    def __init__(self, num_nodes: int, in_dim: int, K: int = 3, d_e: int = 10, heads: int = 4):
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

    def _attn_A(self):
        Q = torch.einsum('nd,dhm->nhm', self.psi_emb, self.W_q)  # (N, H, d_e)
        K = torch.einsum('nd,dhm->nhm', self.psi_emb, self.W_k)  # (N, H, d_e)
        d_e = Q.size(-1)
        attn = torch.einsum('nhd,mhd->hnm', Q, K) / math.sqrt(d_e)  # (H, N, N)
        return F.softmax(attn, dim=-1)

    def _blend(self, A_g, A_attn_h):
        alpha = torch.sigmoid(self.attn_alpha)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def forward(self, x):
        """
        Args:
            x: (B, N, L)
        Returns:
            out: (B, N)
        """
        B, N, L = x.shape
        assert N == self.N

        # Build effective adjacency per head
        A_base = self._gaussian_A()  # (N, N)
        A_attn = self._attn_A()  # (H, N, N)
        A_effs = torch.stack([self._blend(A_base, A_attn[h]) for h in range(self.H)], dim=0)  # (H, N, N)

        # Aggregate over heads
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


# ============================================================================
# MODEL WRAPPERS: BIMamba + GNN
# ============================================================================

class BIMamba_GCN(nn.Module):
    """BIMamba encoder + GCN aggregation"""
    def __init__(self, args: ModelArgs, R: int = 3, d_e: int = 10):
        super().__init__()
        self.bi_mamba = BIMambaBlock(args, R=R)
        self.gcn = GCNLayer(args.d_model, args.seq_len, d_e=d_e)
        self.head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        """
        Args:
            x: (B, L, N)
        Returns:
            mu, log_var: (B,), (B,)
        """
        # Temporal encoding with BIMamba
        y_seq = self.bi_mamba(x)  # (B, L, N)

        # Graph convolution (expects B, N, L)
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node = self.gcn(z_node)  # (B, N)

        # Linear head for prediction
        w = self.head.weight.squeeze(0)  # (N,)
        b = self.head.bias  # scalar

        mu = torch.einsum('bn,n->b', g_node, w) + b

        # Simple variance estimation
        log_var = torch.ones_like(mu) * (-2.0)  # log(0.135) ≈ -2

        return mu, log_var


class BIMamba_GAT(nn.Module):
    """BIMamba encoder + GAT aggregation"""
    def __init__(self, args: ModelArgs, R: int = 3, d_e: int = 10, heads: int = 4):
        super().__init__()
        self.bi_mamba = BIMambaBlock(args, R=R)
        self.gat = GATLayer(args.d_model, args.seq_len, d_e=d_e, heads=heads)
        self.head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        y_seq = self.bi_mamba(x)  # (B, L, N)
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node = self.gat(z_node)  # (B, N)

        w = self.head.weight.squeeze(0)
        b = self.head.bias
        mu = torch.einsum('bn,n->b', g_node, w) + b
        log_var = torch.ones_like(mu) * (-2.0)

        return mu, log_var


class BIMamba_GraphSAGE(nn.Module):
    """BIMamba encoder + GraphSAGE aggregation"""
    def __init__(self, args: ModelArgs, R: int = 3, d_e: int = 10):
        super().__init__()
        self.bi_mamba = BIMambaBlock(args, R=R)
        self.sage = GraphSAGELayer(args.d_model, args.seq_len, d_e=d_e)
        self.head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        y_seq = self.bi_mamba(x)  # (B, L, N)
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node = self.sage(z_node)  # (B, N)

        w = self.head.weight.squeeze(0)
        b = self.head.bias
        mu = torch.einsum('bn,n->b', g_node, w) + b
        log_var = torch.ones_like(mu) * (-2.0)

        return mu, log_var


class BIMamba_MAGAC(nn.Module):
    """BIMamba encoder + MAGAC aggregation (baseline)"""
    def __init__(self, args: ModelArgs, R: int = 3, K: int = 3, d_e: int = 10, heads: int = 4):
        super().__init__()
        self.bi_mamba = BIMambaBlock(args, R=R)
        self.magac = MAGACLayer(args.d_model, args.seq_len, K=K, d_e=d_e, heads=heads)
        self.head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        y_seq = self.bi_mamba(x)  # (B, L, N)
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node = self.magac(z_node)  # (B, N)

        w = self.head.weight.squeeze(0)
        b = self.head.bias
        mu = torch.einsum('bn,n->b', g_node, w) + b
        log_var = torch.ones_like(mu) * (-2.0)

        return mu, log_var


# ============================================================================
# TRAINING WRAPPER
# ============================================================================

def train_gnn_comparison(
    dataset: str = 'IXIC',
    models: Optional[List[str]] = None,
    window: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    loss_type: str = 'auto',
    early_stop_patience: int = 10,
    R: int = 3,
    K: int = 3,
    d_e: int = 10,
    heads: int = 4,
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu'
) -> Dict:
    """
    Compare different GNN layers combined with BIMamba

    Args:
        dataset: Dataset name (IXIC, DJI, NYSE)
        models: List of models to train (None = all 4 models)
        window: Lookback window size
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        loss_type: 'auto', 'nll', 'mse', 'mae', 'huber'
        early_stop_patience: Early stopping patience
        R: Number of BIMamba layers
        K: Chebyshev polynomial order (for MAGAC)
        d_e: Node embedding dimension
        heads: Number of attention heads
        verbose: Print progress
        log_base_dir: Base directory for logs
        device: Device to train on

    Returns:
        Dictionary with results for each model
    """
    if models is None:
        models = ['GCN', 'GAT', 'GraphSAGE', 'MAGAC']

    print("="*80)
    print("MAMBA-GNN ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Models: BIMamba + {', '.join(models)}")
    print(f"Window: {window}, Batch: {batch_size}, Epochs: {epochs}")
    print(f"Loss: {loss_type}, LR: {learning_rate}")
    print(f"R: {R}, K: {K}, d_e: {d_e}, heads: {heads}")
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
    models_dict = {}
    for model_name in models:
        if model_name == 'GCN':
            model = BIMamba_GCN(args, R=R, d_e=d_e)
        elif model_name == 'GAT':
            model = BIMamba_GAT(args, R=R, d_e=d_e, heads=heads)
        elif model_name == 'GraphSAGE':
            model = BIMamba_GraphSAGE(args, R=R, d_e=d_e)
        elif model_name == 'MAGAC':
            model = BIMamba_MAGAC(args, R=R, K=K, d_e=d_e, heads=heads)
        else:
            print(f"Unknown model: {model_name}, skipping...")
            continue

        # Initialize weights
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        models_dict[f"BIMamba+{model_name}"] = model

    # Training config
    config = {
        'epochs': epochs,
        'lr': learning_rate,
        'loss_type': loss_type,
        'patience': early_stop_patience,
        'optimizer_fn': lambda params, lr: torch.optim.Adam(params, lr=lr),
        'scheduler_fn': None
    }

    # Train using unified pipeline
    results = train_models(
        models_dict=models_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=dataset,
        config=config,
        log_base_dir=log_base_dir,
        study_name='mamba_gnn',
        verbose=verbose,
        device=device
    )

    # Improvement analysis
    print("\n" + "="*80)
    print("GNN LAYER COMPARISON")
    print("="*80)

    # Compare each GNN vs MAGAC baseline
    if 'BIMamba+MAGAC' in results:
        magac_ic = results['BIMamba+MAGAC']['ic']
        magac_ric = results['BIMamba+MAGAC']['ric']

        print(f"Baseline (MAGAC):  IC={magac_ic:.4f}, RIC={magac_ric:.4f}")
        print("-" * 80)

        for gnn in ['GCN', 'GAT', 'GraphSAGE']:
            key = f'BIMamba+{gnn}'
            if key in results:
                gnn_ic = results[key]['ic']
                gnn_ric = results[key]['ric']
                ic_diff = ((gnn_ic - magac_ic) / abs(magac_ic)) * 100
                ric_diff = ((gnn_ric - magac_ric) / abs(magac_ric)) * 100

                print(f"{gnn:12s}: IC={gnn_ic:.4f} ({ic_diff:+.2f}%), RIC={gnn_ric:.4f} ({ric_diff:+.2f}%)")

    print("="*80)

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Train GNN comparison models and calculate cross-sectional IC

    Step 1: Train models on multiple datasets
    Step 2: Calculate cross-sectional IC for all models
    """
    from utils.baseline_trainer import calculate_cross_sectional_for_all_models

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Step 1: Train models on multiple datasets
    print("\n" + "="*80)
    print("STEP 1: Training GNN comparison models on multiple datasets")
    print("="*80)

    datasets = ['IXIC', 'DJI', 'NYSE']
    models = ['GCN', 'GAT', 'GraphSAGE', 'MAGAC']

    for dataset in datasets:
        print(f"\n>>> Training on {dataset}...")
        results = train_gnn_comparison(
            dataset=dataset,
            models=models,
            epochs=50,
            loss_type='auto',
            early_stop_patience=10,
            hidden_dim=64,
            window=5,
            batch_size=32,
            learning_rate=0.001,
            R=3,
            K=3,
            d_e=10,
            heads=4,
            verbose=True,
            device=device
        )

    # Step 2: Calculate cross-sectional IC for all models
    print("\n" + "="*80)
    print("STEP 2: Calculating cross-sectional IC for GNN models")
    print("="*80)

    # Model names with prefix for cross-sectional calculation
    model_names = ['BIMamba+GCN', 'BIMamba+GAT', 'BIMamba+GraphSAGE', 'BIMamba+MAGAC']

    cross_results = calculate_cross_sectional_for_all_models(
        models=model_names,
        datasets=datasets,
        study_name='mamba_gnn',
        output_file='logs/mamba_gnn_cross_sectional_summary.txt',
        verbose=True
    )

    print("\n" + "="*80)
    print("✓ MAMBA-GNN Comparison Study Completed!")
    print("="*80)
    print("\nCross-Sectional IC Results:")
    for model_name, result in cross_results.items():
        if result.get('error'):
            print(f"  {model_name}: Error - {result['error']}")
        else:
            print(f"  {model_name}: IC={result['cross_sectional']['ic_mean']:.6f}, "
                  f"RIC={result['cross_sectional']['ric_mean']:.6f}")

    print("\nModel Architecture Summary:")
    print("  BIMamba+GCN:       Bidirectional Mamba + Graph Convolutional Network")
    print("  BIMamba+GAT:       Bidirectional Mamba + Graph Attention Network")
    print("  BIMamba+GraphSAGE: Bidirectional Mamba + GraphSAGE")
    print("  BIMamba+MAGAC:     Bidirectional Mamba + Multi-head Adaptive GAC (baseline)")

    print("\nDetailed results:")
    print("  - Single-dataset results: logs/mamba_gnn/")
    print("  - Cross-sectional IC: logs/mamba_gnn_cross_sectional_summary.txt")

