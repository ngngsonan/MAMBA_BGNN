"""
MAMBA Ablation Models Module - Model Definitions Only

This module contains MAMBA architecture variants for ablation study:
    1. MAMBA     - Single direction Mamba baseline (forward only)
    2. BIMAMBA   - Bidirectional Mamba (forward + backward)
    3. MAMBA+    - Single direction Mamba-2 with SSD
    4. BIMAMBA+  - Bidirectional Mamba-2 with SSD + attention

All models output (mean, log_var) for probabilistic evaluation.

Key Ablation Dimensions:
    - Direction:  Single (forward) vs Bidirectional (forward + backward)
    - SSM:        Mamba (original) vs Mamba-2 (with Structured State-Space Duality)
    - Enhancement: Basic vs Enhanced (with cross-attention)

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
# MAMBA-2 COMPONENTS
# ============================================================================

class Mamba2Block(nn.Module):
    """Enhanced Mamba-2 block with structured state-space duality"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_proj_E * 3, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_proj_E,
            out_channels=args.d_proj_E,
            kernel_size=args.d_conv,
            groups=args.d_proj_E,
            padding=args.d_conv - 1,
            bias=args.conv_bias,
        )

        self.x_proj = nn.Linear(args.d_proj_E, args.dt_rank + args.d_proj_H, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_proj_E, bias=True)

        # Structured state-space matrices
        self.num_blocks = 4
        block_size = args.d_proj_H // self.num_blocks

        A_blocks = []
        for i in range(self.num_blocks):
            A_block = torch.arange(1, block_size + 1).float()
            A_blocks.append(A_block)
        A = torch.block_diag(*[torch.diag(block) for block in A_blocks])
        A = repeat(A, 'h n -> d h n', d=args.d_proj_E)[:, :args.d_proj_H, :args.d_proj_H]
        self.A_log = nn.Parameter(torch.log(A.diagonal(dim1=-2, dim2=-1)))

        self.D = nn.Parameter(torch.ones(args.d_proj_E))
        self.norm = nn.LayerNorm(args.d_proj_E)
        self.out_proj = nn.Linear(args.d_proj_E, args.d_model, bias=args.bias)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        b, l, d = x.shape

        x_proj, z, B_proj = self.in_proj(x).chunk(3, dim=-1)

        x_proj = rearrange(x_proj, 'b l d -> b d l')
        x_proj = self.conv1d(x_proj)[:, :, :l]
        x_proj = rearrange(x_proj, 'b d l -> b l d')
        x_proj = F.silu(x_proj)

        y = self._structured_ssm(x_proj, B_proj)
        y = self.norm(y)
        y = y * torch.sigmoid(z)

        output = self.out_proj(y)
        return output * self.residual_scale

    def _structured_ssm(self, x, B_proj):
        b, l, d = x.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        delta, C = torch.split(x_dbl, [self.args.dt_rank, self.args.d_proj_H], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        B = F.softplus(B_proj)

        y = self._efficient_scan(x, delta, A, B, C, D)
        return y

    def _efficient_scan(self, x, delta, A, B, C, D):
        b, l, d = x.shape
        h = A.shape[-1]

        delta = delta.clamp(min=1e-6, max=1e2)
        A_discrete = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_discrete = delta.unsqueeze(-1) * B.unsqueeze(-1) * x.unsqueeze(-1)

        states = torch.zeros(b, h, device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(l):
            states = A_discrete[:, i, 0] * states + B_discrete[:, i, 0]
            y_i = torch.sum(C[:, i:i+1] * states.unsqueeze(1), dim=-1) + D * x[:, i]
            outputs.append(y_i)

        y = torch.stack(outputs, dim=1)
        return y


class BIMamba2Block(nn.Module):
    """Enhanced Bidirectional Mamba-2 block"""

    def __init__(self, args: ModelArgs, R: int = 3, dropout: float = 0.1):
        super().__init__()
        self.R = R

        self.f_mamba = nn.ModuleList([Mamba2Block(args) for _ in range(R)])
        self.b_mamba = nn.ModuleList([Mamba2Block(args) for _ in range(R)])

        self.norm1 = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])
        self.ffn = nn.ModuleList([self._build_enhanced_ffn(args, dropout) for _ in range(R)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])

        num_heads = 8 if args.d_model % 8 == 0 else (4 if args.d_model % 4 == 0 else 3)
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(args.d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(R)
        ])
        self.norm_attn = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])

    def _build_enhanced_ffn(self, args: ModelArgs, dropout: float):
        hidden_dim = getattr(args, 'd_proj_U', 64)
        return nn.Sequential(
            nn.Linear(args.d_model, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, args.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        for i in range(self.R):
            Y1 = self.f_mamba[i](x)

            x_rev = torch.flip(x, dims=[1]).contiguous()
            Y2_rev = self.b_mamba[i](x_rev)
            Y2 = torch.flip(Y2_rev, dims=[1]).contiguous()

            Y_combined = Y1 + Y2
            Y_attn, _ = self.cross_attention[i](Y_combined, Y_combined, Y_combined)
            Y3 = self.norm_attn[i](Y_combined + Y_attn)

            Y3 = self.norm1[i](x + Y3)
            Yp = self.ffn[i](Y3)
            x = self.norm2[i](Yp + Y3)

        return x


# ============================================================================
# ABLATION MODELS (without graph component)
# ============================================================================

class MAMBAModel(nn.Module):
    """MAMBA - Single direction Mamba baseline"""
    def __init__(self, args: ModelArgs, R: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(args) for _ in range(R)])
        self.norms = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])

        # Pooling and output heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mean_head = nn.Linear(args.d_model, 1)
        self.logvar_head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        # x: (B, L, N)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = norm(x + layer(x))

        # Pool over sequence: (B, L, N) -> (B, N)
        x = x.transpose(1, 2)  # (B, N, L)
        x = self.pool(x).squeeze(-1)  # (B, N)

        mean = self.mean_head(x).squeeze(-1)  # (B,)
        log_var = self.logvar_head(x).squeeze(-1)  # (B,)
        return mean, log_var


class BIMAMBAModel(nn.Module):
    """BIMAMBA - Bidirectional Mamba (forward + backward)"""
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


class MAMBAPlusModel(nn.Module):
    """MAMBA+ - Single direction Mamba-2 with SSD"""
    def __init__(self, args: ModelArgs, R: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([Mamba2Block(args) for _ in range(R)])
        self.norms = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])

        # Pooling and output heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mean_head = nn.Linear(args.d_model, 1)
        self.logvar_head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        # x: (B, L, N)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = norm(x + layer(x))

        # Pool over sequence: (B, L, N) -> (B, N)
        x = x.transpose(1, 2)  # (B, N, L)
        x = self.pool(x).squeeze(-1)  # (B, N)

        mean = self.mean_head(x).squeeze(-1)  # (B,)
        log_var = self.logvar_head(x).squeeze(-1)  # (B,)
        return mean, log_var


class BIMAMBAPlusModel(nn.Module):
    """BIMAMBA+ - Bidirectional Mamba-2 with enhanced SSD and attention"""
    def __init__(self, args: ModelArgs, R: int = 3):
        super().__init__()
        self.bi_mamba2 = BIMamba2Block(args, R=R)

        # Pooling and output heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mean_head = nn.Linear(args.d_model, 1)
        self.logvar_head = nn.Linear(args.d_model, 1)

    def forward(self, x):
        # x: (B, L, N)
        x = self.bi_mamba2(x)  # (B, L, N)

        # Pool over sequence: (B, L, N) -> (B, N)
        x = x.transpose(1, 2)  # (B, N, L)
        x = self.pool(x).squeeze(-1)  # (B, N)

        mean = self.mean_head(x).squeeze(-1)  # (B,)
        log_var = self.logvar_head(x).squeeze(-1)  # (B,)
        return mean, log_var


# ============================================================================
# TRAINING WRAPPER
# ============================================================================

def train_ablation_models(
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
    verbose: bool = True,
    log_base_dir: str = 'logs',
    device: str = 'cpu'
) -> Dict:
    """
    Train MAMBA ablation models using unified training pipeline

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
        R: Number of layers
        verbose: Print progress
        log_base_dir: Base directory for logs
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Dictionary with results for each model
    """
    if models is None:
        models = ['MAMBA', 'BIMAMBA', 'MAMBA+', 'BIMAMBA+']

    print("="*80)
    print("MAMBA ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Window: {window}, Batch: {batch_size}, Epochs: {epochs}")
    print(f"Loss: {loss_type}, LR: {learning_rate}, Layers: {R}")
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

    # Create models
    models_dict = {}
    for model_name in models:
        if model_name == 'MAMBA':
            model = MAMBAModel(args, R=R)
        elif model_name == 'BIMAMBA':
            model = BIMAMBAModel(args, R=R)
        elif model_name == 'MAMBA+':
            model = MAMBAPlusModel(args, R=R)
        elif model_name == 'BIMAMBA+':
            model = BIMAMBAPlusModel(args, R=R)
        else:
            print(f"Unknown model: {model_name}, skipping...")
            continue

        # Initialize weights
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        models_dict[model_name] = model

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
        study_name='mamba_ablation',
        verbose=verbose,
        device=device
    )

    # Add improvement analysis
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)

    # Bidirectional vs Single direction (Mamba)
    if 'MAMBA' in results and 'BIMAMBA' in results:
        mamba_ic = results['MAMBA']['ic']
        bimamba_ic = results['BIMAMBA']['ic']
        improvement = ((bimamba_ic - mamba_ic) / abs(mamba_ic)) * 100
        print(f"BIMAMBA vs MAMBA (IC):       {improvement:+.2f}%  [Bidirectional gain]")

    # Bidirectional vs Single direction (Mamba2)
    if 'MAMBA+' in results and 'BIMAMBA+' in results:
        mambaplus_ic = results['MAMBA+']['ic']
        bimambaplus_ic = results['BIMAMBA+']['ic']
        improvement = ((bimambaplus_ic - mambaplus_ic) / abs(mambaplus_ic)) * 100
        print(f"BIMAMBA+ vs MAMBA+ (IC):     {improvement:+.2f}%  [Bidirectional gain]")

    # Mamba2 vs Mamba (Single direction)
    if 'MAMBA' in results and 'MAMBA+' in results:
        mamba_ic = results['MAMBA']['ic']
        mambaplus_ic = results['MAMBA+']['ic']
        improvement = ((mambaplus_ic - mamba_ic) / abs(mamba_ic)) * 100
        print(f"MAMBA+ vs MAMBA (IC):        {improvement:+.2f}%  [SSD improvement]")

    # Mamba2 vs Mamba (Bidirectional)
    if 'BIMAMBA' in results and 'BIMAMBA+' in results:
        bimamba_ic = results['BIMAMBA']['ic']
        bimambaplus_ic = results['BIMAMBA+']['ic']
        improvement = ((bimambaplus_ic - bimamba_ic) / abs(bimamba_ic)) * 100
        print(f"BIMAMBA+ vs BIMAMBA (IC):    {improvement:+.2f}%  [SSD improvement]")

    # Best overall
    if 'MAMBA' in results and 'BIMAMBA+' in results:
        mamba_ic = results['MAMBA']['ic']
        bimambaplus_ic = results['BIMAMBA+']['ic']
        improvement = ((bimambaplus_ic - mamba_ic) / abs(mamba_ic)) * 100
        print(f"BIMAMBA+ vs MAMBA (IC):      {improvement:+.2f}%  [Overall improvement]")

    print("="*80)

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Train all MAMBA ablation models
    """
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = train_ablation_models(
        dataset='DJI',
        models=['MAMBA', 'BIMAMBA', 'MAMBA+', 'BIMAMBA+'],  # All 4 models
        epochs=50,
        loss_type='auto',
        early_stop_patience=10,
        hidden_dim=64,
        window=5,
        batch_size=32,
        learning_rate=0.001,
        R=3,
        verbose=True,
        device=device
    )

    print("\nAblation study completed!")
    print("Check logs/ablation/ for detailed results and comparison.")
    print("\nModel Architecture Summary:")
    print("  MAMBA:     Single-direction Mamba (forward only)")
    print("  BIMAMBA:   Bidirectional Mamba (forward + backward)")
    print("  MAMBA+:    Single-direction Mamba-2 with SSD")
    print("  BIMAMBA+:  Bidirectional Mamba-2 with SSD + attention")
