# @title MAMBA_BGNN
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

import random
import numpy as np
from datetime import datetime

# Import utilities
from utils.data_processing import data_processing
from utils.trainer import Trainer
from utils.result_plot import plot_analytics

# ---------------------------- Model Arguments ----------------------------
@dataclass
class ModelArgs:
    d_model: int          # = số lượng daily features N
    seq_len: int          # = độ dài lịch sử L
    d_proj_E: int = 64    # E=64 chiều embedding đầu tiên
    d_proj_H: int = 64    # H=64 latent dimension trong SSM
    d_proj_U: int = 32    # U=32 hidden layer trong FFN
    expand: int = 2
    d_state: int = 64     # H = 64
    dt_rank: int | str = 'auto'
    d_conv: int = 3
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_proj_E / 16)

# ---------------------------- Mamba Block ----------------------------
class MambaBlock(nn.Module):
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

# ------------------------- Residual + FFN --------------------------
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


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm = nn.LayerNorm(args.d_model)
        self.mixer = MambaBlock(args)

    def forward(self, x):
        return x + self.mixer(self.norm(x))
# --------------------------- BI-Mamba Stack -------------------------
class BIMambaBlock(nn.Module):
    def __init__(self, args: ModelArgs, R: int = 3, dropout: float = 0.1):
        super().__init__()
        self.R = R
        self.f_mamba = nn.ModuleList([MambaBlock(args) for _ in range(R)])
        self.b_mamba = nn.ModuleList([MambaBlock(args) for _ in range(R)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])
        self.ffn   = nn.ModuleList([FeedForward(args, dropout) for _ in range(R)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])

    def forward(self, x):
        # x: (B, L, N=d_model)
        for i in range(self.R):
            # (8) Y1 = Mamba(X)
            Y1 = self.f_mamba[i](x)

            # (8) Y2 = Mamba(P X); P là phép đảo thời gian theo trục L
            x_rev = torch.flip(x, dims=[1]).contiguous()
            Y2_rev = self.b_mamba[i](x_rev)           # Y2 trên chuỗi đảo
            Y2 = torch.flip(Y2_rev, dims=[1]).contiguous()  # P Y2 để về đúng thứ tự thời gian

            # (9) Y3 = Norm(X + Y1 + P Y2)
            Y3 = self.norm1[i](x + Y1 + Y2)

            # (10) Y' = Projection_L(ReLU(Projection_U(Y3)))
            Yp = self.ffn[i](Y3)

            # (11) Y = Norm(Y' + Y3)
            x = self.norm2[i](Yp + Y3)
        return x

# MAGAC
# --------------------  Adaptive Graph Convolution + MAGAC -------------------------
class MAGAC(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, K: int = 3,
                 d_e: int = 10, heads: int = 4):
        super().__init__()
        self.N      = num_nodes
        self.K      = K
        self.in_dim = in_dim
        self.H      = heads  # số head attention

        # --- Node embedding & Gaussian kernel (như cũ) ---
        self.psi_emb = nn.Parameter(torch.randn(num_nodes, d_e))
        self.psi     = nn.Parameter(torch.tensor(1.0))

        # --- Attention‑based dynamic adjacency ---
        self.W_q = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.W_k = nn.Parameter(torch.randn(d_e, heads, d_e))
        self.attn_alpha = nn.Parameter(torch.tensor(0.5))  # pha trộn Gaussian vs Attention

        # --- Factorised Chebyshev filter weights cho mỗi head ---
        self.F_w = nn.Parameter(torch.randn(heads, d_e, K + 1, in_dim))
        self.f_b = nn.Parameter(torch.randn(heads, d_e))
        self.head_mix = nn.Parameter(torch.ones(heads))  # trọng số tổng hợp head

    # ------------------------------------------------------------------
    def _gaussian_A(self):
        diff = self.psi_emb[:, None, :] - self.psi_emb[None, :, :]
        dist2 = diff.pow(2).sum(-1)
        A = torch.exp(-self.psi * dist2)
        return F.softmax(A, dim=1)  # (N,N)

    def _attn_A(self, psi=None):
        # >>> Bayesian cache (chỉ dùng khi eval) <<<
        if hasattr(self, "_cache_attn") and self._cache_attn is not None and self.training is False:
            return self._cache_attn   
        # Q = torch.einsum('nd,dhm->nhm', self.psi_emb, self.W_q)  # (N,H,d_e)
        # K = torch.einsum('nd,dhm->nhm', self.psi_emb, self.W_k)  # (N,H,d_e)
        psi = self.psi_emb if psi is None else psi
        Q = torch.einsum('nd,dhm->nhm', psi, self.W_q)  # (N,H,d_e)
        K = torch.einsum('nd,dhm->nhm', psi, self.W_k) # (N,H,d_e)
        d_e = Q.size(-1)
        attn = torch.einsum('nhd,mhd->hnm', Q, K) / math.sqrt(d_e)  # (H,N,N)

        attn = F.softmax(attn, dim=-1)                  # row‑wise per head
        # --- lưu cache cho các sample tiếp theo ---
        if hasattr(self, "_cache_attn") and self.training is False:
            self._cache_attn = attn.detach()  
        
        return attn       

    # ---------- helper -------------------------------------------------
    def _blend(self, A_g, A_attn_h):
        """
        self.attn_alpha * A_base + (1 - self.attn_alpha) * A_attn[h]
        """
        alpha = torch.sigmoid(self.attn_alpha)        # bảo đảm (0,1)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def forward(self, x, A_eff_override: torch.Tensor | None = None):
        """x: (B, N, L); A_eff_override: (H, N, N) nếu đã có A_eff"""
        B, N, L = x.shape
        assert N == self.N, "num_nodes mismatch"

        if A_eff_override is not None:
            A_effs = A_eff_override                               # (H,N,N)
        else:
            A_base = self._gaussian_A()                           # (N,N)
            A_attn = self._attn_A()                               # (H,N,N)
            A_effs = torch.stack([self._blend(A_base, A_attn[h])  # (H,N,N)
                                   for h in range(self.H)], dim=0)

        mix_w = F.softmax(self.head_mix, dim=0)                   # (H,)
        out = 0
        for h in range(self.H):
            A_eff = A_effs[h]

            I = torch.eye(N, device=x.device, dtype=x.dtype)
            supports = [I, A_eff]
            for k in range(2, self.K + 1):
                supports.append(2 * A_eff @ supports[-1] - supports[-2])
            supports = torch.stack(supports, dim=0)               # (K+1,N,N)

            W_filter = torch.einsum('nd,dkl->nkl', self.psi_emb, self.F_w[h])  # (N,K+1,L)
            b_filter = self.psi_emb @ self.f_b[h]                                  # (N)

            x_g = torch.einsum('knm,bml->bknl', supports, x)       # (B,K+1,N,L)
            out_h = torch.einsum('bknl,nkl->bn', x_g, W_filter) + b_filter
            out = out + mix_w[h] * out_h
        return out

# BayesianMAGAC v1.1
class BayesianMAGAC(MAGAC):
    def __init__(self, num_nodes, in_dim, K=3, d_e=10, heads=4
                 , mc_train=3, mc_eval=20 #, mc_samples: int = 1
                 , drop_edge_p: float = 0.1, mc_dropout_p: float = 0.2):
        super().__init__(num_nodes, in_dim, K, d_e, heads)
        self.mc_train = mc_train
        self.mc_eval  = mc_eval
        self.mc_samples   = mc_train
        self.drop_edge_p  = drop_edge_p
        self.dropout_emb  = nn.Dropout(p=mc_dropout_p)   # MC-dropout on Ψ
        self.register_buffer("eye_N", torch.eye(num_nodes))  # speed-up
        self._cache_attn  = None  

    # ---------- helper -------------------------------------------------
    def _blend(self, A_g, A_attn_h):
        """α·A_gauss + (1-α)·A_attn  với α∈(0,1)"""
        alpha = torch.sigmoid(self.attn_alpha)        # bảo đảm (0,1)
        return alpha * A_g + (1 - alpha) * A_attn_h

    def train(self, mode: bool = True):
        super().train(mode)
        self.mc_samples = self.mc_train if mode else self.mc_eval
        return self

    def forward(self, x):
        outs = []
        if self.training:
            # cache 1 A_eff cho cả mini-batch để tiết kiệm compute
            A_eff_one = self._sample_A_eff(use_dropout_on_psi=True)
            for _ in range(self.mc_samples):
                outs.append(super().forward(x, A_eff_override=A_eff_one))
        else:
            # eval: mỗi pass là 1 sample A_eff mới (Ψ không dropout)
            for _ in range(self.mc_samples):
                A_eff_s = self._sample_A_eff(use_dropout_on_psi=False)
                outs.append(super().forward(x, A_eff_override=A_eff_s))

        outs = torch.stack(outs, dim=0)                 # (S,B,N)
        mean = outs.mean(0)
        if self.mc_samples == 1:
            log_var = torch.zeros_like(mean)
        else:
            var = outs.var(0, unbiased=False) + 1e-6
            log_var = var.log()
        return mean, log_var
    
    def _sample_A_eff(self, use_dropout_on_psi: bool):
        # (1) stochastic/deterministic node embedding
        psi_stoch = F.dropout(self.psi_emb, p=self.dropout_emb.p, training=use_dropout_on_psi)

        # (2) Gaussian part (row-stochastic)
        diff = psi_stoch[:, None, :] - psi_stoch[None, :, :]
        A_g  = torch.exp(-self.psi * diff.pow(2).sum(-1))
        A_g  = F.softmax(A_g, dim=1)                               # (N,N)

        # (3) Attention part từ Ψ̃
        A_attn = self._attn_A(psi_stoch)                           # (H,N,N)

        # (4) Blend + (eval-only) DropEdge + row-renorm
        A_list = []
        for h in range(self.H):
            A_eff = self._blend(A_g, A_attn[h])                    # (N,N)

            if (self.training is False) and (self.drop_edge_p > 0.0):
                # drop các cạnh ngoài đường chéo, giữ self-loop
                keep = torch.bernoulli((1 - self.drop_edge_p) * torch.ones_like(A_eff))
                keep = keep.fill_diagonal_(1.0)
                A_eff = A_eff * keep
                # renorm hàng để giữ tính row-stochastic (ổn định Chebyshev)
                A_eff = A_eff / (A_eff.sum(dim=1, keepdim=True).clamp_min(1e-6))

            A_list.append(A_eff)
        return torch.stack(A_list, dim=0)                           # (H,N,N)


# --- Modify top-level model -----------------------------------------------
class MAMBA_BayesMAGAC(nn.Module):
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
        y_seq = self.bi_mamba(x)                    # (B,L,N)
        # g_node, log_var_node = self.agc_bayes(      # both (B,N)
        #     y_seq.transpose(1, 2))
        # MAGAC hoạt động ở dạng node-major: (B, N, L)
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node, log_var_node = self.agc_bayes(z_node)  # both (B, N)

        # --- Linear head ---
        w = self.head.weight.squeeze(0)             # (N,)
        b = self.head.bias                          # (1,)

        mu   = torch.einsum('bn,n->b', g_node, w) + b   # (B,)
        var  = torch.einsum('bn,n->b',               # (B,)
                            log_var_node.exp(),      # σ²_node
                            w.pow(2)) + 1e-6
        log_var = var.log()                         # (B,)
        return mu, log_var

# ============================================================================

def main(dataset, loss_type='bayesian'):
    """
    Main training function

    Args:
        dataset: Dataset name ('IXIC', 'DJI', 'NYSE')
        loss_type: Loss function type - 'bayesian' (GaussianNLLLoss), 'mse' (MSELoss), or 'smoothl1' (SmoothL1Loss)
    """
    # >>> MAMBA_BayesMAGAC <<<
    torch.manual_seed(26); np.random.seed(10); random.seed(95)

    # hyper‑parameters
    window      = 5     # history length L
    batch_size  = 128   # batch_size
    data_path = f'Dataset/combined_dataframe_{dataset}.csv'

    N, train_loader, val_loader, test_loader = data_processing(data_path, window, batch_size)
    L = window

    m_args = ModelArgs(d_model=N, seq_len=L, d_state=128)
    args = {
        'epochs': 1500,
        'early_stop': True,
        'early_stop_patience': 20,
        'grad_norm': False,
        'max_grad_norm': 5.0,
        'log_dir': f'logs/FDSE25_{dataset}_{loss_type}_log' + ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': f'MAMBA_BayesMAGAC_{dataset}_{loss_type}',
        'log_step': 20,
        'window_size': 63,
        'step_size': 21
    }
    print(f'data_path: {data_path}')
    print(f'Args {args}, N: {N}, L: {window}, batch_size: {batch_size}, Model_Args: {m_args}')
    print(f'Loss type: {loss_type.upper()}')

    # Initialize model
    print(f" TRAIN MAMBA_BayesMAGAC with {loss_type.upper()}")
    model = MAMBA_BayesMAGAC(m_args, R=3, K=3, d_e=10, mc_train=3, mc_eval=10, drop_edge_p=0.1, mc_dropout_p=0.2)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Select loss function based on loss_type
    if loss_type.lower() == 'bayesian':
        loss_fn = nn.GaussianNLLLoss(full=True, reduction='mean')
    elif loss_type.lower() == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type.lower() == 'smoothl1':
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'bayesian', 'mse', or 'smoothl1'")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[750, 1050, 1350], gamma=0.1)

    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader, test_loader,
                      args=args, lr_scheduler=scheduler, loss_type=loss_type)
    trainer.train()
    trainer.test()
    plot_analytics(trainer.args['log_dir'])

if __name__ == "__main__":
    # Example usage with different loss types:

    # 1. Bayesian mode (default) - uses GaussianNLLLoss with uncertainty quantification
    main('IXIC', loss_type='bayesian')
    main('DJI', loss_type='bayesian')
    main('NYSE', loss_type='bayesian')

    # 2. MSE mode - deterministic training without uncertainty
    # main('IXIC', loss_type='mse')
    # main('DJI', loss_type='mse')
    # main('NYSE', loss_type='mse')

    # 3. SmoothL1 mode - deterministic training with robust loss
    # main('IXIC', loss_type='smoothl1')
    # main('DJI', loss_type='smoothl1')
    # main('NYSE', loss_type='smoothl1')
