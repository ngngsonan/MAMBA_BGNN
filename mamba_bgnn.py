# @title MAMBA_BGNN
# %%writefile /content/MAMBA_BGNN/mamba_bgnn.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

import csv

import random
import numpy as np
import torch.utils.data
import os
import logging
from datetime import datetime
import time
import copy

import sys

import h5py
import argparse
import configparser
import matplotlib.pyplot as plt
import pandas as pd

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, einsum, repeat
from result_plot import plot_analysis

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


# >>> TRAINER - NEW 2 <<<
class Trainer(object):
    """Minimal yet full featured Trainer for MAMBA_BGNN (probabilistic)."""

    # ================= init =================
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, test_loader,
                 args, lr_scheduler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler

        os.makedirs(self.args['log_dir'], exist_ok=True)
        self.logger = self._get_logger()

        self.best_state, self.best_loss = None, float('inf')
        self.not_improved = 0
        self.best_path = os.path.join(args.get('log_dir'), 'best_model.pth')

        # --- CSV paths & headers (MAIN TABLE) ---
        self.val_csv  = os.path.join(self.args['log_dir'], 'val_metrics.csv')
        self.test_csv = os.path.join(self.args['log_dir'], 'test_metrics.csv')
        self.test_pred_csv = os.path.join(self.args['log_dir'], 'test_predictions.csv')
        self.best_val_pred_csv = os.path.join(self.args['log_dir'], 'val_predictions_best.csv')

        self.val_header  = ['epoch','nll','rmse','mae','ic','ric','crps','sharp',
                            'picp90','gap90','picp95','gap95','aurc']
        self.test_header = ['nll','rmse','mae','ic','ric','crps','sharp',
                            'picp90','gap90','picp95','gap95','aurc']

        self._init_csv(self.val_csv,  self.val_header)
        self._init_csv(self.test_csv, self.test_header)

    # ================= logging & csv utils =================
    def _get_logger(self):
        model_name = self.args['model_name'] + ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger = logging.getLogger(model_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.args['log_dir'], model_name + '.log'))
        sh = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', '%Y-%m-%d %H:%M')
        sh.setFormatter(fmt)
        logger.addHandler(sh); logger.addHandler(fh)
        return logger

    def _init_csv(self, path, header):
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(header)

    def _append_csv(self, path, header, row_dict):
        row = []
        for k in header:
            v = row_dict[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            if isinstance(v, (np.generic,)):
                v = float(v)
            row.append(v)
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def _dump_predictions(self, path, mu, sigma, y):
        df = pd.DataFrame({
            'y': y.detach().cpu().numpy().astype(float),
            'mu': mu.detach().cpu().numpy().astype(float),
            'sigma': sigma.detach().cpu().numpy().astype(float),
        })
        df.to_csv(path, index=False)

    # ================= probabilistic helpers =================
    @staticmethod
    def _to_sigma(log_var: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * log_var).clamp_min(1e-8)

    @staticmethod
    def _crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Closed-form CRPS for N(mu, sigma^2):
        # CRPS = sigma * [ z*(2Φ(z)-1) + 2φ(z) - 1/√π ],  z=(y-mu)/sigma
        sigma = sigma.clamp_min(1e-8)
        z = (y - mu) / sigma
        try:
            Phi = torch.special.ndtr(z)
        except AttributeError:
            Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        phi = torch.exp(-0.5 * z**2) / math.sqrt(2*math.pi)
        crps = sigma * (z * (2*Phi - 1) + 2*phi - 1.0/math.sqrt(math.pi))
        return torch.clamp(crps, min=0.0)

    @staticmethod
    def _picp_and_gap(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, q: float):
        # central interval coverage at nominal q (e.g., q=0.90)
        p = (1.0 + q)/2.0
        try:
            z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0*p - 1.0, device=mu.device, dtype=mu.dtype))
        except AttributeError:
            z = math.sqrt(2.0) * torch.special.erfinv(torch.tensor(2.0*p - 1.0, device=mu.device, dtype=mu.dtype))
        lo, hi = mu - z*sigma, mu + z*sigma
        obs = ((y >= lo) & (y <= hi)).float().mean().item()
        gap = abs(obs - q)
        return obs, gap

    @staticmethod
    def _aurc_rmse(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, points: int = 10) -> float:
        # Risk–Coverage AUC for RMSE (lower better)
        idx = torch.argsort(sigma)  # most certain first
        y, mu = y[idx], mu[idx]
        covs = torch.linspace(0.1, 1.0, points, device=y.device)
        prev = None; auc = 0.0
        for i, c in enumerate(covs):
            k = max(1, int(c.item()*y.numel()))
            rmse_c = torch.sqrt(torch.mean((y[:k]-mu[:k])**2)).item()
            if i > 0:
                h = (covs[i] - covs[i-1]).item()
                auc += 0.5 * h * (prev + rmse_c)
            prev = rmse_c
        return auc

    # ================= training loops =================
    def _run_epoch(self, epoch):
        self.model.train(); total = 0.0
        for step, (x, y) in enumerate(self.train_loader):
            self.opt.zero_grad()
            mu, log_var = self.model(x)                               # (B,), (B,)
            loss = self.loss_fn(mu, y.squeeze(), log_var.exp())       # NLL
            loss.backward()
            if self.args['grad_norm']:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.opt.step()
            total += loss.item()
            if step % self.args['log_step'] == 0:
                self.logger.info(f"Epoch {epoch} [{step}/{len(self.train_loader)}] Loss(NLL): {loss.item():.6f}")
        if self.lr_scheduler: self.lr_scheduler.step()
        train_loss = total / len(self.train_loader)
        self.logger.info(f"Epoch {epoch} Train NLL: {train_loss:.6f}")
        return train_loss

    def _validate(self, epoch):
        self.model.eval(); total = 0.0
        preds, trues, logvars = [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                mu, log_var = self.model(x)
                total += self.loss_fn(mu, y.squeeze(), log_var.exp()).item()
                preds.append(mu); trues.append(y.squeeze()); logvars.append(log_var.squeeze())
        nll = total / len(self.val_loader)

        preds   = torch.cat(preds, 0).squeeze(-1)
        trues   = torch.cat(trues, 0).squeeze(-1)
        logvars = torch.cat(logvars, 0).squeeze(-1)
        sigmas  = self._to_sigma(logvars)

        # ---- MAIN metrics ----
        rmse = torch.sqrt(torch.mean((trues - preds)**2))
        mae  = torch.mean(torch.abs(trues - preds))
        ic   = self.pearson(trues, preds)
        ric  = self.ric(trues, preds)
        crps = self._crps_gaussian(preds, sigmas, trues).mean()
        sharp = sigmas.mean()
        picp90, gap90 = self._picp_and_gap(preds, sigmas, trues, q=0.90)
        picp95, gap95 = self._picp_and_gap(preds, sigmas, trues, q=0.95)
        aurc = self._aurc_rmse(trues, preds, sigmas, points=10)

        self.logger.info(
            f"VAL  RMSE:{rmse:.4f}  MAE:{mae:.4f}  IC:{ic:.4f}  RIC:{ric:.4f}  "
            f"NLL:{nll:.5f}  CRPS:{crps:.5f}  Sharp(σ):{sharp:.5f}  "
            f"PICP90:{picp90:.3f}|Gap:{gap90:.3f}  PICP95:{picp95:.3f}|Gap:{gap95:.3f}  "
            f"AURC:{aurc:.5f}"
        )

        # write CSV row
        # row = {'epoch': int(epoch), 'nll': nll.item(), 'rmse': rmse.item(), 'mae': mae.item(),
        #        'ic': ic.item(), 'ric': ric.item(), 'crps': crps.item(), 'sharp': sharp.item(),
        #        'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc}
        row = {'epoch': int(epoch), 'nll': nll, 'rmse': rmse, 'mae': mae,
               'ic': ic, 'ric': ric, 'crps': crps, 'sharp': sharp,
               'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc}
        self._append_csv(self.val_csv, self.val_header, row)

        # return bundle to dump best-val predictions later
        return nll, (preds, sigmas, trues)

    def train(self):
        best_bundle = None
        for epoch in range(1, self.args['epochs'] + 1):
            _ = self._run_epoch(epoch)
            nll, bundle = self._validate(epoch)
            if nll < self.best_loss:
                self.best_loss = nll
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.not_improved = 0
                self.logger.info('--- New best model (by VAL NLL) ---')
                torch.save(self.best_state, self.best_path)
                best_bundle = bundle
            else:
                self.not_improved += 1
            if self.args['early_stop'] and self.not_improved >= self.args['early_stop_patience']:
                self.logger.info('Early stopping triggered.')
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if best_bundle is not None:
            mu_b, sigma_b, y_b = best_bundle
            self._dump_predictions(self.best_val_pred_csv, mu_b, sigma_b, y_b)

    # ================ metrics & test ================
    @staticmethod
    def pearson(x, y):
        vx, vy = x - x.mean(), y - y.mean()
        return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-12)

    @staticmethod
    def rank_tensor(x):
        flat = x.view(-1)
        idx = torch.argsort(flat)
        ranks = torch.empty_like(flat, dtype=torch.float32)
        ranks[idx] = torch.arange(1, flat.numel() + 1, dtype=torch.float32, device=x.device)
        return ranks.view_as(x)

    @staticmethod
    def ric(x, y):
        rx, ry = Trainer.rank_tensor(x), Trainer.rank_tensor(y)
        return Trainer.pearson(rx, ry)

    def test(self):
        self.model.eval(); preds, trues, logvars = [], [], []
        nll_total = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                mu, log_var = self.model(x)
                nll_total += self.loss_fn(mu, y.squeeze(), log_var.exp()).item()
                preds.append(mu); trues.append(y.squeeze()); logvars.append(log_var.squeeze())

        preds   = torch.cat(preds, 0).squeeze(-1)
        trues   = torch.cat(trues, 0).squeeze(-1)
        logvars = torch.cat(logvars, 0).squeeze(-1)
        sigmas  = self._to_sigma(logvars)
        nll = nll_total / max(1, len(self.test_loader))

        rmse = torch.sqrt(torch.mean((trues - preds)**2))
        mae  = torch.mean(torch.abs(trues - preds))
        ic   = self.pearson(trues, preds)
        ric  = self.ric(trues, preds)
        crps = self._crps_gaussian(preds, sigmas, trues).mean()
        sharp = sigmas.mean()
        picp90, gap90 = self._picp_and_gap(preds, sigmas, trues, q=0.90)
        picp95, gap95 = self._picp_and_gap(preds, sigmas, trues, q=0.95)
        aurc = self._aurc_rmse(trues, preds, sigmas, points=10)

        self.logger.info(
            f"TEST RMSE:{rmse:.4f}  MAE:{mae:.4f}  IC:{ic:.4f}  RIC:{ric:.4f}  "
            f"NLL:{nll:.5f}  CRPS:{crps:.5f}  Sharp(σ):{sharp:.5f}  "
            f"PICP90:{picp90:.3f}|Gap:{gap90:.3f}  PICP95:{picp95:.3f}|Gap:{gap95:.3f}  "
            f"AURC:{aurc:.5f}"
        )
        row = { 'nll': nll, 'rmse': rmse, 'mae': mae,
               'ic': ic, 'ric': ric, 'crps': crps, 'sharp': sharp,
               'picp90': picp90, 'gap90': gap90, 'picp95': picp95, 'gap95': gap95, 'aurc': aurc}
        self._append_csv(self.test_csv, self.test_header, row)

        self._dump_predictions(self.test_pred_csv, preds, sigmas, trues)

        return mae, rmse, ic, ric

#  >>> DATA PROCESSING <<<  
# ===========================================================================
from torch.utils.data import DataLoader, TensorDataset

# ---------- Min‑Max scaler (fit on TRAIN features only) ----------
class MinMax01:
    def fit(self, x):
        self.min = x.min(0)
        self.max = x.max(0)
    def transform(self, x):
        return (x - self.min) / (self.max - self.min + 1e-8)
# ---------- DataLoaders ----------
def make_loader(x, y, batch_size):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=False, drop_last=False)

def data_processing(data_path, window, batch_size):
    df = pd.read_csv(data_path,
                     index_col='Date', parse_dates=True) 
    _ = df.pop('Name')
    print(" Data shape:", df.shape)
    # --- LOW VOLUMNE DROP
    # vol_thresh = df['Vol.'].quantile(0.01)
    # df = df[df['Vol.']>vol_thresh]
    # print(" LOW VOLUMNE DROP - Data shape:", df.shape)
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("NaN distribution:\n",df.isnull().sum())
    df.fillna(df.median(), inplace=True)
    df.dropna(inplace=True)
    print(" FILLNA BY MEDIAN - Data shape:", df.shape)
    
    raw_np = df.values.astype('float32')   # (T, F) price = col 0
    # ## --- Tính log-return & winsorize giá (thay khối ret) ---
    # p_low, p_high = np.percentile(raw_np[:,0], [1,99])
    # raw_np[:,0]   = np.clip(raw_np[:,0], p_low, p_high)
    T, Fdim = raw_np.shape
    assert Fdim >= 2, 'Need price + at least 1 feature'
    
    # ---------- chrono split indices ----------
    train_len = int(0.80 * (T - window))
    val_len   = int(0.05 * (T - window))
    
    # prepare training feature matrix for fitting (exclude price col 0)
    train_feat_matrix = []
    for i in range(train_len):
        train_feat_matrix.append(raw_np[i:i+window, 1:])
    train_feat_matrix = np.concatenate(train_feat_matrix, axis=0)  # (train_len*L, N)
    scaler = MinMax01(); 
    scaler.fit(train_feat_matrix)
    
    # ---------- build samples (scaled features) ----------
    X_list, Y_list = [], []
    for i in range(T - window):
        price_prev = raw_np[i+window-1, 0]
        price_cur  = raw_np[i+window,   0]
        if price_prev == 0:                 # avoid div‑zero
            continue
        ret = (price_cur - price_prev) / price_prev
        # ret = math.log(price_cur / price_prev)
        if not np.isfinite(ret):
            continue
        feat_block = scaler.transform(raw_np[i:i+window, 1:])
        X_list.append(torch.tensor(feat_block, dtype=torch.float32))
        Y_list.append(torch.tensor([[ret]], dtype=torch.float32))
    
    XX = torch.stack(X_list)  # (S, L, N)
    YY = torch.stack(Y_list)  # (S, 1, 1)
    
    # recompute lengths after possible filtering
    num_samples = XX.shape[0]
    train_len = int(0.80 * num_samples)
    val_len   = int(0.05 * num_samples)
    
    test_len  = num_samples - train_len - val_len
    X_train, Y_train = XX[:train_len], YY[:train_len]
    X_val,   Y_val   = XX[train_len:train_len+val_len], YY[train_len:train_len+val_len]
    X_test,  Y_test  = XX[-test_len:], YY[-test_len:]
    
    train_loader = make_loader(X_train, Y_train, batch_size)
    val_loader   = make_loader(X_val,   Y_val, batch_size)
    test_loader  = make_loader(X_test,  Y_test, batch_size)

    return XX.shape[2], train_loader, val_loader, test_loader


def main(dataset):
    # >>> TRAIN MAMBA_BayesMAGAC - IXIC <<<  
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
        'log_dir': f'{dataset}_log' + ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") ,
        'model_name': f'{dataset}_v3',
        'log_step': 20,
    }
    print(f'data_path: {data_path}')
    print(f'Args {args}, N: {N}, L: {window}, batch_size: {batch_size}, Model_Args: {m_args}')

    print(" TRAIN MAMBA_BayesMAGAC")
    model_v3  = MAMBA_BayesMAGAC(m_args, R=3, K=3, d_e=10, mc_train=3, mc_eval=10, drop_edge_p=0.1, mc_dropout_p=0.2)
    for p in model_v3.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    loss_fn   = nn.GaussianNLLLoss(full=True, reduction='mean') #nn.MSELoss()
    optimizer = torch.optim.Adam(model_v3.parameters(), lr=1e-3, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[750, 1050, 1350], gamma=0.1)

    trainer_v3 = Trainer(model_v3, loss_fn, optimizer, train_loader, val_loader, test_loader,
                    args=args, lr_scheduler=scheduler)
    trainer_v3.train()
    trainer_v3.test()
    plot_analytics(trainer_v3.args['log_dir'])

if __name__ == "__main__":
    main('IXIC')
    main('DJI')
    main('NYSE')
