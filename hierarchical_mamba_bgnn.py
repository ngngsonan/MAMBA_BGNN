"""Regime-aware extension of MAMBA-BGNN for ICML 2026 preparation.
Implements hierarchical latent regime controller, diffusion-driven graph prior,
and a risk-aware decoder that interfaces with the existing trainer stack.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_bgnn import (
    ModelArgs,
    BIMambaBlock,
    BayesianMAGAC,
)


def _kl_categorical(posterior_logits: torch.Tensor,
                    prior_logits: torch.Tensor,
                    dim: int = -1) -> torch.Tensor:
    """Compute KL divergence between categorical distributions given logits."""
    log_p = F.log_softmax(posterior_logits, dim=dim)
    log_q = F.log_softmax(prior_logits, dim=dim)
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=dim)


class HierarchicalRegimeController(nn.Module):
    """Hierarchical latent regime controller using Gumbel-Softmax sampling."""

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        global_states: int = 4,
        sector_states: int = 6,
        asset_states: int = 4,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = max(64, d_model)
        self.temperature = temperature
        self.seq_len = seq_len
        self.global_states = global_states
        self.sector_states = sector_states
        self.asset_states = asset_states

        self.global_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, global_states),
        )
        self.sector_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model * sector_states),
        )
        self.asset_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model * asset_states),
        )

        self.global_prior = nn.Parameter(torch.zeros(global_states))
        self.sector_prior = nn.Parameter(torch.zeros(sector_states))
        self.asset_prior = nn.Parameter(torch.zeros(asset_states))

        # Feature-gating projections from regime probabilities
        self.global_to_feature = nn.Linear(global_states, d_model, bias=False)
        self.sector_embeddings = nn.Parameter(torch.randn(sector_states, d_model))
        self.asset_embeddings = nn.Parameter(torch.randn(asset_states, d_model))

    def forward(
        self,
        x: torch.Tensor,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Infer hierarchical regimes and produce feature-level gating."""
        # x: (B, L, N)
        batch_size, _, num_features = x.shape
        seq_mean = x.mean(dim=1)  # (B, N)
        temporal_diff = x[:, 1:, :] - x[:, :-1, :]
        diff_summary = temporal_diff.mean(dim=1)

        global_logits = self.global_net(seq_mean)
        sector_logits = self.sector_net(seq_mean).view(batch_size, num_features, self.sector_states)
        asset_logits = self.asset_net(diff_summary).view(batch_size, num_features, self.asset_states)

        global_probs = F.gumbel_softmax(global_logits, tau=self.temperature, hard=hard, dim=-1)
        sector_probs = F.gumbel_softmax(sector_logits, tau=self.temperature, hard=hard, dim=-1)
        asset_probs = F.gumbel_softmax(asset_logits, tau=self.temperature, hard=hard, dim=-1)

        # Compute KL divergences for variational objective
        kl_global = _kl_categorical(global_logits, self.global_prior, dim=-1).mean()
        kl_sector = _kl_categorical(sector_logits, self.sector_prior, dim=-1).mean()
        kl_asset = _kl_categorical(asset_logits, self.asset_prior, dim=-1).mean()

        global_gate = self.global_to_feature(global_probs)
        sector_gate_matrix = torch.einsum('bfs,sd->bfd', sector_probs, self.sector_embeddings)
        sector_gate = sector_gate_matrix.diagonal(dim1=1, dim2=2)
        asset_gate_matrix = torch.einsum('bfs,sd->bfd', asset_probs, self.asset_embeddings)
        asset_gate = asset_gate_matrix.diagonal(dim1=1, dim2=2)

        combined_gate = torch.sigmoid(global_gate + sector_gate + asset_gate)

        return {
            'global_probs': global_probs,
            'sector_probs': sector_probs,
            'asset_probs': asset_probs,
            'feature_gate': combined_gate,  # (B, N)
            'kl_global': kl_global,
            'kl_sector': kl_sector,
            'kl_asset': kl_asset,
        }


class DiffusionGraphPrior(nn.Module):
    """Diffusion-inspired refinement of adjacency matrices guided by regimes."""

    def __init__(
        self,
        num_nodes: int,
        heads: int,
        steps: int = 3,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.heads = heads
        self.steps = steps
        self.step_logits = nn.Parameter(torch.linspace(-1.0, 0.5, steps))
        self.noise_scale = nn.Parameter(torch.full((steps,), 0.05))

    def forward(
        self,
        adjacency: torch.Tensor,
        regime_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Refine adjacency using outer products of regime-aware features."""
        # adjacency: (H, N, N)
        # regime_feature: (B, N)
        context = regime_feature.mean(dim=0)  # (N,)
        context = context / (context.norm(p=2) + 1e-6)
        outer = torch.ger(context, context)
        outer = outer / outer.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        refined = []
        for h in range(self.heads):
            A = adjacency[h]
            for step in range(self.steps):
                alpha = torch.sigmoid(self.step_logits[step])
                noise = self.noise_scale[step] * torch.randn_like(A)
                A = alpha * A + (1 - alpha) * outer + noise
                A = 0.5 * (A + A.transpose(-1, -2))
                A = F.softmax(A, dim=-1)
            refined.append(A)
        return torch.stack(refined, dim=0)


class RegimeAwareBayesianMAGAC(BayesianMAGAC):
    """MAGAC extension using diffusion prior conditioned on regimes."""

    def __init__(self, *args, diffusion_steps: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diffusion_prior = DiffusionGraphPrior(
            num_nodes=args[0],
            heads=self.H,
            steps=diffusion_steps,
        )
        self._regime_feature: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        regime_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._regime_feature = regime_feature
        return super().forward(x)

    def _sample_A_eff(self, use_dropout_on_psi: bool):  # type: ignore[override]
        base = super()._sample_A_eff(use_dropout_on_psi)
        if self._regime_feature is None:
            return base
        return self.diffusion_prior(base, self._regime_feature)


class RiskAwareDecoder(nn.Module):
    """Decoder that predicts returns and tail risk parameters."""

    def __init__(self, input_dim: int, cvar_level: float = 0.9) -> None:
        super().__init__()
        hidden = max(64, input_dim // 2)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)
        self.cvar_head = nn.Linear(hidden, 1)
        self.cvar_level = cvar_level

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.shared(z)
        mean = self.mean_head(features).squeeze(-1)
        log_var = self.logvar_head(features).squeeze(-1)
        cvar_logits = self.cvar_head(features).squeeze(-1)
        cvar = torch.sigmoid(cvar_logits)
        return {
            'mean': mean,
            'log_var': log_var,
            'cvar_multiplier': cvar,
        }

    def compute_cvar(self, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        standard_normal = torch.distributions.Normal(
            torch.zeros(1, device=mean.device, dtype=mean.dtype),
            torch.ones(1, device=mean.device, dtype=mean.dtype),
        )
        z_alpha = standard_normal.icdf(torch.tensor(
            [self.cvar_level], device=mean.device, dtype=mean.dtype
        ))
        pdf = torch.exp(-0.5 * z_alpha**2) / math.sqrt(2 * math.pi)
        tail = mean - sigma * (pdf / (1 - self.cvar_level))
        return tail.squeeze(0)


class HierarchicalMambaBGNN(nn.Module):
    """Full ICML 2026 candidate architecture."""

    def __init__(
        self,
        args: ModelArgs,
        R: int = 3,
        K: int = 3,
        d_e: int = 10,
        heads: int = 4,
        mc_train: int = 4,
        mc_eval: int = 16,
        regime_global_states: int = 4,
        regime_sector_states: int = 6,
        regime_asset_states: int = 4,
        regime_temperature: float = 0.5,
        diffusion_steps: int = 3,
        cvar_level: float = 0.9,
    ) -> None:
        super().__init__()
        self.regime_controller = HierarchicalRegimeController(
            args.d_model,
            args.seq_len,
            global_states=regime_global_states,
            sector_states=regime_sector_states,
            asset_states=regime_asset_states,
            temperature=regime_temperature,
        )
        self.temporal_model = BIMambaBlock(args, R=R)
        self.graph_model = RegimeAwareBayesianMAGAC(
            args.d_model,
            args.seq_len,
            K,
            d_e,
            heads=heads,
            mc_train=mc_train,
            mc_eval=mc_eval,
            diffusion_steps=diffusion_steps,
        )
        self.decoder = RiskAwareDecoder(args.d_model, cvar_level=cvar_level)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        regimes = self.regime_controller(x)
        gated_x = x * regimes['feature_gate'].unsqueeze(1)
        temporal = self.temporal_model(gated_x)
        node_major = temporal.transpose(1, 2).contiguous()
        graph_mean, graph_logvar = self.graph_model(node_major, regimes['feature_gate'])

        decoder_outputs = self.decoder(graph_mean)
        mean = decoder_outputs['mean']
        var = decoder_outputs['log_var'].exp() + torch.exp(graph_logvar).sum(dim=1)
        log_var = var.log()

        aux = {
            'regimes': regimes,
            'decoder': decoder_outputs,
            'graph_log_var': graph_logvar,
        }
        return mean, log_var, aux
