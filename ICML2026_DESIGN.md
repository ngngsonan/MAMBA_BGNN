# ICML 2026 Regime-Aware Extension

This document summarises the new components introduced for the ICML 2026 preparation workstream.

## 1. Architecture Overview

```
Input sequences (B, L, N)
        │
        ▼
HierarchicalRegimeController ──► feature_gate (B, N)
        │                         KL priors (global, sector, asset)
        ▼
Regime-aware gating applied to temporal input
        │
        ▼
BIMambaBlock (temporal SSM backbone)
        │
        ▼
RegimeAwareBayesianMAGAC + DiffusionGraphPrior
        │
        ▼
RiskAwareDecoder (mean/logvar/CVaR heads)
```

- **HierarchicalRegimeController** (`hierarchical_mamba_bgnn.py`): learns global, sector, and asset latent regimes via Gumbel-Softmax. Produces per-feature gates and KL terms used in the loss.
- **DiffusionGraphPrior**: refines MAGAC adjacency matrices with regime-conditioned diffusion steps, interpolating between the Gaussian kernel and regime-induced outer products.
- **RiskAwareDecoder**: predicts mean/log-variance alongside a CVaR multiplier used to penalise downside risk.

## 2. Training Objective

Implemented in `risk_aware_training.py`:

- Base Gaussian NLL on return predictions.
- KL regularisation over the three regime hierarchies with tunable `kl_weight`.
- CVaR-style downside penalty encouraging the predicted tail threshold to sit below observed losses (`risk_weight`).
- Conformal calibration on the best validation checkpoint provides finite-sample coverage guarantees.

Objects of interest:

- `RiskAwareLossConfig`: hyperparameter container for KL and risk weights.
- `RiskAwareLoss`: returns scalar loss plus logging components for dashboards.
- `RiskAwareTrainer`: end-to-end training loop with CUDA support, CSV logging, checkpointing, and risk-aware evaluation.

## 3. Running the Pipeline

```bash
conda run -n py310 python run_icml2026_pipeline.py \
  --dataset IXIC \
  --epochs 150 \
  --lr 3e-4 \
  --kl_weight 5e-4 \
  --risk_weight 0.05 \
  --conformal_alpha 0.1
```

Outputs are written to `<DATASET>_icml2026_log/`:

- `best_model_icml2026.pth`: checkpoint.
- `val_metrics_icml2026.csv`, `test_metrics_icml2026.csv`: scalar metrics including conformal coverage.
- Logs capturing loss components and calibration diagnostics.

## 4. Extending the Work

- **Regime granularity**: adjust `regime_global_states`, `regime_sector_states`, and `regime_asset_states` when instantiating `HierarchicalMambaBGNN` for different market universes.
- **Diffusion depth**: `diffusion_steps` controls how aggressively the adjacency is denoised; increasing the value encourages smoother structural priors.
- **Risk preferences**: experiment with `risk_weight` and `cvar_level` to target alternative tail probabilities; the calibrator automatically adapts interval widths.
- **Ablations**: disable individual components by freezing gates (set `feature_gate` to ones), bypassing diffusion (`diffusion_steps=0`), or dropping the CVaR term (`risk_weight=0`).

## 5. Next Steps Toward Publication

- Empirically validate on cross-market datasets (equities, rates, commodities) to demonstrate robustness.
- Compare against hierarchical VAEs and state-space transformers with risk-aware objectives.
- Derive theoretical guarantees on identifiability of the latent hierarchy and convergence of the diffusion prior.
- Package results into appendices highlighting uncertainty calibration, tail-risk control, and interpretability of learned regimes.

This document should be updated alongside future experiments to keep the ICML 2026 track reproducible and transparent.
