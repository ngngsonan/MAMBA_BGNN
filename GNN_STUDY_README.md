# MAMBA-GNN Ablation Study

## Overview

This ablation study compares different Graph Neural Network (GNN) layers combined with BIMamba (Bidirectional Mamba) for stock return prediction. After establishing BIMamba as the optimal temporal encoder (from `mamba_ablation_study.py`), we now investigate which graph convolution layer provides the best spatial aggregation.

## Models Compared

| Model | Description | Key Features |
|-------|-------------|--------------|
| **BIMamba + GCN** | Graph Convolutional Network | Spectral graph convolution with symmetric normalization |
| **BIMamba + GAT** | Graph Attention Network | Multi-head attention mechanism for adaptive edge weights |
| **BIMamba + GraphSAGE** | GraphSAGE | Neighborhood sampling with mean aggregation |
| **BIMamba + MAGAC** | Multi-head Adaptive GAC | Gaussian kernel + attention + Chebyshev polynomial (baseline) |

## Architecture

All models follow the same structure:

```
Input (B, L, N)
    ↓
BIMamba Encoder (temporal)
    → Forward Mamba
    → Backward Mamba
    → FFN + Residual
    ↓
Transpose (B, L, N) → (B, N, L)
    ↓
GNN Layer (spatial)
    → [GCN / GAT / GraphSAGE / MAGAC]
    ↓
Linear Head
    ↓
Output: (mean, log_var)
```

## Usage

### Quick Start

```bash
# Train all models on IXIC dataset
python run_gnn_study.py --dataset IXIC --epochs 50

# Train specific models
python run_gnn_study.py --dataset DJI --models GCN GAT MAGAC --epochs 100

# Run on all datasets
python run_gnn_study.py --all-datasets --epochs 50
```

### Advanced Options

```bash
python run_gnn_study.py \
    --dataset IXIC \
    --models GCN GAT GraphSAGE MAGAC \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --window 5 \
    --hidden-dim 64 \
    --R 3 \
    --K 3 \
    --d-e 10 \
    --heads 4 \
    --loss-type auto \
    --patience 15 \
    --device cuda
```

### Parameters

**Dataset Options:**
- `--dataset`: Choose from `IXIC`, `DJI`, `NYSE` (default: `IXIC`)
- `--all-datasets`: Run on all datasets sequentially

**Model Selection:**
- `--models`: Select models to train (default: all 4 models)
  - Options: `GCN`, `GAT`, `GraphSAGE`, `MAGAC`

**Training Hyperparameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--window`: Lookback window size (default: 5)
- `--hidden-dim`: Hidden dimension (default: 64)

**Model Architecture:**
- `--R`: Number of BIMamba layers (default: 3)
- `--K`: Chebyshev polynomial order for MAGAC (default: 3)
- `--d-e`: Node embedding dimension (default: 10)
- `--heads`: Number of attention heads (default: 4)

**Training Options:**
- `--loss-type`: Loss function (`auto`, `nll`, `mse`, `mae`, `huber`)
- `--patience`: Early stopping patience (default: 10)

**Logging:**
- `--log-dir`: Base directory for logs (default: `logs`)
- `--quiet`: Reduce output verbosity

**Device:**
- `--device`: Training device (`auto`, `cpu`, `cuda`)

## Programmatic Usage

You can also use the study programmatically:

```python
import torch
from models.mamba_gnn_study import train_gnn_comparison

device = 'cuda' if torch.cuda.is_available() else 'cpu'

results = train_gnn_comparison(
    dataset='IXIC',
    models=['GCN', 'GAT', 'GraphSAGE', 'MAGAC'],
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    hidden_dim=64,
    window=5,
    R=3,              # BIMamba layers
    K=3,              # Chebyshev order
    d_e=10,           # Node embedding dim
    heads=4,          # Attention heads
    loss_type='auto',
    early_stop_patience=10,
    verbose=True,
    device=device
)

# Access results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  IC:  {metrics['ic']:.4f}")
    print(f"  RIC: {metrics['ric']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
```

## Output Metrics

The study reports the following metrics for each model:

- **IC (Information Coefficient)**: Correlation between predictions and actual returns
- **RIC (Rank IC)**: Spearman rank correlation
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **Train/Val/Test Loss**: Loss curves during training

## Results Location

Results are saved to:
```
logs/mamba_gnn/
├── {dataset}/
│   ├── BIMamba+GCN/
│   │   ├── model.pt
│   │   ├── metrics.json
│   │   └── training_curve.png
│   ├── BIMamba+GAT/
│   ├── BIMamba+GraphSAGE/
│   └── BIMamba+MAGAC/
└── comparison_table.csv
```

## Comparison Analysis

The study automatically computes improvement percentages:

```
GNN LAYER COMPARISON
================================================================================
Baseline (MAGAC):  IC=0.0523, RIC=0.0489
--------------------------------------------------------------------------------
GCN         : IC=0.0445 (-14.91%), RIC=0.0421 (-13.91%)
GAT         : IC=0.0501 (-4.21%), RIC=0.0478 (-2.25%)
GraphSAGE   : IC=0.0467 (-10.71%), RIC=0.0443 (-9.41%)
================================================================================
```

## GNN Layer Details

### 1. GCN (Graph Convolutional Network)
- **Aggregation**: Spectral convolution with symmetric normalization
- **Adjacency**: Learned from node embeddings via cosine similarity
- **Normalization**: `D^{-1/2} A D^{-1/2}` (symmetric)
- **Pros**: Simple, effective, well-studied
- **Cons**: Assumes homophily, limited expressiveness

### 2. GAT (Graph Attention Network)
- **Aggregation**: Multi-head attention mechanism
- **Adjacency**: Attention scores computed dynamically
- **Attention**: `α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))`
- **Pros**: Adaptive edge weights, handles heterophily
- **Cons**: Higher computational cost, more parameters

### 3. GraphSAGE
- **Aggregation**: Mean pooling of neighbors
- **Adjacency**: Top-k neighbors from learned embeddings
- **Combination**: Mean of self and neighbor features
- **Pros**: Inductive learning, scalable
- **Cons**: Fixed neighborhood, less adaptive

### 4. MAGAC (Multi-head Adaptive GAC)
- **Aggregation**: Chebyshev polynomial convolution
- **Adjacency**: Blend of Gaussian kernel + multi-head attention
- **Formula**: `α·A_gauss + (1-α)·A_attn`
- **Polynomial**: K-hop aggregation with Chebyshev basis
- **Pros**: Combines multiple inductive biases, highly adaptive
- **Cons**: Most complex, more hyperparameters

## Research Questions

This ablation study addresses:

1. **Does MAGAC outperform standard GNN layers?**
   - Compare MAGAC vs GCN/GAT/GraphSAGE on IC/RIC metrics

2. **What makes MAGAC effective?**
   - Multi-head attention vs single attention (GAT)
   - Gaussian kernel + attention blend vs pure attention
   - Chebyshev polynomials vs 1-hop convolution (GCN)

3. **Is the complexity justified?**
   - Cost-benefit analysis: performance gain vs parameter count
   - Ablate components of MAGAC to identify key contributions

## Expected Outcomes

Based on preliminary results, we expect:

- **MAGAC > GAT > GraphSAGE > GCN** (in terms of IC)
- MAGAC's advantage comes from:
  - Multi-scale aggregation (Chebyshev polynomials)
  - Adaptive adjacency (Gaussian + attention blend)
  - Multi-head architecture
- Trade-off between expressiveness and overfitting risk

## Next Steps

After identifying the best GNN layer:

1. **Bayesian Extension**: Add uncertainty quantification (MC-Dropout, DropEdge)
2. **Hyperparameter Tuning**: Optimize K, d_e, heads for best GNN
3. **Ensemble Methods**: Combine multiple GNN architectures
4. **Interpretability**: Analyze learned graph structures

## References

1. **GCN**: Kipf & Welling (2017) - *Semi-Supervised Classification with Graph Convolutional Networks*
2. **GAT**: Veličković et al. (2018) - *Graph Attention Networks*
3. **GraphSAGE**: Hamilton et al. (2017) - *Inductive Representation Learning on Large Graphs*
4. **Mamba**: Gu & Dao (2023) - *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*

## Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size or hidden dimension
python run_gnn_study.py --batch-size 16 --hidden-dim 32
```

**Slow Training:**
```bash
# Train fewer epochs or use CPU for small datasets
python run_gnn_study.py --epochs 30 --device cpu
```

**NaN Loss:**
```bash
# Reduce learning rate or use gradient clipping
python run_gnn_study.py --lr 0.0001
```

## License

This code is part of the MAMBA-BGNN research project.
