# Baseline Models

This folder contains baseline models for comparison with the main MAMBA_BGNN model.

## 📁 Files

- **baseline_notebook.py** - Complete baseline training script integrated with main project pipeline
- **README.md** - This file

## 🎯 Overview

The baseline models are trained using the **same data processing and evaluation pipeline** as the main MAMBA_BGNN model to ensure fair comparison:

- ✅ Same train/val/test splits (80/5/15)
- ✅ Out-of-time testing (no data leakage)
- ✅ Same metrics (RMSE, MAE, IC, RIC, Sharpe, Max DD, etc.)
- ✅ Same data normalization (MinMax scaler fit only on training data)
- ✅ Comprehensive logging to `logs/baselines/`

## 🏗️ Available Models

All 5 baseline models return probabilistic predictions (mean, log_variance):

1. **Linear** - Simple feedforward neural network
2. **LSTM** - Long Short-Term Memory network
3. **Transformer** - Transformer encoder with positional encoding
4. **AGCRN** - Adaptive Graph Convolution RNN
5. **TemporalGN** - Temporal Graph Network with attention

## 🚀 Quick Start

### From Python Script

```python
from baseline.baseline_notebook import train_all_baselines

# Train all baseline models
results = train_all_baselines(
    dataset='IXIC',
    models=['Linear', 'LSTM', 'Transformer', 'AGCRN', 'TemporalGN'],
    epochs=50,
    loss_type='auto',
    verbose=True
)

# Results contain metrics for each model
print(results['Linear']['rmse'])
print(results['LSTM']['ic'])
```

### From Jupyter Notebook

```python
import sys
sys.path.append('../')  # Add MAMBA_BGNN to path

from baseline.baseline_notebook import train_all_baselines

results = train_all_baselines(
    dataset='IXIC',
    models=['Linear', 'LSTM'],  # Train subset
    epochs=20,
    loss_type='auto'
)
```

### From Command Line

```bash
cd MAMBA_BGNN
python baseline/baseline_notebook.py
```

## ⚙️ Parameters

```python
train_all_baselines(
    dataset='IXIC',              # Dataset: IXIC, DJI, NYSE
    models=None,                 # List of models (None = all 5)
    window=5,                    # Lookback window
    batch_size=32,               # Batch size
    epochs=50,                   # Training epochs
    learning_rate=0.001,         # Learning rate
    hidden_dim=64,               # Hidden dimension
    loss_type='auto',            # Loss: auto, nll, mse, mae, huber
    early_stop_patience=10,      # Early stopping patience
    verbose=True,                # Print progress
    log_base_dir='logs/baselines' # Log directory
)
```

### Loss Function Selection

- **'auto'** (recommended) - Automatically selects optimal loss for each model:
  - Linear/AGCRN/TemporalGN → MSE (simple, stable)
  - LSTM/Transformer → Huber (robust to outliers)
- **'nll'** - Gaussian Negative Log-Likelihood (probabilistic)
- **'mse'** - Mean Squared Error
- **'mae'** - Mean Absolute Error
- **'huber'** - Huber/Smooth L1 Loss

## 📊 Output & Logs

After training, results are saved to `logs/baselines/`:

```
logs/baselines/
├── IXIC_Linear_20231117_183045/
│   ├── best_model.pth
│   ├── val_metrics.csv
│   └── test_metrics.csv
├── IXIC_LSTM_20231117_183045/
│   ├── best_model.pth
│   ├── val_metrics.csv
│   └── test_metrics.csv
├── ...
└── IXIC_summary_20231117_183045/
    ├── baseline_comparison.csv
    └── baseline_summary.txt
```

### Summary Files

**baseline_comparison.csv** - Comparison table of all models:
```csv
Model,RMSE,MAE,IC,RIC,Dir Acc,Sharpe,Max DD
Linear,0.012345,0.009876,0.234,0.198,0.567,0.89,-0.12
LSTM,0.011234,0.008765,0.345,0.289,0.623,1.23,-0.09
...
```

**baseline_summary.txt** - Detailed summary report:
```
================================================================================
BASELINE MODELS COMPARISON SUMMARY
================================================================================
Dataset: IXIC
Timestamp: 20231117_183045
Models trained: Linear, LSTM, Transformer, AGCRN, TemporalGN
================================================================================

METRICS COMPARISON:
--------------------------------------------------------------------------------
   Model      RMSE     MAE      IC     RIC  Dir Acc  Sharpe  Max DD
  Linear  0.012345  0.009876  0.234  0.198   0.567    0.89   -0.12
    LSTM  0.011234  0.008765  0.345  0.289   0.623    1.23   -0.09
...

BEST MODELS BY METRIC:
--------------------------------------------------------------------------------
RMSE        : LSTM            (0.011234)
MAE         : LSTM            (0.008765)
IC          : Transformer     (0.378000)
RIC         : Transformer     (0.312000)
DIR_ACC     : LSTM            (0.623000)
SHARPE      : AGCRN           (1.450000)
```

## 📈 Metrics

Each model is evaluated on comprehensive metrics:

### Basic Metrics
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **IC** - Information Coefficient (Pearson correlation)
- **RIC** - Rank Information Coefficient (Spearman correlation)

### Financial Metrics
- **Dir Acc** - Directional Accuracy
- **Sharpe** - Sharpe Ratio (annualized)
- **Max DD** - Maximum Drawdown
- **Calmar** - Calmar Ratio
- **Hit Rate** - Trading hit rate

### Probabilistic Metrics
- **NLL** - Negative Log-Likelihood
- **CRPS** - Continuous Ranked Probability Score
- **PICP90** - 90% Prediction Interval Coverage Probability
- **Gap90** - Coverage gap from nominal 90%

## 🔧 Integration with Main Project

This baseline training script uses:

- **utils.data_processing** - Same data loading and preprocessing
- **Same train/val/test splits** - 80/5/15 chronological split
- **Same metrics** - Compatible with trainer.py metrics
- **Same evaluation protocol** - Out-of-time testing

This ensures fair comparison between baseline models and MAMBA_BGNN.

## 💡 Tips

1. **Quick testing**: Use fewer epochs (10-20) and subset of models for testing
2. **Full comparison**: Train all 5 models with 50+ epochs for publication
3. **Custom models**: Add your own model class following the BaselineModel interface
4. **GPU**: Models automatically use GPU if available (via PyTorch default)

## 📝 Example Workflows

### Quick Test
```python
# Test with 2 models, 10 epochs
results = train_all_baselines(
    dataset='IXIC',
    models=['Linear', 'LSTM'],
    epochs=10,
    verbose=True
)
```

### Full Comparison
```python
# Train all models for paper
results = train_all_baselines(
    dataset='IXIC',
    models=['Linear', 'LSTM', 'Transformer', 'AGCRN', 'TemporalGN'],
    epochs=100,
    learning_rate=0.0005,
    early_stop_patience=15,
    verbose=True
)
```

### Custom Loss
```python
# Use specific loss for all models
results = train_all_baselines(
    dataset='DJI',
    models=['LSTM', 'Transformer'],
    loss_type='huber',  # Huber loss for all
    epochs=50
)
```

## 🔗 Related Files

- **utils/data_processing.py** - Data loading pipeline
- **utils/trainer.py** - Main training framework
- **Dataset/** - Raw data files

---

For questions or issues, please check the code documentation in [baseline_notebook.py](baseline_notebook.py).
