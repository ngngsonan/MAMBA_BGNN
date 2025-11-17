# Baseline Models

This folder contains all baseline models and related utilities for training and comparison.

## 📁 Contents

- **baseline_models.py** - Implementation of 5 baseline models (Linear, LSTM, Transformer, AGCRN, TemporalGN)
- **train_all_baselines.py** - Script to train all baseline models
- **compare_baselines.py** - Script to compare results across all models
- **test_baseline_setup.py** - Verification script to test setup
- **BASELINE_TRAINING_GUIDE.md** - Comprehensive usage guide (Vietnamese)
- **__init__.py** - Package initialization

## 🚀 Quick Start

**Important**: Run all commands from the root `MAMBA_BGNN/` directory, not from inside the `baseline/` folder.

### Train all baseline models:

```bash
python baseline/train_all_baselines.py --dataset IXIC --epochs 50
```

### Compare results:

```bash
python baseline/compare_baselines.py --dataset IXIC
```

### Test setup:

```bash
python baseline/test_baseline_setup.py
```

## 📖 Documentation

See [BASELINE_TRAINING_GUIDE.md](BASELINE_TRAINING_GUIDE.md) for detailed documentation (in Vietnamese).

## 🏗️ Models

All models are probabilistic, returning both mean and log-variance predictions:

1. **LinearBaseline** - Simple feedforward baseline
2. **LSTMBaseline** - LSTM-based temporal model
3. **TransformerBaseline** - Transformer encoder for sequences
4. **AGCRNBaseline** - Adaptive Graph Convolution RNN
5. **TemporalGNBaseline** - Temporal Graph Network

## 📊 Results

Results are saved to `logs/baselines/` with comprehensive metrics including:
- Probabilistic metrics (NLL, RMSE, MAE, IC, RIC, CRPS, etc.)
- Financial metrics (Sharpe ratio, max drawdown, directional accuracy, etc.)
- Advanced analysis (market regime analysis, stress testing, rolling window evaluation)

## 🔧 Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn

See `../requirements.txt` for complete list.

## 📝 Usage Example

```bash
# From MAMBA_BGNN/ directory

# 1. Train all models
python baseline/train_all_baselines.py \
    --dataset IXIC \
    --epochs 50 \
    --early_stop \
    --early_stop_patience 10

# 2. Compare results
python baseline/compare_baselines.py --dataset IXIC

# 3. View report
cat logs/baselines/comparison/comparison_report.txt
```

## 🔗 Related Files

- `../utils/data_processing.py` - Data loading and preprocessing
- `../utils/trainer.py` - Training and evaluation framework
- `../Dataset/` - Raw data files

---

For detailed instructions, see [BASELINE_TRAINING_GUIDE.md](BASELINE_TRAINING_GUIDE.md).
