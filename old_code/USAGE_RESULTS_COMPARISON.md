# Results Comparison and Summary Report Usage Guide

This guide explains how to use the new summary report and results comparison features added to the MAMBA_BGNN trainer.

## Overview

The trainer now automatically:
1. Tracks training time (per epoch and total)
2. Generates a comprehensive summary report after testing
3. Provides utilities to compare results from multiple model runs

## Features Added

### 1. Training Time Tracking

The `Trainer` class now tracks:
- Time for each epoch
- Total training time
- Average epoch time

These metrics are logged and included in the summary report.

### 2. Automatic Summary Report Generation

After calling `trainer.test()`, a summary report is automatically generated with:
- Training configuration
- Training time statistics
- Test metrics (all available metrics based on loss type)
- Rolling window evaluation results (if available)
- Output file paths

**Output Files:**
- `training_summary.txt` - Comprehensive text report
- All existing CSV files (test_metrics.csv, val_metrics.csv, etc.)

### 3. Results Comparison Utilities

The new `utils/results_comparison.py` module provides functions to:
- Read test metrics from multiple runs
- Compare models side-by-side
- Generate comparison reports
- Auto-discover log directories

---

## Usage Examples

### Example 1: Basic Training with Automatic Summary

```python
from utils.trainer import Trainer
from utils.data_processing import data_processing

# Load data
num_features, train_loader, val_loader, test_loader = data_processing(
    'Dataset/combined_dataframe_IXIC.csv',
    window=5,
    batch_size=32
)

# Setup model, loss, optimizer (your existing code)
# ...

# Create trainer
args = {
    'log_dir': 'logs/IXIC_MAMBA_BGNN_run1',
    'model_name': 'MAMBA_BGNN',
    'dataset': 'IXIC',
    'epochs': 100,
    'early_stop': True,
    'early_stop_patience': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_step': 50,
    'grad_norm': True,
    'max_grad_norm': 5.0,
}

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    args=args,
    loss_type='bayesian'
)

# Train and test
trainer.train()
test_metrics = trainer.test()  # Summary report is automatically generated!

# The summary report is saved at: logs/IXIC_MAMBA_BGNN_run1/training_summary.txt
```

### Example 2: Manual Summary Report Generation

If you want to regenerate the summary report after testing:

```python
# After training and testing
trainer.generate_summary_report()
```

### Example 3: Compare Multiple Model Runs

```python
from utils.results_comparison import generate_comparison_report

# Specify log directories to compare
log_dirs = [
    'logs/IXIC_MAMBA_BGNN_run1',
    'logs/baselines/IXIC_Linear_20250101_120000',
    'logs/baselines/IXIC_LSTM_20250101_120000',
    'logs/baselines/IXIC_Transformer_20250101_120000',
]

model_names = ['MAMBA_BGNN', 'Linear', 'LSTM', 'Transformer']

# Generate comparison report
comparison_df, report_path = generate_comparison_report(
    log_dirs=log_dirs,
    model_names=model_names,
    output_dir='logs/comparison',
    dataset_name='IXIC'
)

# Output:
# - logs/comparison/model_comparison.csv (comparison table)
# - logs/comparison/comparison_summary.txt (detailed report)
```

### Example 4: Auto-Find and Compare All Runs

```python
from utils.results_comparison import find_model_logs, generate_comparison_report

# Auto-find all log directories for IXIC dataset
log_dirs = find_model_logs(base_dir='logs', dataset='IXIC')

print(f"Found {len(log_dirs)} runs:")
for log_dir in log_dirs:
    print(f"  - {log_dir}")

# Compare all found runs
if log_dirs:
    comparison_df, report_path = generate_comparison_report(
        log_dirs=log_dirs,
        output_dir='logs/IXIC_all_models_comparison',
        dataset_name='IXIC'
    )
```

### Example 5: Read Single Run Metrics

```python
from utils.results_comparison import read_test_metrics

# Read metrics from a single run
result = read_test_metrics('logs/IXIC_MAMBA_BGNN_run1')

print(f"Model: {result['model_name']}")
print(f"Test RMSE: {result['test_metrics']['rmse']:.6f}")
print(f"Test IC: {result['test_metrics']['ic']:.6f}")
print(f"Best Epoch: {result['validation_info']['best_epoch']}")
```

### Example 6: Compare with Baselines

```python
from baseline.baseline_notebook import train_all_baselines
from utils.results_comparison import generate_comparison_report

# 1. Train all baselines
baseline_results = train_all_baselines(
    dataset='IXIC',
    models=['Linear', 'LSTM', 'Transformer', 'AGCRN', 'TemporalGN'],
    epochs=50,
    loss_type='auto'
)

# 2. Train MAMBA_BGNN (your existing code)
# trainer.train()
# trainer.test()

# 3. Find all log directories
from utils.results_comparison import find_model_logs

baseline_logs = find_model_logs(base_dir='logs/baselines', dataset='IXIC')
mamba_logs = find_model_logs(base_dir='logs', dataset='IXIC', pattern='MAMBA')

all_logs = baseline_logs + mamba_logs

# 4. Generate comprehensive comparison
comparison_df, report_path = generate_comparison_report(
    log_dirs=all_logs,
    output_dir='logs/final_comparison',
    dataset_name='IXIC - All Models'
)

print("\nFinal comparison saved to:", report_path)
```

---

## Output Structure

After training with the updated trainer, your log directory will contain:

```
logs/IXIC_MAMBA_BGNN_run1/
├── best_model.pth                      # Best model checkpoint
├── val_metrics.csv                     # Validation metrics per epoch
├── test_metrics.csv                    # Test metrics
├── test_predictions.csv                # Test predictions (mu, sigma, y)
├── val_predictions_best.csv            # Best validation predictions
├── rolling_window_results.csv          # Rolling window detailed results
├── rolling_window_stats.csv            # Rolling window statistics
├── training_summary.txt                # ⭐ NEW: Comprehensive summary report
└── MAMBA_BGNN_2025-01-17_12-00-00.log # Training log
```

---

## Summary Report Contents

The `training_summary.txt` includes:

1. **Header**: Model name, dataset, loss type, timestamp
2. **Training Configuration**: Epochs, early stopping, batch size, learning rate
3. **Training Time**: Total time, average/min/max epoch time
4. **Test Metrics**: All available metrics based on loss type
   - Bayesian: NLL, RMSE, MAE, IC, RIC, CRPS, Sharpness, PICP, Gap, AURC
   - Deterministic: Loss, RMSE, MAE, IC, RIC
5. **Rolling Window Evaluation**: Statistics from time-series evaluation
6. **Output Files**: Paths to all generated files

---

## Comparison Report Contents

The comparison report (`comparison_summary.txt`) includes:

1. **Header**: Dataset name, number of models compared
2. **Metrics Comparison Table**: Side-by-side comparison of all models
3. **Best Models by Metric**: Winner for each metric (RMSE, IC, RIC, etc.)
4. **Training Efficiency**: Epochs used by each model
5. **Output Files**: Paths to comparison files

---

## API Reference

### Trainer Methods

#### `generate_summary_report()`
Generates a comprehensive summary report.

**Returns:** Path to summary report file

**Called automatically** after `trainer.test()`

### results_comparison Functions

#### `read_test_metrics(log_dir, model_name=None)`
Read test metrics from a single training run.

**Args:**
- `log_dir` (str): Path to log directory
- `model_name` (str, optional): Model name

**Returns:** Dictionary with test metrics and metadata

---

#### `compare_multiple_runs(log_dirs, model_names=None)`
Compare test metrics from multiple runs.

**Args:**
- `log_dirs` (List[str]): List of log directory paths
- `model_names` (List[str], optional): List of model names

**Returns:** pandas DataFrame with comparison

---

#### `generate_comparison_report(log_dirs, model_names=None, output_dir=None, dataset_name="Comparison")`
Generate comprehensive comparison report.

**Args:**
- `log_dirs` (List[str]): List of log directory paths
- `model_names` (List[str], optional): List of model names
- `output_dir` (str, optional): Output directory
- `dataset_name` (str): Dataset name for report

**Returns:** Tuple of (comparison_df, report_path)

---

#### `find_model_logs(base_dir='logs', dataset=None, pattern=None)`
Find all model log directories matching criteria.

**Args:**
- `base_dir` (str): Base directory to search
- `dataset` (str, optional): Filter by dataset name
- `pattern` (str, optional): Additional pattern to match

**Returns:** List of log directory paths

---

## Benefits

1. **Automatic Tracking**: No need to manually track training time
2. **Comprehensive Reports**: All important info in one place
3. **Easy Comparison**: Quickly compare multiple models/runs
4. **Consistent Format**: Same format as baseline_notebook.py
5. **Auto-discovery**: Find and compare all runs automatically

---

## Integration with Existing Code

The new features are **backward compatible**. Your existing code will work without changes, but you'll automatically get:
- Training time tracking
- Summary reports after testing
- Better logging output

No code changes required! 🎉

---

## Troubleshooting

### Issue: Summary report is empty or missing metrics

**Solution:** Make sure you called `trainer.test()` before `generate_summary_report()`

### Issue: Comparison fails with "Test metrics CSV not found"

**Solution:** Ensure all log directories have completed a test run (with test_metrics.csv)

### Issue: Cannot find log directories automatically

**Solution:** Check that your logs are in the expected structure with `test_metrics.csv` files

---

## Tips

1. **Organize Logs**: Use descriptive log directory names including dataset and timestamp
2. **Comparison Workflow**: Train all models first, then run comparison once
3. **Reusable Reports**: You can regenerate comparison reports anytime without retraining
4. **Filter by Pattern**: Use `pattern` parameter to filter specific model types

---

## Example Output

### Training Summary (Console)
```
================================================================================
TRAINING COMPLETED
================================================================================
Model: MAMBA_BGNN
Total Time: 1234.56s (20.58 min)
Avg Epoch Time: 12.35s
--------------------------------------------------------------------------------
Test RMSE:  0.012345
Test IC:    0.567890
Test RIC:   0.543210
Test NLL:   -1.234567
Test CRPS:  0.008901
================================================================================
Summary saved to: logs/IXIC_MAMBA_BGNN_run1/training_summary.txt
================================================================================
```

### Comparison Summary (Console)
```
================================================================================
MODEL COMPARISON SUMMARY
================================================================================
Dataset: IXIC
Models compared: 4
--------------------------------------------------------------------------------
       Model      RMSE       MAE        IC       RIC
  MAMBA_BGNN  0.012345  0.009876  0.567890  0.543210
      Linear  0.015678  0.012345  0.456789  0.432109
        LSTM  0.013456  0.010234  0.523456  0.512345
 Transformer  0.014567  0.011234  0.498765  0.487654

================================================================================
✓ Comparison saved to: logs/comparison
  - Table: logs/comparison/model_comparison.csv
  - Report: logs/comparison/comparison_summary.txt
================================================================================
```

---

For more examples, see:
- `utils/results_comparison.py` (bottom of file)
- `baseline/baseline_notebook.py` (for baseline training)
