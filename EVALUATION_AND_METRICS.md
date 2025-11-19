# Bayesian vs Non-Bayesian Evaluation Metrics & Visualization

This document summarizes the metrics and plots used to compare **BIMamba-BGNN (Bayesian)** with **BIMamba-MAGAC (Non-Bayesian)**.

---

## 📊 I. BASIC METRICS

### 1. Predictive Performance
| Metric | Description | Better when |
|--------|-------------|-------------|
| **IC** (Information Coefficient) | Pearson correlation between prediction and target | Higher |
| **RIC** (Rank IC) | Spearman correlation - robust to outliers | Higher |
| **RMSE** | Root Mean Square Error | Lower |
| **MAE** | Mean Absolute Error | Lower |
| **Directional Accuracy** | % correct direction predictions (up/down) | Higher |

### 2. Financial Metrics
| Metric | Description | Formula |
|--------|-------------|---------|
| **Sharpe Ratio** | Risk-adjusted returns | (μ - rf) / σ × √252 |
| **Max Drawdown** | Maximum peak-to-trough decline | min(cumulative - running_max) |

---

## 🎯 II. UNCERTAINTY-BASED METRICS (Exploiting Mean + Variance)

### 1. Probabilistic Metrics

#### CRPS (Continuous Ranked Probability Score)
- **Purpose**: Evaluate quality of probabilistic predictions
- **Meaning**: Combines accuracy and calibration
- **Better**: Lower (0 is perfect)

#### Calibration Metrics
- **Coverage@68%/95%/99%**: % samples within prediction interval
- **Calibration Gap**: |observed_coverage - expected_coverage|
- **Mean Calibration Error**: Average gaps across confidence levels
- **Meaning**: Does the model "know what it knows" (well-calibrated)?

#### Sharpness
- **Formula**: mean(σ)
- **Meaning**: Average "confidence" of the model
- **Trade-off**: Want both sharp (low σ) AND calibrated

### 2. Uncertainty-Weighted IC
```python
weights = 1 / (σ + ε)
IC_weighted = correlation(predictions × weights, targets × weights)
```
- **Purpose**: Reward predictions when model is confident AND correct
- **Meaning**: Predictions with low uncertainty are trusted more

### 3. Risk-Adjusted Returns ⭐
```python
position_size = 1 / (1 + σ)  # Inverse uncertainty
returns = position_size × (predictions × targets)
```
- **Strategy**:
  - Low uncertainty → Bet more
  - High uncertainty → Bet less
- **Metrics**: Sharpe ratio before/after adjustment
- **Practical value**: This is how fund managers actually use uncertainty!

### 4. Confidence-Accuracy Relationship
```python
confidence = 1 / σ
correlation(confidence, |error|)
```
- **Expected**: Negative correlation
- **Meaning**: Is the model more accurate when confident?
- **Analysis**: Divide into bins by confidence levels

### 5. Uncertainty Contribution Analysis ⭐⭐⭐
**Key Question**: Does Bayesian uncertainty identify difficult samples?

**Analysis**:
- **When confident** (σ < p25):
  - Bayesian error: Low
  - Non-Bayesian error: Low

- **When uncertain** (σ > p75):
  - Bayesian error: Higher
  - Non-Bayesian error: Much higher
  - **→ Bayesian "knows" these are hard samples!**

**Test**: correlation(σ_Bayesian, error_NonBayesian)
- **Positive** → Uncertainty successfully identifies hard samples ✓

### 6. Prediction Interval Metrics
| Metric | Description |
|--------|-------------|
| **Mean Width (95%)** | Average width of 95% prediction interval |
| **Coverage (95%)** | % targets within 95% PI (ideal: 0.95) |
| **Width StdDev** | Variability of interval widths |

**Trade-off**: Narrow intervals (confident) vs Good coverage (calibrated)

---

## 🌍 III. MARKET REGIME ANALYSIS

### Regime Classification
Divide market into 3 regimes based on rolling volatility:
- **Low volatility** (< 33rd percentile): Stable market
- **Medium volatility** (33-67th percentile): Normal
- **High volatility** (> 67th percentile): Turbulent market

### Performance by Regime
Evaluate IC, RMSE, Directional Accuracy in each regime:
- Does the model perform well in high volatility?
- Does uncertainty increase during turbulent periods?

**Insight**: Bayesian models often perform better in high volatility periods thanks to uncertainty estimates!

---

## 📈 IV. VISUALIZATION PLOTS

### A. Basic Comparison Plots

#### 1. `metrics_comparison.png`
- **Format**: Bar chart
- **Content**: IC, RIC, RMSE, MAE, NLL, CRPS
- **Value labels**: Display values on bars

#### 2. `prediction_comparison.png`
- **Plot 1**: Time series with uncertainty bands (95% CI)
  - Non-Bayesian: Blue band
  - Bayesian: Red band
  - True values: Black scatter
- **Plot 2**: Scatter plot (predicted vs true)
  - Perfect prediction line: y=x

#### 3. `uncertainty_comparison.png`
4 subplots:
- **Sigma distribution**: Histogram comparison
- **Prediction vs Uncertainty**: Scatter plot
- **Error vs Uncertainty**: Correlation check
- **Uncertainty time series**: Rolling mean (50-pt MA)

#### 4. `calibration_comparison.png`
- **Plot 1**: Calibration curve
  - Observed vs Expected coverage
  - Perfect line: y=x
- **Plot 2**: Calibration gaps at 68%, 90%, 95%, 99%

### B. Market Regime Plots

#### 5. `regime_analysis.png`
4 subplots:
- **Regime over time**: Color-coded scatter
- **Regime distribution**: Bar chart
- **Error by regime**: Comparison bars
- **Uncertainty by regime**: Comparison bars

### C. Uncertainty Contribution Plots ⭐

#### 6. `confidence_accuracy_curve.png`
- **2 subplots**: Non-Bayesian | Bayesian
- **X-axis**: Confidence (1/σ) by deciles
- **Y-axis**: Mean absolute error
- **Error bars**: Standard error
- **Annotation**: Correlation + p-value
- **Expected**: Negative slope (high confidence → low error)

#### 7. `risk_adjusted_returns.png`
- **Plot 1**: Sharpe ratio comparison
  - Base Sharpe vs Risk-Adjusted Sharpe
  - Improvement % labels
- **Plot 2**: Cumulative returns
  - Before/after position sizing

**Interpretation**: Does using uncertainty for position sizing improve Sharpe ratio?

#### 8. `uncertainty_contribution.png` ⭐⭐⭐
**MOST IMPORTANT PLOT**

- **Plot 1**: Error bars by uncertainty level
  - Low uncertainty (confident)
  - High uncertainty (uncertain)
  - Sample counts + mean σ

- **Plot 2**: Text summary box
  - Correlation: σ vs error
  - p-value
  - Detailed statistics
  - Interpretation

**Key insight**: Positive correlation confirms Bayesian uncertainty identifies difficult samples!

#### 9. `sharpness_calibration_tradeoff.png`
- **Format**: Scatter plot
- **X-axis**: Sharpness (mean σ) - lower is better
- **Y-axis**: Calibration error - lower is better
- **Ideal region**: Bottom-left corner
- **Reference lines**:
  - Good calibration threshold (0.05)
  - Mean sharpness

**Interpretation**: A good model must be both sharp AND calibrated!

---

## 📁 V. OUTPUT STRUCTURE

### New Structure (v2.0) - With Timestamped Studies

Each experiment run creates a **timestamped study folder**, enabling clean tracking of multiple experiments:

```
logs/
│
├── bayesian_vs_nonbayesian_20251119_143025/  ← Timestamped study folder
│   │
│   ├── IXIC/                                  # Dataset 1
│   │   ├── BIMamba-MAGAC_NonBayesian/
│   │   │   ├── val_metrics.csv
│   │   │   ├── test_metrics.csv
│   │   │   ├── test_predictions.csv           # y, mu, sigma, log_var
│   │   │   └── model_best.pt
│   │   │
│   │   ├── BIMamba-MAGAC_Bayesian/
│   │   │   ├── val_metrics.csv
│   │   │   ├── test_metrics.csv
│   │   │   ├── test_predictions.csv
│   │   │   └── model_best.pt
│   │   │
│   │   ├── comparison_plots/                  ← Generated plots
│   │   │   ├── metrics_comparison.png          # Basic metrics bars
│   │   │   ├── prediction_comparison.png       # Time series + scatter
│   │   │   ├── uncertainty_comparison.png      # 4-panel uncertainty
│   │   │   ├── calibration_comparison.png      # Curve + gaps
│   │   │   ├── regime_analysis.png             # 4-panel regime
│   │   │   ├── confidence_accuracy_curve.png   # ⭐ Confidence bins
│   │   │   ├── risk_adjusted_returns.png       # ⭐ Position sizing
│   │   │   ├── uncertainty_contribution.png    # ⭐⭐⭐ Main insight
│   │   │   └── sharpness_calibration_tradeoff.png
│   │   │
│   │   ├── regime_analysis.csv                 # regime, rolling_std
│   │   ├── uncertainty_analysis.json           # ⭐ All uncertainty metrics
│   │   └── comprehensive_comparison.json       # Full results
│   │
│   ├── DJI/                                    # Dataset 2 (same structure)
│   ├── NYSE/                                   # Dataset 3 (same structure)
│   │
│   ├── cross_dataset_summary.json              # Aggregated stats
│   │   ├── summary
│   │   │   ├── NonBayesian: {metric: {mean, std}}
│   │   │   ├── Bayesian: {metric: {mean, std}}
│   │   │   └── Improvement: {metric: percentage}
│   │   └── detailed_results
│   │
│   ├── cross_sectional_ic_summary.txt          # Cross-sectional IC analysis
│   └── comprehensive_analysis_report.txt       # Full statistical analysis
│
├── bayesian_vs_nonbayesian_20251119_150530/    ← Another experiment run
│   └── ... (same structure)
│
└── bayesian_vs_nonbayesian_20251120_091245/    ← Yet another run
    └── ...
```

**Key Benefits:**
- ✅ **Single timestamp per study** - All datasets use same timestamp
- ✅ **Clean hierarchy** - `study → dataset → model`
- ✅ **Multi-run tracking** - Each run has unique timestamp
- ✅ **No conflicts** - Experiments don't overwrite each other
- ✅ **Easy cross-dataset analysis** - Consistent paths across datasets

**Path Format:**
```
logs/{study_name}_{timestamp}/{dataset}/{model_name}/{file}
     └─ bayesian_vs_nonbayesian_20251119_143025
                                  └─ YYYYMMDD_HHMMSS
```

### Key Files Explained

#### `test_predictions.csv`
```csv
y,mu,log_var,sigma
0.023,0.021,-4.5,0.011
-0.015,-0.012,-4.2,0.015
...
```

#### `uncertainty_analysis.json`
```json
{
  "uncertainty_weighted_ic": {"nb": 0.123, "b": 0.145},
  "risk_adjusted_returns": {
    "nb": {
      "sharpe_base": 0.5,
      "sharpe_adjusted": 0.65,
      "improvement": 30.0
    },
    "b": {
      "sharpe_base": 0.6,
      "sharpe_adjusted": 0.82,
      "improvement": 36.7
    }
  },
  "sharpness": {"nb": 0.025, "b": 0.028},
  "prediction_intervals": {...},
  "confidence_accuracy": {
    "bins": [...],
    "correlation_confidence_vs_error": -0.45,
    "correlation_p_value": 1.2e-15
  },
  "uncertainty_contribution": {
    "high_uncertainty": {
      "bayesian_error": 0.032,
      "nonbayesian_error": 0.045,
      "mean_sigma": 0.05
    },
    "low_uncertainty": {
      "bayesian_error": 0.015,
      "nonbayesian_error": 0.017,
      "mean_sigma": 0.012
    },
    "uncertainty_identifies_difficulty": {
      "correlation": 0.38,
      "p_value": 2.3e-8,
      "interpretation": "Positive correlation means..."
    }
  }
}
```

#### `cross_dataset_summary.json`
```json
{
  "datasets": ["IXIC", "DJI", "NYSE"],
  "summary": {
    "NonBayesian": {
      "ic": {"mean": 0.234, "std": 0.021},
      "ric": {"mean": 0.198, "std": 0.018},
      ...
    },
    "Bayesian": {
      "ic": {"mean": 0.267, "std": 0.019},
      "ric": {"mean": 0.232, "std": 0.015},
      ...
    },
    "Improvement": {
      "ic": 14.1,    // %
      "ric": 17.2,
      ...
    }
  },
  "detailed_results": {...}
}
```

---

## 🚀 VI. USAGE

### 1. Run Comparison on Multiple Datasets
```python
from models.run_bayesian_vs_nonbayesian import run_multi_dataset_comparison

results = run_multi_dataset_comparison(
    datasets=['IXIC', 'DJI', 'NYSE'],
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    R=3, K=3, heads=4,
    mc_train=3, mc_eval=20,
    device='cuda'
)

# Returns dictionary with:
# - results['study_name']: e.g., 'bayesian_vs_nonbayesian_20251119_143025'
# - results['study_timestamp']: e.g., '20251119_143025'
# - results['all_results']: Performance for each dataset
# - results['summary']: Cross-dataset aggregated metrics
# - results['cross_sectional']: Cross-sectional IC results

print(f"Study created: {results['study_name']}")
print(f"Results saved to: logs/{results['study_name']}/")
```

**Console Output**:
```
================================================================================
MULTI-DATASET BAYESIAN VS NON-BAYESIAN COMPARISON
================================================================================
...
[1/4] Computing financial metrics...
  Sharpe Ratio - NB: 0.5234, B: 0.6421
  Max Drawdown - NB: -0.123, B: -0.098
  RIC - NB: 0.234, B: 0.267

[2/4] Computing probabilistic metrics...
  CRPS - NB: 0.012345, B: 0.010234

[3/4] Computing calibration metrics...
  Mean Calibration Error - NB: 0.0456, B: 0.0234
  Coverage@95% - NB: 0.9234, B: 0.9478

[4/4] Analyzing market regimes...
  Low volatility:    35.2%
  Medium volatility: 42.1%
  High volatility:   22.7%

================================================================================
UNCERTAINTY-BASED ANALYSIS (Exploiting Bayesian Mean + Variance)
================================================================================

[1/5] Uncertainty-weighted IC...
  UW-IC - NB: 0.2456, B: 0.2897
  → Rewards confidence when correct

[2/5] Risk-adjusted returns (position sizing by uncertainty)...
  Base Sharpe - NB: 0.5234, B: 0.6421
  Adjusted Sharpe - NB: 0.6821, B: 0.8765
  Improvement - NB: 30.32%, B: 36.51%
  → Lower uncertainty = larger position size

[5/5] Confidence-accuracy relationship...
  Correlation (Confidence vs Error):
    NB: -0.3456 (p=2.34e-12)
    B:  -0.4789 (p=1.23e-18)
  → Expect negative correlation (high confidence = low error)

--------------------------------------------------------------------------------
UNCERTAINTY CONTRIBUTION ANALYSIS
--------------------------------------------------------------------------------

When Bayesian is HIGHLY uncertain (top 25%):
  Bayesian error:     0.032456
  Non-Bayesian error: 0.045234
  Mean sigma:         0.051234

When Bayesian is CONFIDENT (bottom 25%):
  Bayesian error:     0.015234
  Non-Bayesian error: 0.017123
  Mean sigma:         0.012345

Does uncertainty identify difficult samples?
  Correlation (Bayesian σ vs Non-Bayesian error): 0.3812
  p-value: 2.34e-08
  → Positive correlation means Bayesian uncertainty identifies samples
    where Non-Bayesian struggles

✓ Uncertainty analysis saved to: logs/.../uncertainty_analysis.json
```

### 2. Generate All Plots
```python
from utils.result_plot import plot_bayesian_vs_nonbayesian_comparison

# Generate all plots (including uncertainty plots)
plot_bayesian_vs_nonbayesian_comparison('logs/bayesian_vs_nonbayesian')

# Or for specific dataset
plot_bayesian_vs_nonbayesian_comparison(
    'logs/bayesian_vs_nonbayesian',
    dataset='IXIC'
)
```

### 3. Generate Only Uncertainty Plots
```python
from utils.result_plot import plot_uncertainty_based_comparison

plot_uncertainty_based_comparison('logs/bayesian_vs_nonbayesian')
```

---

## 📝 VII. INTERPRETATION GUIDE

### How to Read Results

#### 1. Basic Performance
- **IC/RIC**: Bayesian should be 5-15% higher
- **RMSE**: Bayesian should be 3-10% lower
- **Directional Accuracy**: Slight improvement (1-3%)

#### 2. Uncertainty Quality
- **Calibration Gap < 0.05**: Good
- **Coverage@95% ≈ 0.95**: Well-calibrated
- **CRPS**: Bayesian should be ~10-30% lower

#### 3. Practical Value
- **Risk-Adjusted Sharpe improvement > 20%**: Excellent
- **Uncertainty-weighted IC > Standard IC**: Confirms uncertainty is useful
- **Negative confidence-error correlation**: Model "knows when it knows"

#### 4. Key Insights
**Must check**: `uncertainty_contribution.png`
- **Positive correlation (σ vs NB error)**: ✓ Uncertainty identifies difficulty
- **High σ → Both models struggle**: Normal
- **High σ → Only NB struggles**: ✓✓ Bayesian has advantage
- **Low σ → Low error**: ✓ Model is confident when correct

### Common Patterns

#### ✅ **Good Bayesian Model**
- IC: +10-15% vs Non-Bayesian
- Calibration error: < 0.05
- Risk-adjusted Sharpe improvement: > 20%
- Confidence-error correlation: Negative & significant
- Uncertainty identifies difficulty: Positive correlation

#### ⚠️ **Overconfident Model**
- Sharpness: Very low (good)
- Calibration gap: Large (bad)
- Coverage@95%: < 0.90
- **Fix**: Increase prior variance, tune KL weight

#### ⚠️ **Underconfident Model**
- Sharpness: Very high (too uncertain)
- Calibration gap: Large (bad)
- Coverage@95%: > 0.98
- **Fix**: Decrease prior variance, increase mc_samples

---

## 🎓 VIII. RESEARCH CONTRIBUTIONS

### For Paper Writing

#### Main Claims
1. **Uncertainty quantification**: Bayesian GNN provides well-calibrated uncertainty
2. **Practical value**: Uncertainty improves risk-adjusted returns
3. **Interpretability**: Uncertainty identifies difficult market conditions
4. **Robustness**: Consistent improvement across multiple datasets/regimes

#### Key Tables for Paper

**Table 1**: Cross-Dataset Performance
```
Metric          | Non-Bayesian | Bayesian    | Improvement
----------------|--------------|-------------|------------
IC              | 0.234±0.021  | 0.267±0.019 | +14.1%
RIC             | 0.198±0.018  | 0.232±0.015 | +17.2%
CRPS            | 0.0234±0.003 | 0.0189±0.002| -19.2%
Sharpe (adj)    | 0.682±0.045  | 0.877±0.038 | +28.6%
```

**Table 2**: Uncertainty Contribution
```
Condition        | Bayesian Error | Non-Bayesian Error | Ratio
-----------------|----------------|--------------------|-----------
Low σ (confident)| 0.0152         | 0.0171            | 0.89
High σ (uncertain)| 0.0325        | 0.0452            | 0.72
Improvement      | -              | -                 | 19.0%
```

#### Key Figures for Paper
1. **Figure 1**: Calibration comparison (from `calibration_comparison.png`)
2. **Figure 2**: Uncertainty contribution (from `uncertainty_contribution.png`)
3. **Figure 3**: Risk-adjusted returns (from `risk_adjusted_returns.png`)
4. **Figure 4**: Regime analysis (from `regime_analysis.png`)

#### Statistical Tests
All correlations reported with:
- Spearman's ρ (robust to outliers)
- p-values (significance test)
- Sample sizes

**Example**: "Bayesian uncertainty significantly correlates with Non-Bayesian error (ρ = 0.381, p < 1e-7, n = 1247), confirming that uncertainty successfully identifies difficult samples."

---

## ⚙️ IX. TECHNICAL NOTES

### Code Robustness
All metrics handle edge cases:
- ✅ Division by zero: `+ 1e-6` or `+ 1e-8`
- ✅ Empty arrays: `if mask.sum() > 0`
- ✅ NaN values: Explicit checks
- ✅ Small samples: Return defaults
- ✅ File I/O: `os.path.exists()` checks
- ✅ JSON serialization: Explicit `float()`, `int()` casts

### Performance Tips
- Use GPU for training: `device='cuda'`
- Reduce `mc_eval` for faster inference (trade-off: accuracy)
- Use fewer epochs for quick testing
- Process datasets in parallel (modify code if needed)

### Customization
Modify hyperparameters:
```python
run_multi_dataset_comparison(
    R=3,           # BiMamba depth
    K=3,           # Chebyshev order
    heads=4,       # Attention heads
    mc_train=3,    # MC samples during training
    mc_eval=20,    # MC samples during evaluation (↑ = better uncertainty)
    hidden_dim=64  # Model capacity
)
```

---

## 📚 References & Citations

### Key Papers
- **Bayesian Neural Networks**: Blundell et al. (2015) - Weight Uncertainty in Neural Networks
- **Calibration**: Guo et al. (2017) - On Calibration of Modern Neural Networks
- **CRPS**: Gneiting & Raftery (2007) - Strictly Proper Scoring Rules
- **Financial Applications**: Mnih & Rezende (2016) - Variational Inference for Monte Carlo

### Metrics Standards
- **IC/RIC**: Qlib (Microsoft), WorldQuant
- **Sharpe Ratio**: Finance industry standard
- **Calibration**: ML uncertainty quantification standard
- **CRPS**: Weather forecasting, now ML standard

---

## ✉️ Contact & Support

For issues or questions:
1. Check code documentation
2. Review this evaluation guide
3. Examine example outputs in `logs/`

**Happy Evaluating!** 🎉
