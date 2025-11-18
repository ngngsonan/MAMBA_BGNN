# Baseline Models Summary – NYSE, DJI, IXIC

## 0. Metrics: Definitions and How to Compare Models

- **RMSE (Root Mean Squared Error)**
  - Measures average magnitude of prediction errors (squared, then square root).
  - **Lower is better**. Penalizes large errors more strongly than MAE.

- **MAE (Mean Absolute Error)**
  - Average absolute difference between predictions and true values.
  - **Lower is better**. More robust to outliers than RMSE.

- **IC (Information Coefficient)**
  - Correlation between model predictions and actual outcomes (Pearson correlation).
  - **Higher is better** (closer to 1). Indicates how well predictions preserve ordering of returns.
  - **Single-Asset IC**: Time-series correlation for individual assets (typically 0.3-0.9)
  - **Cross-Sectional IC**: Correlation across multiple assets at each time point (typically 0.01-0.05)

- **RIC (Rank Information Coefficient)**
  - Correlation between *ranks* of predictions and actual returns.
  - **Higher is better**. Focuses purely on ranking ability, important for relative-return strategies.

- **Dir Acc (Directional Accuracy)**
  - Fraction of times the model correctly predicts the direction (up/down) of the move.
  - **Higher is better** (closer to 1). Important for long/short or binary decision strategies.

- **Sharpe (Sharpe Ratio)**
  - Risk-adjusted return metric: average return divided by its standard deviation (often annualized).
  - **Higher is better**. Balances profitability and volatility.

- **Max DD (Maximum Drawdown)**
  - Largest peak-to-trough loss over the evaluation period.
  - **Lower is better**. Measures worst-case loss scenario.

- **Calmar (Calmar Ratio)**
  - Sharpe ratio divided by Maximum Drawdown.
  - **Higher is better**. Measures return per unit of downside risk.

- **CRPS (Continuous Ranked Probability Score)**
  - Probabilistic forecasting metric measuring prediction accuracy.
  - **Lower is better**. Evaluates both point predictions and uncertainty.

- **Avg Time (s/epoch)** and **Total Time (s)**
  - Average training time per epoch and total training time.
  - **Lower is better**, but must be traded off against predictive performance.

**How to compare models**

- For **error metrics (RMSE, MAE, CRPS)** → choose models with **lower** values.
- For **signal quality metrics (IC, RIC)** → choose models with **higher** values.
- For **trading-oriented metrics (Dir Acc, Sharpe, Calmar)**:
  - Prefer **higher Dir Acc, Sharpe, and Calmar**, **lower Max DD**.
- Combine performance with **training cost**:
  - When several models have similar metrics, prefer the one with **lower training time**.
  - For production, pick a model that balances accuracy, stability (Sharpe/Max DD), and compute cost.

---

## 1. Detailed Results Tables

### 1.1. IXIC

| Model       |   RMSE   |   MAE    |    IC    |    RIC    | Dir Acc | Sharpe  | Max DD | Calmar  |  CRPS   | Avg Time (s/epoch) | Total Time (s) |
|-------------|----------|----------|----------|-----------|---------|---------|--------|---------|---------|---------------------|----------------|
| Linear      | 0.014162 | 0.011097 | 0.605496 | 0.551295  | 0.667308| 7.0892  | 0.0571 | 28.8117 | 0.203716| 0.09                | 1.4            |
| LSTM        | 0.015453 | 0.012178 | 0.397450 | 0.372582  | 0.588462| 3.9226  | 0.0794 | 12.3241 | 0.218493| 0.64                | 21.7           |
| Transformer | 0.017113 | 0.013361 | 0.043085 | 0.041830  | 0.490385| 0.3750  | 0.3227 | 0.2992  | 0.136165| 0.94                | 22.7           |
| AGCRN       | 0.017600 | 0.013816 |-0.137604 |-0.127954  | 0.494231|-0.1648  | 0.4013 |-0.1058  | 0.243926| 0.32                | 4.8            |
| TemporalGN  | 0.014705 | 0.011321 | 0.428085 | 0.398525  | 0.634615| 5.2504  | 0.0675 | 18.9577 | 0.223801| 0.28                | 9.1            |

- Total training time (all models): **59.7s (~1.0 minute)**

---

### 1.2. DJI

| Model       |   RMSE   |   MAE    |    IC    |    RIC    | Dir Acc | Sharpe  | Max DD | Calmar  |  CRPS   | Avg Time (s/epoch) | Total Time (s) |
|-------------|----------|----------|----------|-----------|---------|---------|--------|---------|---------|---------------------|----------------|
| Linear      | 0.012404 | 0.009957 | 0.053386 | 0.051360  | 0.511538|-0.0712  | 0.2606 |-0.0445  | 0.265464| 0.08                | 1.4            |
| LSTM        | 0.010845 | 0.008598 | 0.280188 | 0.248379  | 0.548077| 2.0637  | 0.1641 | 2.0302  | 0.229385| 0.57                | 21.0           |
| Transformer | 0.013535 | 0.010984 |-0.011121 |-0.014519  | 0.513462|-0.5587  | 0.2973 |-0.3062  | 0.185503| 0.93                | 13.1           |
| AGCRN       | 0.010392 | 0.007790 |-0.090756 |-0.100441  | 0.467308|-1.7918  | 0.4767 |-0.6103  | 0.228094| 0.32                | 3.5            |
| TemporalGN  | 0.010530 | 0.007882 |-0.095118 |-0.081390  | 0.471154|-1.8613  | 0.5396 |-0.5596  | 0.246736| 0.32                | 3.5            |

- Total training time (all models): **42.5s (~0.7 minute)**

---

### 1.3. NYSE

| Model       |   RMSE   |   MAE    |    IC    |    RIC    | Dir Acc | Sharpe  | Max DD | Calmar  |  CRPS   | Avg Time (s/epoch) | Total Time (s) |
|-------------|----------|----------|----------|-----------|---------|---------|--------|---------|---------|---------------------|----------------|
| Linear      | 0.007983 | 0.006087 | 0.755718 | 0.708335  | 0.719231| 8.1317  | 0.0405 | 29.9402 | 0.244566| 0.08                | 2.4            |
| LSTM        | 0.010418 | 0.007929 | 0.379425 | 0.353079  | 0.615385| 4.2695  | 0.0673 | 10.4834 | 0.234128| 0.55                | 15.4           |
| Transformer | 0.011411 | 0.008743 |-0.188118 |-0.189360  | 0.448077|-2.7682  | 0.6645 |-0.7074  | 0.228755| 0.96                | 18.3           |
| AGCRN       | 0.011600 | 0.008973 |-0.101766 |-0.101308  | 0.496154|-0.2739  | 0.3301 |-0.1426  | 0.218674| 0.31                | 3.4            |
| TemporalGN  | 0.010660 | 0.008142 | 0.239069 | 0.225401  | 0.561538| 2.4428  | 0.1288 | 3.2119  | 0.243238| 0.25                | 4.0            |

- Total training time (all models): **43.5s (~0.7 minute)**

---

## 2. Cross-Sectional IC Analysis (Multi-Asset)

Cross-sectional IC measures correlation **across assets** at each time point, unlike single-asset IC which measures correlation **across time** for individual assets.

| Model       | CS-IC (Mean) | CS-IC (Median) | CS-IC (% Positive) | CS-RIC (Mean) | CS-RIC (Median) | CS-RIC (% Positive) |
|-------------|--------------|----------------|--------------------|---------------|-----------------|---------------------|
| TemporalGN  | 0.217386     | 0.663534       | 61.5%              | 0.168269      | 0.500000        | 61.2%               |
| LSTM        | 0.175235     | 0.334396       | 60.8%              | 0.139423      | 0.500000        | 59.8%               |
| Linear      | 0.116685     | 0.281883       | 56.7%              | 0.115385      | 0.500000        | 58.3%               |
| Transformer | 0.011787     | -0.005651      | 49.8%              | -0.005769     | -0.500000       | 48.7%               |
| AGCRN       | 0.000791     | 0.071459       | 51.3%              | 0.021154      | 0.500000        | 50.4%               |

**Note**: Assets analyzed: IXIC, DJI, NYSE across 520 trading days (test set).

---

## 3. Best Models per Dataset (by Metric)

### 3.1. IXIC

- **RMSE**: Linear – 0.014162
- **MAE**: Linear – 0.011097
- **IC**: Linear – 0.605496
- **RIC**: Linear – 0.551295
- **Dir Acc**: Linear – 0.667308
- **Sharpe**: Linear – 7.0892
- **Calmar**: Linear – 28.8117
- **CRPS**: Transformer – 0.136165

### 3.2. DJI

- **RMSE**: AGCRN – 0.010392
- **MAE**: AGCRN – 0.007790
- **IC**: LSTM – 0.280188
- **RIC**: LSTM – 0.248379
- **Dir Acc**: LSTM – 0.548077
- **Sharpe**: LSTM – 2.0637
- **Calmar**: LSTM – 2.0302
- **CRPS**: Transformer – 0.185503

### 3.3. NYSE

- **RMSE**: Linear – 0.007983
- **MAE**: Linear – 0.006087
- **IC**: Linear – 0.755718
- **RIC**: Linear – 0.708335
- **Dir Acc**: Linear – 0.719231
- **Sharpe**: Linear – 8.1317
- **Calmar**: Linear – 29.9402
- **CRPS**: AGCRN – 0.218674

### 3.4. Cross-Sectional (Multi-Asset)

- **CS-IC (Mean)**: TemporalGN – 0.217386
- **CS-RIC (Mean)**: TemporalGN – 0.168269
- **CS-IC (% Positive)**: TemporalGN – 61.5%

---

## 4. Statistical Observations

### 4.1. IXIC

- Linear achieves best performance across 7/8 primary metrics (RMSE, MAE, IC, RIC, Dir Acc, Sharpe, Calmar)
- Linear outperforms all models on Sharpe ratio by 35% (7.0892 vs 5.2504 for TemporalGN)
- Linear achieves Calmar ratio of 28.8117, 52% higher than TemporalGN (18.9577)
- AGCRN shows negative IC (-0.137604) and negative Sharpe (-0.1648)
- Transformer training time (22.7s) is 16× longer than Linear (1.4s) but achieves lower performance on all metrics except CRPS
- TemporalGN achieves second-best performance with IC=0.428085 and training time of 9.1s

### 4.2. DJI

- No single model dominates; performance is distributed across models
- LSTM achieves best IC (0.280188), RIC (0.248379), Dir Acc (0.548077), and Sharpe (2.0637)
- AGCRN achieves lowest RMSE (0.010392) and MAE (0.007790) but negative IC (-0.090756) and Sharpe (-1.7918)
- All models except LSTM show negative Sharpe ratios, indicating poor risk-adjusted returns
- Linear shows near-zero Sharpe (-0.0712) with IC of only 0.053386
- AGCRN and TemporalGN both show negative IC despite low error metrics
- Transformer CRPS (0.185503) is 19% lower than LSTM (0.229385) despite worse IC performance

### 4.3. NYSE

- Linear achieves best performance across 7/8 primary metrics (RMSE, MAE, IC, RIC, Dir Acc, Sharpe, Calmar)
- Linear achieves IC of 0.755718, 99% higher than LSTM (0.379425)
- Linear achieves Calmar ratio of 29.9402, 186% higher than LSTM (10.4834)
- Transformer and AGCRN both show negative IC and Sharpe ratios
- Linear training time (2.4s) is 6.4× faster than LSTM (15.4s) with superior performance
- LSTM Dir Acc (0.615385) is 14% lower than Linear (0.719231)

### 4.4. Cross-Dataset Performance Patterns

- **Linear**: Dominates on IXIC and NYSE (IC > 0.6, Sharpe > 7) but weak on DJI (IC = 0.053, Sharpe = -0.071)
- **LSTM**: Consistent moderate performance across all datasets (IC range: 0.280-0.397, Sharpe range: 2.06-4.27)
- **Transformer**: Poor single-asset IC on all datasets (range: -0.188 to 0.043) but lowest CRPS on DJI (0.185) and IXIC (0.136)
- **AGCRN**: Strong error metrics (RMSE, MAE) but consistently negative IC across all datasets
- **TemporalGN**: Mixed performance; positive IC on IXIC (0.428) and NYSE (0.239), negative on DJI (-0.095)

### 4.5. Cross-Sectional IC Analysis

- TemporalGN achieves highest cross-sectional IC (0.217386) and RIC (0.168269), with 61.5% positive IC days
- LSTM ranks second with CS-IC of 0.175235 and 60.8% positive IC days
- Linear achieves CS-IC of 0.116685, 46% lower than TemporalGN
- Transformer shows near-zero cross-sectional performance (CS-IC = 0.011787, 49.8% positive days)
- AGCRN shows minimal cross-sectional predictive power (CS-IC = 0.000791)
- Cross-sectional IC values (0.0-0.22) are significantly lower than single-asset IC values (0.0-0.76), which is expected

### 4.6. Training Efficiency

- Linear is fastest across all datasets: 1.4s (IXIC), 1.4s (DJI), 2.4s (NYSE)
- AGCRN and TemporalGN show similar training times: 3.4-4.8s per dataset
- LSTM training times: 15.4-21.7s per dataset, 8-15× slower than Linear
- Transformer is slowest on IXIC (22.7s) and NYSE (18.3s), providing lowest performance-to-cost ratio
- Total training time for all 5 models: 59.7s (IXIC), 42.5s (DJI), 43.5s (NYSE)

---

## 5. Summary

**Best Overall Model by Dataset:**
- **IXIC**: Linear (highest IC, Sharpe, Calmar, Dir Acc)
- **DJI**: LSTM (only model with positive IC and Sharpe > 2)
- **NYSE**: Linear (highest IC, Sharpe, Calmar, Dir Acc)

**Best Cross-Sectional Model:**
- **TemporalGN** (CS-IC = 0.217, 61.5% positive days)

**Most Consistent Model:**
- **LSTM** (positive IC and Sharpe across all datasets, moderate performance)

**Fastest Model:**
- **Linear** (1.4-2.4s total training time, competitive performance on IXIC and NYSE)

**Key Finding:**
- Model performance is highly dataset-dependent. Linear excels on IXIC and NYSE but fails on DJI, while LSTM provides stable positive performance across all datasets.
