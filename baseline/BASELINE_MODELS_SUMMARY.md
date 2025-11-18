# Baseline Models Summary – NYSE, DJI, IXIC

## 0. Metrics: Definitions and How to Compare Models

- **RMSE (Root Mean Squared Error)**  
  - Measures average magnitude of prediction errors (squared, then square root).  
  - **Lower is better**. Penalizes large errors more strongly than MAE.

- **MAE (Mean Absolute Error)**  
  - Average absolute difference between predictions and true values.  
  - **Lower is better**. More robust to outliers than RMSE.

- **IC (Information Coefficient)**  
  - Correlation between model predictions and actual outcomes (often rank/Pearson correlation).  
  - **Higher is better** (closer to 1). Indicates how well predictions preserve ordering of returns.

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

- **Avg Time (s/epoch)** and **Total Time (s)**  
  - Average training time per epoch and total training time.  
  - **Lower is better**, but must be traded off against predictive performance.

**How to compare models**

- For **error metrics (RMSE, MAE)** → choose models with **lower** values.  
- For **signal quality metrics (IC, RIC)** → choose models with **higher** values.  
- For **trading-oriented metrics (Dir Acc, Sharpe, Max DD)**:
  - Prefer **higher Dir Acc and Sharpe**, **lower Max DD**.  
- Combine performance with **training cost**:
  - When several models have similar metrics, prefer the one with **lower training time**.
  - For production, pick a model that balances accuracy, stability (Sharpe/Max DD), and compute cost.

---

## 1. Detailed Results Tables

### 1.1. NYSE

| Model       |   RMSE   |   MAE    |    IC    |   RIC    | Dir Acc | Sharpe  | Max DD | Avg Time (s/epoch) | Total Time (s) |
|------------|----------|----------|----------|----------|---------|---------|--------|---------------------|----------------|
| Linear     | 0.008273 | 0.006368 | 0.715716 | 0.670291 | 0.696154 |  7.8294 | 0.0388 | 0.15                | 4.0            |
| LSTM       | 0.002654 | 0.002057 | 0.989890 | 0.986255 | 0.923077 | 15.4815 | 0.0064 | 0.61                | 52.0           |
| Transformer| 0.010881 | 0.008233 | 0.158771 | 0.149696 | 0.534615 |  1.2542 | 0.1582 | 1.35                | 29.7           |
| AGCRN      | 0.002210 | 0.001730 | 0.981631 | 0.978330 | 0.932692 | 15.5128 | 0.0082 | 0.56                | 47.5           |
| TemporalGN | 0.012026 | 0.009533 | 0.331378 | 0.292127 | 0.526923 |  0.6416 | 0.1928 | 0.39                | 10.8           |

- Total training time (all models): **144.1s (~2.4 minutes)**

---

### 1.2. DJI

| Model       |   RMSE   |   MAE    |    IC    |    RIC    | Dir Acc | Sharpe  | Max DD | Avg Time (s/epoch) | Total Time (s) |
|------------|----------|----------|----------|-----------|---------|---------|--------|---------------------|----------------|
| Linear     | 0.011319 | 0.008819 | 0.283870 | 0.236958  | 0.534615 |  1.2776 | 0.1363 | 0.15                | 2.7            |
| LSTM       | 0.009312 | 0.007110 | 0.442777 | 0.403133  | 0.625000 |  5.0311 | 0.0510 | 0.60                | 14.4           |
| Transformer| 0.011423 | 0.008822 |-0.058610 |-0.041689  | 0.505769 | -0.5170 | 0.2668 | 1.32                | 22.4           |
| AGCRN      | 0.010442 | 0.007920 | 0.090743 | 0.095452  | 0.507692 |  0.3722 | 0.1890 | 0.53                | 7.9            |
| TemporalGN | 0.009738 | 0.007405 | 0.334857 | 0.284510  | 0.596154 |  4.0962 | 0.0497 | 0.42                | 10.8           |

- Total training time (all models): **58.3s (~1.0 minute)**

---

### 1.3. IXIC

| Model       |   RMSE   |   MAE    |    IC    |    RIC    | Dir Acc | Sharpe  | Max DD | Avg Time (s/epoch) | Total Time (s) |
|------------|----------|----------|----------|-----------|---------|---------|--------|---------------------|----------------|
| Linear     | 0.015872 | 0.013638 | 0.835495 | 0.794874  | 0.634615 |  6.3398 | 0.1089 | 0.16                | 5.4            |
| LSTM       | 0.015387 | 0.012157 | 0.406920 | 0.379268  | 0.598077 |  4.1532 | 0.0773 | 0.61                | 12.8           |
| Transformer| 0.009766 | 0.007345 | 0.921686 | 0.928672  | 0.876923 | 15.0479 | 0.0319 | 1.32                | 105.4          |
| AGCRN      | 0.016170 | 0.012538 | 0.138147 | 0.125782  | 0.525000 |  0.6159 | 0.2377 | 0.53                | 8.5            |
| TemporalGN | 0.014576 | 0.011356 | 0.444235 | 0.408945  | 0.632692 |  6.0409 | 0.0796 | 0.43                | 8.1            |

- Total training time (all models): **140.2s (~2.3 minutes)**

---

## 2. Best Models per Dataset (by Metric)

### 2.1. NYSE

- RMSE: **AGCRN** – 0.002210  
- MAE: **AGCRN** – 0.001730  
- IC: **LSTM** – 0.989890  
- RIC: **LSTM** – 0.986255  
- Dir Acc: **AGCRN** – 0.932692  
- Sharpe: **AGCRN** – 15.5128  

### 2.2. DJI

- RMSE: **LSTM** – 0.009312  
- MAE: **LSTM** – 0.007110  
- IC: **LSTM** – 0.442777  
- RIC: **LSTM** – 0.403133  
- Dir Acc: **LSTM** – 0.625000  
- Sharpe: **LSTM** – 5.0311  

### 2.3. IXIC

- RMSE: **Transformer** – 0.009766  
- MAE: **Transformer** – 0.007345  
- IC: **Transformer** – 0.921686  
- RIC: **Transformer** – 0.928672  
- Dir Acc: **Transformer** – 0.876923  
- Sharpe: **Transformer** – 15.0479  

---

## 3. High-Level Observations

- **NYSE**  
  - **AGCRN** dominates on error metrics (RMSE, MAE) and trading metrics (Dir Acc, Sharpe), with very low Max DD.  
  - **LSTM** slightly outperforms AGCRN on IC/RIC.  
  - Overall, AGCRN is the best choice when prioritizing trading performance and stability.

- **DJI**  
  - **LSTM** is clearly the best model across all major metrics (RMSE, MAE, IC, RIC, Dir Acc, Sharpe).  
