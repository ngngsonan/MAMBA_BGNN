# Hướng Dẫn Train và So Sánh Baseline Models

Hướng dẫn này giúp bạn train tất cả baseline models và so sánh kết quả một cách toàn diện.

## 📁 Cấu Trúc Folder

Tất cả file liên quan đến baseline đã được tổ chức trong folder `baseline/`:

```
MAMBA_BGNN/
├── baseline/
│   ├── __init__.py                    # Package initialization
│   ├── baseline_models.py             # Model definitions
│   ├── train_all_baselines.py         # Training script
│   ├── compare_baselines.py           # Comparison script
│   ├── test_baseline_setup.py         # Verification script
│   └── BASELINE_TRAINING_GUIDE.md     # This guide
├── utils/
│   ├── data_processing.py
│   └── trainer.py
└── Dataset/
```

**Lưu ý**: Tất cả lệnh phải chạy từ thư mục gốc `MAMBA_BGNN/`, không phải từ trong folder `baseline/`.

## 📋 Tổng Quan

Hệ thống bao gồm:
- **5 Baseline Models**: Linear, LSTM, Transformer, AGCRN, TemporalGN
- **Comprehensive Metrics**: NLL, RMSE, MAE, IC, RIC, CRPS, Directional Accuracy, Sharpe Ratio, và nhiều hơn nữa
- **Advanced Evaluation**: Market regime analysis, stress testing, rolling window evaluation

## 🚀 Cách Sử Dụng

### 1. Train Tất Cả Baseline Models

#### Cách đơn giản nhất (sử dụng cấu hình mặc định):

```bash
# Chạy từ thư mục gốc MAMBA_BGNN/
python baseline/train_all_baselines.py --dataset IXIC
```

#### Với cấu hình tùy chỉnh:

```bash
python baseline/train_all_baselines.py \
    --dataset IXIC \
    --window 5 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --hidden_dim 64 \
    --early_stop \
    --early_stop_patience 10
```

#### Train chỉ một số models cụ thể:

```bash
python baseline/train_all_baselines.py \
    --dataset IXIC \
    --models LSTM Transformer AGCRN
```

#### Các tham số quan trọng:

- `--dataset`: Dataset để sử dụng (IXIC, DJI, NYSE)
- `--window`: Lookback window size (mặc định: 5)
- `--batch_size`: Batch size (mặc định: 32)
- `--epochs`: Số epochs (mặc định: 50)
- `--learning_rate`: Learning rate (mặc định: 0.001)
- `--hidden_dim`: Hidden dimension cho models (mặc định: 64)
- `--models`: Danh sách models cần train (mặc định: tất cả)
- `--early_stop`: Sử dụng early stopping
- `--early_stop_patience`: Patience cho early stopping (mặc định: 10)
- `--base_log_dir`: Thư mục lưu logs (mặc định: logs/baselines)

### 2. So Sánh Kết Quả

Sau khi train xong, chạy script so sánh:

```bash
python baseline/compare_baselines.py --dataset IXIC
```

#### Với cấu hình tùy chỉnh:

```bash
python baseline/compare_baselines.py \
    --base_log_dir logs/baselines \
    --dataset IXIC \
    --output_dir logs/baselines/comparison
```

Script này sẽ tạo:
- **comparison_table.csv**: Bảng so sánh tất cả metrics
- **model_rankings.csv**: Xếp hạng models dựa trên average rank
- **comparison_report.txt**: Báo cáo chi tiết dạng text
- **visualizations/**: Thư mục chứa các biểu đồ so sánh
  - `probabilistic_metrics.png`: So sánh các metrics xác suất
  - `financial_metrics.png`: So sánh các metrics tài chính
  - `radar_comparison.png`: Radar chart so sánh tổng thể

## 📊 Kết Quả Được Lưu

### Cho mỗi model, kết quả được lưu tại:

```
logs/baselines/
├── IXIC_Linear/
│   ├── comprehensive_metrics.csv       # Tất cả metrics chính
│   ├── test_metrics.csv               # Test metrics cơ bản
│   ├── val_metrics.csv                # Validation metrics theo epoch
│   ├── test_predictions.csv           # Predictions trên test set
│   ├── regime_analysis.csv            # Market regime analysis
│   ├── stress_test.csv                # Stress test kết quả
│   ├── rolling_window_results.csv     # Rolling window evaluation
│   ├── rolling_window_stats.csv       # Rolling window statistics
│   ├── evaluation_summary.txt         # Tóm tắt đánh giá
│   ├── best_model.pth                 # Best model weights
│   └── Linear_*.log                   # Training logs
├── IXIC_LSTM/
├── IXIC_Transformer/
├── IXIC_AGCRN/
└── IXIC_TemporalGN/
```

### Kết quả so sánh:

```
logs/baselines/comparison/
├── comparison_table.csv               # Bảng so sánh toàn diện
├── model_rankings.csv                 # Xếp hạng models
├── comparison_report.txt              # Báo cáo chi tiết
└── visualizations/
    ├── probabilistic_metrics.png      # Biểu đồ metrics xác suất
    ├── financial_metrics.png          # Biểu đồ metrics tài chính
    └── radar_comparison.png           # Radar chart tổng thể
```

## 📈 Metrics Được Tính Toán

### Probabilistic Metrics:
- **NLL**: Negative Log-Likelihood (càng thấp càng tốt)
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **IC**: Information Coefficient (Pearson correlation)
- **RIC**: Rank Information Coefficient (Spearman correlation)
- **CRPS**: Continuous Ranked Probability Score
- **Sharpness**: Average prediction uncertainty
- **PICP90/95**: Prediction Interval Coverage Probability
- **AURC**: Area Under Risk-Coverage curve

### Financial Metrics:
- **Directional Accuracy**: Tỷ lệ dự đoán đúng hướng
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Mức sụt giảm tối đa
- **Calmar Ratio**: Return/max drawdown
- **Information Ratio**: Active return/tracking error
- **Hit Rate**: Tỷ lệ giao dịch có lãi
- **Tail Ratio**: Upside/downside tail ratio
- **Total/Net Return**: Lợi nhuận trước/sau chi phí giao dịch

### Advanced Analysis:
- **Market Regime Analysis**: Hiệu suất trong các điều kiện thị trường khác nhau
- **Stress Testing**: Hiệu suất trong các giai đoạn khủng hoảng
- **Rolling Window Evaluation**: Hiệu suất theo thời gian

## 🔍 Ví Dụ Workflow Hoàn Chỉnh

### Train và so sánh cho dataset IXIC:

```bash
# Bước 1: Train tất cả models
python baseline/train_all_baselines.py \
    --dataset IXIC \
    --epochs 50 \
    --early_stop \
    --early_stop_patience 10

# Bước 2: So sánh kết quả
python baseline/compare_baselines.py --dataset IXIC

# Bước 3: Xem kết quả
cat logs/baselines/comparison/comparison_report.txt
```

### Train cho nhiều datasets:

```bash
# Train cho IXIC
python baseline/train_all_baselines.py --dataset IXIC --epochs 50

# Train cho DJI
python baseline/train_all_baselines.py --dataset DJI --epochs 50

# Train cho NYSE
python baseline/train_all_baselines.py --dataset NYSE --epochs 50

# So sánh từng dataset
python baseline/compare_baselines.py --dataset IXIC
python baseline/compare_baselines.py --dataset DJI
python baseline/compare_baselines.py --dataset NYSE
```

### Train nhanh để test (epochs ít hơn):

```bash
python baseline/train_all_baselines.py \
    --dataset IXIC \
    --epochs 10 \
    --models Linear LSTM \
    --batch_size 64
```

## 💡 Tips

1. **Early Stopping**: Luôn sử dụng `--early_stop` để tránh overfitting và tiết kiệm thời gian
2. **Batch Size**: Tăng batch size nếu có đủ GPU memory để training nhanh hơn
3. **Learning Rate**: Nếu model không converge, thử giảm learning rate xuống 0.0005 hoặc 0.0001
4. **Hidden Dim**: Models phức tạp hơn (Transformer, AGCRN) có thể cần hidden_dim lớn hơn (128 hoặc 256)
5. **Window Size**: Thử nghiệm với window=3, 5, 10, 20 để tìm cấu hình tốt nhất

## 🐛 Troubleshooting

### Lỗi "No module named 'utils.data_processing'":
```bash
# Kiểm tra file data_processing.py có trong utils/ không
ls -la utils/data_processing.py
```

### Lỗi CUDA out of memory:
```bash
# Giảm batch size hoặc hidden dim
python train_all_baselines.py --batch_size 16 --hidden_dim 32
```

### Training quá lâu:
```bash
# Giảm epochs hoặc sử dụng early stopping
python train_all_baselines.py --epochs 30 --early_stop --early_stop_patience 5
```

## 📝 Cấu Trúc Code

### Main Components:

1. **baseline/baseline_models.py**: Định nghĩa 5 baseline models
   - `LinearBaseline`: Simple linear model
   - `LSTMBaseline`: LSTM-based model
   - `TransformerBaseline`: Transformer encoder
   - `AGCRNBaseline`: Adaptive Graph Convolution RNN
   - `TemporalGNBaseline`: Temporal Graph Network

2. **utils/data_processing.py**: Data loading và preprocessing
   - Load CSV data
   - Train/val/test split (80/5/15)
   - Feature normalization (MinMax scaler fit on train only)
   - Create PyTorch DataLoaders

3. **utils/trainer.py**: Training và evaluation framework
   - Training loop với early stopping
   - Comprehensive metrics calculation
   - Market regime analysis
   - Stress testing
   - Rolling window evaluation
   - Automatic CSV logging

4. **baseline/train_all_baselines.py**: Script train tất cả models
   - Load data
   - Train từng model
   - Save metrics

5. **baseline/compare_baselines.py**: Script so sánh kết quả
   - Load metrics từ tất cả models
   - Tạo comparison tables
   - Generate visualizations
   - Create reports

## 🎯 Expected Results

Thời gian training (trên CPU, 50 epochs):
- **Linear**: ~2-3 phút
- **LSTM**: ~5-10 phút
- **Transformer**: ~8-15 phút
- **AGCRN**: ~10-20 phút
- **TemporalGN**: ~8-15 phút

Performance ranking (thường thấy):
1. **LSTM** hoặc **Transformer**: Tốt nhất cho temporal patterns
2. **AGCRN** hoặc **TemporalGN**: Tốt nếu có graph structure
3. **Linear**: Baseline đơn giản

## 📧 Support

Nếu gặp vấn đề, kiểm tra:
1. Python version >= 3.8
2. Tất cả dependencies được cài đặt: `pip install -r requirements.txt`
3. Dataset files tồn tại trong `Dataset/`
4. Logs directory có quyền write

---

**Happy Training!** 🚀
