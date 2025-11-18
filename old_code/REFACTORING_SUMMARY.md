# Refactoring Summary - MAMBA_BGNN

## 📋 Tổng quan
Đã tái cấu trúc dự án MAMBA_BGNN để tách biệt code model và utilities, đồng thời thêm hỗ trợ multi-loss training.

## 🔄 Thay đổi chính

### 1. **Cấu trúc dự án mới**
```
MAMBA_BGNN/
├── mamba_bgnn.py          # Main file - chỉ chứa model definitions và main()
├── utils/                 # Utilities package
│   ├── __init__.py       # Package initialization
│   ├── data_processing.py # Data loading và preprocessing
│   ├── trainer.py        # Training framework với multi-loss support
│   └── result_plot.py    # Visualization utilities
├── Dataset/              # Data files
└── logs/                 # Training logs và results
```

### 2. **File mamba_bgnn.py - Giảm từ 1024 → 435 dòng**

#### ✅ Giữ lại:
- **Model Arguments** (`ModelArgs`)
- **Mamba Block** (`MambaBlock`)
- **FeedForward & Residual Blocks**
- **BI-Mamba Stack** (`BIMambaBlock`)
- **MAGAC** & **BayesianMAGAC**
- **MAMBA_BayesMAGAC** (top-level model)
- **main()** function

#### ❌ Di chuyển vào utils:
- `Trainer` class → `utils/trainer.py`
- `data_processing()` → `utils/data_processing.py`
- `MinMax01` scaler → `utils/data_processing.py`
- `make_loader()` → `utils/data_processing.py`

### 3. **utils/trainer.py - Tính năng mới**

#### 🆕 Multi-Loss Support
Trainer hiện hỗ trợ 3 loại loss functions:

1. **Bayesian Mode** (`loss_type='bayesian'`)
   - Loss: `nn.GaussianNLLLoss()`
   - Output: `(mu, log_var)` - Uncertainty quantification
   - Metrics: NLL, RMSE, MAE, IC, RIC, CRPS, Sharpness, PICP, AURC

2. **MSE Mode** (`loss_type='mse'`)
   - Loss: `nn.MSELoss()`
   - Output: `mu` only - Deterministic
   - Metrics: Loss, RMSE, MAE, IC, RIC

3. **SmoothL1 Mode** (`loss_type='smoothl1'`)
   - Loss: `nn.SmoothL1Loss()`
   - Output: `mu` only - Robust loss
   - Metrics: Loss, RMSE, MAE, IC, RIC

#### 📊 Adaptive CSV Headers
CSV headers tự động điều chỉnh theo `loss_type`:
- **Bayesian**: Đầy đủ probabilistic metrics
- **MSE/SmoothL1**: Chỉ deterministic metrics

### 4. **Cách sử dụng**

#### Import
```python
from utils.data_processing import data_processing
from utils.trainer import Trainer
from utils.result_plot import plot_analytics
```

#### Training với các loss types
```python
# 1. Bayesian mode (default)
main('IXIC', loss_type='bayesian')

# 2. MSE mode
main('IXIC', loss_type='mse')

# 3. SmoothL1 mode
main('IXIC', loss_type='smoothl1')
```

## 🎯 Lợi ích

### Code Organization
- ✅ Tách biệt model logic và training utilities
- ✅ Dễ maintain và extend
- ✅ Tái sử dụng code giữa các experiments

### Flexibility
- ✅ Dễ dàng switch giữa các loss functions
- ✅ Tự động điều chỉnh metrics theo loss type
- ✅ Không cần thay đổi model architecture

### Performance Testing
- ✅ So sánh Bayesian vs Deterministic approaches
- ✅ Test robust loss (SmoothL1) cho outliers
- ✅ Comprehensive evaluation framework

## 📝 Breaking Changes

### Imports cần cập nhật
```python
# Cũ (trong mamba_bgnn.py)
trainer = Trainer(...)  # Defined locally

# Mới (import từ utils)
from utils.trainer import Trainer
trainer = Trainer(..., loss_type='bayesian')
```

### Trainer initialization
```python
# Thêm parameter loss_type
trainer = Trainer(
    model, loss_fn, optimizer,
    train_loader, val_loader, test_loader,
    args=args,
    lr_scheduler=scheduler,
    loss_type='bayesian'  # NEW!
)
```

## 🧪 Testing

### Syntax Check
```bash
python -m py_compile mamba_bgnn.py
```

### Import Test
```python
from utils import Trainer, data_processing
print("✓ Imports successful")
```

### Training Test
```python
# Run with default Bayesian mode
python mamba_bgnn.py
```

## 📚 Dependencies
Xem `requirements.txt`:
- torch>=2.0.0
- numpy, pandas
- einops
- scipy, scikit-learn
- matplotlib, seaborn

## 🚀 Next Steps

1. **Test các loss types** trên datasets khác nhau
2. **So sánh performance** giữa Bayesian vs Deterministic
3. **Tune hyperparameters** cho từng loss type
4. **Document results** trong paper/report

## 📧 Support
Nếu có vấn đề, check:
1. Python version (>=3.8)
2. Dependencies installed (`pip install -r requirements.txt`)
3. CUDA availability (if using GPU)

---
*Last updated: 2025-11-17*
