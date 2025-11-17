#  >>> DATA PROCESSING <<<
# ===========================================================================
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# ---------- Min‑Max scaler (fit on TRAIN features only) ----------
class MinMax01:
    def fit(self, x):
        self.min = x.min(0)
        self.max = x.max(0)
    def transform(self, x):
        return (x - self.min) / (self.max - self.min + 1e-8)
# ---------- DataLoaders ----------
def make_loader(x, y, batch_size):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=False, drop_last=False)

def data_processing(data_path, window, batch_size):
    """
    Process financial data with proper train/val/test splits and feature normalization
    
    Args:
        data_path: path to CSV file with Date index and price in first column
        window: lookback window size L
        batch_size: batch size for dataloaders
        
    Returns:
        num_features: number of features (excluding price)
        train_loader, val_loader, test_loader: PyTorch dataloaders
    """
    # Load and prepare data
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True) 
    _ = df.pop('Name')

    # 1. REMOVE HIGHLY LEAKY FEATURES
    leaky_features = []#['mom','mom1','mom2','mom3', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20']
    features_removed = []
    for feat in leaky_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])
            features_removed.append(feat)
    print(f"   ❌ Removed {len(features_removed)} highly leaky features: {features_removed}")

    print(" Data shape:", df.shape)
    
    print("NaN distribution:\n", df.isnull().sum())
    df.fillna(df.median(), inplace=True)
    df.dropna(inplace=True)
    print(" FILLNA BY MEDIAN - Data shape:", df.shape)
    
    # Split data into price and features
    raw_np = df.values.astype('float32')
    prices = raw_np[:, 0]     # Price column
    features = raw_np[:, 1:]  # Feature columns
    
    T, Fdim = raw_np.shape
    assert Fdim >= 2, 'Need price + at least 1 feature'
    
    # Calculate initial split lengths
    train_len = int(0.80 * (T - window))
    val_len = int(0.05 * (T - window))
    
    # Prepare and fit scaler on training features only
    train_feat_matrix = []
    for i in range(train_len):
        train_feat_matrix.append(features[i:i+window])
    train_feat_matrix = np.concatenate(train_feat_matrix, axis=0)
    
    scaler = MinMax01()
    scaler.fit(train_feat_matrix)
    
    # Build samples with proper temporal alignment
    X_list, Y_list = [], []
    for i in range(T - window):
        # Calculate target return using t and t-1 prices
        price_prev = prices[i+window-1]  # Price at t-1
        price_cur = prices[i+window]     # Price at t
        ret = (price_cur - price_prev) / price_prev
        ret = torch.FloatTensor([[ret]])  # Shape: (1,1)
        
        # Get feature window from t-window to t-1 (no lookahead)
        feat_block = scaler.transform(features[i:i+window])
        feat_block = torch.FloatTensor(feat_block)  # Shape: (L,N)
        
        X_list.append(feat_block)
        Y_list.append(ret)
    
    # Stack all samples
    XX = torch.stack(X_list)  # Shape: (num_samples, L, N)
    YY = torch.stack(Y_list)  # Shape: (num_samples, 1, 1)
    
    # Split ensuring no temporal overlap
    num_samples = len(XX)
    train_len = int(0.80 * num_samples)
    val_len = int(0.05 * num_samples)
    test_len = num_samples - train_len - val_len
    
    # Create train/val/test splits in chronological order
    X_train, Y_train = XX[:train_len], YY[:train_len]
    X_val, Y_val = XX[train_len:train_len+val_len], YY[train_len:train_len+val_len]
    X_test, Y_test = XX[-test_len:], YY[-test_len:]
    
    # Create data loaders
    train_loader = make_loader(X_train, Y_train, batch_size)
    val_loader = make_loader(X_val, Y_val, batch_size)
    test_loader = make_loader(X_test, Y_test, batch_size)

    return features.shape[1], train_loader, val_loader, test_loader