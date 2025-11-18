"""
Data Processing Utilities for MAMBA-BGNN

Provides data loading and preprocessing for financial time series:
    - CSV loading
    - Train/Val/Test split
    - PyTorch Dataset & DataLoader creation
    - Feature normalization

Usage:
    from utils.data_processing import data_processing

    num_features, train_loader, val_loader, test_loader = data_processing(
        'Dataset/combined_dataframe_IXIC.csv',
        window=5,
        batch_size=32
    )
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """
    Time series dataset for financial prediction

    Args:
        data: (T, N+1) numpy array - features + target
        window: Lookback window size
        normalize: Whether to normalize features
    """

    def __init__(self, data: np.ndarray, window: int = 5, normalize: bool = False):
        self.data = torch.FloatTensor(data)
        self.window = window
        self.normalize = normalize

        if normalize:
            # Normalize features (not target)
            features = data[:, :-1]
            target = data[:, -1:]

            scaler = StandardScaler()
            features_norm = scaler.fit_transform(features)

            data_norm = np.concatenate([features_norm, target], axis=1)
            self.data = torch.FloatTensor(data_norm)

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        """
        Returns:
            x: (window, num_features) - historical features
            y: scalar - target return
        """
        x = self.data[idx:idx + self.window, :-1]  # Features
        y = self.data[idx + self.window, -1]       # Target

        return x, y


def data_processing(
    data_path: str,
    window: int = 5,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    normalize: bool = False,
    shuffle_train: bool = True
) -> Tuple[int, DataLoader, DataLoader, DataLoader]:
    """
    Process financial time series data

    Args:
        data_path: Path to CSV file
        window: Lookback window size (L)
        batch_size: Batch size for dataloaders
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        normalize: Whether to normalize features
        shuffle_train: Whether to shuffle training data

    Returns:
        num_features: Number of features (N)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
    """

    # Load CSV
    df = pd.read_csv(data_path)

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Convert to numpy
    data = df.values  # (T, N+1) where last column is target

    # Split
    n = len(data)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, window, normalize)
    val_dataset = TimeSeriesDataset(val_data, window, normalize)
    test_dataset = TimeSeriesDataset(test_data, window, normalize)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    num_features = data.shape[1] - 1  # Exclude target

    return num_features, train_loader, val_loader, test_loader


def load_multiple_datasets(
    datasets: list,
    window: int = 5,
    batch_size: int = 32
) -> dict:
    """
    Load multiple datasets at once

    Args:
        datasets: List of dataset names (e.g., ['IXIC', 'DJI', 'NYSE'])
        window: Lookback window
        batch_size: Batch size

    Returns:
        data_dict: {dataset_name: (num_features, train_loader, val_loader, test_loader)}
    """
    data_dict = {}

    for dataset in datasets:
        data_path = f'Dataset/combined_dataframe_{dataset}.csv'

        try:
            num_features, train_loader, val_loader, test_loader = data_processing(
                data_path, window, batch_size
            )

            data_dict[dataset] = {
                'num_features': num_features,
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader
            }

            print(f"✓ Loaded {dataset}: {num_features} features, "
                  f"{len(train_loader.dataset)} train samples")

        except Exception as e:
            print(f"✗ Error loading {dataset}: {e}")

    return data_dict


# ============================================================================
# Cross-Sectional Data Processing (for multi-stock scenarios)
# ============================================================================

class CrossSectionalDataset(Dataset):
    """
    Cross-sectional dataset for multiple stocks

    Args:
        data: (T, N, F) - Time × Stocks × Features
        targets: (T, N) - Time × Stocks targets
        window: Lookback window
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray, window: int = 5):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        """
        Returns:
            x: (window, num_stocks, num_features)
            y: (num_stocks,) - target returns for all stocks
        """
        x = self.data[idx:idx + self.window]
        y = self.targets[idx + self.window]

        return x, y


def cross_sectional_data_processing(
    data_path: str,
    window: int = 5,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[int, int, DataLoader, DataLoader, DataLoader]:
    """
    Process cross-sectional data (multiple stocks)

    Args:
        data_path: Path to data file
        window: Lookback window
        batch_size: Batch size
        train_ratio: Train set ratio
        val_ratio: Val set ratio

    Returns:
        num_stocks: Number of stocks
        num_features: Number of features per stock
        train_loader, val_loader, test_loader
    """

    # Load data (placeholder - adjust based on actual data format)
    # Expected format: (T, N*F) where N is num_stocks, F is features per stock
    df = pd.read_csv(data_path)
    data = df.values

    # Placeholder: Assume single stock for now
    # In multi-stock scenario, reshape appropriately
    num_stocks = 1
    num_features = data.shape[1] - 1

    # For now, use same processing as regular time series
    _, train_loader, val_loader, test_loader = data_processing(
        data_path, window, batch_size, train_ratio, val_ratio
    )

    return num_stocks, num_features, train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data processing
    print("Testing data processing...")

    try:
        num_features, train_loader, val_loader, test_loader = data_processing(
            'Dataset/combined_dataframe_IXIC.csv',
            window=5,
            batch_size=32
        )

        print(f"✓ Data processing successful!")
        print(f"  Features: {num_features}")
        print(f"  Train: {len(train_loader.dataset)} samples")
        print(f"  Val: {len(val_loader.dataset)} samples")
        print(f"  Test: {len(test_loader.dataset)} samples")

        # Test batch
        for batch_x, batch_y in train_loader:
            print(f"  Batch shape: x={batch_x.shape}, y={batch_y.shape}")
            break

    except Exception as e:
        print(f"✗ Error: {e}")
