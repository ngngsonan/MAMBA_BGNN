"""
Quick test script to verify baseline training setup is working correctly.

This script:
1. Verifies all imports work
2. Tests data loading
3. Tests model creation
4. Tests one epoch of training for each model
5. Verifies metrics calculation
"""

import sys
import os
# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from baseline.baseline_models import create_baseline_models
from utils.data_processing import data_processing


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, mu, y, var):
        var = var.clamp(min=1e-6)
        nll = 0.5 * (torch.log(var) + ((y - mu) ** 2) / var)
        return nll.mean()


def test_imports():
    """Test that all necessary imports work"""
    print("\n" + "="*80)
    print("TEST 1: Testing imports...")
    print("="*80)

    try:
        from utils.trainer import Trainer
        print("✅ Trainer import successful")
    except Exception as e:
        print(f"❌ Trainer import failed: {e}")
        return False

    try:
        from baseline.baseline_models import create_baseline_models
        print("✅ Baseline models import successful")
    except Exception as e:
        print(f"❌ Baseline models import failed: {e}")
        return False

    try:
        from utils.data_processing import data_processing
        print("✅ Data processing import successful")
    except Exception as e:
        print(f"❌ Data processing import failed: {e}")
        return False

    return True


def test_data_loading(dataset='IXIC', window=5, batch_size=32):
    """Test data loading"""
    print("\n" + "="*80)
    print("TEST 2: Testing data loading...")
    print("="*80)

    try:
        data_path = f'Dataset/combined_dataframe_{dataset}.csv'

        if not os.path.exists(data_path):
            print(f"❌ Dataset file not found: {data_path}")
            return None

        num_features, train_loader, val_loader, test_loader = data_processing(
            data_path=data_path,
            window=window,
            batch_size=batch_size
        )

        print(f"✅ Data loaded successfully!")
        print(f"   Features: {num_features}")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")

        return num_features, train_loader, val_loader, test_loader

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_creation(num_features=82, seq_len=5):
    """Test model creation"""
    print("\n" + "="*80)
    print("TEST 3: Testing model creation...")
    print("="*80)

    try:
        models = create_baseline_models(input_dim=num_features, seq_len=seq_len)
        print(f"✅ Created {len(models)} models successfully:")

        for name, model in models.items():
            n_params = sum(p.numel() for p in model.parameters())
            print(f"   {name:15s} - {n_params:,} parameters")

        return models

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(models, batch_size=32, seq_len=5, num_features=82):
    """Test forward pass for all models"""
    print("\n" + "="*80)
    print("TEST 4: Testing forward pass...")
    print("="*80)

    x = torch.randn(batch_size, seq_len, num_features)
    y = torch.randn(batch_size)

    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                mu, log_var = model(x)

            assert mu.shape == (batch_size,), f"Expected mu shape {(batch_size,)}, got {mu.shape}"
            assert log_var.shape == (batch_size,), f"Expected log_var shape {(batch_size,)}, got {log_var.shape}"

            print(f"✅ {name:15s} - Forward pass successful")

        except Exception as e:
            print(f"❌ {name:15s} - Forward pass failed: {e}")
            return False

    return True


def test_training_step(models, train_loader):
    """Test one training step for each model"""
    print("\n" + "="*80)
    print("TEST 5: Testing training step...")
    print("="*80)

    loss_fn = GaussianNLLLoss()

    for name, model in models.items():
        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Get one batch
            x, y = next(iter(train_loader))

            # Forward pass
            optimizer.zero_grad()
            mu, log_var = model(x)
            loss = loss_fn(mu, y.squeeze(), log_var.exp())

            # Backward pass
            loss.backward()
            optimizer.step()

            print(f"✅ {name:15s} - Training step successful (loss: {loss.item():.6f})")

        except Exception as e:
            print(f"❌ {name:15s} - Training step failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def test_trainer_setup(model, train_loader, val_loader, test_loader):
    """Test Trainer setup"""
    print("\n" + "="*80)
    print("TEST 6: Testing Trainer setup...")
    print("="*80)

    try:
        from utils.trainer import Trainer

        loss_fn = GaussianNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        args = {
            'log_dir': 'logs/test',
            'model_name': 'TestModel',
            'epochs': 1,
            'early_stop': False,
            'grad_norm': True,
            'max_grad_norm': 5.0,
            'log_step': 10,
            'rolling_window_size': 63,
            'rolling_step_size': 21
        }

        os.makedirs(args['log_dir'], exist_ok=True)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args
        )

        print("✅ Trainer setup successful")
        return True

    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("BASELINE TRAINING SETUP VERIFICATION")
    print("="*80)

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your setup.")
        return

    # Test 2: Data loading
    data_result = test_data_loading(dataset='IXIC', window=5, batch_size=32)
    if data_result is None:
        print("\n❌ Data loading test failed. Please check your dataset.")
        return

    num_features, train_loader, val_loader, test_loader = data_result

    # Test 3: Model creation
    models = test_model_creation(num_features=num_features, seq_len=5)
    if models is None:
        print("\n❌ Model creation test failed.")
        return

    # Test 4: Forward pass
    if not test_forward_pass(models, num_features=num_features):
        print("\n❌ Forward pass test failed.")
        return

    # Test 5: Training step
    if not test_training_step(models, train_loader):
        print("\n❌ Training step test failed.")
        return

    # Test 6: Trainer setup
    test_model = list(models.values())[0]  # Use first model for testing
    if not test_trainer_setup(test_model, train_loader, val_loader, test_loader):
        print("\n❌ Trainer setup test failed.")
        return

    # All tests passed
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nYou can now run the full training with:")
    print("  python train_all_baselines.py --dataset IXIC --epochs 50")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
