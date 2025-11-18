"""
Quick test script to verify GNN models can be instantiated and run forward pass

Usage:
    python test_gnn_models.py
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamba_gnn_study import (
    ModelArgs,
    BIMamba_GCN,
    BIMamba_GAT,
    BIMamba_GraphSAGE,
    BIMamba_MAGAC
)


def test_model(model_class, model_name, args, **kwargs):
    """Test a single model"""
    print(f"\nTesting {model_name}...")
    print("-" * 60)

    try:
        # Initialize model
        model = model_class(args, **kwargs)
        print(f"✓ Model initialized")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Create dummy input
        batch_size = 4
        L = args.seq_len
        N = args.d_model
        x = torch.randn(batch_size, L, N)
        print(f"✓ Input created: {x.shape}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            mu, log_var = model(x)

        print(f"✓ Forward pass successful")
        print(f"  Output shapes: mu={mu.shape}, log_var={log_var.shape}")
        print(f"  mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
        print(f"  log_var range: [{log_var.min().item():.4f}, {log_var.max().item():.4f}]")

        # Check for NaNs
        if torch.isnan(mu).any() or torch.isnan(log_var).any():
            print("⚠ Warning: NaN detected in outputs")
            return False

        print(f"✓ {model_name} test PASSED")
        return True

    except Exception as e:
        print(f"✗ {model_name} test FAILED")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("GNN MODELS UNIT TEST")
    print("="*60)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Model arguments (small size for testing)
    num_features = 20  # Simulating 20 stocks
    seq_len = 5       # 5-day lookback
    args = ModelArgs(
        d_model=num_features,
        seq_len=seq_len,
        d_proj_E=32,      # Smaller for testing
        d_proj_H=32,
        d_proj_U=16,
        d_state=32
    )

    print(f"\nModel Args:")
    print(f"  d_model (N): {args.d_model}")
    print(f"  seq_len (L): {args.seq_len}")
    print(f"  d_proj_E: {args.d_proj_E}")

    # Test each model
    results = {}

    # 1. BIMamba + GCN
    results['GCN'] = test_model(
        BIMamba_GCN, "BIMamba+GCN", args,
        R=2, d_e=8
    )

    # 2. BIMamba + GAT
    results['GAT'] = test_model(
        BIMamba_GAT, "BIMamba+GAT", args,
        R=2, d_e=8, heads=2
    )

    # 3. BIMamba + GraphSAGE
    results['GraphSAGE'] = test_model(
        BIMamba_GraphSAGE, "BIMamba+GraphSAGE", args,
        R=2, d_e=8
    )

    # 4. BIMamba + MAGAC
    results['MAGAC'] = test_model(
        BIMamba_MAGAC, "BIMamba+MAGAC", args,
        R=2, K=2, d_e=8, heads=2
    )

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {model_name:12s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
