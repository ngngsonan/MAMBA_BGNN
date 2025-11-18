"""
MAMBA Models Package

This package contains MAMBA architecture implementations for financial forecasting:
- mamba_bgnn.py: Main MAMBA-BGNN model with graph components
- mamba_models.py: MAMBA ablation model variants (MAMBA, BIMAMBA, MAMBA+, BIMAMBA+)
- mamba_study.py: Convenient functions for running ablation studies

Usage:
    from models.mamba_bgnn import MAMBA_BayesMAGAC
    from models.mamba_models import MAMBAModel, BIMAMBAModel, MAMBAPlusModel, BIMAMBAPlusModel
    from models.mamba_study import run_full_ablation
"""

# Import main models for easy access
try:
    from .mamba_bgnn import (
        MAMBA_BayesMAGAC,
        ModelArgs,
        MambaBlock,
        BIMambaBlock,
        BayesianMAGAC
    )
except ImportError:
    pass  # Models may not be available yet

try:
    from .mamba_models import (
        MAMBAModel,
        BIMAMBAModel,
        MAMBAPlusModel,
        BIMAMBAPlusModel,
        train_ablation_models
    )
except ImportError:
    pass

try:
    from .mamba_study import (
        run_full_ablation,
        run_direction_ablation,
        run_architecture_ablation,
        run_quick_test
    )
except ImportError:
    pass

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = [
    # Main model
    'MAMBA_BayesMAGAC',
    'ModelArgs',
    'MambaBlock',
    'BIMambaBlock',
    'BayesianMAGAC',
    # Ablation models
    'MAMBAModel',
    'BIMAMBAModel',
    'MAMBAPlusModel',
    'BIMAMBAPlusModel',
    'train_ablation_models',
    # Study functions
    'run_full_ablation',
    'run_direction_ablation',
    'run_architecture_ablation',
    'run_quick_test',
]
