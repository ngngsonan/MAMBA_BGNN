"""
Baseline Models Package

This package contains baseline models and utilities for training and comparison.

Modules:
- baseline_models: Implementation of all baseline models
- train_all_baselines: Script to train all baseline models
- compare_baselines: Script to compare baseline model results
- test_baseline_setup: Verification script for setup
"""

from .baseline_models import (
    BaselineModel,
    LinearBaseline,
    LSTMBaseline,
    TransformerBaseline,
    AGCRNBaseline,
    TemporalGNBaseline,
    create_baseline_models
)

__all__ = [
    'BaselineModel',
    'LinearBaseline',
    'LSTMBaseline',
    'TransformerBaseline',
    'AGCRNBaseline',
    'TemporalGNBaseline',
    'create_baseline_models'
]
