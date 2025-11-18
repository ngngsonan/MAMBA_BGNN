"""
Utilities package for MAMBA-BGNN

Modules:
    - baseline_trainer: Training and evaluation utilities
    - result_plot: Plotting and visualization
    - data_processing: Data loading and preprocessing
"""

from .baseline_trainer import (
    train_models,
    regime_analysis,
    compute_financial_metrics,
    compare_bayesian_vs_nonbayesian,
    calculate_cross_sectional_for_all_models
)

from .result_plot import (
    plot_analytics,
    plot_bayesian_vs_nonbayesian_comparison,
    generate_comparison_report
)

__all__ = [
    'train_models',
    'regime_analysis',
    'compute_financial_metrics',
    'compare_bayesian_vs_nonbayesian',
    'calculate_cross_sectional_for_all_models',
    'plot_analytics',
    'plot_bayesian_vs_nonbayesian_comparison',
    'generate_comparison_report'
]
