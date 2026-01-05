"""
ACRB Analysis Module

Statistical analysis tools for ACRB evaluation results.

Includes:
- Threshold sensitivity analysis for fairness thresholds
- Mixed-effects modeling for controlling confounders
- Bootstrap confidence intervals
- Effect size calculations
"""

from .sensitivity import (
    threshold_sensitivity_analysis,
    disparity_stability_analysis,
    FairnessThresholdResult,
)
from .mixed_effects import (
    fit_mixed_effects_model,
    MixedEffectsResult,
    compute_icc,
)

__all__ = [
    # Sensitivity Analysis
    "threshold_sensitivity_analysis",
    "disparity_stability_analysis",
    "FairnessThresholdResult",
    # Mixed Effects Modeling
    "fit_mixed_effects_model",
    "MixedEffectsResult",
    "compute_icc",
]
