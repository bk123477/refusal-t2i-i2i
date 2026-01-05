"""
ACRB Preprocessing Module

Tools for preparing images and data for ACRB evaluation.

Includes:
- Visibility controls for I2I disability marker evaluation
- Image filtering and validation
- Dataset preprocessing utilities
"""

from .visibility import (
    filter_visible_regions,
    VisibilityResult,
    compute_visibility_score,
    DISABILITY_BODY_PARTS,
)

__all__ = [
    "filter_visible_regions",
    "VisibilityResult",
    "compute_visibility_score",
    "DISABILITY_BODY_PARTS",
]
