"""
ACRB: Attribute-Conditioned Refusal Bias Framework

A comprehensive toolkit for auditing fairness in I2I generative models
by measuring attribute-conditioned refusal bias across demographic and cultural attributes.

Key Components:
- pipeline: Main ACRB pipeline (Algorithm 1 implementation)
- models: I2I model wrappers
- metrics: Hard refusal detection + soft refusal (cue erasure) scoring
- analysis: Sensitivity analysis and mixed-effects modeling
- preprocessing: I2I visibility controls
- utils: Data card generation for reproducibility
- config: Paper-aligned configuration

Quick Start:
    from acrb.pipeline import ACRBPipeline, ACRBConfig

    config = ACRBConfig(
        model_name="flux-2-dev",
        max_base_prompts=10,
        llm_model="gemini-3-flash-preview"
    )

    pipeline = ACRBPipeline(config)
    result = pipeline.run()

    print(f"Delta_refusal: {result.delta_refusal:.4f}")
    print(f"Delta_erasure: {result.delta_erasure:.4f}")
"""

__version__ = "0.2.0"
__author__ = "ACRB Research Team"

# Main pipeline (Algorithm 1)
from .pipeline import ACRBPipeline, ACRBConfig, ACRBResult

# Model wrappers
from .models import I2IModelWrapper

# Metrics
from .metrics import (
    RefusalDetector,
    CueRetentionScorer,
    DisparityMetric,
    ErasureCalculator,
    RefusalResult,
    CueRetentionResult,
    DisparityResult,
    ErasureStats,
    compute_judge_stability,
)

# Analysis modules
from .analysis import (
    threshold_sensitivity_analysis,
    disparity_stability_analysis,
    fit_mixed_effects_model,
    compute_icc,
    FairnessThresholdResult,
    MixedEffectsResult,
)

# Preprocessing
from .preprocessing import (
    filter_visible_regions,
    compute_visibility_score,
    VisibilityResult,
)

# Utilities
from .utils import (
    generate_data_card,
    DataCard,
    ExperimentMetadata,
)

# Configuration
from .config import (
    MODELS,
    DATASET_CONFIG,
    DOMAINS,
    ATTRIBUTES,
    REFUSAL_THRESHOLDS,
    VLM_CONFIG,
    get_model_config,
    get_refusal_threshold,
    list_supported_models,
)

__all__ = [
    # Pipeline
    "ACRBPipeline",
    "ACRBConfig",
    "ACRBResult",

    # Models
    "I2IModelWrapper",

    # Metrics
    "RefusalDetector",
    "CueRetentionScorer",
    "DisparityMetric",
    "ErasureCalculator",
    "RefusalResult",
    "CueRetentionResult",
    "DisparityResult",
    "ErasureStats",
    "compute_judge_stability",

    # Analysis
    "threshold_sensitivity_analysis",
    "disparity_stability_analysis",
    "fit_mixed_effects_model",
    "compute_icc",
    "FairnessThresholdResult",
    "MixedEffectsResult",

    # Preprocessing
    "filter_visible_regions",
    "compute_visibility_score",
    "VisibilityResult",

    # Utilities
    "generate_data_card",
    "DataCard",
    "ExperimentMetadata",

    # Configuration
    "MODELS",
    "DATASET_CONFIG",
    "DOMAINS",
    "ATTRIBUTES",
    "REFUSAL_THRESHOLDS",
    "VLM_CONFIG",
    "get_model_config",
    "get_refusal_threshold",
    "list_supported_models",
]
