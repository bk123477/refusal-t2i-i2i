"""Metrics modules for refusal and cue erasure evaluation.

Enhanced with IJCAI reviewer feedback:
- Per-model CLIP threshold calibration
- Multi-VLM ensemble for soft-refusal detection
- Bonferroni correction and bootstrap CI
- Threshold sensitivity analysis
"""

from .refusal_detector import (
    RefusalDetector,
    RefusalResult,
    RefusalType,
    MODEL_CLIP_THRESHOLDS,
    PROVIDER_REFUSAL_SIGNALS,
    threshold_sensitivity_analysis,
    calibrate_threshold_for_model,
)
from .cue_retention_scorer import (
    CueRetentionScorer,
    CueRetentionResult,
    JudgmentAgreement,
    DEFAULT_CONFIDENCE_THRESHOLD,
    FAIRJUDGE_SYSTEM_PROMPT,
    compute_judge_stability,
)
from .erasure_calculator import ErasureCalculator, ErasureStats
from .disparity_metric import (
    DisparityMetric,
    DisparityResult,
    interpret_effect_size,
    EFFECT_SIZE_THRESHOLDS,
)

__all__ = [
    # Refusal Detection
    "RefusalDetector",
    "RefusalResult",
    "RefusalType",
    "MODEL_CLIP_THRESHOLDS",
    "PROVIDER_REFUSAL_SIGNALS",
    "threshold_sensitivity_analysis",
    "calibrate_threshold_for_model",
    # Cue Retention / Soft Refusal
    "CueRetentionScorer",
    "CueRetentionResult",
    "JudgmentAgreement",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "FAIRJUDGE_SYSTEM_PROMPT",
    "compute_judge_stability",
    # Erasure Calculator
    "ErasureCalculator",
    "ErasureStats",
    # Disparity Metrics
    "DisparityMetric",
    "DisparityResult",
    "interpret_effect_size",
    "EFFECT_SIZE_THRESHOLDS",
]
