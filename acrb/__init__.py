"""
I2I-T2I-Bias-Refusal: Unified Framework for Attribute-Conditioned Refusal Bias Evaluation

Measures both hard refusal (explicit blocking) and soft refusal (cue erasure)
across Text-to-Image and Image-to-Image generative models.
"""

__version__ = "2.0.0"

from .evaluation import ACRBPipeline, EvaluationResult

__all__ = ["ACRBPipeline", "EvaluationResult"]
