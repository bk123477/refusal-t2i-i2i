"""
VLM (Vision-Language Model) backends for ACRB.

Provides Qwen VLM for attribute detection and cue retention scoring.
"""

from .qwen_backend import QwenVLMBackend, AttributeDetectionResult

__all__ = ["QwenVLMBackend", "AttributeDetectionResult"]
