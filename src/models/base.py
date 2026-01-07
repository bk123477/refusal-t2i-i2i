"""
Base classes for I2I model wrappers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image
from enum import Enum


class RefusalType(Enum):
    """Types of model refusal."""
    NONE = "none"  # No refusal, image generated
    HARD = "hard"  # Explicit refusal (API error, policy message)
    SOFT = "soft"  # Generated but attribute missing/erased
    EMPTY = "empty"  # Empty or unchanged image


@dataclass
class EditResult:
    """Result of an I2I edit request."""
    success: bool
    refusal_type: RefusalType
    output_image: Optional[Image.Image] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[dict] = None
    latency_ms: float = 0.0

    @property
    def is_refused(self) -> bool:
        return self.refusal_type != RefusalType.NONE


class I2IModel(ABC):
    """Abstract base class for I2I models."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._loaded = False

    @abstractmethod
    def load(self):
        """Load model weights."""
        pass

    @abstractmethod
    def edit(
        self,
        source_image: Image.Image,
        prompt: str,
        **kwargs
    ) -> EditResult:
        """
        Apply edit prompt to source image.

        Args:
            source_image: PIL Image to edit
            prompt: Edit instruction
            **kwargs: Model-specific parameters

        Returns:
            EditResult with output image or refusal info
        """
        pass

    def batch_edit(
        self,
        source_images: list[Image.Image],
        prompts: list[str],
        **kwargs
    ) -> list[EditResult]:
        """Batch edit multiple images with same or different prompts."""
        results = []
        for img, prompt in zip(source_images, prompts):
            result = self.edit(img, prompt, **kwargs)
            results.append(result)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded
