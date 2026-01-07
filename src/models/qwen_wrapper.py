"""
Qwen-Image-Edit-2511 I2I Model Wrapper
Provider: Alibaba/Qwen
Model: https://huggingface.co/Qwen/Qwen-Image-Edit-2511

Enhanced version with better consistency, integrated LoRA, improved character consistency.
"""

import time
from typing import Optional, List, Union
from PIL import Image

from .base import I2IModel, EditResult, RefusalType


class QwenImageEditWrapper(I2IModel):
    """
    Wrapper for Qwen-Image-Edit-2511 model.

    HuggingFace: https://huggingface.co/Qwen/Qwen-Image-Edit-2511

    Key features:
    - Mitigated image drift
    - Improved character consistency
    - Integrated LoRA capabilities
    - Enhanced industrial design generation
    - Strengthened geometric reasoning

    Setup:
        pip install git+https://github.com/huggingface/diffusers
    """

    MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
    MODEL_URL = "https://huggingface.co/Qwen/Qwen-Image-Edit-2511"

    def __init__(self, device: str = "cuda"):
        super().__init__(model_name="Qwen-Image-Edit-2511", device=device)
        self.pipe = None

    def load(self):
        """Load Qwen-Image-Edit-2511 model."""
        try:
            from diffusers import QwenImageEditPlusPipeline
            import torch

            print(f"Loading Qwen-Image-Edit-2511 on {self.device}...")
            print(f"Model URL: {self.MODEL_URL}")

            self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16
            )
            self.pipe = self.pipe.to(self.device)
            self.pipe.set_progress_bar_config(disable=None)

            self._loaded = True
            print("Qwen-Image-Edit-2511 loaded successfully")

        except ImportError as e:
            print(f"Failed to import QwenImageEditPlusPipeline: {e}")
            print("\nPlease install latest diffusers:")
            print("  pip install git+https://github.com/huggingface/diffusers")
            raise

        except Exception as e:
            print(f"Failed to load Qwen-Image-Edit-2511: {e}")
            raise

    def edit(
        self,
        source_image: Union[Image.Image, List[Image.Image]],
        prompt: str,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 40,
        negative_prompt: str = " ",
        seed: Optional[int] = None,
        **kwargs
    ) -> EditResult:
        """
        Apply edit to image using Qwen-Image-Edit-2511.

        Args:
            source_image: Input image(s) - supports multi-image input
            prompt: Edit instruction
            true_cfg_scale: True CFG scale (default 4.0)
            guidance_scale: Guidance scale (default 1.0)
            num_inference_steps: Number of denoising steps
            negative_prompt: Negative prompt
            seed: Random seed (optional)
        """
        if not self._loaded:
            self.load()

        import torch
        start_time = time.time()

        try:
            # Handle single or multiple images
            if isinstance(source_image, Image.Image):
                images = [source_image.convert("RGB")]
            else:
                images = [img.convert("RGB") for img in source_image]

            # Setup generator
            if seed is not None:
                generator = torch.manual_seed(seed)
            else:
                generator = torch.manual_seed(0)

            # Run inference
            inputs = {
                "image": images,
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():
                output = self.pipe(**inputs)

            latency_ms = (time.time() - start_time) * 1000
            output_image = output.images[0]

            return EditResult(
                success=True,
                refusal_type=RefusalType.NONE,
                output_image=output_image,
                latency_ms=latency_ms
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e).lower()

            # Qwen may have Chinese error messages
            if any(kw in error_msg for kw in [
                "safety", "policy", "inappropriate", "blocked",
                "nsfw", "违规", "不当", "敏感", "禁止"
            ]):
                return EditResult(
                    success=False,
                    refusal_type=RefusalType.HARD,
                    error_message=str(e),
                    latency_ms=latency_ms
                )

            return EditResult(
                success=False,
                refusal_type=RefusalType.HARD,
                error_message=str(e),
                latency_ms=latency_ms
            )
