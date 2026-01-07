"""
FLUX.2 [dev] I2I Model Wrapper
Provider: Black Forest Labs
Model: https://huggingface.co/black-forest-labs/FLUX.2-dev

Supports 4-bit quantization and remote text encoder for consumer GPUs (RTX 4090/5090).
"""

import time
from typing import Optional, List, Union
from PIL import Image

from .base import I2IModel, EditResult, RefusalType


class FluxWrapper(I2IModel):
    """
    Wrapper for FLUX.2-dev image editing model.

    HuggingFace: https://huggingface.co/black-forest-labs/FLUX.2-dev

    Supports:
    - Full precision (requires high VRAM)
    - 4-bit quantization for consumer GPUs
    - Remote text encoder to reduce memory

    Setup:
        pip install git+https://github.com/huggingface/diffusers
    """

    MODEL_ID = "black-forest-labs/FLUX.2-dev"
    MODEL_ID_QUANTIZED = "diffusers/FLUX.2-dev-bnb-4bit"
    MODEL_URL = "https://huggingface.co/black-forest-labs/FLUX.2-dev"
    REMOTE_TEXT_ENCODER_URL = "https://remote-text-encoder-flux-2.huggingface.co/predict"

    def __init__(
        self,
        device: str = "cuda",
        use_quantized: bool = False,
        use_remote_text_encoder: bool = False
    ):
        super().__init__(model_name="FLUX.2-dev", device=device)
        self.pipe = None
        self.use_quantized = use_quantized
        self.use_remote_text_encoder = use_remote_text_encoder

    def load(self):
        """Load FLUX.2 model."""
        try:
            from diffusers import Flux2Pipeline
            import torch

            print(f"Loading FLUX.2-dev on {self.device}...")
            print(f"Model URL: {self.MODEL_URL}")
            print(f"Quantized: {self.use_quantized}")
            print(f"Remote text encoder: {self.use_remote_text_encoder}")

            model_id = self.MODEL_ID_QUANTIZED if self.use_quantized else self.MODEL_ID

            if self.use_remote_text_encoder:
                # Load without text encoder (will use remote)
                self.pipe = Flux2Pipeline.from_pretrained(
                    model_id,
                    tt_encoder=None,
                    torch_dtype=torch.bfloat16
                )
            else:
                self.pipe = Flux2Pipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16
                )

            self.pipe = self.pipe.to(self.device)
            self._loaded = True
            print("FLUX.2-dev loaded successfully")

        except ImportError as e:
            print(f"Failed to import Flux2Pipeline: {e}")
            print("\nPlease install latest diffusers:")
            print("  pip install git+https://github.com/huggingface/diffusers")
            raise

        except Exception as e:
            print(f"Failed to load FLUX.2-dev: {e}")
            raise

    def _get_remote_text_embeddings(self, prompt: str):
        """Get text embeddings from remote encoder."""
        import requests
        import io
        import torch
        from huggingface_hub import get_token

        response = requests.post(
            self.REMOTE_TEXT_ENCODER_URL,
            json={"prompt": [prompt]},
            headers={
                "Authorization": f"Bearer {get_token()}",
                "Content-Type": "application/json"
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"Remote text encoder failed: {response.text}")

        prompt_embeds = torch.load(io.BytesIO(response.content))
        return prompt_embeds.to(self.device)

    def edit(
        self,
        source_image: Union[Image.Image, List[Image.Image]],
        prompt: str,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        **kwargs
    ) -> EditResult:
        """
        Apply edit to image using FLUX.2.

        Args:
            source_image: Input image(s) for context
            prompt: Edit instruction
            guidance_scale: Guidance scale (default 4.0)
            num_inference_steps: Number of denoising steps (28 is good trade-off)
            seed: Random seed (optional)
        """
        if not self._loaded:
            self.load()

        import torch
        start_time = time.time()

        try:
            # Prepare input images
            if isinstance(source_image, Image.Image):
                images = [source_image.convert("RGB")]
            else:
                images = [img.convert("RGB") for img in source_image]

            # Setup generator
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = torch.Generator(device=self.device).manual_seed(42)

            # Prepare kwargs
            gen_kwargs = {
                "image": images,
                "generator": generator,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }

            # Use remote text encoder or local
            if self.use_remote_text_encoder:
                gen_kwargs["prompt_embeds"] = self._get_remote_text_embeddings(prompt)
            else:
                gen_kwargs["prompt"] = prompt

            # Generate
            output = self.pipe(**gen_kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Check for NSFW filtering
            if hasattr(output, 'nsfw_content_detected') and output.nsfw_content_detected:
                if output.nsfw_content_detected[0]:
                    return EditResult(
                        success=False,
                        refusal_type=RefusalType.HARD,
                        error_message="NSFW content detected - safety filter triggered",
                        latency_ms=latency_ms
                    )

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

            if any(kw in error_msg for kw in ["safety", "policy", "inappropriate", "blocked", "nsfw"]):
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
