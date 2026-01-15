"""
Step1X-Edit-v1p2 I2I Model Wrapper
Provider: StepFun
Model: https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2

Native reasoning edit model with thinking and reflection modes.
Requires custom diffusers branch: https://github.com/Peyton-Chen/diffusers (branch: step1xedit_v1p2)
"""

import time
from typing import Optional, List, Union
from PIL import Image

from .base import I2IModel, EditResult, RefusalType


class Step1XWrapper(I2IModel):
    """
    Wrapper for Step1X-Edit-v1p2 image editing model.

    HuggingFace: https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2

    Setup:
        git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git
        cd diffusers && pip install -e .
        pip install RegionE  # optional, for faster inference
    """

    MODEL_ID = "stepfun-ai/Step1X-Edit-v1p2"
    MODEL_URL = "https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2"

    def __init__(
        self,
        device: str = "cuda",
        enable_thinking: bool = False,  # Disabled for debugging
        enable_reflection: bool = False,  # Disabled for debugging
        use_region_e: bool = False  # Disabled for debugging
    ):
        super().__init__(model_name="Step1X-Edit-v1p2", device=device)
        self.pipe = None
        self.region_helper = None
        self.enable_thinking = enable_thinking
        self.enable_reflection = enable_reflection
        self.use_region_e = use_region_e

    def load(self):
        """Load Step1X-Edit-v1p2 model."""
        try:
            from diffusers import Step1XEditPipelineV1P2
            import torch

            print(f"Loading Step1X-Edit-v1p2 on {self.device}...")
            print(f"Model URL: {self.MODEL_URL}")

            self.pipe = Step1XEditPipelineV1P2.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16
            )
            # Use sequential CPU offload to fit in 24GB VRAM
            self.pipe.enable_sequential_cpu_offload()

            # Setup RegionE for faster inference (optional)
            if self.use_region_e:
                try:
                    from RegionE import RegionEHelper
                    self.region_helper = RegionEHelper(self.pipe)
                    self.region_helper.set_params()
                    self.region_helper.enable()
                    print("RegionE enabled for faster inference")
                except ImportError:
                    print("RegionE not installed - continuing without it")
                    self.region_helper = None

            self._loaded = True
            print("Step1X-Edit-v1p2 loaded successfully")
            print(f"  Thinking mode: {self.enable_thinking}")
            print(f"  Reflection mode: {self.enable_reflection}")

        except ImportError as e:
            print(f"Failed to import Step1XEditPipelineV1P2: {e}")
            print("\nPlease install the custom diffusers branch:")
            print("  git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git")
            print("  cd diffusers && pip install -e .")
            raise

        except Exception as e:
            print(f"Failed to load Step1X-Edit-v1p2: {e}")
            raise

    def edit(
        self,
        source_image: Union[Image.Image, List[Image.Image]],
        prompt: str,
        true_cfg_scale: float = 6.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        **kwargs
    ) -> EditResult:
        """
        Apply edit to image using Step1X-Edit-v1p2.

        Args:
            source_image: Input image(s) - supports multi-image input
            prompt: Edit instruction
            true_cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed (optional)
        """
        if not self._loaded:
            self.load()

        import torch
        start_time = time.time()

        try:
            # Handle single or multiple images
            is_multi_image = False
            if isinstance(source_image, Image.Image):
                images = source_image.convert("RGB")
            else:
                # Multi-image input: concatenate side-by-side
                # Step1X doesn't support list of images, so we merge them
                is_multi_image = True
                converted_images = [img.convert("RGB") for img in source_image]
                
                # Get dimensions
                widths = [img.width for img in converted_images]
                heights = [img.height for img in converted_images]
                
                # Create a new image by concatenating horizontally
                total_width = sum(widths)
                max_height = max(heights)
                
                concatenated = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for img in converted_images:
                    concatenated.paste(img, (x_offset, 0))
                    x_offset += img.width
                
                images = concatenated

            # Add prefix for multi-image inputs to help model understand
            if is_multi_image:
                prompt_prefix = "The input image contains two separate people side by side (left and right). Keep both people visible in the output. "
                full_prompt = prompt_prefix + prompt
            else:
                full_prompt = prompt

            # Setup generator
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(42)

            # Run inference
            pipe_output = self.pipe(
                image=images,
                prompt=full_prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
                enable_thinking_mode=self.enable_thinking,
                enable_reflection_mode=self.enable_reflection,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Get best output (with reflection) or first image
            if self.enable_reflection and hasattr(pipe_output, 'final_images'):
                output_image = pipe_output.final_images[0]
            else:
                output_image = pipe_output.images[0]

            # Collect thinking info if available
            raw_response = {}
            if self.enable_thinking and hasattr(pipe_output, 'reformat_prompt'):
                raw_response['reformat_prompt'] = pipe_output.reformat_prompt
            if self.enable_reflection and hasattr(pipe_output, 'think_info'):
                raw_response['think_info'] = str(pipe_output.think_info[0]) if pipe_output.think_info else None
                raw_response['best_info'] = str(pipe_output.best_info[0]) if hasattr(pipe_output, 'best_info') else None

            return EditResult(
                success=True,
                refusal_type=RefusalType.NONE,
                output_image=output_image,
                latency_ms=latency_ms,
                raw_response=raw_response if raw_response else None
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e).lower()

            # Print detailed error for debugging - CRITICAL for Step1X debugging
            print("🚨 STEP1X CRITICAL ERROR 🚨")
            print(f"❌ Error Type: {type(e).__name__}")
            print(f"❌ Error Message: {str(e)}")
            print(f"❌ Prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
            print(f"❌ Image Size: {getattr(source_image, 'size', 'Unknown')}")
            print(f"❌ Config: steps={num_inference_steps}, cfg={true_cfg_scale}")
            print(f"❌ Thinking: {self.enable_thinking}, Reflection: {self.enable_reflection}")
            print("❌ Full Traceback:")
            import traceback
            traceback.print_exc()
            print("🚨 END CRITICAL ERROR 🚨")

            if any(kw in error_msg for kw in ["safety", "policy", "inappropriate", "blocked", "nsfw"]):
                return EditResult(
                    success=False,
                    refusal_type=RefusalType.HARD,
                    error_message=str(e),
                    latency_ms=latency_ms
                )

            return EditResult(
                success=False,
                refusal_type=RefusalType.API_ERROR,  # Use API_ERROR instead of HARD for debugging
                error_message=str(e),
                latency_ms=latency_ms
            )

    def cleanup(self):
        """Cleanup resources."""
        if self.region_helper is not None:
            self.region_helper.disable()
