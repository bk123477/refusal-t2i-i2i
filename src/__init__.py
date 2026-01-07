"""
I2I Refusal Bias Study - IJCAI 2026
Silent Discrimination: Race-Conditioned Refusal Bias in Image-to-Image Editing Models

Models:
- FLUX.2-dev: https://huggingface.co/black-forest-labs/FLUX.2-dev
- Step1X-Edit-v1p2: https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2
- Qwen-Image-Edit-2511: https://huggingface.co/Qwen/Qwen-Image-Edit-2511

Dataset:
- FairFace: https://huggingface.co/datasets/HuggingFaceM4/FairFace
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

from .config import PathConfig, NamingConfig, ExperimentConfig, MODEL_CONFIGS
