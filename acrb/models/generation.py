#!/usr/bin/env python3
"""
ACRB Image Generation Module

Centralized image generation logic for T2I and I2I models.
Consolidates functionality from various generate_*.py scripts.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch
from tqdm import tqdm

from ..models.t2i_wrapper import T2IModelWrapper
from ..models.i2i_wrapper import I2IModelWrapper
from ..metrics.refusal_detector import RefusalDetector

logger = logging.getLogger(__name__)

class ACRBImageGenerator:
    """Unified image generation orchestrator for ACRB."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model wrappers
        self.t2i_models = {}
        self.i2i_models = {}

        # Initialize refusal detector
        self.refusal_detector = RefusalDetector(threshold=config.get('refusal_threshold', 0.25))

    def load_t2i_model(self, model_name: str) -> T2IModelWrapper:
        """Load or get cached T2I model."""
        if model_name not in self.t2i_models:
            self.t2i_models[model_name] = T2IModelWrapper(model_name, device=self.device)
        return self.t2i_models[model_name]

    def load_i2i_model(self, model_name: str) -> I2IModelWrapper:
        """Load or get cached I2I model."""
        if model_name not in self.i2i_models:
            self.i2i_models[model_name] = I2IModelWrapper(model_name, device=self.device)
        return self.i2i_models[model_name]

    def generate_batch(self, prompts: List[Dict], model_name: str, mode: str = "t2i",
                      batch_size: int = 4, output_dir: Path = None) -> List[Dict]:
        """
        Generate images for a batch of prompts.

        Args:
            prompts: List of prompt dictionaries
            model_name: Name of the model to use
            mode: "t2i" or "i2i"
            batch_size: Batch size for generation
            output_dir: Directory to save images

        Returns:
            List of generation results with image paths and metadata
        """
        if mode == "t2i":
            model = self.load_t2i_model(model_name)
        elif mode == "i2i":
            model = self.load_i2i_model(model_name)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        results = []

        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating {mode.upper()}"):
            batch = prompts[i:i+batch_size]

            for prompt_data in batch:
                try:
                    prompt_text = prompt_data['expanded_text']
                    prompt_id = prompt_data['id']

                    # Generate image
                    if mode == "t2i":
                        image = model.generate(prompt_text)
                    else:
                        # For I2I, we need source image
                        source_image_path = prompt_data.get('source_image')
                        if not source_image_path:
                            logger.warning(f"No source image for I2I: {prompt_id}")
                            continue
                        image = model.edit_image(source_image_path, prompt_text)

                    # Detect refusal
                    is_refusal = self.refusal_detector.detect_refusal(image, prompt_text)

                    # Save image if not refused
                    image_path = None
                    if not is_refusal and output_dir:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        image_path = output_dir / f"{prompt_id}.png"
                        image.save(image_path)

                    result = {
                        'prompt_id': prompt_id,
                        'prompt_text': prompt_text,
                        'model': model_name,
                        'mode': mode,
                        'is_refusal': is_refusal,
                        'image_path': str(image_path) if image_path else None,
                        'metadata': prompt_data
                    }

                    results.append(result)

                except Exception as e:
                    logger.error(f"Generation failed for {prompt_data['id']}: {e}")
                    results.append({
                        'prompt_id': prompt_data['id'],
                        'error': str(e),
                        'model': model_name,
                        'mode': mode
                    })

        return results

    def generate_dataset(self, prompts_file: Path, models: List[str],
                        modes: List[str] = ["t2i"], output_dir: Path = None) -> Dict:
        """
        Generate complete dataset across multiple models and modes.

        Args:
            prompts_file: Path to prompts JSON file
            models: List of model names to use
            modes: List of modes ("t2i", "i2i")
            output_dir: Base output directory

        Returns:
            Summary of generation results
        """
        # Load prompts
        with open(prompts_file) as f:
            all_prompts = json.load(f)

        # Filter for generation (exclude neutral for now)
        generation_prompts = [p for p in all_prompts if p['attribute_type'] != 'neutral']

        summary = {
            'total_prompts': len(generation_prompts),
            'models': models,
            'modes': modes,
            'results': {}
        }

        for model in models:
            for mode in modes:
                logger.info(f"Generating with {model} in {mode} mode")

                # Create output directory
                model_output_dir = output_dir / model / mode if output_dir else None

                # Generate
                results = self.generate_batch(
                    generation_prompts, model, mode,
                    output_dir=model_output_dir
                )

                # Store results
                key = f"{model}_{mode}"
                summary['results'][key] = {
                    'generated': len([r for r in results if not r.get('error')]),
                    'refusals': len([r for r in results if r.get('is_refusal')]),
                    'errors': len([r for r in results if r.get('error')]),
                    'details': results
                }

        return summary

    def generate_quick_test(self, model_name: str, num_samples: int = 5,
                           output_dir: Path = None) -> Dict:
        """Quick test generation with simple prompts."""
        test_prompts = [
            {"id": f"test_{i}", "expanded_text": f"A simple test image {i}",
             "attribute_type": "test", "attribute_value": "test"}
            for i in range(num_samples)
        ]

        results = self.generate_batch(test_prompts, model_name, "t2i",
                                    output_dir=output_dir)

        return {
            'model': model_name,
            'samples': num_samples,
            'successful': len([r for r in results if not r.get('error')]),
            'results': results
        }

