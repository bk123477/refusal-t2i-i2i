#!/usr/bin/env python3
"""
ACRB Image Generation Module (I2I Focus)

Centralized image generation logic for I2I models.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch
from tqdm import tqdm

from .i2i_wrapper import I2IModelWrapper
from ..metrics.refusal_detector import RefusalDetector

logger = logging.getLogger(__name__)


class ACRBImageGenerator:
    """Unified I2I image generation orchestrator for ACRB."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model wrappers
        self.i2i_models = {}

        # Initialize refusal detector
        self.refusal_detector = RefusalDetector(threshold=config.get('refusal_threshold', 0.25))

    def load_i2i_model(self, model_name: str) -> I2IModelWrapper:
        """Load or get cached I2I model."""
        if model_name not in self.i2i_models:
            self.i2i_models[model_name] = I2IModelWrapper(model_name, device=self.device)
        return self.i2i_models[model_name]

    def generate_batch(self, prompts: List[Dict], model_name: str,
                       source_images_dir: Path,
                       batch_size: int = 4, output_dir: Path = None) -> List[Dict]:
        """
        Generate I2I edits for a batch of prompts.

        Args:
            prompts: List of prompt dictionaries
            model_name: Name of the model to use
            source_images_dir: Directory containing source images
            batch_size: Batch size for generation
            output_dir: Directory to save images

        Returns:
            List of generation results with image paths and metadata
        """
        model = self.load_i2i_model(model_name)
        results = []

        # Get available source images
        source_images = list(source_images_dir.glob("*.jpg")) + \
                        list(source_images_dir.glob("*.png"))

        if not source_images:
            logger.error(f"No source images found in {source_images_dir}")
            return results

        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating I2I"):
            batch = prompts[i:i+batch_size]

            for j, prompt_data in enumerate(batch):
                try:
                    prompt_text = prompt_data.get('expanded_text', prompt_data.get('text', ''))
                    prompt_id = prompt_data.get('id', f"prompt_{i+j}")

                    # Select source image deterministically
                    source_idx = (i + j) % len(source_images)
                    source_image_path = str(source_images[source_idx])

                    # Generate I2I edit
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
                        'source_image': source_image_path,
                        'is_refusal': is_refusal,
                        'image_path': str(image_path) if image_path else None,
                        'metadata': prompt_data
                    }

                    results.append(result)

                except Exception as e:
                    logger.error(f"Generation failed for {prompt_data.get('id', 'unknown')}: {e}")
                    results.append({
                        'prompt_id': prompt_data.get('id', 'unknown'),
                        'error': str(e),
                        'model': model_name
                    })

        return results

    def generate_dataset(self, prompts_file: Path, models: List[str],
                         source_images_dir: Path,
                         output_dir: Path = None) -> Dict:
        """
        Generate complete I2I dataset across multiple models.

        Args:
            prompts_file: Path to prompts JSON file
            models: List of model names to use
            source_images_dir: Directory containing source images
            output_dir: Base output directory

        Returns:
            Summary of generation results
        """
        # Load prompts
        with open(prompts_file) as f:
            all_prompts = json.load(f)

        # Filter for generation (exclude neutral for now)
        generation_prompts = [p for p in all_prompts if p.get('attribute_type') != 'neutral']

        summary = {
            'total_prompts': len(generation_prompts),
            'models': models,
            'results': {}
        }

        for model in models:
            logger.info(f"Generating with {model}")

            # Create output directory
            model_output_dir = output_dir / model if output_dir else None

            # Generate
            results = self.generate_batch(
                generation_prompts, model, source_images_dir,
                output_dir=model_output_dir
            )

            # Store results
            summary['results'][model] = {
                'generated': len([r for r in results if not r.get('error')]),
                'refusals': len([r for r in results if r.get('is_refusal')]),
                'errors': len([r for r in results if r.get('error')]),
                'details': results
            }

        return summary

    def generate_quick_test(self, model_name: str, source_images_dir: Path,
                            num_samples: int = 5, output_dir: Path = None) -> Dict:
        """Quick test generation with simple prompts."""
        test_prompts = [
            {
                "id": f"test_{i}",
                "expanded_text": f"Add a subtle change to the image {i}",
                "attribute_type": "test",
                "attribute_value": "test"
            }
            for i in range(num_samples)
        ]

        results = self.generate_batch(
            test_prompts, model_name, source_images_dir,
            output_dir=output_dir
        )

        return {
            'model': model_name,
            'samples': num_samples,
            'successful': len([r for r in results if not r.get('error')]),
            'results': results
        }
