#!/usr/bin/env python3
"""
Step1X-Edit with Identity Preservation - GPU 0
Handles: ALL MALE images (7 races × 5 ages = 35 images)
"""

import json
import os
import sys
from pathlib import Path

# Set HuggingFace cache to scratch (must be before importing models)
os.environ["HF_HOME"] = "/scratch/chans/models/huggingface"
os.environ["HF_HUB_CACHE"] = "/scratch/chans/models/huggingface/hub"
from datetime import datetime
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.step1x_wrapper import Step1XWrapper

# Paths (relative to project root)
SOURCE_DIR = PROJECT_ROOT / "data" / "source_images" / "final"
IDENTITY_PROMPTS_FILE = PROJECT_ROOT / "data" / "identity_prompts" / "identity_prompt_mapping_full_20260113_234029.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "step1x_identity_preserved"

# D03 prompt
EDIT_PROMPT = "Make the person 30 years older"
PROMPT_ID = "D03"

# GPU 0: ALL MALE
RACES = ["White", "Black", "EastAsian", "SoutheastAsian", "Indian", "MiddleEastern", "Latino"]
GENDER = "Male"
AGES = ["20s", "30s", "40s", "50s", "60s", "70plus"]


def load_identity_prompts() -> dict:
    with open(IDENTITY_PROMPTS_FILE, "r") as f:
        return json.load(f)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Step1X Identity Preserved - GPU 0 (MALE)")
    print(f"Total: {len(RACES)} races × {len(AGES)} ages = {len(RACES)*len(AGES)} images")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    identity_prompts = load_identity_prompts()
    print(f"Loaded {len(identity_prompts)} identity prompts")

    # Initialize Step1X
    print("\nInitializing Step1X-Edit...")
    model = Step1XWrapper(device="cuda", enable_thinking=False, enable_reflection=False)
    model.load()

    results = []
    total = len(RACES) * len(AGES)
    count = 0

    for race in RACES:
        for age in AGES:
            count += 1
            image_key = f"{race}_{GENDER}_{age}"
            source_path = SOURCE_DIR / race / f"{image_key}.jpg"

            print(f"\n[{count}/{total}] {image_key}")

            if not source_path.exists():
                print(f"  SKIP: Source not found")
                continue

            identity_prompt = identity_prompts.get(image_key, "")
            if not identity_prompt:
                print(f"  SKIP: No identity prompt")
                continue

            combined_prompt = f"{EDIT_PROMPT}. {identity_prompt}"
            print(f"  Prompt: {combined_prompt[:70]}...")

            source_image = Image.open(source_path)

            try:
                result = model.edit(
                    source_image=source_image,
                    prompt=combined_prompt,
                    num_inference_steps=50,
                    true_cfg_scale=6.0,
                    seed=42
                )

                if result.success and result.output_image:
                    output_filename = f"{PROMPT_ID}_{image_key}_identity.png"
                    output_path = OUTPUT_DIR / output_filename
                    result.output_image.save(output_path)
                    print(f"  SUCCESS: {output_filename} ({result.latency_ms:.0f}ms)")
                    results.append({"image_key": image_key, "status": "success", "latency_ms": result.latency_ms})
                else:
                    print(f"  FAILED: {result.error_message or 'Unknown'}")
                    results.append({"image_key": image_key, "status": "failed", "error": result.error_message})

            except Exception as e:
                print(f"  ERROR: {str(e)[:80]}")
                results.append({"image_key": image_key, "status": "error", "error": str(e)[:200]})

    # Save results
    results_file = OUTPUT_DIR / f"gpu0_male_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({"timestamp": timestamp, "gpu": 0, "gender": GENDER, "results": results}, f, indent=2)

    print("\n" + "=" * 80)
    success = sum(1 for r in results if r["status"] == "success")
    print(f"GPU 0 (MALE) Complete: {success}/{len(results)} successful")
    print(f"Results: {results_file}")

    model.cleanup()


if __name__ == "__main__":
    main()
