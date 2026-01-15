#!/usr/bin/env python3
"""
Category B Pilot - GPU 1
B02 (Military) - Balanced distribution across races/genders/ages
PRESERVED ONLY
"""
import os
import json
import sys
from pathlib import Path

os.environ["HF_HOME"] = "/scratch/chans/models/huggingface"
os.environ["HF_HUB_CACHE"] = "/scratch/chans/models/huggingface/hub"

from datetime import datetime
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.step1x_wrapper import Step1XWrapper

# Paths
SOURCE_DIR = PROJECT_ROOT / "data" / "source_images" / "final"
IDENTITY_PROMPTS_FILE = PROJECT_ROOT / "data" / "identity_prompts" / "identity_prompt_mapping_full_20260113_234029.json"
PROMPTS_FILE = PROJECT_ROOT / "data" / "prompts" / "i2i_prompts.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "category_b_pilot"

# B02 - Balanced distribution (7 races × mixed gender/age)
B02_TARGETS = [
    # Each race: 1 male + 1 female, varied ages
    "White_Male_30s",
    "White_Female_40s",
    "Black_Male_20s",
    "Black_Female_30s",
    "EastAsian_Male_40s",
    "EastAsian_Female_20s",
    "SoutheastAsian_Male_30s",
    "SoutheastAsian_Female_50s",
    "Indian_Male_50s",
    "Indian_Female_20s",
    "MiddleEastern_Male_20s",
    "MiddleEastern_Female_30s",
    "Latino_Male_40s",
    "Latino_Female_20s",
]


def load_identity_prompts() -> dict:
    with open(IDENTITY_PROMPTS_FILE, "r") as f:
        return json.load(f)


def load_prompt(prompt_id: str) -> dict:
    with open(PROMPTS_FILE, "r") as f:
        data = json.load(f)
        for p in data["prompts"]:
            if p["id"] == prompt_id:
                return p
    return None


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Category B Pilot - GPU 1 - B02 (Military) PRESERVED")
    print(f"Targets: {len(B02_TARGETS)} balanced distribution")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    preserved_dir = OUTPUT_DIR / "preserved"
    preserved_dir.mkdir(parents=True, exist_ok=True)

    identity_prompts = load_identity_prompts()
    print(f"Loaded {len(identity_prompts)} identity prompts")

    prompt_info = load_prompt("B02")
    prompt_text = prompt_info["prompt"]
    print(f"B02: {prompt_text[:60]}...")

    # Initialize Step1X
    print("\nInitializing Step1X-Edit...")
    model = Step1XWrapper(device="cuda", enable_thinking=False, enable_reflection=False)
    model.load()

    results = []
    total = len(B02_TARGETS)

    print(f"\n--- B02 PRESERVED ({total} targets) ---")
    for count, image_key in enumerate(B02_TARGETS, 1):
        # Parse race from image_key
        race = image_key.split("_")[0]
        source_path = SOURCE_DIR / race / f"{image_key}.jpg"

        print(f"\n[{count}/{total}] {image_key}")

        if not source_path.exists():
            print(f"  SKIP: Source not found: {source_path}")
            continue

        identity_prompt = identity_prompts.get(image_key, "")
        if not identity_prompt:
            print(f"  SKIP: No identity prompt")
            continue

        source_image = Image.open(source_path)
        combined_prompt = f"{prompt_text}. {identity_prompt}"

        try:
            result = model.edit(
                source_image=source_image,
                prompt=combined_prompt,
                num_inference_steps=50,
                true_cfg_scale=6.0,
                seed=42
            )

            if result.success and result.output_image:
                output_filename = f"B02_{image_key}_preserved.png"
                output_path = preserved_dir / output_filename
                result.output_image.save(output_path)
                print(f"  SUCCESS: {output_filename} ({result.latency_ms:.0f}ms)")
                results.append({
                    "prompt_id": "B02",
                    "image_key": image_key,
                    "status": "success",
                    "latency_ms": result.latency_ms
                })
            else:
                print(f"  FAILED: {result.error_message or 'Unknown'}")
                results.append({
                    "prompt_id": "B02",
                    "image_key": image_key,
                    "status": "failed",
                    "error": result.error_message
                })

        except Exception as e:
            print(f"  ERROR: {str(e)[:80]}")
            results.append({
                "prompt_id": "B02",
                "image_key": image_key,
                "status": "error",
                "error": str(e)[:200]
            })

    # Save results
    results_file = OUTPUT_DIR / f"gpu1_B02_preserved_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "gpu": 1,
            "prompt": "B02",
            "targets": B02_TARGETS,
            "results": results
        }, f, indent=2)

    print("\n" + "=" * 80)
    success = sum(1 for r in results if r["status"] == "success")
    print(f"GPU 1 Complete: {success}/{total} B02 preserved images")
    print(f"Results: {results_file}")

    model.cleanup()


if __name__ == "__main__":
    main()
