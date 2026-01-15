#!/usr/bin/env python3
"""
Category C Pilot - GPU 0 (MALE)
New ambiguous "traditional" prompts for stereotype detection
Generates: Edited (baseline) + Preserved (with identity prompt)
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

# Paths (relative to project root)
SOURCE_DIR = PROJECT_ROOT / "data" / "source_images" / "final"
IDENTITY_PROMPTS_FILE = PROJECT_ROOT / "data" / "identity_prompts" / "identity_prompt_mapping_full_20260113_234029.json"
PROMPTS_FILE = PROJECT_ROOT / "data" / "prompts" / "category_c_new.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "category_c_pilot"

# GPU 0: MALE only, 20s only for quick pilot
RACES = ["White", "Black", "EastAsian", "SoutheastAsian", "Indian", "MiddleEastern", "Latino"]
GENDER = "Male"
AGE = "20s"  # Quick pilot: single age


def load_identity_prompts() -> dict:
    with open(IDENTITY_PROMPTS_FILE, "r") as f:
        return json.load(f)


def load_category_c_prompts() -> list:
    with open(PROMPTS_FILE, "r") as f:
        data = json.load(f)
        return data["prompts"]


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Category C Pilot - GPU 0 (MALE)")
    print("Ambiguous 'traditional' prompts for stereotype detection")
    print(f"Images: {len(RACES)} races × 1 gender × 1 age = {len(RACES)} images")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    # Create output directories
    edited_dir = OUTPUT_DIR / "edited"
    preserved_dir = OUTPUT_DIR / "preserved"
    edited_dir.mkdir(parents=True, exist_ok=True)
    preserved_dir.mkdir(parents=True, exist_ok=True)

    identity_prompts = load_identity_prompts()
    print(f"Loaded {len(identity_prompts)} identity prompts")

    prompts = load_category_c_prompts()
    print(f"Loaded {len(prompts)} Category C prompts")

    # Initialize Step1X
    print("\nInitializing Step1X-Edit...")
    model = Step1XWrapper(device="cuda", enable_thinking=False, enable_reflection=False)
    model.load()

    results = []
    total = len(RACES) * len(prompts) * 2  # × 2 for edited + preserved
    count = 0

    for prompt_info in prompts:
        prompt_id = prompt_info["id"]
        prompt_text = prompt_info["prompt"]
        domain = prompt_info["domain"]

        print(f"\n{'='*60}")
        print(f"Prompt {prompt_id} ({domain}): {prompt_text}")
        print("=" * 60)

        # ===== PASS 1: All races EDITED (baseline) =====
        print(f"\n--- {prompt_id} EDITED (all races) ---")
        for race in RACES:
            image_key = f"{race}_{GENDER}_{AGE}"
            source_path = SOURCE_DIR / race / f"{image_key}.jpg"

            if not source_path.exists():
                print(f"  SKIP: Source not found: {source_path}")
                continue

            count += 1
            print(f"[{count}/{total}] {image_key} - EDITED")

            source_image = Image.open(source_path)

            try:
                result = model.edit(
                    source_image=source_image,
                    prompt=prompt_text,
                    num_inference_steps=50,
                    true_cfg_scale=6.0,
                    seed=42
                )

                if result.success and result.output_image:
                    output_filename = f"{prompt_id}_{image_key}_edited.png"
                    output_path = edited_dir / output_filename
                    result.output_image.save(output_path)
                    print(f"  SUCCESS: {output_filename} ({result.latency_ms:.0f}ms)")
                    results.append({
                        "prompt_id": prompt_id,
                        "domain": domain,
                        "image_key": image_key,
                        "type": "edited",
                        "status": "success",
                        "latency_ms": result.latency_ms
                    })
                else:
                    print(f"  FAILED: {result.error_message or 'Unknown'}")
                    results.append({
                        "prompt_id": prompt_id,
                        "image_key": image_key,
                        "type": "edited",
                        "status": "failed",
                        "error": result.error_message
                    })

            except Exception as e:
                print(f"  ERROR: {str(e)[:80]}")
                results.append({
                    "prompt_id": prompt_id,
                    "image_key": image_key,
                    "type": "edited",
                    "status": "error",
                    "error": str(e)[:200]
                })

        # ===== PASS 2: All races PRESERVED (with identity prompt) =====
        print(f"\n--- {prompt_id} PRESERVED (all races) ---")
        for race in RACES:
            image_key = f"{race}_{GENDER}_{AGE}"
            source_path = SOURCE_DIR / race / f"{image_key}.jpg"

            if not source_path.exists():
                continue

            identity_prompt = identity_prompts.get(image_key, "")
            if not identity_prompt:
                print(f"  SKIP: No identity prompt for {image_key}")
                continue

            count += 1
            print(f"[{count}/{total}] {image_key} - PRESERVED")

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
                    output_filename = f"{prompt_id}_{image_key}_preserved.png"
                    output_path = preserved_dir / output_filename
                    result.output_image.save(output_path)
                    print(f"  SUCCESS: {output_filename} ({result.latency_ms:.0f}ms)")
                    results.append({
                        "prompt_id": prompt_id,
                        "domain": domain,
                        "image_key": image_key,
                        "type": "preserved",
                        "status": "success",
                        "latency_ms": result.latency_ms
                    })
                else:
                    print(f"  FAILED: {result.error_message or 'Unknown'}")
                    results.append({
                        "prompt_id": prompt_id,
                        "image_key": image_key,
                        "type": "preserved",
                        "status": "failed",
                        "error": result.error_message
                    })

            except Exception as e:
                print(f"  ERROR: {str(e)[:80]}")
                results.append({
                    "prompt_id": prompt_id,
                    "image_key": image_key,
                    "type": "preserved",
                    "status": "error",
                    "error": str(e)[:200]
                })

    # Save results
    results_file = OUTPUT_DIR / f"gpu0_male_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "gpu": 0,
            "gender": GENDER,
            "age": AGE,
            "prompts": [p["id"] for p in prompts],
            "results": results
        }, f, indent=2)

    print("\n" + "=" * 80)
    edited_success = sum(1 for r in results if r["type"] == "edited" and r["status"] == "success")
    preserved_success = sum(1 for r in results if r["type"] == "preserved" and r["status"] == "success")
    print(f"GPU 0 (MALE) Complete:")
    print(f"  Edited: {edited_success}/{len(RACES) * len(prompts)}")
    print(f"  Preserved: {preserved_success}/{len(RACES) * len(prompts)}")
    print(f"Results: {results_file}")

    model.cleanup()


if __name__ == "__main__":
    main()
