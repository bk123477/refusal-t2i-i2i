#!/usr/bin/env python3
"""
Generate JSON files with AWS S3 image URLs for each experiment.

Output:
  - data/survey/exp1_items.json  - VLM Scoring items
  - data/survey/exp2_items.json  - Pairwise A/B comparison items
  - data/survey/exp3_items.json  - WinoBias evaluation items
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "survey"

S3_BASE_URL = "https://i2i-refusal.s3.us-east-2.amazonaws.com"

MODELS = ["flux", "qwen", "step1x"]
CATEGORIES = ["A_neutral", "B_occupation", "C_cultural", "D_vulnerability", "E_harmful"]
RACES = ["Black", "EastAsian", "Indian", "Latino", "MiddleEastern", "SoutheastAsian", "White"]


def generate_exp1_json():
    """Generate JSON for Experiment 1: VLM Scoring (sampling-based evaluation)."""
    print("Generating Experiment 1 JSON...")

    items = []

    for model in MODELS:
        model_dir = RESULTS_DIR / "exp1_sampling" / f"{model}_organized" / "by_category"
        if not model_dir.exists():
            print(f"  [SKIP] {model}: directory not found")
            continue

        for category in CATEGORIES:
            cat_dir = model_dir / category
            if not cat_dir.exists():
                continue

            for img_file in cat_dir.glob("*.png"):
                # Parse filename: {PromptID}_{Race}_{Gender}_{Age}_{status}.png
                parts = img_file.stem.split("_")
                if len(parts) < 5:
                    continue

                prompt_id = parts[0]
                race = parts[1]
                gender = parts[2]
                age = parts[3]
                status = parts[4] if len(parts) > 4 else "unknown"

                # Build S3 URL
                s3_url = f"{S3_BASE_URL}/{model}/by_category/{category}/{img_file.name}"

                item = {
                    "id": f"{model}_{img_file.stem}",
                    "model": model,
                    "category": category.split("_")[0],  # A, B, C, D, E
                    "categoryName": category,
                    "promptId": prompt_id,
                    "race": race,
                    "gender": gender,
                    "age": age,
                    "status": status,
                    "filename": img_file.name,
                    "s3Url": s3_url
                }
                items.append(item)

        print(f"  {model}: {len([i for i in items if i['model'] == model])} images")

    # Sort by model, category, prompt, race
    items.sort(key=lambda x: (x["model"], x["category"], x["promptId"], x["race"]))

    output_file = OUTPUT_DIR / "exp1_items.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp1_vlm_scoring",
            "description": "VLM-based evaluation of I2I editing quality",
            "totalItems": len(items),
            "models": list(set(i["model"] for i in items)),
            "categories": list(set(i["categoryName"] for i in items)),
            "items": items
        }, f, indent=2)

    print(f"  Saved: {output_file} ({len(items)} items)")
    return len(items)


def generate_exp2_json():
    """Generate JSON for Experiment 2: Pairwise A/B comparison (edited vs preserved)."""
    print("\nGenerating Experiment 2 JSON...")

    items = []
    exp2_dir = RESULTS_DIR / "exp2_pairwise"

    # Process step1x pairwise data organized by prompt
    step1x_dir = exp2_dir / "step1x"
    if step1x_dir.exists():
        # Iterate through prompt folders (B01, B02, D03)
        for prompt_dir in step1x_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            prompt_id = prompt_dir.name
            edited_dir = prompt_dir / "edited"
            preserved_dir = prompt_dir / "preserved"

            if not preserved_dir.exists():
                continue

            # Get all preserved images
            preserved_files = list(preserved_dir.glob("*.png"))
            print(f"  Step1X {prompt_id}: {len(preserved_files)} preserved files")

            for preserved_file in preserved_files:
                # Parse: {PromptID}_{Race}_{Gender}_{Age}_{preserved|identity}.png
                parts = preserved_file.stem.split("_")
                if len(parts) < 4:
                    continue

                race = parts[1]
                gender = parts[2]
                age = parts[3]
                suffix = parts[4] if len(parts) > 4 else "preserved"

                # Find matching edited file
                edited_pattern = f"{prompt_id}_{race}_{gender}_{age}_edited.png"
                edited_file = edited_dir / edited_pattern if edited_dir.exists() else None
                has_edited = edited_file is not None and edited_file.exists()

                # Determine category
                category_code = prompt_id[0]
                category_name = {
                    "B": "B_occupation",
                    "D": "D_vulnerability"
                }.get(category_code, "unknown")

                # Build S3 URLs (organized by prompt)
                preserved_url = f"{S3_BASE_URL}/pairwise/step1x/{prompt_id}/preserved/{preserved_file.name}"
                edited_url = f"{S3_BASE_URL}/pairwise/step1x/{prompt_id}/edited/{edited_pattern}" if has_edited else None
                # Use final folder images (curated 84 images) instead of fairface
                source_url = f"{S3_BASE_URL}/source/final/{race}/{race}_{gender}_{age}.jpg"

                item = {
                    "id": f"exp2_step1x_{preserved_file.stem}",
                    "model": "step1x",
                    "promptId": prompt_id,
                    "category": category_code,
                    "categoryName": category_name,
                    "race": race,
                    "gender": gender,
                    "age": age,
                    "preservedFilename": preserved_file.name,
                    "editedFilename": edited_pattern if has_edited else None,
                    "sourceImageUrl": source_url,
                    "preservedImageUrl": preserved_url,
                    "editedImageUrl": edited_url,
                    "hasEditedPair": has_edited
                }
                items.append(item)

    # Sort by prompt, race, gender, age
    items.sort(key=lambda x: (x["promptId"], x["race"], x["gender"], x["age"]))

    output_file = OUTPUT_DIR / "exp2_items.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp2_pairwise_comparison",
            "description": "Pairwise A/B comparison: edited vs identity-preserved images for soft erasure detection",
            "purpose": "Evaluate if model preserves identity when requested vs standard editing",
            "targetPrompts": ["B01", "B02", "D03"],
            "totalItems": len(items),
            "models": list(set(i["model"] for i in items)),
            "items": items
        }, f, indent=2)

    print(f"  Saved: {output_file} ({len(items)} items)")
    return len(items)


def generate_exp3_json():
    """Generate JSON for Experiment 3: WinoBias binary evaluation."""
    print("\nGenerating Experiment 3 JSON...")

    # Load WinoBias prompts
    prompts_file = PROMPTS_DIR / "winobias_prompts_with_stereotype.json"
    if not prompts_file.exists():
        print("  [ERROR] winobias_prompts_with_stereotype.json not found")
        return 0

    with open(prompts_file) as f:
        winobias_prompts = json.load(f)

    items = []

    # Check WinoBias output folder for Qwen
    qwen_dir = RESULTS_DIR / "exp3_winobias" / "qwen"

    if qwen_dir.exists():
        # List actual files
        qwen_files = list(qwen_dir.glob("prompt_*.png"))
        print(f"  Qwen folder: {len(qwen_files)} images")

        for prompt_data in winobias_prompts:
            prompt_id = prompt_data["id"]
            prompt_num = str(prompt_id).zfill(3)

            # Find matching file (format: prompt_001_YYYYMMDD_HHMMSS.png)
            matching_files = [f for f in qwen_files if f.name.startswith(f"prompt_{prompt_num}_")]

            if matching_files:
                filename = matching_files[0].name

                # Build S3 URL - Note: WinoBias images need to be uploaded separately
                s3_url = f"{S3_BASE_URL}/winobias/qwen/{filename}"

                item = {
                    "id": f"exp3_qwen_{prompt_num}",
                    "model": "qwen",
                    "promptId": prompt_id,
                    "promptText": prompt_data["prompt"],
                    "stereotype": prompt_data["gender_stereotype"],
                    "inputImage1": prompt_data["input_image_1"],
                    "inputImage2": prompt_data["input_image_2"],
                    "filename": filename,
                    "s3Url": s3_url,
                    "sourceInput1Url": f"{S3_BASE_URL}/source/fairface/{prompt_data['input_image_1']}",
                    "sourceInput2Url": f"{S3_BASE_URL}/source/fairface/{prompt_data['input_image_2']}"
                }
                items.append(item)

    # Sort by prompt ID
    items.sort(key=lambda x: x["promptId"])

    output_file = OUTPUT_DIR / "exp3_items.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp3_winobias_evaluation",
            "description": "WinoBias gender stereotype evaluation - determine if model preserved stereotype",
            "totalPrompts": len(winobias_prompts),
            "totalItems": len(items),
            "models": ["qwen"],
            "items": items
        }, f, indent=2)

    print(f"  Saved: {output_file} ({len(items)} items)")
    return len(items)


def main():
    print("=" * 60)
    print("Generating Experiment JSON Files with S3 URLs")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp1_count = generate_exp1_json()
    exp2_count = generate_exp2_json()
    exp3_count = generate_exp3_json()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Experiment 1 (VLM Scoring): {exp1_count} items")
    print(f"  Experiment 2 (Pairwise A/B): {exp2_count} items")
    print(f"  Experiment 3 (WinoBias): {exp3_count} items")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
