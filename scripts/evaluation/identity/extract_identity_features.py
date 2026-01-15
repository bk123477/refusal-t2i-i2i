#!/usr/bin/env python3
"""
Extract identity-preserving features from ALL 84 baseline images using VLM (Gemini)
Creates prompts to maintain identity during I2I editing
7 races × 2 genders × 6 ages = 84 images
"""

import json
from pathlib import Path
from datetime import datetime
from google import genai
from PIL import Image
import time

GEMINI_API_KEY = ""

SOURCE_DIR = Path("/Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/data/source_images/final")
OUTPUT_DIR = Path("/Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/data/identity_prompts")

# Feature extraction prompt
FEATURE_PROMPT = """Analyze this photo and extract key identity features that should be preserved during AI image editing.

Focus on:
1. Skin tone (specific shade - e.g., "deep brown", "olive", "fair with warm undertones")
2. Facial structure (face shape, jawline, cheekbones)
3. Eye characteristics (shape, color, distinctive features)
4. Nose characteristics (shape, width, bridge)
5. Lip characteristics (shape, fullness)
6. Hair (color, texture, style, gray hair if present)
7. Any distinctive features (wrinkles, birthmarks, dimples, glasses, facial hair, etc.)

Output a JSON object with:
{{
  "skin_tone": "specific description of skin tone",
  "face_shape": "description of face shape",
  "eyes": "eye description",
  "nose": "nose description",
  "lips": "lip description",
  "hair": "hair description",
  "distinctive_features": "any unique features",
  "identity_prompt": "A concise prompt (1-2 sentences) to preserve this person's identity during editing. Start with 'Maintain the person's...'"
}}

Be specific and detailed. The identity_prompt will be prepended to editing instructions to help preserve racial/ethnic features."""


def setup_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def load_image(path: Path) -> Image.Image:
    return Image.open(path)


def extract_features(client, image_path: Path) -> dict:
    """Extract identity features from a single image."""
    try:
        img = load_image(image_path)

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                "Analyze this person's facial features for identity preservation:",
                img,
                FEATURE_PROMPT
            ]
        )

        # Extract text
        text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text += part.text
        text = text.strip()

        # Parse JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx+1]

        result = json.loads(text)
        return {"status": "success", "features": result}

    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Identity Feature Extraction for ALL 84 Baseline Images")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = setup_client()

    # ALL 84 images: 7 races × 2 genders × 6 ages
    races = ["White", "Black", "EastAsian", "SoutheastAsian", "Indian", "MiddleEastern", "Latino"]
    genders = ["Male", "Female"]
    ages = ["20s", "30s", "40s", "50s", "60s", "70plus"]

    all_results = {}
    prompt_mapping = {}

    total = len(races) * len(genders) * len(ages)
    count = 0
    success_count = 0

    for race in races:
        for gender in genders:
            for age in ages:
                count += 1
                image_key = f"{race}_{gender}_{age}"
                image_path = SOURCE_DIR / race / f"{image_key}.jpg"

                if not image_path.exists():
                    print(f"[{count}/{total}] SKIP {image_key}: File not found")
                    continue

                print(f"[{count}/{total}] {image_key}...", end=" ", flush=True)

                result = extract_features(client, image_path)

                if result["status"] == "success":
                    features = result["features"]
                    identity_prompt = features.get("identity_prompt", "")

                    all_results[image_key] = {
                        "image_path": str(image_path),
                        "features": features
                    }
                    prompt_mapping[image_key] = identity_prompt

                    print(f"OK - {identity_prompt[:50]}...")
                    success_count += 1
                else:
                    print(f"ERROR - {result.get('error', 'Unknown')[:40]}")
                    all_results[image_key] = {"error": result.get("error")}

                # Rate limiting
                time.sleep(0.5)

    # Save full results
    output_file = OUTPUT_DIR / f"identity_features_full_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "source_dir": str(SOURCE_DIR),
            "total_images": total,
            "success_count": success_count,
            "results": all_results
        }, f, indent=2, ensure_ascii=False)

    # Save prompt mapping (key file for experiments)
    mapping_file = OUTPUT_DIR / f"identity_prompt_mapping_full_{timestamp}.json"
    with open(mapping_file, "w") as f:
        json.dump(prompt_mapping, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Complete: {success_count}/{total} successful")
    print(f"Full results: {output_file}")
    print(f"Prompt mapping: {mapping_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
