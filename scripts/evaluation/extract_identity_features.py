#!/usr/bin/env python3
"""
Extract identity-preserving features from baseline images using VLM (Gemini)
Creates prompts to maintain identity during I2I editing
"""

import json
from pathlib import Path
from datetime import datetime
from google import genai
from PIL import Image

GEMINI_API_KEY = "AIzaSyCYNx5TMv-2cPsPIBdHCZ6PuG0c_HQXNxY"

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
6. Hair (color, texture, style)
7. Any distinctive features (birthmarks, dimples, etc.)

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
    print("Identity Feature Extraction for Baseline Images")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = setup_client()

    # Image configurations
    races = ["White", "Black", "EastAsian", "SoutheastAsian", "Indian", "MiddleEastern", "Latino"]
    genders = ["Male", "Female"]
    age = "20s"

    all_results = {}

    for race in races:
        all_results[race] = {}

        for gender in genders:
            image_path = SOURCE_DIR / race / f"{race}_{gender}_{age}.jpg"

            if not image_path.exists():
                print(f"\n[SKIP] {race} {gender}: File not found")
                continue

            print(f"\n[{race} {gender}]")
            print(f"  Processing: {image_path.name}")

            result = extract_features(client, image_path)

            if result["status"] == "success":
                features = result["features"]
                print(f"  Skin tone: {features.get('skin_tone', 'N/A')}")
                print(f"  Identity prompt: {features.get('identity_prompt', 'N/A')[:60]}...")

                all_results[race][gender] = {
                    "image_path": str(image_path),
                    "features": features
                }
            else:
                print(f"  Error: {result.get('error', 'Unknown')[:50]}")
                all_results[race][gender] = {"error": result.get("error")}

    # Save results
    output_file = OUTPUT_DIR / f"identity_features_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "source_dir": str(SOURCE_DIR),
            "results": all_results
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")

    # Generate identity prompt mapping file
    prompt_mapping = {}
    for race in races:
        for gender in genders:
            key = f"{race}_{gender}_{age}"
            if race in all_results and gender in all_results[race]:
                data = all_results[race][gender]
                if "features" in data and "identity_prompt" in data["features"]:
                    prompt_mapping[key] = data["features"]["identity_prompt"]

    mapping_file = OUTPUT_DIR / f"identity_prompt_mapping_{timestamp}.json"
    with open(mapping_file, "w") as f:
        json.dump(prompt_mapping, f, indent=2, ensure_ascii=False)

    print(f"Prompt mapping saved to: {mapping_file}")
    print("=" * 80)

    # Print summary
    print("\n=== IDENTITY PROMPTS SUMMARY ===\n")
    for key, prompt in prompt_mapping.items():
        print(f"{key}:")
        print(f"  {prompt}\n")


if __name__ == "__main__":
    main()
