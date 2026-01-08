#!/usr/bin/env python3
"""
Generate Mock Results for Testing Analysis Pipeline
Creates synthetic experiment results with controlled bias patterns
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import PathConfig


def generate_mock_results(
    model_name: str,
    output_dir: Path,
    bias_strength: float = 0.2,
    seed: int = 42
):
    """
    Generate mock experiment results with controlled bias patterns.

    Args:
        model_name: Name of the model
        output_dir: Output directory
        bias_strength: Strength of bias (0 = no bias, 1 = max bias)
        seed: Random seed
    """
    np.random.seed(seed)

    races = ["White", "Black", "East Asian", "Southeast Asian",
             "Indian", "Middle Eastern", "Latino_Hispanic"]

    # Race-specific base refusal rates (simulating bias)
    race_bias = {
        "White": 0.0,  # baseline
        "Black": bias_strength * 0.3,
        "East Asian": bias_strength * 0.15,
        "Southeast Asian": bias_strength * 0.2,
        "Indian": bias_strength * 0.18,
        "Middle Eastern": bias_strength * 0.25,
        "Latino_Hispanic": bias_strength * 0.22
    }

    # Category-specific refusal rates
    category_rates = {
        "A": 0.02,  # Neutral - very low
        "B": 0.10,  # Occupation - low
        "C": 0.25,  # Cultural - medium
        "D": 0.30,  # Vulnerability - medium-high
        "E": 0.60   # Harmful - high
    }

    categories = ["A", "B", "C", "D", "E"]
    genders = ["Male", "Female"]
    age_codes = ["20s", "30s", "40s"]

    results = []
    request_idx = 0

    # Generate 10 prompts per category
    for cat in categories:
        for prompt_num in range(1, 11):
            prompt_id = f"{cat}{prompt_num:02d}"

            # For each race × gender × age combination
            for race in races:
                for gender in genders:
                    for age in age_codes:
                        request_idx += 1

                        # Calculate refusal probability
                        base_rate = category_rates[cat]
                        race_effect = race_bias[race]

                        # Add interaction effect for cultural prompts
                        if cat == "C" and race != "White":
                            # Cultural prompts show stronger bias
                            race_effect *= 1.5

                        refusal_prob = min(base_rate + race_effect, 0.95)

                        # Simulate refusal
                        is_refused = np.random.random() < refusal_prob

                        # Generate CLIP similarity score (inversely related to refusal)
                        if is_refused:
                            clip_similarity = np.random.uniform(0.70, 0.92)
                        else:
                            clip_similarity = np.random.uniform(0.93, 0.99)

                        # Refusal type
                        if is_refused:
                            refusal_types = ["api_error", "unchanged", "empty_image", "safety_filter"]
                            refusal_type = np.random.choice(refusal_types, p=[0.4, 0.3, 0.2, 0.1])
                            confidence = np.random.uniform(0.7, 1.0)
                        else:
                            refusal_type = "none"
                            confidence = np.random.uniform(0.8, 1.0)

                        # Generate image ID for bootstrap analysis
                        image_id = f"{race.replace(' ', '')}_{gender}_{age}"

                        # Generate result
                        result = {
                            "request_idx": request_idx,
                            "model": model_name,
                            "prompt_id": prompt_id,
                            "prompt_text": f"Mock prompt {prompt_id}",
                            "category": cat,
                            "hypothesis": f"H{cat}",
                            "race_code": race.replace(" ", ""),
                            "race": race,
                            "gender": gender,
                            "age_code": age,
                            "image_id": image_id,  # For bootstrap analysis
                            "source_image": f"data/source_images/fairface/final/{race}/{race}_{gender}_{age}.jpg",
                            "output_image": None if is_refused else f"data/results/{model_name}/mock/images/{race}/{prompt_id}_{race}_{gender}_{age}_success.png",
                            "success": not is_refused,
                            "is_refused": is_refused,
                            "refusal_type": refusal_type,
                            "refusal_confidence": confidence,
                            "clip_similarity": clip_similarity,  # For threshold sensitivity
                            "latency_ms": int(np.random.uniform(5000, 15000)),
                            "error_message": "Mock error" if is_refused else None,
                            "timestamp": datetime.now().isoformat()
                        }

                        results.append(result)

    # Save results
    experiment_id = f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = output_dir / model_name / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    results_file = exp_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Generate summary
    total = len(results)
    refused = sum(1 for r in results if r["is_refused"])
    by_race = {}

    for race in races:
        race_results = [r for r in results if r["race"] == race]
        race_refused = sum(1 for r in race_results if r["is_refused"])
        by_race[race] = {
            "total": len(race_results),
            "refused": race_refused,
            "refusal_rate": race_refused / len(race_results) if race_results else 0
        }

    by_category = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_refused = sum(1 for r in cat_results if r["is_refused"])
        by_category[cat] = {
            "total": len(cat_results),
            "refused": cat_refused,
            "refusal_rate": cat_refused / len(cat_results) if cat_results else 0
        }

    summary = {
        "experiment_id": experiment_id,
        "model": model_name,
        "total_requests": total,
        "successful": total - refused,
        "refused": refused,
        "overall_refusal_rate": refused / total,
        "by_race": by_race,
        "by_category": by_category,
        "bias_strength": bias_strength
    }

    summary_file = exp_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGenerated mock results for {model_name}")
    print(f"  Total requests: {total}")
    print(f"  Refusal rate: {refused/total:.2%}")
    print(f"  Saved to: {exp_dir}")

    print(f"\nRefusal rates by race:")
    for race, data in by_race.items():
        print(f"  {race}: {data['refusal_rate']:.2%}")

    print(f"\nRefusal rates by category:")
    for cat, data in by_category.items():
        print(f"  {cat}: {data['refusal_rate']:.2%}")

    return results_file


def main():
    parser = argparse.ArgumentParser(description="Generate mock experiment results")
    parser.add_argument("--model", type=str, default="flux",
                       help="Model name")
    parser.add_argument("--output-dir", type=str, default="data/results",
                       help="Output directory")
    parser.add_argument("--bias-strength", type=float, default=0.2,
                       help="Bias strength (0-1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--models", type=str, nargs="+",
                       help="Generate for multiple models")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.models:
        for model in args.models:
            generate_mock_results(
                model_name=model,
                output_dir=output_dir,
                bias_strength=args.bias_strength,
                seed=args.seed
            )
    else:
        generate_mock_results(
            model_name=args.model,
            output_dir=output_dir,
            bias_strength=args.bias_strength,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
