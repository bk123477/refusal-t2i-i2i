#!/usr/bin/env python3
"""
FairFace Dataset Download and Sampling Script

Downloads FairFace from HuggingFace and samples 30 images per race (210 total)
for the I2I refusal bias experiment.

Usage:
    python scripts/download_fairface.py
    python scripts/download_fairface.py --samples-per-race 50
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# FairFace race categories
RACE_CATEGORIES = [
    "White",
    "Black",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
    "Latino_Hispanic"
]

# Age range for sampling (20-50 for aging experiment validity)
AGE_RANGE = ["20-29", "30-39", "40-49"]


def download_and_sample_fairface(
    output_dir: str = "data/fairface",
    samples_per_race: int = 30,
    seed: int = 42,
    balanced_gender: bool = True
):
    """
    Download FairFace and sample balanced images per race.

    Args:
        output_dir: Directory to save sampled images
        samples_per_race: Number of images per race category
        seed: Random seed for reproducibility
        balanced_gender: If True, sample 50% male / 50% female per race
    """
    random.seed(seed)
    output_path = Path(output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FairFace Dataset Download & Sampling")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"Samples per race: {samples_per_race}")
    print(f"Total target: {samples_per_race * len(RACE_CATEGORIES)} images")
    print(f"Balanced gender: {balanced_gender}")
    print("=" * 60)

    # Load FairFace dataset from HuggingFace
    print("\nLoading FairFace dataset from HuggingFace...")
    dataset = load_dataset("HuggingFaceM4/FairFace", split="train")
    print(f"Total images in dataset: {len(dataset)}")

    # Group images by race and gender
    print("\nGrouping images by race and gender...")
    race_gender_groups = defaultdict(lambda: defaultdict(list))

    for idx, item in enumerate(tqdm(dataset, desc="Indexing")):
        race = item["race"]
        gender = item["gender"]
        age = item["age"]

        # Filter by age range (20-50)
        if age in AGE_RANGE and race in RACE_CATEGORIES:
            race_gender_groups[race][gender].append(idx)

    # Print statistics
    print("\nDataset statistics (age 20-50):")
    for race in RACE_CATEGORIES:
        male_count = len(race_gender_groups[race]["Male"])
        female_count = len(race_gender_groups[race]["Female"])
        print(f"  {race}: {male_count} Male, {female_count} Female")

    # Sample images
    print(f"\nSampling {samples_per_race} images per race...")
    sampled_data = []

    for race in RACE_CATEGORIES:
        if balanced_gender:
            # Sample half male, half female
            n_per_gender = samples_per_race // 2
            male_indices = race_gender_groups[race]["Male"]
            female_indices = race_gender_groups[race]["Female"]

            if len(male_indices) < n_per_gender:
                print(f"  Warning: {race} has only {len(male_indices)} males, using all")
                sampled_males = male_indices
            else:
                sampled_males = random.sample(male_indices, n_per_gender)

            if len(female_indices) < n_per_gender:
                print(f"  Warning: {race} has only {len(female_indices)} females, using all")
                sampled_females = female_indices
            else:
                sampled_females = random.sample(female_indices, n_per_gender)

            race_samples = sampled_males + sampled_females

            # If odd number requested, add one more randomly
            if samples_per_race % 2 == 1:
                remaining = [i for i in male_indices + female_indices if i not in race_samples]
                if remaining:
                    race_samples.append(random.choice(remaining))
        else:
            # Random sampling without gender balance
            all_indices = race_gender_groups[race]["Male"] + race_gender_groups[race]["Female"]
            race_samples = random.sample(all_indices, min(samples_per_race, len(all_indices)))

        # Save sampled images
        race_clean = race.replace(" ", "_").replace("/", "_")
        for i, idx in enumerate(race_samples):
            item = dataset[idx]
            image = item["image"]
            gender = item["gender"]
            age = item["age"]

            # Generate filename
            filename = f"{race_clean}_{i:03d}_{gender}_{age}.jpg"
            filepath = images_path / filename

            # Save image
            if isinstance(image, Image.Image):
                image.save(filepath, "JPEG", quality=95)

            # Record metadata
            sampled_data.append({
                "filename": filename,
                "race": race,
                "gender": gender,
                "age": age,
                "original_index": idx
            })

        print(f"  {race}: {len(race_samples)} images sampled")

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "total_samples": len(sampled_data),
            "samples_per_race": samples_per_race,
            "races": RACE_CATEGORIES,
            "age_range": AGE_RANGE,
            "balanced_gender": balanced_gender,
            "seed": seed,
            "samples": sampled_data
        }, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")

    # Create summary statistics
    summary_path = output_path / "summary.json"
    summary = {
        "total": len(sampled_data),
        "by_race": {},
        "by_gender": {"Male": 0, "Female": 0}
    }

    for item in sampled_data:
        race = item["race"]
        gender = item["gender"]
        summary["by_race"][race] = summary["by_race"].get(race, 0) + 1
        summary["by_gender"][gender] += 1

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Sampling Complete!")
    print("=" * 60)
    print(f"Total images: {len(sampled_data)}")
    print(f"Images directory: {images_path}")
    print(f"Metadata file: {metadata_path}")
    print("\nBreakdown by race:")
    for race, count in summary["by_race"].items():
        print(f"  {race}: {count}")
    print(f"\nBreakdown by gender:")
    print(f"  Male: {summary['by_gender']['Male']}")
    print(f"  Female: {summary['by_gender']['Female']}")

    return sampled_data


def main():
    parser = argparse.ArgumentParser(description="Download and sample FairFace dataset")
    parser.add_argument("--output-dir", type=str, default="data/fairface",
                        help="Output directory for sampled images")
    parser.add_argument("--samples-per-race", type=int, default=30,
                        help="Number of samples per race category")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-gender-balance", action="store_true",
                        help="Disable gender balancing")

    args = parser.parse_args()

    download_and_sample_fairface(
        output_dir=args.output_dir,
        samples_per_race=args.samples_per_race,
        seed=args.seed,
        balanced_gender=not args.no_gender_balance
    )


if __name__ == "__main__":
    main()
