#!/usr/bin/env python3
"""
Download FairFace dataset from HuggingFace
https://huggingface.co/datasets/HuggingFaceM4/FairFace
"""

import argparse
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download FairFace dataset")
    parser.add_argument("--output-dir", type=str,
                       default="data/datasets/fairface",
                       help="Output directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "validation"],
                       help="Dataset split to download")
    parser.add_argument("--padding", type=str, default="1.25",
                       choices=["0.25", "1.25"],
                       help="Padding ratio (1.25 = more context around face)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Downloading FairFace Dataset")
    print("Source: https://huggingface.co/datasets/HuggingFaceM4/FairFace")
    print("="*60)

    # Load dataset
    print(f"\nLoading {args.split} split (padding={args.padding})...")
    dataset = load_dataset("HuggingFaceM4/FairFace", args.padding, split=args.split)

    print(f"Dataset size: {len(dataset)} images")
    print(f"Features: {dataset.features}")

    # Show distribution
    print("\nRace distribution:")
    races = {}
    for item in dataset:
        race = item["race"]
        races[race] = races.get(race, 0) + 1

    for race, count in sorted(races.items()):
        print(f"  {race}: {count}")

    # Save dataset info
    info_path = output_dir / "dataset_info.txt"
    with open(info_path, "w") as f:
        f.write(f"FairFace Dataset\n")
        f.write(f"Source: https://huggingface.co/datasets/HuggingFaceM4/FairFace\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Total images: {len(dataset)}\n\n")
        f.write("Race distribution:\n")
        for race, count in sorted(races.items()):
            f.write(f"  {race}: {count}\n")

    print(f"\nDataset info saved to {info_path}")
    print(f"Dataset cached by HuggingFace datasets library")
    print("\nTo sample 84 images for the experiment, run:")
    print("  python scripts/sample_fairface.py")


if __name__ == "__main__":
    main()
