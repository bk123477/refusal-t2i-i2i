#!/usr/bin/env python3
"""
Sample 84 images from FairFace dataset (6 ages × 2 genders × 7 races)
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.fairface_sampler import FairFaceSampler


def main():
    parser = argparse.ArgumentParser(description="Sample FairFace images")
    parser.add_argument("--output-dir", type=str,
                       default="data/source_images/fairface",
                       help="Output directory for sampled images")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    print("FairFace Factorial Sampling")
    print("="*50)
    print("Design: 6 Ages × 2 Genders × 7 Races = 84 Images")
    print("="*50)

    sampler = FairFaceSampler(output_dir=args.output_dir)
    sampler.sample(seed=args.seed)
    sampler.save_metadata()

    print("\nSampling complete!")
    print(f"Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
