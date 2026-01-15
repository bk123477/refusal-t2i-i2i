#!/usr/bin/env python3
"""
Organize I2I experiment results for S3 upload.

Consolidates multiple experiment folders into a clean structure:
  - by_category/  (A_neutral, B_occupation, etc.)
  - by_race/      (Black, White, etc.)
  - metadata/     (merged results.json, config.json)

Usage:
    python scripts/data/organize_for_s3.py --model step1x
    python scripts/data/organize_for_s3.py --model flux
    python scripts/data/organize_for_s3.py --model qwen --source /custom/path
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Category mappings
CATEGORY_MAP = {
    "A": "A_neutral",
    "B": "B_occupation",
    "C": "C_cultural",
    "D": "D_vulnerability",
    "E": "E_harmful"
}

CATEGORY_NAMES = {
    "A": "Neutral Baseline",
    "B": "Occupational Stereotype",
    "C": "Cultural/Religious",
    "D": "Vulnerability",
    "E": "Harmful/Safety"
}

DEFAULT_SOURCE = Path.home() / "Downloads" / "t2i-bias-refusal-result-image"


def find_experiment_dirs(source_dir: Path) -> list[Path]:
    """Find all experiment directories (with images/ subfolder)."""
    exp_dirs = []
    for d in sorted(source_dir.iterdir()):
        if d.is_dir() and (d / "images").exists():
            exp_dirs.append(d)
    return exp_dirs


def parse_image_filename(filename: str) -> dict:
    """
    Parse image filename into components.
    Format: {PromptID}_{Race}_{Gender}_{Age}_{status}.png
    Example: A01_Black_Female_20s_success.png
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 5:
        return None

    prompt_id = parts[0]
    race = parts[1]
    gender = parts[2]
    age = parts[3]
    status = parts[4] if len(parts) > 4 else "unknown"

    category = prompt_id[0] if prompt_id else "?"

    return {
        "prompt_id": prompt_id,
        "category": category,
        "race": race,
        "gender": gender,
        "age": age,
        "status": status,
        "filename": filename
    }


def organize_images(source_dir: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """
    Organize images from multiple experiment folders.
    Returns statistics about the organization.
    """
    exp_dirs = find_experiment_dirs(source_dir)

    if not exp_dirs:
        print(f"  No experiment directories found in {source_dir}")
        return {}

    print(f"  Found {len(exp_dirs)} experiment directories")

    # Create output directories
    by_category_dir = output_dir / "by_category"
    by_race_dir = output_dir / "by_race"
    metadata_dir = output_dir / "metadata"

    if not dry_run:
        for cat_code, cat_name in CATEGORY_MAP.items():
            (by_category_dir / cat_name).mkdir(parents=True, exist_ok=True)

        for race in ["Black", "EastAsian", "Indian", "Latino", "MiddleEastern", "SoutheastAsian", "White"]:
            (by_race_dir / race).mkdir(parents=True, exist_ok=True)

        metadata_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results and images
    all_results = []
    stats = defaultdict(lambda: defaultdict(int))
    image_count = 0

    for exp_dir in exp_dirs:
        print(f"  Processing {exp_dir.name}...")

        # Load results.json if exists
        results_file = exp_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results_data = json.load(f)
                if isinstance(results_data, list):
                    all_results.extend(results_data)
                elif isinstance(results_data, dict) and "results" in results_data:
                    all_results.extend(results_data["results"])

        # Process images
        images_dir = exp_dir / "images"
        for race_dir in images_dir.iterdir():
            if not race_dir.is_dir():
                continue

            race = race_dir.name
            for img_file in race_dir.glob("*.png"):
                parsed = parse_image_filename(img_file.name)
                if not parsed:
                    continue

                category = parsed["category"]
                cat_folder = CATEGORY_MAP.get(category)

                if not cat_folder:
                    print(f"    Warning: Unknown category '{category}' in {img_file.name}")
                    continue

                # Copy to by_category
                cat_dest = by_category_dir / cat_folder / img_file.name
                if not dry_run and not cat_dest.exists():
                    shutil.copy2(img_file, cat_dest)

                # Copy to by_race
                race_dest = by_race_dir / race / img_file.name
                if not dry_run and not race_dest.exists():
                    shutil.copy2(img_file, race_dest)

                # Update stats
                stats[category][race] += 1
                stats[category]["total"] += 1
                stats["total"][race] += 1
                stats["total"]["total"] += 1
                image_count += 1

    # Save merged results
    if not dry_run and all_results:
        merged_results = {
            "model": source_dir.name,
            "organized_at": datetime.now().isoformat(),
            "total_images": image_count,
            "results": all_results
        }
        with open(metadata_dir / "results.json", "w") as f:
            json.dump(merged_results, f, indent=2)

        # Save stats
        with open(metadata_dir / "stats.json", "w") as f:
            json.dump(dict(stats), f, indent=2)

    return dict(stats)


def print_stats(stats: dict, model: str):
    """Print organization statistics."""
    if not stats:
        return

    print(f"\n{'='*60}")
    print(f"  {model.upper()} Organization Summary")
    print(f"{'='*60}")

    # Header
    races = ["Black", "EastAsian", "Indian", "Latino", "MiddleEastern", "SoutheastAsian", "White", "total"]
    header = f"{'Category':<15}" + "".join(f"{r[:6]:>8}" for r in races)
    print(header)
    print("-" * len(header))

    # Rows
    for cat in ["A", "B", "C", "D", "E", "total"]:
        if cat not in stats:
            continue
        row = f"{CATEGORY_NAMES.get(cat, cat):<15}"
        for race in races:
            count = stats[cat].get(race, 0)
            row += f"{count:>8}"
        print(row)

    print(f"\n  Total images: {stats.get('total', {}).get('total', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Organize I2I results for S3")
    parser.add_argument("--model", required=True, choices=["step1x", "flux", "qwen"],
                        help="Model name (step1x, flux, qwen)")
    parser.add_argument("--source", type=Path, default=None,
                        help=f"Source directory (default: {DEFAULT_SOURCE}/<model>)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <source>_organized)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without copying")

    args = parser.parse_args()

    # Set paths
    source_dir = args.source or (DEFAULT_SOURCE / args.model)
    output_dir = args.output or source_dir.parent / f"{args.model}_organized"

    print(f"\n{'='*60}")
    print(f"  I2I Results Organizer for S3")
    print(f"{'='*60}")
    print(f"  Model:  {args.model}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Mode:   {'DRY RUN' if args.dry_run else 'COPY'}")
    print(f"{'='*60}\n")

    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return 1

    # Organize
    stats = organize_images(source_dir, output_dir, dry_run=args.dry_run)

    # Print summary
    print_stats(stats, args.model)

    if not args.dry_run:
        print(f"\n  Output saved to: {output_dir}")
        print(f"\n  S3 upload command:")
        print(f"    aws s3 sync {output_dir} s3://YOUR-BUCKET/i2i-bias-results/{args.model}/")

    return 0


if __name__ == "__main__":
    exit(main())
