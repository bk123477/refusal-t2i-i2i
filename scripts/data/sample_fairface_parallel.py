#!/usr/bin/env python3
"""
Parallel FairFace Sampler
Sample multiple versions of 84 images using different random seeds.

Usage:
    # Sample 3 versions in parallel
    python scripts/sample_fairface_parallel.py --versions 3 --workers 4

    # Sample specific versions
    python scripts/sample_fairface_parallel.py --version-ids V2 V3 V4
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datasets import load_dataset
import random
import multiprocessing as mp


@dataclass
class SamplingResult:
    """Result of a single version sampling."""
    version_id: str
    success: bool
    image_count: int
    output_dir: str
    seed: int
    error: str = None


def sample_single_version(args: tuple) -> SamplingResult:
    """
    Sample one version of 84 images.
    Designed to run in a separate process.
    """
    version_id, output_base_dir, seed = args

    RACES = ['Black', 'EastAsian', 'Indian', 'Latino', 'MiddleEastern', 'SoutheastAsian', 'White']
    GENDERS = ['Male', 'Female']
    AGES = ['20s', '30s', '40s', '50s', '60s', '70plus']

    RACE_MAP = {
        'Black': 'Black',
        'EastAsian': 'East Asian',
        'Indian': 'Indian',
        'Latino': 'Latino_Hispanic',
        'MiddleEastern': 'Middle Eastern',
        'SoutheastAsian': 'Southeast Asian',
        'White': 'White'
    }

    AGE_MAP = {
        '20s': '20-29',
        '30s': '30-39',
        '40s': '40-49',
        '50s': '50-59',
        '60s': '60-69',
        '70plus': 'more than 70'
    }

    try:
        # Create output directory
        output_dir = Path(output_base_dir) / version_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        print(f"[{version_id}] Loading FairFace dataset...")
        dataset = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train")

        race_names = dataset.features["race"].names
        gender_names = dataset.features["gender"].names
        age_names = dataset.features["age"].names

        random.seed(seed)
        images_metadata = []
        count = 0
        total = len(RACES) * len(GENDERS) * len(AGES)

        print(f"[{version_id}] Sampling {total} images (seed={seed})...")

        for race_code in RACES:
            # Create race subfolder
            race_dir = output_dir / race_code
            race_dir.mkdir(exist_ok=True)

            race_label = RACE_MAP[race_code]
            race_idx = race_names.index(race_label)

            for gender in GENDERS:
                gender_idx = gender_names.index(gender)

                for age_code in AGES:
                    age_label = AGE_MAP[age_code]
                    age_idx = age_names.index(age_label)

                    # Filter candidates
                    filtered = dataset.filter(
                        lambda x: x["race"] == race_idx and x["gender"] == gender_idx and x["age"] == age_idx
                    )

                    if len(filtered) == 0:
                        print(f"[{version_id}] WARNING: No candidates for {race_code}/{gender}/{age_code}")
                        continue

                    # Random selection
                    idx = random.randint(0, len(filtered) - 1)
                    item = filtered[idx]

                    # Save image
                    filename = f"{race_code}_{gender}_{age_code}.jpg"
                    filepath = race_dir / filename
                    item["image"].save(filepath)

                    images_metadata.append({
                        'filename': filename,
                        'race': race_label,
                        'race_code': race_code,
                        'gender': gender,
                        'age': age_label,
                        'age_code': age_code,
                        'path': str(filepath)
                    })

                    count += 1

        # Save metadata
        metadata = {
            'version': version_id,
            'seed': seed,
            'created_at': datetime.now().isoformat(),
            'total_images': len(images_metadata),
            'factorial_design': '7 races x 2 genders x 6 ages = 84 images',
            'folder_structure': '{RaceCode}/{RaceCode}_{Gender}_{AgeCode}.jpg',
            'images': images_metadata
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[{version_id}] Done! Sampled {count} images")

        return SamplingResult(
            version_id=version_id,
            success=True,
            image_count=count,
            output_dir=str(output_dir),
            seed=seed
        )

    except Exception as e:
        print(f"[{version_id}] ERROR: {e}")
        return SamplingResult(
            version_id=version_id,
            success=False,
            image_count=0,
            output_dir=str(output_dir) if 'output_dir' in dir() else "",
            seed=seed,
            error=str(e)
        )


def main():
    parser = argparse.ArgumentParser(description="Parallel FairFace Sampler")
    parser.add_argument("--output-dir", type=str,
                       default="data/source_images/fairface",
                       help="Base output directory")
    parser.add_argument("--versions", type=int, default=3,
                       help="Number of versions to sample")
    parser.add_argument("--version-ids", nargs="+", type=str,
                       help="Specific version IDs (e.g., V2 V3 V4)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--base-seed", type=int, default=1000,
                       help="Base seed (each version uses base_seed + version_num)")

    args = parser.parse_args()

    # Determine version IDs
    if args.version_ids:
        version_ids = args.version_ids
    else:
        # Generate version IDs: V2, V3, V4, ... (V1 already exists)
        version_ids = [f"V{i+2}" for i in range(args.versions)]

    # Determine workers
    workers = args.workers or min(mp.cpu_count(), len(version_ids))

    print("=" * 60)
    print("Parallel FairFace Sampler")
    print("=" * 60)
    print(f"Versions to sample: {version_ids}")
    print(f"Workers: {workers}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Prepare tasks
    tasks = []
    for i, version_id in enumerate(version_ids):
        seed = args.base_seed + i
        tasks.append((version_id, args.output_dir, seed))

    # Run in parallel
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(sample_single_version, task): task[0] for task in tasks}

        for future in as_completed(futures):
            version_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[{version_id}] Failed with exception: {e}")
                results.append(SamplingResult(
                    version_id=version_id,
                    success=False,
                    image_count=0,
                    output_dir="",
                    seed=0,
                    error=str(e)
                ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = 0
    for result in sorted(results, key=lambda x: x.version_id):
        status = "SUCCESS" if result.success else f"FAILED ({result.error})"
        print(f"  {result.version_id}: {status} - {result.image_count} images (seed={result.seed})")
        if result.success:
            success_count += 1

    print(f"\nTotal: {success_count}/{len(results)} versions completed")
    print("=" * 60)


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    main()
