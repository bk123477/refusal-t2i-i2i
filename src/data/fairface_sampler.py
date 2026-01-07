"""
FairFace Dataset Sampler for Factorial Design
6 Ages × 2 Genders × 7 Races = 84 Images
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datasets import load_dataset
from PIL import Image
import random


@dataclass
class ImageMetadata:
    """Metadata for a sampled image."""
    race: str
    gender: str
    age: str
    image_path: str
    original_idx: int


class FairFaceSampler:
    """Sample images from FairFace dataset with factorial design."""

    RACES = ["White", "Black", "East Asian", "Southeast Asian", "Indian", "Middle Eastern", "Latino_Hispanic"]
    GENDERS = ["Male", "Female"]
    AGES = ["20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]

    # Map FairFace age labels to our labels
    AGE_MAP = {
        "20-29": "20-29",
        "30-39": "30-39",
        "40-49": "40-49",
        "50-59": "50-59",
        "60-69": "60-69",
        "more than 70": "70+"
    }

    def __init__(self, output_dir: str = "data/source_images/fairface"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        self.sampled_images: list[ImageMetadata] = []

    def load_dataset(self):
        """Load FairFace dataset from HuggingFace."""
        print("Loading FairFace dataset from HuggingFace...")
        self.dataset = load_dataset("HuggingFaceM4/FairFace", split="train")
        print(f"Loaded {len(self.dataset)} images")

    def _get_candidates(self, race: str, gender: str, age: str) -> list[int]:
        """Get candidate indices for a specific combination."""
        candidates = []
        for idx, item in enumerate(self.dataset):
            if (item["race"] == race and
                item["gender"] == gender and
                item["age"] == age):
                candidates.append(idx)
        return candidates

    def sample(self, seed: int = 42) -> list[ImageMetadata]:
        """Sample one image per (race, gender, age) combination."""
        if self.dataset is None:
            self.load_dataset()

        random.seed(seed)
        self.sampled_images = []

        print(f"Sampling 84 images (7 races × 2 genders × 6 ages)...")

        for race in self.RACES:
            for gender in self.GENDERS:
                for age in self.AGES:
                    candidates = self._get_candidates(race, gender, age)

                    if not candidates:
                        print(f"WARNING: No candidates for {race}/{gender}/{age}")
                        continue

                    # Select one random candidate
                    idx = random.choice(candidates)

                    # Save image
                    age_label = self.AGE_MAP[age]
                    filename = f"{race}_{gender}_{age_label}.jpg"
                    filepath = self.output_dir / filename

                    image = self.dataset[idx]["image"]
                    image.save(filepath)

                    metadata = ImageMetadata(
                        race=race,
                        gender=gender,
                        age=age_label,
                        image_path=str(filepath),
                        original_idx=idx
                    )
                    self.sampled_images.append(metadata)

        print(f"Sampled {len(self.sampled_images)} images")
        return self.sampled_images

    def save_metadata(self, output_path: Optional[str] = None):
        """Save sampling metadata to JSON."""
        if output_path is None:
            output_path = self.output_dir / "metadata.json"

        metadata = {
            "total_images": len(self.sampled_images),
            "dimensions": {
                "races": self.RACES,
                "genders": self.GENDERS,
                "ages": list(self.AGE_MAP.values())
            },
            "images": [
                {
                    "race": img.race,
                    "gender": img.gender,
                    "age": img.age,
                    "path": img.image_path,
                    "original_idx": img.original_idx
                }
                for img in self.sampled_images
            ]
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {output_path}")


if __name__ == "__main__":
    sampler = FairFaceSampler()
    sampler.sample()
    sampler.save_metadata()
