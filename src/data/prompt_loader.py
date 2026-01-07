"""
Prompt Loader for I2I Refusal Bias Study
Loads and manages 50 prompts across 5 categories
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Prompt:
    """Single edit prompt."""
    id: str
    category: str
    prompt: str
    hypothesis: str
    congruent_races: Optional[list[str]] = None
    incongruent_races: Optional[list[str]] = None
    overt_domain: Optional[str] = None


class PromptLoader:
    """Load and manage I2I edit prompts."""

    CATEGORIES = {
        "A": "Neutral Baseline",
        "B": "Occupational Stereotype",
        "C": "Cultural/Religious Expression",
        "D": "Vulnerability Attributes",
        "E": "Harmful/Safety-Triggering"
    }

    def __init__(self, prompts_path: str = "data/prompts/i2i_prompts.json"):
        self.prompts_path = Path(prompts_path)
        self.prompts: list[Prompt] = []
        self.prompts_by_category: dict[str, list[Prompt]] = {
            cat: [] for cat in self.CATEGORIES
        }
        self._load()

    def _load(self):
        """Load prompts from JSON file."""
        with open(self.prompts_path) as f:
            data = json.load(f)

        for p in data["prompts"]:
            prompt = Prompt(
                id=p["id"],
                category=p["category"],
                prompt=p["prompt"],
                hypothesis=p["hypothesis"],
                congruent_races=p.get("congruent_races"),
                incongruent_races=p.get("incongruent_races"),
                overt_domain=p.get("overt_domain")
            )
            self.prompts.append(prompt)
            self.prompts_by_category[p["category"]].append(prompt)

    def get_all(self) -> list[Prompt]:
        """Get all prompts."""
        return self.prompts

    def get_by_category(self, category: str) -> list[Prompt]:
        """Get prompts by category (A, B, C, D, or E)."""
        return self.prompts_by_category.get(category, [])

    def get_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Get prompt by ID (e.g., 'A01', 'B05')."""
        for p in self.prompts:
            if p.id == prompt_id:
                return p
        return None

    def get_cultural_prompts_with_congruence(self) -> list[Prompt]:
        """Get Category C prompts with congruence information."""
        return [p for p in self.prompts_by_category["C"]
                if p.congruent_races is not None]

    def get_harmful_prompts(self) -> list[Prompt]:
        """Get Category E (harmful) prompts."""
        return self.prompts_by_category["E"]

    def is_stereotype_congruent(self, prompt: Prompt, race: str) -> Optional[bool]:
        """Check if a prompt is stereotype-congruent for a given race."""
        if prompt.congruent_races is None:
            return None

        if race in prompt.congruent_races:
            return True
        elif race in prompt.incongruent_races:
            return False
        return None  # Neutral

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)


if __name__ == "__main__":
    loader = PromptLoader()
    print(f"Loaded {len(loader)} prompts")

    for cat, name in loader.CATEGORIES.items():
        prompts = loader.get_by_category(cat)
        print(f"  Category {cat} ({name}): {len(prompts)} prompts")
