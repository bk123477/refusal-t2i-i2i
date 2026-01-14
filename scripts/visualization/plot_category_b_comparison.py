#!/usr/bin/env python3
"""
Category B Pilot Comparison Plot
Base vs Edited vs Preserved for Occupational Stereotype prompts
Shows identity preservation across 7 races
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source_images" / "final"
PILOT_DIR = PROJECT_ROOT / "data" / "results" / "category_b_pilot"
PROMPTS_FILE = PROJECT_ROOT / "data" / "prompts" / "i2i_prompts.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "category_b_pilot" / "plots"

RACES = ["White", "Black", "EastAsian", "SoutheastAsian", "Indian", "MiddleEastern", "Latino"]
DISPLAY = {
    "White": "White", "Black": "Black", "EastAsian": "E.Asian",
    "SoutheastAsian": "SE.Asian", "Indian": "Indian",
    "MiddleEastern": "M.East", "Latino": "Latino"
}

# Short names for prompts
PROMPT_NAMES = {
    "B01": "CEO",
    "B02": "Military",
    "B03": "Doctor",
    "B04": "Teacher",
    "B05": "Housekeeper",
    "B06": "Security",
    "B07": "Politician",
    "B08": "Athlete",
    "B09": "Model",
    "B10": "Janitor"
}

AGE = "20s"


def load_img(path):
    if path.exists():
        return mpimg.imread(str(path))
    return None


def load_prompts():
    with open(PROMPTS_FILE, "r") as f:
        data = json.load(f)
        return [p for p in data["prompts"] if p["id"].startswith("B")]


def create_prompt_plot(prompt_id: str, prompt_text: str, gender: str):
    """Create comparison plot for a single prompt."""
    short_name = PROMPT_NAMES.get(prompt_id, prompt_id)

    fig = plt.figure(figsize=(20, 9))

    fig.suptitle(
        f'{prompt_id} "{short_name}" - {gender} {AGE}\n'
        f'Base vs Edited (Baseline) vs Preserved (+Identity Prompt)',
        fontsize=14, fontweight='bold', y=0.98
    )

    n = len(RACES)
    w, h = 0.12, 0.26
    x_start = 0.06
    gap = 0.13

    rows = [
        ("BASE\n(Source)", 0.68, "#4CAF50",
         lambda r: SOURCE_DIR / r / f"{r}_{gender}_{AGE}.jpg"),
        ("EDITED\n(Baseline)", 0.38, "#F44336",
         lambda r: PILOT_DIR / "edited" / f"{prompt_id}_{r}_{gender}_{AGE}_edited.png"),
        ("PRESERVED\n(+ID Prompt)", 0.08, "#2196F3",
         lambda r: PILOT_DIR / "preserved" / f"{prompt_id}_{r}_{gender}_{AGE}_preserved.png"),
    ]

    for i, race in enumerate(RACES):
        x = x_start + i * gap

        for j, (label, y, color, path_fn) in enumerate(rows):
            ax = fig.add_axes([x, y, w, h])

            img = load_img(path_fn(race))
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                ax.set_facecolor('#f5f5f5')

            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)

            if j == 0:
                ax.set_title(DISPLAY[race], fontsize=11, fontweight='bold', pad=3)
            if i == 0:
                fig.text(0.02, y + h/2, label, fontsize=10, fontweight='bold',
                        va='center', ha='center', rotation=90)

    # Arrows
    fig.text(0.5, 0.66, '▼ Step1X I2I Edit (Baseline) ▼', fontsize=10, ha='center', color='#888')
    fig.text(0.5, 0.36, '▼ + Identity Preservation Prompt ▼', fontsize=10, ha='center', color='#888')

    # Legend
    for k, (c, l) in enumerate([("#4CAF50", "Base"), ("#F44336", "Edited"), ("#2196F3", "Preserved")]):
        fig.patches.append(plt.Rectangle((0.30 + k*0.15, 0.02), 0.02, 0.015,
                                         facecolor=c, transform=fig.transFigure))
        fig.text(0.33 + k*0.15, 0.027, l, fontsize=10, va='center')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{prompt_id}_{gender}_{AGE}_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
    plt.close()


def create_combined_gender_plot(prompt_id: str, prompt_text: str):
    """Create single plot with both Male and Female."""
    short_name = PROMPT_NAMES.get(prompt_id, prompt_id)

    fig = plt.figure(figsize=(20, 16))

    fig.suptitle(
        f'{prompt_id} "{short_name}"\n'
        f'Base → Edited → Preserved (+Identity Prompt)',
        fontsize=14, fontweight='bold', y=0.98
    )

    n = len(RACES)
    w, h = 0.115, 0.13
    x_start = 0.06
    gap = 0.125

    # Male section (top)
    male_rows = [
        ("MALE\nBase", 0.82, "#4CAF50",
         lambda r: SOURCE_DIR / r / f"{r}_Male_{AGE}.jpg"),
        ("MALE\nEdited", 0.67, "#F44336",
         lambda r: PILOT_DIR / "edited" / f"{prompt_id}_{r}_Male_{AGE}_edited.png"),
        ("MALE\nPreserved", 0.52, "#2196F3",
         lambda r: PILOT_DIR / "preserved" / f"{prompt_id}_{r}_Male_{AGE}_preserved.png"),
    ]

    # Female section (bottom)
    female_rows = [
        ("FEMALE\nBase", 0.35, "#4CAF50",
         lambda r: SOURCE_DIR / r / f"{r}_Female_{AGE}.jpg"),
        ("FEMALE\nEdited", 0.20, "#F44336",
         lambda r: PILOT_DIR / "edited" / f"{prompt_id}_{r}_Female_{AGE}_edited.png"),
        ("FEMALE\nPreserved", 0.05, "#2196F3",
         lambda r: PILOT_DIR / "preserved" / f"{prompt_id}_{r}_Female_{AGE}_preserved.png"),
    ]

    all_rows = male_rows + female_rows

    for i, race in enumerate(RACES):
        x = x_start + i * gap

        for j, (label, y, color, path_fn) in enumerate(all_rows):
            ax = fig.add_axes([x, y, w, h])

            img = load_img(path_fn(race))
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=9)
                ax.set_facecolor('#f5f5f5')

            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)

            if j == 0:
                ax.set_title(DISPLAY[race], fontsize=10, fontweight='bold', pad=2)

            if i == 0:
                fig.text(0.02, y + h/2, label, fontsize=8, fontweight='bold',
                        va='center', ha='center', rotation=90)

    # Section divider
    fig.text(0.5, 0.49, '─' * 60, fontsize=10, ha='center', color='#ccc')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{prompt_id}_ALL_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
    plt.close()


def main():
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} Category B prompts")

    for prompt_info in prompts:
        prompt_id = prompt_info["id"]
        prompt_text = prompt_info["prompt"]

        print(f"\n{'='*60}")
        print(f"Creating plots for {prompt_id}: {PROMPT_NAMES.get(prompt_id, prompt_id)}")

        # Individual gender plots
        create_prompt_plot(prompt_id, prompt_text, "Male")
        create_prompt_plot(prompt_id, prompt_text, "Female")

        # Combined plot
        create_combined_gender_plot(prompt_id, prompt_text)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
