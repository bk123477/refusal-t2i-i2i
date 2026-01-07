#!/usr/bin/env python3
"""
ACRB Prompt Expander: OVERT prompts + Attribute Expansion

Takes OVERT's verified benign-but-triggering prompts and adds
demographic/cultural attributes to create minimal-pair counterfactuals.

No LLM needed - simple template-based insertion for consistency.
"""

import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


# ============================================================================
# ATTRIBUTE DEFINITIONS (from ACRB paper)
# ============================================================================

ATTRIBUTES = {
    "culture": [
        {"value": "Nigerian", "marker": "Nigerian"},
        {"value": "Korean", "marker": "Korean"},
        {"value": "Kenyan", "marker": "Kenyan"},
        {"value": "Chinese", "marker": "Chinese"},
        {"value": "Indian", "marker": "Indian"},
        {"value": "American", "marker": "American"},  # baseline
    ],
    "gender": [
        {"value": "male", "marker": "male"},
        {"value": "female", "marker": "female"},
        {"value": "non-binary", "marker": "non-binary"},
    ],
    "disability": [
        {"value": "wheelchair_user", "marker": "wheelchair-using"},
        {"value": "visually_impaired", "marker": "visually impaired"},
        {"value": "hearing_impaired", "marker": "deaf"},
        {"value": "prosthetic_limb", "marker": "with a prosthetic limb"},
    ],
    "religion": [
        {"value": "Muslim", "marker": "Muslim"},
        {"value": "Jewish", "marker": "Jewish"},
        {"value": "Christian", "marker": "Christian"},
        {"value": "Hindu", "marker": "Hindu"},
        {"value": "Buddhist", "marker": "Buddhist"},
        {"value": "Sikh", "marker": "Sikh"},
    ],
    "age": [
        {"value": "elderly", "marker": "elderly"},
        {"value": "middle_aged", "marker": "middle-aged"},
        {"value": "young_adult", "marker": "young adult"},
    ],
}

# Categories from OVERT
OVERT_CATEGORIES = [
    "violence",
    "sexual_content",
    "self_harm",
    "illegal_activities",
    "privacy_individual",
    "privacy_public",
    "copyright_violations",
    "discrimination",
    "unethical_unsafe_action",
]


@dataclass
class ExpandedPrompt:
    """Single expanded prompt with attribute."""
    id: str
    base_prompt: str
    expanded_prompt: str
    category: str
    attribute_type: str
    attribute_value: str
    is_baseline: bool  # True if neutral/American baseline


def insert_attribute(prompt: str, marker: str) -> str:
    """
    Insert attribute marker into prompt.

    Strategy:
    1. Find "a person", "a man", "a woman", "someone" -> replace with attributed version
    2. If no direct person reference, prepend attribute naturally
    """
    prompt_lower = prompt.lower()

    # Pattern 1: Replace generic person references
    patterns = [
        (r'\ba person\b', f'a {marker} person'),
        (r'\bthe person\b', f'the {marker} person'),
        (r'\ba man\b', f'a {marker} man'),
        (r'\bthe man\b', f'the {marker} man'),
        (r'\ba woman\b', f'a {marker} woman'),
        (r'\bthe woman\b', f'the {marker} woman'),
        (r'\bsomeone\b', f'a {marker} person'),
        (r'\ba character\b', f'a {marker} character'),
        (r'\bthe character\b', f'the {marker} character'),
        (r'\ban individual\b', f'a {marker} individual'),
        (r'\bthe individual\b', f'the {marker} individual'),
        (r'\ba teenager\b', f'a {marker} teenager'),
        (r'\bteenagers\b', f'{marker} teenagers'),
        (r'\ba student\b', f'a {marker} student'),
        (r'\bstudents\b', f'{marker} students'),
        (r'\byoung people\b', f'{marker} young people'),
        (r'\ba young person\b', f'a {marker} young person'),
        (r'\bpeople\b', f'{marker} people'),
        (r'\ba child\b', f'a {marker} child'),
        (r'\bchildren\b', f'{marker} children'),
        (r'\ba doctor\b', f'a {marker} doctor'),
        (r'\ba nurse\b', f'a {marker} nurse'),
        (r'\ba teacher\b', f'a {marker} teacher'),
        (r'\ba worker\b', f'a {marker} worker'),
        (r'\ba user\b', f'a {marker} user'),
        (r'\bparents\b', f'{marker} parents'),
        (r'\ba parent\b', f'a {marker} parent'),
    ]

    result = prompt
    matched = False

    for pattern, replacement in patterns:
        new_result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
        if new_result != result:
            result = new_result
            matched = True
            break  # Only replace first match

    # Pattern 2: If no match, prepend attribute naturally
    if not matched:
        # Remove common starting phrases and add attribute
        action_verbs = ('create', 'generate', 'draw', 'depict', 'show',
                        'illustrate', 'produce', 'capture', 'design', 'make')

        if prompt_lower.startswith(action_verbs):
            # Find where the actual content starts
            # e.g., "create an image of X" -> "create an image of a {marker} person doing X"
            result = f"{prompt} (featuring a {marker} person)"
        else:
            result = f"A {marker} person in this scene: {prompt}"

    return result


def load_overt_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load OVERT prompts from CSV."""
    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                'seed_prompt': row.get('seed_prompt', ''),
                'image_prompt': row['image_prompt'],
                'category': row['category'],
            })
    return prompts


def load_overt_json(json_dir: Path) -> List[Dict[str, str]]:
    """Load OVERT prompts from JSON files."""
    prompts = []
    for json_file in json_dir.glob('*_full.json'):
        category = json_file.stem.replace('_full', '')
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                for img_prompt in item.get('image_prompts', []):
                    prompts.append({
                        'seed_prompt': item.get('seed_prompt', ''),
                        'image_prompt': img_prompt,
                        'category': category,
                    })
    return prompts


def expand_prompts(
    overt_prompts: List[Dict[str, str]],
    attribute_types: List[str] = None,
    max_prompts_per_category: int = None,
) -> List[ExpandedPrompt]:
    """
    Expand OVERT prompts with attributes.

    Args:
        overt_prompts: List of OVERT prompts
        attribute_types: Which attribute types to expand (default: all)
        max_prompts_per_category: Limit prompts per category (for testing)

    Returns:
        List of ExpandedPrompt objects
    """
    if attribute_types is None:
        attribute_types = list(ATTRIBUTES.keys())

    # Group by category
    by_category = {}
    for p in overt_prompts:
        cat = p['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p)

    expanded = []
    prompt_idx = 0

    for category, cat_prompts in by_category.items():
        # Limit if specified
        if max_prompts_per_category:
            cat_prompts = cat_prompts[:max_prompts_per_category]

        for base_prompt_data in cat_prompts:
            base_prompt = base_prompt_data['image_prompt']

            # Add neutral baseline
            expanded.append(ExpandedPrompt(
                id=f"acrb_{prompt_idx:05d}_neutral",
                base_prompt=base_prompt,
                expanded_prompt=base_prompt,  # unchanged
                category=category,
                attribute_type="neutral",
                attribute_value="neutral",
                is_baseline=True,
            ))

            # Expand with each attribute type
            for attr_type in attribute_types:
                for attr in ATTRIBUTES[attr_type]:
                    attr_value = attr['value']
                    marker = attr['marker']

                    # Determine if this is baseline (American for culture, etc.)
                    is_baseline = attr_value in ['American', 'Christian', 'male']

                    expanded_text = insert_attribute(base_prompt, marker)

                    expanded.append(ExpandedPrompt(
                        id=f"acrb_{prompt_idx:05d}_{attr_type}_{attr_value}",
                        base_prompt=base_prompt,
                        expanded_prompt=expanded_text,
                        category=category,
                        attribute_type=attr_type,
                        attribute_value=attr_value,
                        is_baseline=is_baseline,
                    ))

            prompt_idx += 1

    return expanded


def save_expanded_prompts(prompts: List[ExpandedPrompt], output_path: Path):
    """Save expanded prompts to JSON."""
    data = [asdict(p) for p in prompts]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(prompts)} prompts to {output_path}")


def generate_minimal_pairs(prompts: List[ExpandedPrompt]) -> List[Dict]:
    """
    Generate minimal pairs for direct comparison.

    Returns pairs like:
    {
        "base_prompt": "...",
        "category": "violence",
        "pairs": [
            {"attribute": "Nigerian", "prompt": "..."},
            {"attribute": "American", "prompt": "..."},
        ]
    }
    """
    # Group by base prompt
    by_base = {}
    for p in prompts:
        if p.base_prompt not in by_base:
            by_base[p.base_prompt] = {
                "base_prompt": p.base_prompt,
                "category": p.category,
                "variants": []
            }
        by_base[p.base_prompt]["variants"].append({
            "attribute_type": p.attribute_type,
            "attribute_value": p.attribute_value,
            "expanded_prompt": p.expanded_prompt,
            "is_baseline": p.is_baseline,
        })

    return list(by_base.values())


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Expand OVERT prompts with ACRB attributes')
    parser.add_argument('--overt-path', type=str,
                        default='/Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/external/OVERT-main/data',
                        help='Path to OVERT data directory')
    parser.add_argument('--output-dir', type=str,
                        default='/Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/data/prompts/acrb_expanded',
                        help='Output directory for expanded prompts')
    parser.add_argument('--use-csv', action='store_true',
                        help='Use CSV instead of JSON')
    parser.add_argument('--max-per-category', type=int, default=None,
                        help='Max prompts per category (for testing)')
    parser.add_argument('--attributes', type=str, nargs='+',
                        default=['culture'],  # Start with culture only
                        choices=list(ATTRIBUTES.keys()),
                        help='Which attribute types to expand')

    args = parser.parse_args()

    overt_path = Path(args.overt_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OVERT prompts
    print("Loading OVERT prompts...")
    if args.use_csv:
        overt_prompts = load_overt_csv(overt_path / 'OVERT_full.csv')
    else:
        overt_prompts = load_overt_json(overt_path / 'OVERT-full')

    print(f"Loaded {len(overt_prompts)} OVERT prompts")

    # Show category distribution
    categories = {}
    for p in overt_prompts:
        cat = p['category']
        categories[cat] = categories.get(cat, 0) + 1
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Expand prompts
    print(f"\nExpanding with attributes: {args.attributes}")
    expanded = expand_prompts(
        overt_prompts,
        attribute_types=args.attributes,
        max_prompts_per_category=args.max_per_category,
    )

    print(f"Generated {len(expanded)} expanded prompts")

    # Save full expanded set
    save_expanded_prompts(expanded, output_dir / 'acrb_full.json')

    # Save minimal pairs format
    pairs = generate_minimal_pairs(expanded)
    with open(output_dir / 'acrb_minimal_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} minimal pair groups to acrb_minimal_pairs.json")

    # Show sample
    print("\n" + "="*60)
    print("SAMPLE EXPANSION:")
    print("="*60)
    if expanded:
        sample_base = expanded[0].base_prompt
        print(f"\nBase: {sample_base}\n")
        for p in expanded[:10]:
            if p.base_prompt == sample_base:
                print(f"  [{p.attribute_type}:{p.attribute_value}]")
                print(f"  → {p.expanded_prompt}\n")


if __name__ == '__main__':
    main()
