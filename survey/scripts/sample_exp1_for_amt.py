#!/usr/bin/env python3
"""
AMT Evaluation Sampling Script for Exp1 Dataset

Goal: Sample exactly 500 items from 5,040 total samples with balanced constraints:
- 25 images per prompt (20 prompts × 25 = 500)
- Balanced across 3 models (~167 per model, or 8-9 per prompt per model)
- Within each model, balanced across gender, race, age

Author: Chan Seo
Date: 2026-01-17
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent  # I2I-T2I-Bias-Refusal/
DATA_FILE = BASE_DIR / "survey/public/data/exp1_items.json"
OUTPUT_DIR = BASE_DIR / "data/amt_sampling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling parameters
TOTAL_SAMPLES = 500
SAMPLES_PER_PROMPT = 25  # 20 prompts × 25 = 500
MODELS = ['flux', 'qwen', 'step1x']


def load_data():
    """Load exp1_items.json"""
    with open(DATA_FILE) as f:
        data = json.load(f)
    return data['items']


RACES = ['Black', 'EastAsian', 'Indian', 'Latino', 'MiddleEastern', 'SoutheastAsian', 'White']
GENDERS = ['Male', 'Female']
AGES = ['20s', '30s', '40s', '50s', '60s', '70plus']


def sample_balanced(items):
    """
    Main sampling function with STRICT prompt balance AND global demographic balance.

    Strategy:
    1. Strict: 25 items per prompt (20 prompts × 25 = 500)
    2. Global balance tracking across all selections
    3. Prioritize GLOBAL deficits, not per-prompt deficits

    Target distribution for 500 samples:
    - 20 prompts × 25 items = 500 (STRICT)
    - ~167 per model (166, 167, 167)
    - 250 per gender
    - ~71-72 per race
    - ~83-84 per age
    """
    print(f"Total items available: {len(items)}")

    # Group items by promptId
    prompt_groups = defaultdict(list)
    for item in items:
        prompt_groups[item['promptId']].append(item)

    prompts = sorted(prompt_groups.keys())
    print(f"Unique prompts: {len(prompts)}")

    # Create global lookup by (promptId, model, race, gender, age)
    item_lookup = defaultdict(list)
    for item in items:
        key = (item['promptId'], item['model'], item['race'], item['gender'], item['age'])
        item_lookup[key].append(item)

    # Shuffle within each cell
    for key in item_lookup:
        random.shuffle(item_lookup[key])

    # GLOBAL tracking
    global_counts = {
        'model': Counter(),
        'race': Counter(),
        'gender': Counter(),
        'age': Counter(),
        'prompt': Counter()
    }

    # Global targets
    target_model = TOTAL_SAMPLES / len(MODELS)  # 166.67
    target_race = TOTAL_SAMPLES / len(RACES)  # 71.43
    target_gender = TOTAL_SAMPLES / len(GENDERS)  # 250
    target_age = TOTAL_SAMPLES / len(AGES)  # 83.33

    sampled = []

    # Process prompts in rounds to ensure global balance
    # Round-robin: pick 1 item from each prompt, repeat 25 times
    for round_num in range(SAMPLES_PER_PROMPT):
        for pid in prompts:
            if global_counts['prompt'][pid] >= SAMPLES_PER_PROMPT:
                continue

            # Create all possible combinations for this prompt
            all_combos = []
            for model in MODELS:
                for race in RACES:
                    for gender in GENDERS:
                        for age in AGES:
                            key = (pid, model, race, gender, age)
                            if item_lookup[key]:
                                all_combos.append((model, race, gender, age))

            if not all_combos:
                continue

            # Priority function based on GLOBAL deficits
            def get_priority(combo):
                model, race, gender, age = combo
                model_deficit = target_model - global_counts['model'][model]
                race_deficit = target_race - global_counts['race'][race]
                gender_deficit = target_gender - global_counts['gender'][gender]
                age_deficit = target_age - global_counts['age'][age]
                # Weight all equally for global balance
                return model_deficit + race_deficit + gender_deficit + age_deficit

            # Sort by priority and add randomness for ties
            random.shuffle(all_combos)
            all_combos.sort(key=get_priority, reverse=True)

            # Pick the best combo
            for combo in all_combos:
                model, race, gender, age = combo
                key = (pid, model, race, gender, age)
                if item_lookup[key]:
                    item = item_lookup[key].pop()
                    sampled.append(item)
                    global_counts['model'][model] += 1
                    global_counts['race'][race] += 1
                    global_counts['gender'][gender] += 1
                    global_counts['age'][age] += 1
                    global_counts['prompt'][pid] += 1
                    break

    # Verify prompt counts
    for pid in prompts:
        cnt = global_counts['prompt'][pid]
        if cnt != SAMPLES_PER_PROMPT:
            print(f"  Warning: {pid} has {cnt} items (expected {SAMPLES_PER_PROMPT})")

    print(f"Sampled {len(sampled)} items")
    return sampled


def compute_distributions(sampled_items):
    """Compute distribution tables for gender, race, age."""
    df = pd.DataFrame(sampled_items)

    results = {}

    # Overall distributions
    results['overall'] = {
        'gender': df['gender'].value_counts().to_dict(),
        'race': df['race'].value_counts().to_dict(),
        'age': df['age'].value_counts().to_dict(),
        'model': df['model'].value_counts().to_dict(),
        'category': df['category'].value_counts().to_dict(),
    }

    # Per-model distributions
    results['per_model'] = {}
    for model in MODELS:
        model_df = df[df['model'] == model]
        results['per_model'][model] = {
            'total': len(model_df),
            'gender': model_df['gender'].value_counts().to_dict(),
            'race': model_df['race'].value_counts().to_dict(),
            'age': model_df['age'].value_counts().to_dict(),
        }

    return results, df


def print_distributions(results):
    """Pretty print distribution tables."""
    print("\n" + "="*60)
    print("SAMPLING RESULTS")
    print("="*60)

    print("\n📊 OVERALL DISTRIBUTIONS")
    print("-"*40)

    for attr in ['model', 'category', 'gender', 'race', 'age']:
        print(f"\n{attr.upper()}:")
        dist = results['overall'][attr]
        total = sum(dist.values())
        for k, v in sorted(dist.items()):
            pct = v / total * 100
            bar = '█' * int(pct / 2)
            print(f"  {k:15} {v:4} ({pct:5.1f}%) {bar}")

    print("\n" + "-"*40)
    print("📊 PER-MODEL DISTRIBUTIONS")
    print("-"*40)

    for model in MODELS:
        model_data = results['per_model'][model]
        print(f"\n🔹 {model.upper()} (n={model_data['total']})")

        for attr in ['gender', 'race', 'age']:
            print(f"  {attr}:")
            dist = model_data[attr]
            total = sum(dist.values())
            for k, v in sorted(dist.items()):
                pct = v / total * 100
                print(f"    {k:15} {v:3} ({pct:5.1f}%)")


def visualize_distributions(df, output_dir):
    """Create bar charts for distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('AMT Sampling Distributions (n=500)', fontsize=14, fontweight='bold')

    # Overall distributions
    # Gender
    ax = axes[0, 0]
    gender_counts = df['gender'].value_counts()
    bars = ax.bar(gender_counts.index, gender_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax.set_title('Gender Distribution (Overall)')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, gender_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val), ha='center')

    # Race
    ax = axes[0, 1]
    race_counts = df['race'].value_counts().sort_index()
    bars = ax.bar(range(len(race_counts)), race_counts.values, color='#45B7D1')
    ax.set_title('Race Distribution (Overall)')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(race_counts)))
    ax.set_xticklabels(race_counts.index, rotation=45, ha='right')
    for bar, val in zip(bars, race_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha='center', fontsize=8)

    # Age
    ax = axes[0, 2]
    age_order = ['20s', '30s', '40s', '50s', '60s', '70plus']
    age_counts = df['age'].value_counts().reindex(age_order)
    bars = ax.bar(age_counts.index, age_counts.values, color='#96CEB4')
    ax.set_title('Age Distribution (Overall)')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, age_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha='center')

    # Per-model distributions
    # Gender per model
    ax = axes[1, 0]
    gender_model = df.groupby(['model', 'gender']).size().unstack(fill_value=0)
    gender_model.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
    ax.set_title('Gender Distribution per Model')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Gender')

    # Race per model
    ax = axes[1, 1]
    race_model = df.groupby(['model', 'race']).size().unstack(fill_value=0)
    race_model.plot(kind='bar', ax=ax, colormap='Set3')
    ax.set_title('Race Distribution per Model')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Race', fontsize=7, loc='upper right')

    # Age per model
    ax = axes[1, 2]
    age_model = df.groupby(['model', 'age']).size().unstack(fill_value=0)
    age_model = age_model[age_order]
    age_model.plot(kind='bar', ax=ax, colormap='Pastel1')
    ax.set_title('Age Distribution per Model')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Age', fontsize=8)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / 'amt_sampling_distributions.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Saved distribution chart to: {fig_path}")

    plt.close()


def save_results(sampled_items, df, results, output_dir):
    """Save sampling results to files."""

    # Save sampled items as JSON
    json_path = output_dir / 'exp1_amt_sampled.json'
    output_data = {
        'experiment': 'exp1_amt_evaluation',
        'description': 'Stratified sample of 500 items for AMT human evaluation',
        'sampling_seed': RANDOM_SEED,
        'total_items': len(sampled_items),
        'items_per_prompt': SAMPLES_PER_PROMPT,
        'models': MODELS,
        'items': sampled_items
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"📄 Saved JSON to: {json_path}")

    # Save as CSV
    csv_path = output_dir / 'exp1_amt_sampled.csv'
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV to: {csv_path}")

    # Save distribution summary
    summary_path = output_dir / 'sampling_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Saved summary to: {summary_path}")


def analyze_balance(df):
    """Analyze how balanced the sampling is."""
    print("\n" + "="*60)
    print("BALANCE ANALYSIS")
    print("="*60)

    # Check model balance
    model_counts = df['model'].value_counts()
    model_ideal = len(df) / len(MODELS)
    print(f"\nModel balance (ideal: {model_ideal:.1f} each):")
    for model, count in model_counts.items():
        diff = count - model_ideal
        print(f"  {model}: {count} (diff: {diff:+.1f})")

    # Check prompt balance
    prompt_counts = df['promptId'].value_counts()
    print(f"\nPrompt balance (ideal: {SAMPLES_PER_PROMPT} each):")
    print(f"  Min: {prompt_counts.min()}, Max: {prompt_counts.max()}, Mean: {prompt_counts.mean():.1f}")

    # Check demographic balance within each model
    for model in MODELS:
        model_df = df[df['model'] == model]
        n = len(model_df)

        print(f"\n{model.upper()} demographic balance:")

        # Gender
        gender_counts = model_df['gender'].value_counts()
        gender_ideal = n / 2
        print(f"  Gender (ideal: {gender_ideal:.1f} each):")
        for g, c in gender_counts.items():
            print(f"    {g}: {c} (diff: {c - gender_ideal:+.1f})")

        # Race
        race_counts = model_df['race'].value_counts()
        race_ideal = n / 7
        print(f"  Race (ideal: {race_ideal:.1f} each):")
        max_diff = max(abs(c - race_ideal) for c in race_counts.values)
        print(f"    Max deviation from ideal: {max_diff:.1f}")

        # Age
        age_counts = model_df['age'].value_counts()
        age_ideal = n / 6
        print(f"  Age (ideal: {age_ideal:.1f} each):")
        max_diff = max(abs(c - age_ideal) for c in age_counts.values)
        print(f"    Max deviation from ideal: {max_diff:.1f}")


def main():
    print("="*60)
    print("AMT Evaluation Sampling Script")
    print("="*60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target samples: {TOTAL_SAMPLES}")

    # Load data
    print("\n📂 Loading data...")
    items = load_data()
    print(f"Loaded {len(items)} items")

    # Sample
    print("\n🎯 Performing stratified sampling...")
    sampled_items = sample_balanced(items)
    print(f"Sampled {len(sampled_items)} items")

    # Compute distributions
    print("\n📊 Computing distributions...")
    results, df = compute_distributions(sampled_items)

    # Print distributions
    print_distributions(results)

    # Analyze balance
    analyze_balance(df)

    # Visualize
    print("\n📈 Creating visualizations...")
    visualize_distributions(df, OUTPUT_DIR)

    # Save results
    print("\n💾 Saving results...")
    save_results(sampled_items, df, results, OUTPUT_DIR)

    print("\n" + "="*60)
    print("✅ SAMPLING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
