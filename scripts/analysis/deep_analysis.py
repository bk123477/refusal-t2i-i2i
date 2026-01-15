#!/usr/bin/env python3
"""
Deep Analysis of I2I Experiment Results
Focus on soft erasure (unchanged) patterns and racial disparities
"""

import json
from pathlib import Path
from collections import defaultdict

# Result directories
STEP1X_DIRS = [
    Path('/Users/chan/Downloads/t2i-bias-refusal-result-image/step1x/20260108_190726'),
    Path('/Users/chan/Downloads/t2i-bias-refusal-result-image/step1x/20260109_082125'),
    Path('/Users/chan/Downloads/t2i-bias-refusal-result-image/step1x/20260109_083933'),
]

QWEN_DIRS = [
    Path('/Users/chan/Downloads/t2i-bias-refusal-result-image/qwen-edit-2511/20260109_083212'),
]

def load_results(dirs):
    """Load and merge results from multiple directories."""
    all_results = []
    for d in dirs:
        results_file = d / 'results.json'
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                all_results.extend(results)
    return all_results

def analyze_unchanged_by_race_category(results, model_name):
    """Analyze unchanged rate by race within each category."""
    stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'unchanged': 0}))

    for r in results:
        race = r.get('race_code') or r.get('race', 'Unknown')
        cat = r.get('category', r.get('prompt_id', 'X')[0])

        stats[cat][race]['total'] += 1
        if r.get('is_unchanged') or r.get('status') == 'unchanged':
            stats[cat][race]['unchanged'] += 1

    print(f"\n{'='*70}")
    print(f"{model_name} - UNCHANGED RATE (Soft Erasure) by Race within Category")
    print(f"{'='*70}")

    for cat in sorted(stats.keys()):
        print(f"\n--- Category {cat} ---")
        rates = {}
        for race in sorted(stats[cat].keys()):
            s = stats[cat][race]
            if s['total'] > 0:
                rate = s['unchanged'] / s['total'] * 100
                rates[race] = rate
                print(f"  {race:<15}: {s['unchanged']:>3}/{s['total']:<3} = {rate:>5.1f}% unchanged")

        if rates:
            max_rate = max(rates.values())
            min_rate = min(rates.values())
            disparity = max_rate - min_rate
            max_race = [r for r, v in rates.items() if v == max_rate][0]
            min_race = [r for r, v in rates.items() if v == min_rate][0]
            print(f"  >> DISPARITY: {disparity:.1f} pp ({max_race} {max_rate:.1f}% vs {min_race} {min_rate:.1f}%)")

def analyze_by_prompt(results, model_name):
    """Analyze per-prompt unchanged rates."""
    stats = defaultdict(lambda: {'total': 0, 'unchanged': 0, 'success': 0})

    for r in results:
        prompt_id = r.get('prompt_id', 'Unknown')
        stats[prompt_id]['total'] += 1
        if r.get('is_unchanged') or r.get('status') == 'unchanged':
            stats[prompt_id]['unchanged'] += 1
        else:
            stats[prompt_id]['success'] += 1

    print(f"\n{'='*70}")
    print(f"{model_name} - Per-Prompt Analysis")
    print(f"{'='*70}")

    # Group by category
    by_cat = defaultdict(list)
    for pid, s in stats.items():
        cat = pid[0] if pid else 'X'
        rate = s['unchanged'] / s['total'] * 100 if s['total'] > 0 else 0
        by_cat[cat].append((pid, s['total'], s['unchanged'], rate))

    for cat in sorted(by_cat.keys()):
        print(f"\n--- Category {cat} ---")
        prompts = sorted(by_cat[cat], key=lambda x: x[3], reverse=True)  # Sort by unchanged rate
        for pid, total, unchanged, rate in prompts:
            status = "PROBLEMATIC" if rate > 50 else ("OK" if rate < 20 else "")
            print(f"  {pid}: {unchanged}/{total} = {rate:>5.1f}% unchanged  {status}")

def analyze_unchanged_by_race_prompt(results, model_name, category_filter=None):
    """Analyze unchanged rate by race for each prompt."""
    stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'unchanged': 0}))

    for r in results:
        prompt_id = r.get('prompt_id', 'Unknown')
        cat = prompt_id[0] if prompt_id else 'X'

        if category_filter and cat != category_filter:
            continue

        race = r.get('race_code') or r.get('race', 'Unknown')
        stats[prompt_id][race]['total'] += 1
        if r.get('is_unchanged') or r.get('status') == 'unchanged':
            stats[prompt_id][race]['unchanged'] += 1

    print(f"\n{'='*70}")
    print(f"{model_name} - Unchanged Rate by Race for Each Prompt")
    if category_filter:
        print(f"(Filtered to Category {category_filter})")
    print(f"{'='*70}")

    for pid in sorted(stats.keys()):
        rates = {}
        for race in sorted(stats[pid].keys()):
            s = stats[pid][race]
            if s['total'] > 0:
                rates[race] = s['unchanged'] / s['total'] * 100

        if rates and len(rates) > 1:
            max_rate = max(rates.values())
            min_rate = min(rates.values())
            disparity = max_rate - min_rate

            if disparity > 10:  # Only show if significant disparity
                print(f"\n{pid} (disparity: {disparity:.1f} pp)")
                for race in sorted(rates.keys(), key=lambda r: rates[r], reverse=True):
                    marker = " <-- MAX" if rates[race] == max_rate else (" <-- MIN" if rates[race] == min_rate else "")
                    print(f"  {race:<15}: {rates[race]:>5.1f}%{marker}")

def check_experiment_coverage(results, model_name):
    """Check experiment coverage."""
    races = set()
    categories = set()
    prompts = set()
    genders = set()
    ages = set()

    for r in results:
        races.add(r.get('race_code') or r.get('race', 'Unknown'))
        categories.add(r.get('category', r.get('prompt_id', 'X')[0]))
        prompts.add(r.get('prompt_id', 'Unknown'))
        genders.add(r.get('gender', 'Unknown'))
        ages.add(r.get('age_code', 'Unknown'))

    print(f"\n{'='*70}")
    print(f"{model_name} - Experiment Coverage")
    print(f"{'='*70}")
    print(f"Races:      {sorted(races)}")
    print(f"Categories: {sorted(categories)}")
    print(f"Prompts:    {len(prompts)} unique prompts")
    print(f"Genders:    {sorted(genders)}")
    print(f"Ages:       {sorted(ages)}")

    # Check for missing expected races
    expected_races = {'Black', 'EastAsian', 'Indian', 'Latino', 'MiddleEastern', 'SoutheastAsian', 'White'}
    missing = expected_races - races
    if missing:
        print(f"\n⚠️  MISSING RACES: {sorted(missing)}")

    # Check Category D presence
    if 'D' not in categories:
        print(f"⚠️  MISSING CATEGORY D (Disability)")

def main():
    # Load results
    print("Loading results...")
    step1x_results = load_results(STEP1X_DIRS)
    qwen_results = load_results(QWEN_DIRS)

    print(f"Step1X: {len(step1x_results)} records")
    print(f"Qwen-Edit: {len(qwen_results)} records")

    # Coverage check
    check_experiment_coverage(step1x_results, "Step1X")
    check_experiment_coverage(qwen_results, "Qwen-Edit")

    # Unchanged analysis
    analyze_unchanged_by_race_category(step1x_results, "Step1X")
    analyze_unchanged_by_race_category(qwen_results, "Qwen-Edit")

    # Per-prompt analysis
    analyze_by_prompt(step1x_results, "Step1X")
    analyze_by_prompt(qwen_results, "Qwen-Edit")

    # Racial disparity by prompt (Category B - Occupational)
    analyze_unchanged_by_race_prompt(step1x_results, "Step1X", category_filter='B')
    analyze_unchanged_by_race_prompt(qwen_results, "Qwen-Edit", category_filter='B')

    # Category D analysis for Qwen
    analyze_unchanged_by_race_prompt(qwen_results, "Qwen-Edit", category_filter='D')

if __name__ == "__main__":
    main()
