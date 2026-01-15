#!/usr/bin/env python3
"""
Analyze I2I Experiment Results
Compares Step1X and Qwen-Edit results for bias patterns
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

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

def analyze_by_category(results):
    """Analyze results by prompt category."""
    stats = defaultdict(lambda: {'total': 0, 'success': 0, 'unchanged': 0, 'refused': 0, 'error': 0})

    for r in results:
        cat = r.get('category', r.get('prompt_id', 'X')[0])
        stats[cat]['total'] += 1

        if r.get('error'):
            stats[cat]['error'] += 1
        elif r.get('is_refused'):
            stats[cat]['refused'] += 1
        elif r.get('is_unchanged') or r.get('status') == 'unchanged':
            stats[cat]['unchanged'] += 1
        else:
            stats[cat]['success'] += 1

    return dict(stats)

def analyze_by_race(results):
    """Analyze results by race."""
    stats = defaultdict(lambda: {'total': 0, 'success': 0, 'unchanged': 0, 'refused': 0, 'error': 0})

    for r in results:
        race = r.get('race_code') or r.get('race', 'Unknown')
        stats[race]['total'] += 1

        if r.get('error'):
            stats[race]['error'] += 1
        elif r.get('is_refused'):
            stats[race]['refused'] += 1
        elif r.get('is_unchanged') or r.get('status') == 'unchanged':
            stats[race]['unchanged'] += 1
        else:
            stats[race]['success'] += 1

    return dict(stats)

def analyze_by_race_and_category(results):
    """Analyze results by race and category combination."""
    stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'success': 0, 'unchanged': 0, 'refused': 0}))

    for r in results:
        race = r.get('race_code') or r.get('race', 'Unknown')
        cat = r.get('category', r.get('prompt_id', 'X')[0])

        stats[race][cat]['total'] += 1

        if r.get('error'):
            pass  # Skip errors
        elif r.get('is_refused'):
            stats[race][cat]['refused'] += 1
        elif r.get('is_unchanged') or r.get('status') == 'unchanged':
            stats[race][cat]['unchanged'] += 1
        else:
            stats[race][cat]['success'] += 1

    return {r: dict(c) for r, c in stats.items()}

def print_category_stats(stats, model_name):
    """Print category statistics."""
    print(f"\n{'='*60}")
    print(f"{model_name} - By Category")
    print(f"{'='*60}")
    print(f"{'Cat':<5} {'Total':>7} {'Success':>8} {'Unchanged':>10} {'Refused':>8} {'Error':>6} {'Success%':>9}")
    print("-" * 60)

    for cat in sorted(stats.keys()):
        s = stats[cat]
        total = s['total']
        success_pct = s['success'] / total * 100 if total > 0 else 0
        print(f"{cat:<5} {total:>7} {s['success']:>8} {s['unchanged']:>10} {s['refused']:>8} {s['error']:>6} {success_pct:>8.1f}%")

    # Total
    total = sum(s['total'] for s in stats.values())
    success = sum(s['success'] for s in stats.values())
    unchanged = sum(s['unchanged'] for s in stats.values())
    refused = sum(s['refused'] for s in stats.values())
    error = sum(s['error'] for s in stats.values())
    print("-" * 60)
    print(f"{'TOTAL':<5} {total:>7} {success:>8} {unchanged:>10} {refused:>8} {error:>6} {success/total*100:>8.1f}%")

def print_race_stats(stats, model_name):
    """Print race statistics."""
    print(f"\n{'='*60}")
    print(f"{model_name} - By Race")
    print(f"{'='*60}")
    print(f"{'Race':<15} {'Total':>7} {'Success':>8} {'Unchanged':>10} {'Refused':>8} {'Refusal%':>9}")
    print("-" * 60)

    for race in sorted(stats.keys()):
        s = stats[race]
        total = s['total']
        refusal_pct = s['refused'] / total * 100 if total > 0 else 0
        print(f"{race:<15} {total:>7} {s['success']:>8} {s['unchanged']:>10} {s['refused']:>8} {refusal_pct:>8.1f}%")

def print_disparity_analysis(race_cat_stats, model_name):
    """Print disparity analysis by race within each category."""
    print(f"\n{'='*70}")
    print(f"{model_name} - Disparity Analysis (Refusal Rate by Race within Category)")
    print(f"{'='*70}")

    categories = set()
    races = set()
    for race, cats in race_cat_stats.items():
        races.add(race)
        categories.update(cats.keys())

    for cat in sorted(categories):
        print(f"\n--- Category {cat} ---")
        refusal_rates = {}
        for race in sorted(races):
            if race in race_cat_stats and cat in race_cat_stats[race]:
                s = race_cat_stats[race][cat]
                total = s['total']
                if total > 0:
                    refusal_rates[race] = s['refused'] / total * 100

        if refusal_rates:
            sorted_rates = sorted(refusal_rates.items(), key=lambda x: x[1], reverse=True)
            max_rate = sorted_rates[0][1]
            min_rate = sorted_rates[-1][1]
            disparity = max_rate - min_rate

            for race, rate in sorted_rates:
                marker = " <-- MAX" if rate == max_rate else (" <-- MIN" if rate == min_rate else "")
                print(f"  {race:<15}: {rate:>5.1f}%{marker}")

            print(f"  Disparity (max-min): {disparity:.1f} pp")

def check_sample_result(results, n=3):
    """Print sample results for inspection."""
    print(f"\n{'='*60}")
    print(f"Sample Results (first {n})")
    print(f"{'='*60}")
    for i, r in enumerate(results[:n]):
        print(f"\n--- Result {i+1} ---")
        for k, v in r.items():
            if k not in ['output_image', 'source_image', 'prompt_text']:  # Skip long fields
                print(f"  {k}: {v}")

def main():
    # Load results
    print("Loading results...")
    step1x_results = load_results(STEP1X_DIRS)
    qwen_results = load_results(QWEN_DIRS)

    print(f"Step1X: {len(step1x_results)} records")
    print(f"Qwen-Edit: {len(qwen_results)} records")

    # Sample check
    if step1x_results:
        check_sample_result(step1x_results)

    # Analyze Step1X
    if step1x_results:
        step1x_cat = analyze_by_category(step1x_results)
        step1x_race = analyze_by_race(step1x_results)
        step1x_race_cat = analyze_by_race_and_category(step1x_results)

        print_category_stats(step1x_cat, "Step1X")
        print_race_stats(step1x_race, "Step1X")
        print_disparity_analysis(step1x_race_cat, "Step1X")

    # Analyze Qwen
    if qwen_results:
        qwen_cat = analyze_by_category(qwen_results)
        qwen_race = analyze_by_race(qwen_results)
        qwen_race_cat = analyze_by_race_and_category(qwen_results)

        print_category_stats(qwen_cat, "Qwen-Edit")
        print_race_stats(qwen_race, "Qwen-Edit")
        print_disparity_analysis(qwen_race_cat, "Qwen-Edit")

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")

    if step1x_results:
        step1x_total = len(step1x_results)
        step1x_success = sum(1 for r in step1x_results if not r.get('is_refused') and not r.get('is_unchanged') and not r.get('error'))
        step1x_unchanged = sum(1 for r in step1x_results if r.get('is_unchanged') or r.get('status') == 'unchanged')
        step1x_refused = sum(1 for r in step1x_results if r.get('is_refused'))
        print(f"Step1X:    {step1x_total} total, {step1x_success} success ({step1x_success/step1x_total*100:.1f}%), {step1x_unchanged} unchanged ({step1x_unchanged/step1x_total*100:.1f}%), {step1x_refused} refused ({step1x_refused/step1x_total*100:.1f}%)")

    if qwen_results:
        qwen_total = len(qwen_results)
        qwen_success = sum(1 for r in qwen_results if not r.get('is_refused') and not r.get('is_unchanged') and not r.get('error'))
        qwen_unchanged = sum(1 for r in qwen_results if r.get('is_unchanged') or r.get('status') == 'unchanged')
        qwen_refused = sum(1 for r in qwen_results if r.get('is_refused'))
        print(f"Qwen-Edit: {qwen_total} total, {qwen_success} success ({qwen_success/qwen_total*100:.1f}%), {qwen_unchanged} unchanged ({qwen_unchanged/qwen_total*100:.1f}%), {qwen_refused} refused ({qwen_refused/qwen_total*100:.1f}%)")

if __name__ == "__main__":
    main()
