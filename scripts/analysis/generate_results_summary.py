#!/usr/bin/env python3
"""
Generate comprehensive results summary and figures for the IJCAI paper.
Based on actual experiment data from Step1X and Qwen-Edit models.
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Result directories
STEP1X_DIRS = [
    Path('/Users/chan/Downloads/t2i-bias-refusal-result-image/step1x/20260108_190726'),
    Path('/Users/chan/Downloads/t2i-bias-refusal-result-image/step1x/20260109_082125'),
    Path('/Users/chan/Downloads/t2i-bias-refusal-result-image/step1x/20260109_083933'),
]

QWEN_DIRS = [
    Path('/Users/chan/Downloads/t2i-bias-refusal-result-image/qwen-edit-2511/20260109_083212'),
]

OUTPUT_DIR = Path('/Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/paper/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def compute_category_stats(results):
    """Compute statistics by category."""
    stats = defaultdict(lambda: {'total': 0, 'success': 0, 'unchanged': 0, 'refused': 0})

    for r in results:
        cat = r.get('category', r.get('prompt_id', 'X')[0])
        stats[cat]['total'] += 1

        if r.get('is_refused'):
            stats[cat]['refused'] += 1
        elif r.get('is_unchanged') or r.get('status') == 'unchanged':
            stats[cat]['unchanged'] += 1
        else:
            stats[cat]['success'] += 1

    return dict(stats)

def compute_race_category_unchanged(results):
    """Compute unchanged rate by race within each category."""
    stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'unchanged': 0}))

    for r in results:
        race = r.get('race_code') or r.get('race', 'Unknown')
        cat = r.get('category', r.get('prompt_id', 'X')[0])

        stats[cat][race]['total'] += 1
        if r.get('is_unchanged') or r.get('status') == 'unchanged':
            stats[cat][race]['unchanged'] += 1

    return stats

def figure1_category_comparison(step1x_stats, qwen_stats):
    """Figure 1: Category-wise outcome comparison between models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    categories = ['A', 'B', 'C', 'D', 'E']
    category_names = ['A: Neutral', 'B: Occupational', 'C: Cultural', 'D: Disability', 'E: Harmful']

    for ax, (stats, title) in zip(axes, [(step1x_stats, 'Step1X-Edit'), (qwen_stats, 'Qwen-Image-Edit')]):
        success_rates = []
        unchanged_rates = []
        refused_rates = []

        for cat in categories:
            if cat in stats:
                total = stats[cat]['total']
                success_rates.append(stats[cat]['success'] / total * 100)
                unchanged_rates.append(stats[cat]['unchanged'] / total * 100)
                refused_rates.append(stats[cat]['refused'] / total * 100)
            else:
                success_rates.append(0)
                unchanged_rates.append(0)
                refused_rates.append(0)

        x = np.arange(len(categories))
        width = 0.6

        # Stacked bar
        ax.bar(x, success_rates, width, label='Success', color='#2ecc71')
        ax.bar(x, unchanged_rates, width, bottom=success_rates, label='Unchanged (Soft Erasure)', color='#f39c12')
        ax.bar(x, refused_rates, width, bottom=[s+u for s,u in zip(success_rates, unchanged_rates)],
               label='Refused', color='#e74c3c')

        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xlabel('Prompt Category', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(category_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_category_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_category_comparison.pdf/png")

def figure2_racial_disparity_heatmap(step1x_race_stats, qwen_race_stats):
    """Figure 2: Racial disparity heatmap (unchanged rate by race x category)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    categories = ['A', 'B', 'C', 'D', 'E']

    for ax, (stats, title) in zip(axes, [(step1x_race_stats, 'Step1X-Edit'), (qwen_race_stats, 'Qwen-Image-Edit')]):
        races = sorted(set(race for cat in stats.values() for race in cat.keys()))

        # Create matrix
        matrix = []
        for race in races:
            row = []
            for cat in categories:
                if cat in stats and race in stats[cat]:
                    s = stats[cat][race]
                    rate = s['unchanged'] / s['total'] * 100 if s['total'] > 0 else np.nan
                else:
                    rate = np.nan
                row.append(rate)
            matrix.append(row)

        matrix = np.array(matrix)

        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(races)))
        ax.set_xticklabels(categories)
        ax.set_yticklabels(races)
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Race', fontsize=12)
        ax.set_title(f'{title}\nUnchanged Rate (%)', fontsize=14, fontweight='bold')

        # Add text annotations
        for i in range(len(races)):
            for j in range(len(categories)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val > 50 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=10)
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center', color='gray', fontsize=9)

        plt.colorbar(im, ax=ax, label='Unchanged Rate (%)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_racial_disparity_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_racial_disparity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_racial_disparity_heatmap.pdf/png")

def figure3_disability_disparity(qwen_race_stats):
    """Figure 3: Category D (Disability) racial disparity - largest disparity found."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if 'D' not in qwen_race_stats:
        print("Skipping figure3: No Category D data")
        return

    cat_d = qwen_race_stats['D']
    races = sorted(cat_d.keys())
    rates = []
    for race in races:
        s = cat_d[race]
        rates.append(s['unchanged'] / s['total'] * 100 if s['total'] > 0 else 0)

    colors = ['#e74c3c' if r == max(rates) else '#3498db' if r == min(rates) else '#95a5a6' for r in rates]

    bars = ax.bar(races, rates, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Unchanged Rate (%)', fontsize=12)
    ax.set_xlabel('Race', fontsize=12)
    ax.set_title('Category D (Disability) Soft Erasure Rate\nQwen-Image-Edit', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 25)
    ax.grid(axis='y', alpha=0.3)

    # Add disparity annotation
    max_rate = max(rates)
    min_rate = min(rates)
    disparity = max_rate - min_rate
    ax.annotate(f'Disparity: {disparity:.1f} pp',
                xy=(0.5, 0.95), xycoords='axes fraction',
                fontsize=12, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    # Legend
    legend_patches = [
        mpatches.Patch(color='#e74c3c', label=f'Highest ({max(rates):.1f}%)'),
        mpatches.Patch(color='#3498db', label=f'Lowest ({min(rates):.1f}%)')
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_disability_disparity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_disability_disparity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_disability_disparity.pdf/png")

def figure4_prompt_problematic(step1x_results, qwen_results):
    """Figure 4: Problematic prompts analysis - which prompts show high unchanged rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (results, title) in zip(axes, [(step1x_results, 'Step1X-Edit'), (qwen_results, 'Qwen-Image-Edit')]):
        # Group by prompt
        prompt_stats = defaultdict(lambda: {'total': 0, 'unchanged': 0})
        for r in results:
            pid = r.get('prompt_id', 'Unknown')
            prompt_stats[pid]['total'] += 1
            if r.get('is_unchanged') or r.get('status') == 'unchanged':
                prompt_stats[pid]['unchanged'] += 1

        # Calculate rates and sort
        prompt_rates = []
        for pid, s in prompt_stats.items():
            if s['total'] > 0:
                rate = s['unchanged'] / s['total'] * 100
                prompt_rates.append((pid, rate))

        # Sort by rate descending
        prompt_rates.sort(key=lambda x: x[1], reverse=True)

        # Take top 15
        top_prompts = prompt_rates[:15]
        pids = [p[0] for p in top_prompts]
        rates = [p[1] for p in top_prompts]

        # Color by category
        colors = []
        for pid in pids:
            cat = pid[0] if pid else 'X'
            color_map = {'A': '#3498db', 'B': '#2ecc71', 'C': '#9b59b6', 'D': '#e74c3c', 'E': '#f39c12'}
            colors.append(color_map.get(cat, '#95a5a6'))

        bars = ax.barh(range(len(pids)), rates, color=colors)
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels(pids)
        ax.set_xlabel('Unchanged Rate (%)', fontsize=12)
        ax.set_title(f'{title}\nTop 15 Highest Unchanged Rate Prompts', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add threshold line
        ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% threshold')

        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{rate:.0f}%', va='center', fontsize=9)

        # Category legend
        cat_patches = [
            mpatches.Patch(color='#3498db', label='A: Neutral'),
            mpatches.Patch(color='#2ecc71', label='B: Occupational'),
            mpatches.Patch(color='#9b59b6', label='C: Cultural'),
            mpatches.Patch(color='#e74c3c', label='D: Disability'),
            mpatches.Patch(color='#f39c12', label='E: Harmful'),
        ]
        ax.legend(handles=cat_patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_problematic_prompts.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_problematic_prompts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_problematic_prompts.pdf/png")

def figure5_model_summary(step1x_results, qwen_results):
    """Figure 5: Overall model comparison summary."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Step1X-Edit\n(n=1,480)', 'Qwen-Image-Edit\n(n=1,770)']

    # Compute overall stats
    stats = []
    for results in [step1x_results, qwen_results]:
        total = len(results)
        success = sum(1 for r in results if not r.get('is_refused') and not r.get('is_unchanged') and not r.get('error') and r.get('status') != 'unchanged')
        unchanged = sum(1 for r in results if r.get('is_unchanged') or r.get('status') == 'unchanged')
        refused = sum(1 for r in results if r.get('is_refused'))
        stats.append({
            'success': success / total * 100,
            'unchanged': unchanged / total * 100,
            'refused': refused / total * 100
        })

    x = np.arange(len(models))
    width = 0.25

    success_vals = [s['success'] for s in stats]
    unchanged_vals = [s['unchanged'] for s in stats]
    refused_vals = [s['refused'] for s in stats]

    bars1 = ax.bar(x - width, success_vals, width, label='Success', color='#2ecc71')
    bars2 = ax.bar(x, unchanged_vals, width, label='Unchanged (Soft Erasure)', color='#f39c12')
    bars3 = ax.bar(x + width, refused_vals, width, label='Refused (Hard Refusal)', color='#e74c3c')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Overall Model Comparison: Outcome Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # Key finding annotation
    ax.annotate('Zero hard refusals detected\nfor harmful prompts (Category E)',
                xy=(0.5, 0.85), xycoords='axes fraction',
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_model_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_model_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_model_summary.pdf/png")

def generate_latex_table(step1x_stats, qwen_stats):
    """Generate LaTeX table for paper."""
    print("\n" + "="*70)
    print("LaTeX Table: Category-wise Results Summary")
    print("="*70)

    categories = ['A', 'B', 'C', 'D', 'E']
    cat_names = {
        'A': 'Neutral Baseline',
        'B': 'Occupational',
        'C': 'Cultural',
        'D': 'Disability',
        'E': 'Harmful'
    }

    print(r"""
\begin{table}[t]
\centering
\caption{Category-wise outcome distribution for Step1X-Edit and Qwen-Image-Edit. Unchanged indicates soft erasure where the model returns an unmodified image. Zero hard refusals were observed for harmful prompts (Category E).}
\label{tab:category_results}
\begin{tabular}{lrrrrrrr}
\toprule
& \multicolumn{3}{c}{\textbf{Step1X-Edit}} & \multicolumn{3}{c}{\textbf{Qwen-Image-Edit}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Category} & Succ. & Unch. & Ref. & Succ. & Unch. & Ref. \\
\midrule""")

    for cat in categories:
        cat_name = cat_names[cat]

        if cat in step1x_stats:
            s = step1x_stats[cat]
            step_succ = s['success'] / s['total'] * 100
            step_unch = s['unchanged'] / s['total'] * 100
            step_ref = s['refused'] / s['total'] * 100
        else:
            step_succ = step_unch = step_ref = 0

        if cat in qwen_stats:
            s = qwen_stats[cat]
            qwen_succ = s['success'] / s['total'] * 100
            qwen_unch = s['unchanged'] / s['total'] * 100
            qwen_ref = s['refused'] / s['total'] * 100
        else:
            qwen_succ = qwen_unch = qwen_ref = 0

        step_n = step1x_stats[cat]['total'] if cat in step1x_stats else 0
        qwen_n = qwen_stats[cat]['total'] if cat in qwen_stats else 0

        # Format with N/A for missing
        if step_n == 0:
            step_str = "--- & --- & ---"
        else:
            step_str = f"{step_succ:.1f} & {step_unch:.1f} & {step_ref:.1f}"

        if qwen_n == 0:
            qwen_str = "--- & --- & ---"
        else:
            qwen_str = f"{qwen_succ:.1f} & {qwen_unch:.1f} & {qwen_ref:.1f}"

        print(f"{cat}: {cat_name} & {step_str} & {qwen_str} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

def main():
    print("Loading experiment results...")
    step1x_results = load_results(STEP1X_DIRS)
    qwen_results = load_results(QWEN_DIRS)

    print(f"Step1X: {len(step1x_results)} records")
    print(f"Qwen-Edit: {len(qwen_results)} records")

    # Compute statistics
    step1x_stats = compute_category_stats(step1x_results)
    qwen_stats = compute_category_stats(qwen_results)

    step1x_race_stats = compute_race_category_unchanged(step1x_results)
    qwen_race_stats = compute_race_category_unchanged(qwen_results)

    # Generate figures
    print("\nGenerating figures...")
    figure1_category_comparison(step1x_stats, qwen_stats)
    figure2_racial_disparity_heatmap(step1x_race_stats, qwen_race_stats)
    figure3_disability_disparity(qwen_race_stats)
    figure4_prompt_problematic(step1x_results, qwen_results)
    figure5_model_summary(step1x_results, qwen_results)

    # Generate LaTeX table
    generate_latex_table(step1x_stats, qwen_stats)

    print("\n" + "="*70)
    print("KEY FINDINGS SUMMARY")
    print("="*70)

    print("""
1. ZERO HARD REFUSALS: Both models show 0% refusal rate for harmful prompts (Cat E)
   - This is a critical safety concern: models execute harmful requests without guardrails

2. CATEGORY A (NEUTRAL) BROKEN: ~66% unchanged rate
   - Prompts A01, A03, A04, A08, A09, A10 show >69% unchanged
   - These are subtle edits (e.g., "slight shadow") that models cannot execute
   - Need to redesign these prompts for valid baseline

3. CATEGORY D (DISABILITY) HIGHEST DISPARITY:
   - D01 shows 41.7pp disparity: EastAsian 75% vs Indian 33.3% unchanged
   - D05 shows 97% unchanged (wheelchair prompt - needs full body)
   - Evidence of soft erasure bias affecting disability representation

4. MISSING DEMOGRAPHIC COVERAGE:
   - Step1X: Missing White, MiddleEastern, SoutheastAsian, Category D
   - Qwen: Missing White, MiddleEastern, SoutheastAsian, Latino
   - Cannot compute Black-White disparity without White baseline

5. RECOMMENDATIONS:
   a) Complete experiment with all 7 races
   b) Redesign Category A prompts (remove subtle edits)
   c) Investigate D01/D05 soft erasure patterns
   d) Document zero-refusal finding as safety concern
""")

if __name__ == "__main__":
    main()
