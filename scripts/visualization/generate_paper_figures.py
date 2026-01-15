#!/usr/bin/env python3
"""
Generate Paper Figures for IJCAI 2026
Based on Step1X pilot experiment data
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style for academic papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (colorblind-friendly)
COLORS = {
    'success': '#2ecc71',      # Green
    'unchanged': '#f39c12',    # Orange
    'refused': '#e74c3c',      # Red
    'neutral': '#95a5a6',      # Gray
}

CATEGORY_NAMES = {
    'A': 'Neutral\nBaseline',
    'B': 'Occupational\nStereotype',
    'C': 'Cultural\nExpression',
    'D': 'Vulnerability\nAttributes',
    'E': 'Harmful\nContent'
}


def load_pilot_data():
    """Load pilot experiment data."""
    data_path = Path(__file__).parent.parent.parent / "analysis" / "step1x_black_viz_data.json"
    with open(data_path) as f:
        return json.load(f)


def figure1_category_outcomes(data, output_dir):
    """
    Figure 1: Stacked bar chart of outcomes by category
    Shows success/unchanged/refused rates for each category
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    categories = ['A', 'B', 'C', 'D']
    cat_data = {c['category']: c for c in data['by_category']}

    x = np.arange(len(categories))
    width = 0.6

    success_rates = [cat_data[c]['success_rate'] * 100 for c in categories]
    unchanged_rates = [cat_data[c]['unchanged_rate'] * 100 for c in categories]
    refused_rates = [cat_data[c]['refusal_rate'] * 100 for c in categories]

    # Stacked bars
    bars1 = ax.bar(x, success_rates, width, label='Success', color=COLORS['success'])
    bars2 = ax.bar(x, unchanged_rates, width, bottom=success_rates, label='Unchanged', color=COLORS['unchanged'])
    bars3 = ax.bar(x, refused_rates, width, bottom=[s+u for s,u in zip(success_rates, unchanged_rates)],
                   label='Refused', color=COLORS['refused'])

    # Add percentage labels
    for i, (s, u, r) in enumerate(zip(success_rates, unchanged_rates, refused_rates)):
        if s > 10:
            ax.text(i, s/2, f'{s:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if u > 10:
            ax.text(i, s + u/2, f'{u:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
        if r > 5:
            ax.text(i, s + u + r/2, f'{r:.0f}%', ha='center', va='center', fontsize=8, color='white')

    ax.set_ylabel('Percentage of Requests')
    ax.set_xlabel('Prompt Category')
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_NAMES[c] for c in categories])
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')
    ax.set_title('Step1X-Edit Response Distribution by Category\n(Black demographic pilot, n=254)')

    # Add sample sizes
    for i, c in enumerate(categories):
        n = cat_data[c]['total']
        ax.text(i, -8, f'n={n}', ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_category_outcomes.pdf')
    plt.savefig(output_dir / 'fig1_category_outcomes.png')
    plt.close()
    print(f"Saved: fig1_category_outcomes.pdf/png")


def figure2_category_a_breakdown(data, output_dir):
    """
    Figure 2: Category A prompt-level unchanged rates
    Shows which neutral prompts fail most often
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    prompts = data['category_a_prompts']

    # Sort by unchanged rate
    prompts_sorted = sorted(prompts, key=lambda x: x['unchanged_rate'], reverse=True)

    prompt_labels = [p['prompt_id'] for p in prompts_sorted]
    unchanged_rates = [p['unchanged_rate'] * 100 for p in prompts_sorted]

    # Color bars by severity
    colors = ['#e74c3c' if r > 80 else '#f39c12' if r > 50 else '#2ecc71' for r in unchanged_rates]

    y_pos = np.arange(len(prompt_labels))
    bars = ax.barh(y_pos, unchanged_rates, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(prompt_labels)
    ax.set_xlabel('Unchanged Rate (%)')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # Add prompt descriptions as annotations
    short_desc = {
        'A01': 'Reading glasses',
        'A02': 'Lighting/white balance',
        'A03': 'Reduce noise',
        'A04': 'Subtle smile',
        'A05': 'Black & white',
        'A06': 'Gray backdrop',
        'A07': '1:1 crop',
        'A08': 'Skin blemishes',
        'A09': 'Image clarity',
        'A10': 'Contrast/exposure'
    }

    for i, (pid, rate) in enumerate(zip(prompt_labels, unchanged_rates)):
        ax.text(rate + 2, i, f'{short_desc[pid]}', va='center', fontsize=8)

    # Add threshold line
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(52, len(prompts_sorted)-0.5, 'Baseline threshold', fontsize=8, color='gray')

    ax.set_title('Category A (Neutral) Prompt Failure Analysis\nHigh unchanged rate indicates model limitation, not refusal')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='Unusable (>80%)'),
        mpatches.Patch(facecolor='#f39c12', label='Problematic (50-80%)'),
        mpatches.Patch(facecolor='#2ecc71', label='Acceptable (<50%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_category_a_breakdown.pdf')
    plt.savefig(output_dir / 'fig2_category_a_breakdown.png')
    plt.close()
    print(f"Saved: fig2_category_a_breakdown.pdf/png")


def figure3_occupation_comparison(data, output_dir):
    """
    Figure 3: Category B occupational prompt success rates
    Shows prestige vs labor role differences
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    prompts = data['category_b_prompts']

    # Define prestige levels
    prestige_roles = ['B01', 'B02', 'B03', 'B07']  # CEO, Military, Medical, Politician
    labor_roles = ['B05', 'B10']  # Housekeeper, Janitor
    neutral_roles = ['B04', 'B06', 'B08', 'B09']  # Teacher, Security, Athlete, Model

    role_names = {
        'B01': 'CEO',
        'B02': 'Military',
        'B03': 'Medical',
        'B04': 'Teacher',
        'B05': 'Housekeeper',
        'B06': 'Security',
        'B07': 'Politician',
        'B08': 'Athlete',
        'B09': 'Model',
        'B10': 'Janitor'
    }

    prompt_dict = {p['prompt_id']: p for p in prompts}

    # Sort by success rate
    sorted_ids = sorted(prompt_dict.keys(), key=lambda x: prompt_dict[x]['success_rate'], reverse=True)

    y_pos = np.arange(len(sorted_ids))
    success_rates = [prompt_dict[pid]['success_rate'] * 100 for pid in sorted_ids]

    # Color by role type
    colors = []
    for pid in sorted_ids:
        if pid in prestige_roles:
            colors.append('#3498db')  # Blue for prestige
        elif pid in labor_roles:
            colors.append('#9b59b6')  # Purple for labor
        else:
            colors.append('#1abc9c')  # Teal for neutral

    bars = ax.barh(y_pos, success_rates, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{pid}: {role_names[pid]}" for pid in sorted_ids])
    ax.set_xlabel('Success Rate (%)')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # Add percentage labels
    for i, rate in enumerate(success_rates):
        ax.text(rate + 1, i, f'{rate:.0f}%', va='center', fontsize=9)

    ax.set_title('Category B (Occupational) Success Rates\nNo significant prestige-based refusal disparity observed')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', label='High-prestige roles'),
        mpatches.Patch(facecolor='#9b59b6', label='Labor roles'),
        mpatches.Patch(facecolor='#1abc9c', label='Neutral roles')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_occupation_comparison.pdf')
    plt.savefig(output_dir / 'fig3_occupation_comparison.png')
    plt.close()
    print(f"Saved: fig3_occupation_comparison.pdf/png")


def figure4_tri_modal_framework(output_dir):
    """
    Figure 4: Conceptual diagram of tri-modal bias framework
    Hard Refusal -> Soft Erasure -> Stereotype Replacement
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Three boxes
    boxes = [
        (1, 1.5, 'Hard Refusal', '#e74c3c', 'Explicit denial\n"Cannot process\nthis request"'),
        (4, 1.5, 'Soft Erasure', '#f39c12', 'Silent omission\nRequested attribute\nnot rendered'),
        (7, 1.5, 'Stereotype\nReplacement', '#9b59b6', 'Identity drift\nDemographic shift\ntoward stereotypes'),
    ]

    for x, y, title, color, desc in boxes:
        # Main box
        rect = plt.Rectangle((x-0.9, y-0.6), 1.8, 1.2,
                             facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y+0.3, title, ha='center', va='center', fontweight='bold', fontsize=10)
        ax.text(x, y-0.25, desc, ha='center', va='center', fontsize=7, style='italic')

    # Arrows between boxes
    arrow_style = dict(arrowstyle='->', color='gray', lw=1.5)
    ax.annotate('', xy=(3.0, 1.5), xytext=(2.1, 1.5), arrowprops=arrow_style)
    ax.annotate('', xy=(6.0, 1.5), xytext=(5.1, 1.5), arrowprops=arrow_style)

    # Labels above arrows
    ax.text(2.55, 1.8, 'Bypass\ndetection', ha='center', fontsize=7, color='gray')
    ax.text(5.55, 1.8, 'Conform to\nstereotype', ha='center', fontsize=7, color='gray')

    # Title
    ax.text(5, 2.7, 'Tri-Modal Bias Framework: Failure Modes in I2I Safety Alignment',
            ha='center', fontsize=11, fontweight='bold')

    # Subtitle
    ax.text(5, 0.3, 'Increasing subtlety of bias manifestation →',
            ha='center', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_trimodal_framework.pdf')
    plt.savefig(output_dir / 'fig4_trimodal_framework.png')
    plt.close()
    print(f"Saved: fig4_trimodal_framework.pdf/png")


def figure5_experimental_design(output_dir):
    """
    Figure 5: Experimental design overview
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Experimental Design: Factorial I2I Bias Evaluation',
            ha='center', fontsize=12, fontweight='bold')

    # Source Images box
    rect1 = plt.Rectangle((0.5, 3.5), 2.5, 1.5, facecolor='#3498db', alpha=0.3,
                          edgecolor='#3498db', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.75, 4.7, 'Source Images', ha='center', fontweight='bold', fontsize=10)
    ax.text(1.75, 4.2, '7 races × 2 genders', ha='center', fontsize=9)
    ax.text(1.75, 3.9, '× 6 ages = 84 images', ha='center', fontsize=9)

    # Prompts box
    rect2 = plt.Rectangle((0.5, 1.5), 2.5, 1.5, facecolor='#2ecc71', alpha=0.3,
                          edgecolor='#2ecc71', linewidth=2)
    ax.add_patch(rect2)
    ax.text(1.75, 2.7, 'Edit Prompts', ha='center', fontweight='bold', fontsize=10)
    ax.text(1.75, 2.2, '5 categories', ha='center', fontsize=9)
    ax.text(1.75, 1.9, '54 prompts total', ha='center', fontsize=9)

    # I2I Models box
    rect3 = plt.Rectangle((4, 2.5), 2, 1.5, facecolor='#9b59b6', alpha=0.3,
                          edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 3.7, 'I2I Models', ha='center', fontweight='bold', fontsize=10)
    ax.text(5, 3.2, 'FLUX.2-dev', ha='center', fontsize=8)
    ax.text(5, 2.9, 'Step1X-Edit', ha='center', fontsize=8)
    ax.text(5, 2.6, 'Qwen-Image-Edit', ha='center', fontsize=8)

    # Output box
    rect4 = plt.Rectangle((7, 2.5), 2.5, 1.5, facecolor='#e74c3c', alpha=0.3,
                          edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(rect4)
    ax.text(8.25, 3.7, 'Evaluation', ha='center', fontweight='bold', fontsize=10)
    ax.text(8.25, 3.2, 'VLM Ensemble', ha='center', fontsize=8)
    ax.text(8.25, 2.9, '+ Human Validation', ha='center', fontsize=8)
    ax.text(8.25, 2.6, '(450 samples)', ha='center', fontsize=8)

    # Arrows
    arrow_style = dict(arrowstyle='->', color='gray', lw=2)
    ax.annotate('', xy=(3.9, 3.25), xytext=(3.1, 4.0), arrowprops=arrow_style)
    ax.annotate('', xy=(3.9, 3.25), xytext=(3.1, 2.5), arrowprops=arrow_style)
    ax.annotate('', xy=(6.9, 3.25), xytext=(6.1, 3.25), arrowprops=arrow_style)

    # Scale annotation
    ax.text(5, 1.3, '84 × 54 × 3 = 13,608 total requests', ha='center',
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Categories legend
    categories = [
        ('A: Neutral Baseline', '#95a5a6'),
        ('B: Occupational', '#3498db'),
        ('C: Cultural/Religious', '#2ecc71'),
        ('D: Vulnerability', '#f39c12'),
        ('E: Harmful/Safety', '#e74c3c'),
    ]
    for i, (cat, color) in enumerate(categories):
        ax.add_patch(plt.Rectangle((0.5 + i*1.8, 0.3), 0.3, 0.3, facecolor=color))
        ax.text(0.9 + i*1.8, 0.45, cat, fontsize=7, va='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_experimental_design.pdf')
    plt.savefig(output_dir / 'fig5_experimental_design.png')
    plt.close()
    print(f"Saved: fig5_experimental_design.pdf/png")


def main():
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "paper" / "figures"
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load data
    data = load_pilot_data()
    print(f"Loaded pilot data: {data['total_requests']} requests")

    # Generate figures
    figure1_category_outcomes(data, output_dir)
    figure2_category_a_breakdown(data, output_dir)
    figure3_occupation_comparison(data, output_dir)
    figure4_tri_modal_framework(output_dir)
    figure5_experimental_design(output_dir)

    print("\nAll figures generated successfully!")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
