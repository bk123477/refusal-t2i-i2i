#!/usr/bin/env python3
"""
ACRB Plotting Utility
Generates paper-ready figures from audit results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def plot_refusal_heatmap(summary_path: Path, output_path: Path):
    """Generates a heatmap of refusal rates by attribute."""
    # Placeholder for actual plotting logic
    print(f"Generating refusal heatmap from {summary_path}...")
    plt.figure(figsize=(10, 6))
    plt.title("Attribute-Conditioned Refusal Bias Heatmap")
    plt.text(0.5, 0.5, "Heatmap Placeholder\n(Culture vs. Domain)", ha='center', va='center')
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

def main():
    # Example usage
    results_dir = Path("experiments/results")
    latest_run = sorted(results_dir.glob("*/*"))
    if not latest_run:
        print("No results found to plot.")
        return
        
    summary_file = latest_run[-1] / "summary.json"
    if summary_file.exists():
        plot_refusal_heatmap(summary_file, Path("figs/results/refusal_heatmap.png"))

if __name__ == "__main__":
    main()
