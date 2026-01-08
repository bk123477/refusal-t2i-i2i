#!/usr/bin/env python3
"""
Compare I2I Refusal Bias Across Multiple Models
Generate comparative analysis and visualizations
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import DisparityMetrics
from src.analysis.statistical import StatisticalAnalyzer
from src.analysis.visualization import ResultsVisualizer


def load_model_results(results_dir: Path, model_name: str) -> pd.DataFrame:
    """Load results for a specific model."""
    model_dir = results_dir / model_name

    if not model_dir.exists():
        print(f"Warning: {model_dir} does not exist")
        return pd.DataFrame()

    # Find most recent experiment
    exp_dirs = sorted(model_dir.glob("*/"), key=lambda x: x.name, reverse=True)

    if not exp_dirs:
        print(f"Warning: No experiments found for {model_name}")
        return pd.DataFrame()

    results_file = exp_dirs[0] / "results.json"

    if not results_file.exists():
        print(f"Warning: No results.json in {exp_dirs[0]}")
        return pd.DataFrame()

    print(f"Loading {model_name} from {results_file}")

    with open(results_file) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # Normalize race field
    race_mapping = {
        "Black": "Black",
        "White": "White",
        "EastAsian": "East Asian",
        "SoutheastAsian": "Southeast Asian",
        "Indian": "Indian",
        "MiddleEastern": "Middle Eastern",
        "Latino": "Latino_Hispanic",
        "Latino_Hispanic": "Latino_Hispanic"
    }

    if "race_code" in df.columns and "race" not in df.columns:
        df["race"] = df["race_code"]

    if "race" in df.columns:
        df["race"] = df["race"].map(lambda x: race_mapping.get(x, x))

    return df


def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--results-dir", type=str, default="data/results",
                       help="Base results directory")
    parser.add_argument("--models", type=str, nargs="+",
                       default=["flux", "step1x", "qwen"],
                       help="Models to compare")
    parser.add_argument("--output-dir", type=str, default="results/comparison",
                       help="Output directory")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all models
    results_by_model = {}
    for model in args.models:
        df = load_model_results(results_dir, model)
        if not df.empty:
            results_by_model[model] = df

    if not results_by_model:
        print("Error: No model results loaded")
        return

    print(f"\nLoaded {len(results_by_model)} models:")
    for model, df in results_by_model.items():
        print(f"  {model}: {len(df)} results")

    # Initialize analyzers
    disparity = DisparityMetrics()
    analyzer = StatisticalAnalyzer()
    visualizer = ResultsVisualizer(output_dir=str(output_dir / "figures"))

    print("\n" + "="*60)
    print("MODEL COMPARISON ANALYSIS")
    print("="*60)

    # 1. Overall Refusal Rates
    print("\n1. OVERALL REFUSAL RATES")
    print("-"*40)
    overall_rates = {}
    for model, df in results_by_model.items():
        rate = df["is_refused"].mean()
        overall_rates[model] = rate
        print(f"   {model}: {rate:.2%}")

    # 2. Disparity Comparison
    print("\n2. DISPARITY COMPARISON")
    print("-"*40)
    disparity_comparison = {}

    for model, df in results_by_model.items():
        disparity_comparison[model] = {}

        for cat in ["A", "B", "C", "D", "E"]:
            cat_df = df[df["category"] == cat]
            if len(cat_df) > 0:
                disp_result = disparity.calculate_disparity(cat_df.to_dict("records"))
                disparity_comparison[model][cat] = {
                    "disparity": disp_result.disparity,
                    "max_race": disp_result.max_race,
                    "min_race": disp_result.min_race,
                    "p_value": disp_result.p_value
                }

        print(f"\n   {model}:")
        for cat in ["A", "B", "C", "D", "E"]:
            if cat in disparity_comparison[model]:
                d = disparity_comparison[model][cat]
                print(f"      Cat {cat}: Δ = {d['disparity']:.2%} "
                      f"({d['min_race']} → {d['max_race']}, p={d['p_value']:.3f})")

    # 3. Race-wise Refusal Rates
    print("\n3. RACE-WISE REFUSAL RATES")
    print("-"*40)

    races = ["White", "Black", "East Asian", "Southeast Asian",
             "Indian", "Middle Eastern", "Latino_Hispanic"]

    race_rates = {race: {} for race in races}

    for model, df in results_by_model.items():
        for race in races:
            race_df = df[df["race"] == race]
            if len(race_df) > 0:
                race_rates[race][model] = race_df["is_refused"].mean()

    for race in races:
        if race_rates[race]:
            print(f"\n   {race}:")
            for model, rate in race_rates[race].items():
                print(f"      {model}: {rate:.2%}")

    # 4. Category-wise Refusal Rates
    print("\n4. CATEGORY-WISE REFUSAL RATES")
    print("-"*40)

    categories = ["A", "B", "C", "D", "E"]
    category_names = {
        "A": "Neutral",
        "B": "Occupation",
        "C": "Cultural",
        "D": "Vulnerability",
        "E": "Harmful"
    }

    cat_rates = {cat: {} for cat in categories}

    for model, df in results_by_model.items():
        for cat in categories:
            cat_df = df[df["category"] == cat]
            if len(cat_df) > 0:
                cat_rates[cat][model] = cat_df["is_refused"].mean()

    for cat in categories:
        if cat_rates[cat]:
            print(f"\n   {cat} ({category_names[cat]}):")
            for model, rate in cat_rates[cat].items():
                print(f"      {model}: {rate:.2%}")

    # 5. Generate Visualizations
    print("\n5. GENERATING VISUALIZATIONS")
    print("-"*40)

    visualizer.plot_model_comparison(results_by_model)
    print("   ✓ Model comparison")

    # 6. Save comparison report
    comparison_report = {
        "models": list(results_by_model.keys()),
        "overall_rates": overall_rates,
        "disparity_by_model": disparity_comparison,
        "race_rates": race_rates,
        "category_rates": cat_rates
    }

    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison_report, f, indent=2, default=str)

    print(f"\n✓ Comparison report saved to {report_path}")
    print(f"✓ Figures saved to {output_dir / 'figures'}")

    # 7. Generate LaTeX comparison table
    print("\n6. GENERATING LATEX COMPARISON TABLE")
    print("-"*40)

    latex = "\\begin{table}[htbp]\n\\centering\n"
    latex += "\\caption{Model Comparison: Overall Refusal Rates}\n"
    latex += "\\begin{tabular}{l" + "c" * len(categories) + "c}\n"
    latex += "\\toprule\n"
    latex += "Model & " + " & ".join([category_names[c] for c in categories]) + " & Overall \\\\\n"
    latex += "\\midrule\n"

    for model in results_by_model.keys():
        row = [model]
        for cat in categories:
            rate = cat_rates[cat].get(model, 0)
            row.append(f"{rate:.1%}")
        row.append(f"{overall_rates[model]:.1%}")
        latex += " & ".join(row) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"

    latex_path = output_dir / "model_comparison_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)

    print(f"   ✓ LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
