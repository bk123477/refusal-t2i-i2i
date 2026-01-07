#!/usr/bin/env python3
"""
Analyze I2I Refusal Bias Experiment Results
Statistical analysis + Visualization
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prompt_loader import PromptLoader
from src.evaluation.metrics import DisparityMetrics, StereotypeCongruenceScore
from src.analysis.statistical import StatisticalAnalyzer
from src.analysis.visualization import ResultsVisualizer


def load_results(results_dir: str, model: str = None) -> pd.DataFrame:
    """Load results from JSON files."""
    results_dir = Path(results_dir)

    if model:
        files = [results_dir / f"{model}_results.json"]
    else:
        files = list(results_dir.glob("*_results.json"))

    all_results = []
    for f in files:
        if f.exists():
            with open(f) as fp:
                results = json.load(fp)
                all_results.extend(results)

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, default="data/results",
                       help="Directory with results JSON files")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to analyze (or all if not specified)")
    parser.add_argument("--output-dir", type=str, default="results/analysis",
                       help="Output directory for analysis")
    parser.add_argument("--prompts", type=str,
                       default="data/prompts/i2i_prompts.json",
                       help="Path to prompts JSON")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading results...")
    df = load_results(args.results_dir, args.model)
    print(f"Loaded {len(df)} results")

    # Initialize analyzers
    analyzer = StatisticalAnalyzer()
    visualizer = ResultsVisualizer(output_dir=str(output_dir / "figures"))
    disparity = DisparityMetrics()

    # Load prompts for SCS calculation
    prompts = PromptLoader(args.prompts)
    cultural_prompts = [
        {"id": p.id, "congruent_races": p.congruent_races, "incongruent_races": p.incongruent_races}
        for p in prompts.get_cultural_prompts_with_congruence()
    ]
    scs_calculator = StereotypeCongruenceScore(cultural_prompts)

    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    # 1. Baseline Validation (H1)
    print("\n1. BASELINE VALIDATION (Category A)")
    print("-"*40)
    baseline = analyzer.baseline_validation(df)
    print(f"   Disparity: {baseline['disparity']:.2%}")
    print(f"   p-value: {baseline['p_value']:.4f}")
    print(f"   Status: {baseline['interpretation']}")

    # 2. Main Effect: Race
    print("\n2. MAIN EFFECT: RACE")
    print("-"*40)
    race_anova = analyzer.one_way_anova_race(df, "is_refused")
    print(f"   F({race_anova.df_between}, {race_anova.df_within}) = {race_anova.f_statistic:.2f}")
    print(f"   p-value: {race_anova.p_value:.4f}")
    print(f"   η² = {race_anova.effect_size:.3f}")
    print(f"   Significant: {race_anova.is_significant}")

    # 3. Main Effect: Category
    print("\n3. MAIN EFFECT: CATEGORY")
    print("-"*40)
    cat_anova = analyzer.one_way_anova_category(df, "is_refused")
    print(f"   F({cat_anova.df_between}, {cat_anova.df_within}) = {cat_anova.f_statistic:.2f}")
    print(f"   p-value: {cat_anova.p_value:.4f}")
    print(f"   η² = {cat_anova.effect_size:.3f}")
    print(f"   Significant: {cat_anova.is_significant}")

    # 4. Interaction Effect
    print("\n4. RACE × CATEGORY INTERACTION")
    print("-"*40)
    interaction = analyzer.two_way_anova(df, "is_refused")
    if "error" not in interaction:
        print(f"   Race effect: F = {interaction['race_effect']['f_stat']:.2f}, p = {interaction['race_effect']['p_value']:.4f}")
        print(f"   Category effect: F = {interaction['category_effect']['f_stat']:.2f}, p = {interaction['category_effect']['p_value']:.4f}")
        print(f"   Interaction: F = {interaction['interaction']['f_stat']:.2f}, p = {interaction['interaction']['p_value']:.4f}")

    # 5. Disparity by Category
    print("\n5. DISPARITY BY CATEGORY")
    print("-"*40)
    for cat in ["A", "B", "C", "D", "E"]:
        cat_df = df[df["category"] == cat]
        cat_disp = disparity.calculate_disparity(cat_df.to_dict("records"))
        print(f"   Category {cat}: Δ = {cat_disp.disparity:.2%} ({cat_disp.min_race} → {cat_disp.max_race})")

    # 6. Stereotype Congruence Score
    print("\n6. STEREOTYPE CONGRUENCE SCORE (Category C)")
    print("-"*40)
    baseline_rate = df[df["category"] == "A"]["is_refused"].mean()
    scs_results = scs_calculator.calculate(df.to_dict("records"), baseline_rate)
    print(f"   Overall SCS: {scs_results['overall_scs']:.3f}")
    print(f"   Interpretation: {scs_results['interpretation']}")

    # 7. Generate Visualizations
    print("\n7. GENERATING VISUALIZATIONS")
    print("-"*40)
    visualizer.plot_refusal_heatmap(df)
    print("   ✓ Refusal heatmap")
    visualizer.plot_disparity_bars(df)
    print("   ✓ Disparity bars")
    if scs_results["per_prompt"]:
        visualizer.plot_scs_scores(scs_results)
        print("   ✓ SCS scores")

    # 8. Save analysis report
    report = {
        "baseline_validation": baseline,
        "race_effect": {
            "f_statistic": race_anova.f_statistic,
            "p_value": race_anova.p_value,
            "effect_size": race_anova.effect_size,
            "significant": race_anova.is_significant
        },
        "category_effect": {
            "f_statistic": cat_anova.f_statistic,
            "p_value": cat_anova.p_value,
            "effect_size": cat_anova.effect_size,
            "significant": cat_anova.is_significant
        },
        "scs": scs_results
    }

    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Analysis report saved to {report_path}")
    print(f"✓ Figures saved to {output_dir / 'figures'}")

    # 9. Generate LaTeX table
    latex_table = visualizer.generate_latex_table(df)
    latex_path = output_dir / "refusal_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
