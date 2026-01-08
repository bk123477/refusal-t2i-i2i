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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.prompt_loader import PromptLoader
from src.evaluation.metrics import DisparityMetrics, StereotypeCongruenceScore
from src.analysis.statistical import StatisticalAnalyzer, VLMCalibration
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.visualization import ResultsVisualizer


def load_results(results_dir: str, model: str = None) -> pd.DataFrame:
    """Load results from JSON files."""
    results_dir = Path(results_dir)

    if model:
        # Look for experiment directories
        model_dir = results_dir / model
        if model_dir.exists():
            # Find most recent experiment
            exp_dirs = sorted(model_dir.glob("*/"), key=lambda x: x.name, reverse=True)
            if exp_dirs:
                results_file = exp_dirs[0] / "results.json"
                if results_file.exists():
                    files = [results_file]
                else:
                    files = []
            else:
                files = []
        else:
            files = [results_dir / f"{model}_results.json"]
    else:
        # Find all results.json files in experiment directories
        files = list(results_dir.glob("*/*/results.json"))
        if not files:
            files = list(results_dir.glob("*_results.json"))

    all_results = []
    for f in files:
        if f.exists():
            with open(f) as fp:
                results = json.load(fp)
                all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Normalize race field (race_code -> race)
    if "race_code" in df.columns and "race" not in df.columns:
        df["race"] = df["race_code"]

    # Map race codes to full names if needed
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

    if "race" in df.columns:
        df["race"] = df["race"].map(lambda x: race_mapping.get(x, x))

    return df


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
    sensitivity = SensitivityAnalyzer()

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

    # 7. Odds Ratios
    print("\n7. ODDS RATIOS (vs White)")
    print("-"*40)
    odds_ratios = analyzer.calculate_all_odds_ratios(df, reference_race="White")
    for or_result in odds_ratios:
        if or_result["odds_ratio"] is not None:
            print(f"   {or_result['race1']}: OR = {or_result['odds_ratio']:.2f} "
                  f"(95% CI: [{or_result['ci_lower']:.2f}, {or_result['ci_upper']:.2f}])")
            print(f"      {or_result['interpretation']}")

    # 8. Pairwise Comparisons
    print("\n8. PAIRWISE COMPARISONS (Bonferroni corrected)")
    print("-"*40)
    pairwise = analyzer.pairwise_comparisons(df, correction="bonferroni")
    significant_pairs = [p for p in pairwise if p.is_significant]
    if significant_pairs:
        for pair in significant_pairs:
            print(f"   {pair.group1} vs {pair.group2}: Δ = {pair.mean_diff:.3f}, p = {pair.p_value:.4f}")
    else:
        print("   No significant pairwise differences after correction")

    # 9. Generate Visualizations
    print("\n9. GENERATING VISUALIZATIONS")
    print("-"*40)
    visualizer.plot_refusal_heatmap(df)
    print("   ✓ Refusal heatmap")
    visualizer.plot_disparity_bars(df)
    print("   ✓ Disparity bars")
    visualizer.plot_interaction_heatmap(df)
    print("   ✓ Interaction heatmap")
    visualizer.plot_race_bars(df)
    print("   ✓ Overall race bars")
    for cat in ["A", "B", "C", "D", "E"]:
        visualizer.plot_race_bars(df, category=cat)
    print("   ✓ Category-specific race bars")
    visualizer.plot_odds_ratios(odds_ratios, reference_race="White")
    print("   ✓ Odds ratios plot")
    if pairwise:
        visualizer.plot_effect_sizes(pairwise)
        print("   ✓ Effect sizes plot")
    if scs_results["per_prompt"]:
        visualizer.plot_scs_scores(scs_results)
        print("   ✓ SCS scores")

    # 10. Save comprehensive analysis report
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
        "interaction_effect": interaction,
        "odds_ratios": odds_ratios,
        "pairwise_comparisons": [
            {
                "group1": p.group1,
                "group2": p.group2,
                "mean_diff": p.mean_diff,
                "p_value": p.p_value,
                "ci_lower": p.ci_lower,
                "ci_upper": p.ci_upper,
                "significant": p.is_significant
            }
            for p in pairwise
        ],
        "scs": scs_results
    }

    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Analysis report saved to {report_path}")
    print(f"✓ Figures saved to {output_dir / 'figures'}")

    # 11. Mixed-Effects Logistic Regression
    print("\n11. MIXED-EFFECTS LOGISTIC REGRESSION")
    print("-"*40)
    mixed_effects_results = {}
    if "image_id" in df.columns:
        print("   Running with image_id as random effect...")
        me_result = analyzer.mixed_effects_logistic(
            df,
            formula="is_refused ~ C(race) + C(category)",
            random_effects=["image_id"]
        )
        if "error" not in me_result:
            print(f"   Model converged: {me_result['converged']}")
            print(f"   AIC: {me_result['model_fit']['aic']:.2f}")
            print(f"   BIC: {me_result['model_fit']['bic']:.2f}")
            print(f"   Number of groups: {me_result['n_groups']}")
            print("\n   Fixed Effects:")
            for param, stats in me_result['fixed_effects'].items():
                sig = "***" if stats['significant'] else ""
                print(f"      {param}: β = {stats['coefficient']:.4f}, p = {stats['p_value']:.4f} {sig}")
            mixed_effects_results = me_result
        else:
            print(f"   Error: {me_result['error']}")
    else:
        print("   Skipping (no image_id column)")

    # 12. Threshold Sensitivity Analysis
    print("\n12. THRESHOLD SENSITIVITY ANALYSIS")
    print("-"*40)
    sensitivity_results = {}
    if "clip_similarity" in df.columns:
        threshold_analysis = sensitivity.threshold_ablation(
            df.to_dict("records"),
            thresholds=[0.90, 0.92, 0.94, 0.95, 0.96, 0.98]
        )
        print(f"   Disparity range: {threshold_analysis['disparity_range']['min']:.3f} - {threshold_analysis['disparity_range']['max']:.3f}")
        print(f"   Disparity std: {threshold_analysis['disparity_range']['std']:.3f}")
        print(f"   Avg ranking correlation: {threshold_analysis['avg_ranking_correlation']:.3f}")
        print(f"   {threshold_analysis['interpretation']}")
        sensitivity_results["threshold_ablation"] = threshold_analysis
    else:
        print("   Skipping (no clip_similarity column)")

    # 13. Bootstrap Confidence Intervals
    print("\n13. BOOTSTRAP ANALYSIS")
    print("-"*40)
    bootstrap_results = {}
    if "image_id" in df.columns:
        print("   Running bootstrap (1000 iterations)...")
        bootstrap_result = sensitivity.bootstrap_disparity(
            df,
            n_bootstrap=1000,
            sample_unit="image"
        )
        print(f"   Disparity: {bootstrap_result['disparity'].mean:.3f}")
        print(f"   95% CI: [{bootstrap_result['disparity'].ci_lower:.3f}, {bootstrap_result['disparity'].ci_upper:.3f}]")
        print(f"   SE: {bootstrap_result['disparity'].std_error:.3f}")
        bootstrap_results = bootstrap_result
    else:
        print("   Skipping (no image_id column)")

    # 14. SCS with Log-Odds Normalization
    print("\n14. STEREOTYPE CONGRUENCE SCORE (LOG-ODDS)")
    print("-"*40)
    scs_log_odds_results = scs_calculator.calculate_scs_log_odds(df.to_dict("records"))
    print(f"   Overall SCS (log-odds): {scs_log_odds_results['overall_scs_log_odds']:.3f}")
    print(f"   Interpretation: {scs_log_odds_results['interpretation']}")

    # 15. Generate LaTeX tables
    print("\n15. GENERATING LATEX TABLES")
    print("-"*40)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    latex_table = visualizer.generate_latex_table(df)
    latex_path = tables_dir / "refusal_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"   ✓ Refusal rates table: {latex_path}")

    disparity_table = visualizer.generate_disparity_latex_table(df)
    disparity_latex_path = tables_dir / "disparity_table.tex"
    with open(disparity_latex_path, "w") as f:
        f.write(disparity_table)
    print(f"   ✓ Disparity table: {disparity_latex_path}")

    # 16. Save extended analysis results
    print("\n16. SAVING EXTENDED ANALYSIS RESULTS")
    print("-"*40)

    # Mixed-effects results
    if mixed_effects_results:
        me_path = output_dir / "mixed_effects_results.json"
        with open(me_path, "w") as f:
            json.dump(mixed_effects_results, f, indent=2, default=str)
        print(f"   ✓ Mixed-effects results: {me_path}")

    # Sensitivity analysis
    if sensitivity_results:
        sens_path = output_dir / "sensitivity_analysis.json"
        with open(sens_path, "w") as f:
            json.dump(sensitivity_results, f, indent=2, default=str)
        print(f"   ✓ Sensitivity analysis: {sens_path}")

    # Bootstrap results
    if bootstrap_results:
        bootstrap_summary = {
            "disparity": {
                "mean": bootstrap_results['disparity'].mean,
                "ci_lower": bootstrap_results['disparity'].ci_lower,
                "ci_upper": bootstrap_results['disparity'].ci_upper,
                "std_error": bootstrap_results['disparity'].std_error
            },
            "rates_by_race": {
                race: {
                    "mean": result.mean,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "std_error": result.std_error
                }
                for race, result in bootstrap_results['rates_by_race'].items()
            },
            "n_bootstrap": bootstrap_results['n_bootstrap'],
            "sample_unit": bootstrap_results['sample_unit']
        }
        bootstrap_path = output_dir / "bootstrap_results.json"
        with open(bootstrap_path, "w") as f:
            json.dump(bootstrap_summary, f, indent=2, default=str)
        print(f"   ✓ Bootstrap results: {bootstrap_path}")

    # SCS log-odds
    scs_log_path = output_dir / "scs_log_odds.json"
    with open(scs_log_path, "w") as f:
        json.dump(scs_log_odds_results, f, indent=2, default=str)
    print(f"   ✓ SCS log-odds results: {scs_log_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - analysis_report.json (main results)")
    print(f"  - mixed_effects_results.json (reviewer feedback #1)")
    print(f"  - sensitivity_analysis.json (reviewer feedback #2)")
    print(f"  - bootstrap_results.json (reviewer feedback #5)")
    print(f"  - scs_log_odds.json (reviewer feedback #4)")
    print(f"  - tables/ (LaTeX tables)")
    print(f"  - figures/ (all visualizations)")


if __name__ == "__main__":
    main()
