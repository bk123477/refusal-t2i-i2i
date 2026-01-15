#!/usr/bin/env python3
"""
Analyze I2I Refusal Bias Experiment Results - Complete Pipeline
Supports both VLM-only and Human-Corrected analyses

Workflow:
1. Load VLM ensemble results
2. Load human review corrections (if available)
3. Generate final corrected dataset
4. Run comprehensive statistical analysis
5. Create visualizations and reports
6. Export publication-ready figures/tables
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.prompt_loader import PromptLoader
from src.evaluation.metrics import DisparityMetrics, StereotypeCongruenceScore
from src.analysis.statistical import StatisticalAnalyzer, VLMCalibration
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.visualization import ResultsVisualizer
from src.config import PathConfig


def load_human_review_results_from_firebase() -> pd.DataFrame:
    """
    Load human review results from Firebase/survey app.
    Falls back to local files if Firebase unavailable.
    """
    try:
        # Try to import Firebase functions
        import firebase_admin
        from firebase_admin import credentials, firestore

        # Initialize Firebase (if not already initialized)
        if not firebase_admin._apps:
            # Try to find service account key
            key_paths = [
                Path(__file__).parent.parent.parent / "firebase-service-account.json",
                Path.home() / "firebase-service-account.json"
            ]

            cred = None
            for key_path in key_paths:
                if key_path.exists():
                    cred = credentials.Certificate(str(key_path))
                    break

            if cred:
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized for data loading")
            else:
                print("⚠️ Firebase service account key not found, using local fallback")
                return load_human_review_results_from_local()

        # Load data from Firestore
        db = firestore.client()
        results_ref = db.collection('human_review_results')
        docs = results_ref.stream()

        human_results = []
        for doc in docs:
            data = doc.to_dict()
            human_results.append({
                'case_id': data.get('caseId') or data.get('case_id'),
                'human_judgment': data.get('human_judgment'),
                'qwen_response': data.get('qwen_response'),
                'gemini_response': data.get('gemini_response'),
                'ensemble_result': data.get('ensemble_result'),
                'attribute': data.get('attribute'),
                'timestamp': data.get('timestamp'),
                'saved_at': data.get('saved_at'),
                'reviewer': data.get('reviewer', 'unknown')
            })

        return pd.DataFrame(human_results)

    except Exception as e:
        print(f"❌ Firebase loading failed: {e}")
        return load_human_review_results_from_local()


def load_human_review_results_from_local() -> pd.DataFrame:
    """
    Load human review results from local survey app files.
    """
    survey_dir = Path(__file__).parent.parent.parent / "survey" / "public"

    # Try to find human review result files
    result_files = list(survey_dir.glob("human_review_results*.json"))

    if not result_files:
        print("⚠️ No human review result files found")
        return pd.DataFrame()

    # Load most recent file
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            human_results = data
        else:
            human_results = data.get('results', [])

        return pd.DataFrame(human_results)

    except Exception as e:
        print(f"❌ Local file loading failed: {e}")
        return pd.DataFrame()


def combine_vlm_and_human_results(vlm_results: pd.DataFrame,
                                human_results: pd.DataFrame) -> pd.DataFrame:
    """
    Combine VLM ensemble results with human corrections.

    Args:
        vlm_results: DataFrame with VLM ensemble evaluations
        human_results: DataFrame with human review corrections

    Returns:
        DataFrame: Final corrected dataset
    """
    if human_results.empty:
        print("ℹ️ No human corrections found, using VLM results only")
        final_results = vlm_results.copy()
        final_results['final_judgment'] = final_results['ensemble_result']
        final_results['correction_type'] = 'none'
        return final_results

    # Create mapping of human corrections
    human_corrections = {}
    for _, row in human_results.iterrows():
        case_key = row['case_id']
        human_corrections[case_key] = {
            'human_judgment': row['human_judgment'],
            'original_ensemble': row['ensemble_result'],
            'qwen_response': row['qwen_response'],
            'gemini_response': row['gemini_response']
        }

    # Apply corrections to VLM results
    final_results = vlm_results.copy()
    final_results['final_judgment'] = final_results.apply(
        lambda row: apply_human_correction(row, human_corrections), axis=1
    )
    final_results['correction_type'] = final_results.apply(
        lambda row: get_correction_type(row, human_corrections), axis=1
    )

    return final_results


def apply_human_correction(row: pd.Series, human_corrections: Dict) -> str:
    """Apply human correction if available, otherwise use ensemble result."""
    case_id = getattr(row, 'case_id', None) or getattr(row, 'id', None)
    if case_id and case_id in human_corrections:
        human_judgment = human_corrections[case_id]['human_judgment']
        if human_judgment and human_judgment != 'SKIPPED':
            return human_judgment
    return row['ensemble_result']


def get_correction_type(row: pd.Series, human_corrections: Dict) -> str:
    """Determine type of correction applied."""
    case_id = getattr(row, 'case_id', None) or getattr(row, 'id', None)
    if case_id and case_id in human_corrections:
        human_judgment = human_corrections[case_id]['human_judgment']
        original = row['ensemble_result']

        if human_judgment == 'SKIPPED':
            return 'skipped'
        elif human_judgment != original:
            return 'corrected'
        else:
            return 'confirmed'
    return 'no_human_review'


def load_vlm_results(results_dir: str, model: str = None) -> pd.DataFrame:
    """Load VLM ensemble results from JSON files."""
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
                for r in results:
                    r["results_file"] = str(f)
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


def load_edit_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Attach edit-difficulty metrics from edit_difficulty.json files."""
    if "output_image" not in df.columns or "results_file" not in df.columns:
        return df

    difficulty_map = {}
    for results_file in df["results_file"].dropna().unique():
        diff_path = Path(results_file).parent / "edit_difficulty.json"
        if not diff_path.exists():
            continue

        with diff_path.open() as f:
            payload = json.load(f)
        for item in payload.get("metrics", []):
            key = item.get("output_image")
            metrics = item.get("metrics", {})
            if key and key not in difficulty_map:
                difficulty_map[key] = metrics

    if not difficulty_map:
        return df

    df["edit_l1"] = df["output_image"].map(lambda p: difficulty_map.get(p, {}).get("l1_mean"))
    df["edit_l2"] = df["output_image"].map(lambda p: difficulty_map.get(p, {}).get("l2_mean"))
    df["edit_psnr"] = df["output_image"].map(lambda p: difficulty_map.get(p, {}).get("psnr"))
    df["edit_ssim"] = df["output_image"].map(lambda p: difficulty_map.get(p, {}).get("ssim"))
    df["edit_hash_diff"] = df["output_image"].map(lambda p: difficulty_map.get(p, {}).get("hash_diff"))

    return df


def summarize_unchanged(df: pd.DataFrame) -> dict:
    """Summarize unchanged rates by race/category for edit difficulty diagnostics."""
    if "is_unchanged" not in df.columns:
        return {}

    summary = {
        "overall_rate": float(df["is_unchanged"].mean())
    }

    by_race = df.groupby("race")["is_unchanged"].mean().to_dict()
    by_category = df.groupby("category")["is_unchanged"].mean().to_dict()

    summary["by_race"] = {k: float(v) for k, v in by_race.items()}
    summary["by_category"] = {k: float(v) for k, v in by_category.items()}
    return summary


def summarize_identity_drift(df: pd.DataFrame) -> dict:
    """Summarize identity drift metrics (racial drift, gender drift, body transformation)."""
    if "racial_drift" not in df.columns:
        return {}

    # Overall drift rates
    summary = {
        "total_samples": len(df),
        "identity_preserved_rate": float(df.get("identity_preserved", pd.Series([True]*len(df))).mean()),
        "racial_drift_detected": 0,
        "gender_drift_detected": 0,
        "body_transform_detected": 0,
        "stereotype_replacement_detected": 0
    }

    # Racial drift breakdown
    if "racial_drift" in df.columns:
        racial_drift_counts = df["racial_drift"].value_counts().to_dict()
        summary["racial_drift_breakdown"] = {k: int(v) for k, v in racial_drift_counts.items()}
        summary["racial_drift_detected"] = len(df[df["racial_drift"] != "SAME"])

    # Gender drift breakdown
    if "gender_drift" in df.columns:
        gender_drift_counts = df["gender_drift"].value_counts().to_dict()
        summary["gender_drift_breakdown"] = {k: int(v) for k, v in gender_drift_counts.items()}
        summary["gender_drift_detected"] = len(df[df["gender_drift"] != "SAME"])

    # Body transformation breakdown
    if "body_transform" in df.columns:
        body_transform_counts = df["body_transform"].value_counts().to_dict()
        summary["body_transform_breakdown"] = {k: int(v) for k, v in body_transform_counts.items()}
        summary["body_transform_detected"] = len(df[df["body_transform"] != "NONE"])

    # Stereotype replacement
    if "stereotype_replacement" in df.columns:
        summary["stereotype_replacement_detected"] = int(df["stereotype_replacement"].sum())
        summary["stereotype_replacement_rate"] = float(df["stereotype_replacement"].mean())

    # Drift by race (source)
    if "race" in df.columns and "racial_drift" in df.columns:
        drift_by_source_race = df.groupby("race")["racial_drift"].apply(
            lambda x: (x != "SAME").sum() / len(x) if len(x) > 0 else 0
        ).to_dict()
        summary["racial_drift_by_source_race"] = {k: float(v) for k, v in drift_by_source_race.items()}

    # Drift by category
    if "category" in df.columns and "racial_drift" in df.columns:
        drift_by_category = df.groupby("category")["racial_drift"].apply(
            lambda x: (x != "SAME").sum() / len(x) if len(x) > 0 else 0
        ).to_dict()
        summary["racial_drift_by_category"] = {k: float(v) for k, v in drift_by_category.items()}

    # Stereotype replacement by category
    if "category" in df.columns and "stereotype_replacement" in df.columns:
        sr_by_category = df.groupby("category")["stereotype_replacement"].mean().to_dict()
        summary["stereotype_replacement_by_category"] = {k: float(v) for k, v in sr_by_category.items()}

    return summary


def analyze_edit_difficulty(df: pd.DataFrame, analyzer: StatisticalAnalyzer) -> dict:
    """Analyze edit difficulty metrics and control for them."""
    required_cols = ["edit_l1", "edit_hash_diff"]
    if not all(col in df.columns for col in required_cols):
        return {}

    results = {}

    # Correlations with refusal and erasure
    for metric in ["edit_l1", "edit_hash_diff", "edit_ssim", "edit_psnr"]:
        if metric in df.columns:
            valid = df[[metric, "is_refused"]].dropna()
            if not valid.empty:
                corr, p_value = stats.spearmanr(valid[metric], valid["is_refused"])
                results[f"{metric}_refusal_corr"] = {"rho": float(corr), "p_value": float(p_value)}

            if "is_erased" in df.columns:
                non_refused = df[~df["is_refused"]]
                valid = non_refused[[metric, "is_erased"]].dropna()
                if not valid.empty:
                    corr, p_value = stats.spearmanr(valid[metric], valid["is_erased"])
                    results[f"{metric}_erasure_corr"] = {"rho": float(corr), "p_value": float(p_value)}

    # Logistic regression with edit difficulty controls
    if "is_refused" in df.columns:
        formula = "is_refused ~ C(race) + C(category) + edit_l1 + edit_hash_diff"
        results["refusal_model_with_difficulty"] = analyzer._logistic_regression(df, formula)

    if "is_erased" in df.columns:
        non_refused = df[~df["is_refused"]]
        formula = "is_erased ~ C(race) + C(category) + edit_l1 + edit_hash_diff"
        results["erasure_model_with_difficulty"] = analyzer._logistic_regression(non_refused, formula)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze I2I Refusal Bias Results - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic VLM analysis (uses 30B model by default)
  python scripts/analysis/analyze_results.py --results-dir data/results

  # Use 8B model for local testing
  python scripts/analysis/analyze_results.py --qwen-model 8B

  # Include human corrections
  python scripts/analysis/analyze_results.py --include-human-review

  # Full pipeline with publication-ready outputs
  python scripts/analysis/analyze_results.py --include-human-review --publication-ready
        """
    )
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Directory with VLM results JSON files")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to analyze (or all if not specified)")
    parser.add_argument("--output-dir", type=str, default="results/analysis",
                       help="Output directory for analysis")
    parser.add_argument("--prompts", type=str,
                       default="data/prompts/i2i_prompts.json",
                       help="Path to prompts JSON")
    parser.add_argument("--qwen-model", type=str, default="30B", choices=["30B", "8B"],
                       help="Qwen VLM model size: 30B (default, production) or 8B (local testing)")
    parser.add_argument("--include-human-review", action="store_true",
                       help="Include human review corrections from Firebase/survey")
    parser.add_argument("--include-edit-difficulty", action="store_true",
                       help="Attach edit-difficulty metrics (edit_difficulty.json)")
    parser.add_argument("--publication-ready", action="store_true",
                       help="Generate publication-ready figures and tables")
    parser.add_argument("--export-csv", action="store_true",
                       help="Export final dataset as CSV for further analysis")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = args.results_dir or str(PathConfig().results_dir)

    # Load VLM results
    print("🔍 Loading VLM ensemble results...")
    vlm_df = load_vlm_results(results_dir, args.model)
    print(f"✅ Loaded {len(vlm_df)} VLM results")

    # Load and integrate human review results if requested
    if args.include_human_review:
        print("🔍 Loading human review corrections...")
        human_df = load_human_review_results_from_firebase()
        print(f"✅ Loaded {len(human_df)} human review results")

        print("🔄 Combining VLM and human results...")
        final_df = combine_vlm_and_human_results(vlm_df, human_df)
        print(f"✅ Created final corrected dataset with {len(final_df)} entries")

        # Show correction statistics
        correction_stats = final_df['correction_type'].value_counts()
        print("\n📊 Correction Statistics:")
        for correction_type, count in correction_stats.items():
            print(f"  {correction_type}: {count} ({count/len(final_df)*100:.1f}%)")
    else:
        print("ℹ️ Using VLM results only (no human corrections)")
        final_df = vlm_df.copy()
        final_df['final_judgment'] = final_df['ensemble_result']
        final_df['correction_type'] = 'none'

    # Optionally attach edit-difficulty metrics
    if args.include_edit_difficulty:
        final_df = load_edit_difficulty(final_df)

    # Export final dataset if requested
    if args.export_csv:
        csv_path = output_dir / "final_corrected_dataset.csv"
        final_df.to_csv(csv_path, index=False)
        print(f"💾 Exported final dataset to: {csv_path}")

    # Normalize image_id for repeated-measures analysis
    if "image_id" not in final_df.columns and "source_image" in final_df.columns:
        final_df["image_id"] = final_df["source_image"].apply(
            lambda p: Path(p).stem if isinstance(p, str) and p else None
        )

    # Initialize analyzers
    analyzer = StatisticalAnalyzer()
    visualizer = ResultsVisualizer(output_dir=str(output_dir / "figures"))
    disparity = DisparityMetrics()
    sensitivity = SensitivityAnalyzer()

    print(f"📊 Starting comprehensive analysis on {len(final_df)} entries...")

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

    df = final_df.copy()

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

    # 10. Mixed-Effects Logistic Regression
    print("\n10. MIXED-EFFECTS LOGISTIC REGRESSION")
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

    # 11. Threshold Sensitivity Analysis
    print("\n11. THRESHOLD SENSITIVITY ANALYSIS")
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

    # 12. Bootstrap Confidence Intervals
    print("\n12. BOOTSTRAP ANALYSIS")
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

    # 13. SCS with Log-Odds Normalization
    print("\n13. STEREOTYPE CONGRUENCE SCORE (LOG-ODDS)")
    print("-"*40)
    scs_log_odds_results = scs_calculator.calculate_scs_log_odds(df.to_dict("records"))
    print(f"   Overall SCS (log-odds): {scs_log_odds_results['overall_scs_log_odds']:.3f}")
    print(f"   Interpretation: {scs_log_odds_results['interpretation']}")

    # 14. SCS with Risk Ratio Normalization
    print("\n14. STEREOTYPE CONGRUENCE SCORE (RISK RATIO)")
    print("-"*40)
    scs_risk_ratio_results = scs_calculator.calculate_scs_risk_ratio(df.to_dict("records"))
    print(f"   Overall SCS (risk ratio): {scs_risk_ratio_results['overall_scs_risk_ratio']:.3f}")
    print(f"   Interpretation: {scs_risk_ratio_results['interpretation']}")

    # 15. Edit Difficulty Diagnostics (optional)
    edit_difficulty_results = {}
    unchanged_summary = {}
    if args.include_edit_difficulty:
        print("\n15. EDIT DIFFICULTY DIAGNOSTICS")
        print("-"*40)
        unchanged_summary = summarize_unchanged(df)
        if unchanged_summary:
            print(f"   Unchanged rate (overall): {unchanged_summary['overall_rate']:.2%}")
        edit_difficulty_results = analyze_edit_difficulty(df, analyzer)
        if edit_difficulty_results:
            print("   ✓ Edit difficulty controls computed")

    # 16. Identity Drift Analysis (NEW)
    identity_drift_summary = {}
    if "racial_drift" in df.columns or "gender_drift" in df.columns:
        print("\n16. IDENTITY DRIFT ANALYSIS")
        print("-"*40)
        identity_drift_summary = summarize_identity_drift(df)

        if identity_drift_summary:
            print(f"   Identity preserved rate: {identity_drift_summary.get('identity_preserved_rate', 0):.2%}")
            print(f"   Racial drift detected: {identity_drift_summary.get('racial_drift_detected', 0)} cases")
            print(f"   Gender drift detected: {identity_drift_summary.get('gender_drift_detected', 0)} cases")
            print(f"   Body transformation detected: {identity_drift_summary.get('body_transform_detected', 0)} cases")

            if "stereotype_replacement_rate" in identity_drift_summary:
                print(f"   Stereotype replacement rate: {identity_drift_summary['stereotype_replacement_rate']:.2%}")
                print(f"   Stereotype replacement cases: {identity_drift_summary['stereotype_replacement_detected']}")

            # Detailed breakdown
            if "racial_drift_breakdown" in identity_drift_summary:
                print("\n   Racial Drift Breakdown:")
                for drift_type, count in identity_drift_summary["racial_drift_breakdown"].items():
                    if drift_type != "SAME":
                        print(f"      {drift_type}: {count}")

            if "racial_drift_by_source_race" in identity_drift_summary:
                print("\n   Racial Drift by Source Race:")
                for race, rate in identity_drift_summary["racial_drift_by_source_race"].items():
                    if rate > 0:
                        print(f"      {race}: {rate:.2%}")

            if "stereotype_replacement_by_category" in identity_drift_summary:
                print("\n   Stereotype Replacement by Category:")
                for cat, rate in identity_drift_summary["stereotype_replacement_by_category"].items():
                    if rate > 0:
                        print(f"      Category {cat}: {rate:.2%}")

    # 17. Save comprehensive analysis report
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
        "scs": scs_results,
        "scs_log_odds": scs_log_odds_results,
        "scs_risk_ratio": scs_risk_ratio_results,
        "edit_difficulty_analysis": edit_difficulty_results,
        "unchanged_summary": unchanged_summary,
        "identity_drift_analysis": identity_drift_summary
    }

    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Analysis report saved to {report_path}")
    print(f"✓ Figures saved to {output_dir / 'figures'}")

    # 18. Generate LaTeX tables
    print("\n18. GENERATING LATEX TABLES")
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

    # 19. Save extended analysis results
    print("\n19. SAVING EXTENDED ANALYSIS RESULTS")
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

    # SCS risk ratio
    scs_rr_path = output_dir / "scs_risk_ratio.json"
    with open(scs_rr_path, "w") as f:
        json.dump(scs_risk_ratio_results, f, indent=2, default=str)
    print(f"   ✓ SCS risk ratio results: {scs_rr_path}")

    if edit_difficulty_results:
        edit_path = output_dir / "edit_difficulty_analysis.json"
        with open(edit_path, "w") as f:
            json.dump(edit_difficulty_results, f, indent=2, default=str)
        print(f"   ✓ Edit difficulty analysis: {edit_path}")

    if unchanged_summary:
        unchanged_path = output_dir / "unchanged_summary.json"
        with open(unchanged_path, "w") as f:
            json.dump(unchanged_summary, f, indent=2, default=str)
        print(f"   ✓ Unchanged summary: {unchanged_path}")

    # Identity drift results
    if identity_drift_summary:
        identity_drift_path = output_dir / "identity_drift_analysis.json"
        with open(identity_drift_path, "w") as f:
            json.dump(identity_drift_summary, f, indent=2, default=str)
        print(f"   ✓ Identity drift analysis: {identity_drift_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - analysis_report.json (main results)")
    print(f"  - mixed_effects_results.json (reviewer feedback #1)")
    print(f"  - sensitivity_analysis.json (reviewer feedback #2)")
    print(f"  - bootstrap_results.json (reviewer feedback #5)")
    print(f"  - scs_log_odds.json (reviewer feedback #4)")
    print(f"  - identity_drift_analysis.json (NEW: stereotype replacement)")
    print(f"  - tables/ (LaTeX tables)")
    print(f"  - figures/ (all visualizations)")

    # Save comprehensive analysis summary
    analysis_summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "analysis_type": "human_corrected" if args.include_human_review else "vlm_only",
            "total_samples": len(final_df),
            "human_corrections": args.include_human_review,
            "publication_ready": args.publication_ready
        },
        "dataset_stats": {
            "total_entries": len(final_df),
            "unique_races": final_df['race'].nunique() if 'race' in final_df.columns else None,
            "unique_genders": final_df['gender'].nunique() if 'gender' in final_df.columns else None,
            "unique_ages": final_df['age'].nunique() if 'age' in final_df.columns else None,
            "correction_stats": final_df['correction_type'].value_counts().to_dict() if 'correction_type' in final_df.columns else None
        },
        "files_generated": {
            "figures": len(list(output_dir.glob('figures/*.png'))),
            "interactive_plots": len(list(output_dir.glob('figures/*.html'))),
            "statistical_results": len(list(output_dir.glob('*.json'))),
            "csv_export": args.export_csv
        }
    }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(analysis_summary, f, indent=2, default=str)

    print(f"📋 Analysis summary: {summary_path}")

    if args.include_human_review:
        correction_count = len(final_df[final_df['correction_type'] == 'corrected'])
        print(f"🤖 Human-AI collaboration: {correction_count} corrections applied")

    if args.publication_ready:
        print("📄 Publication-ready outputs generated")
        print("   → Use figures/ for paper illustrations")
        print("   → Use *.json files for statistical tables")
        print("   → Use analysis_summary.json for metadata")


if __name__ == "__main__":
    main()
