#!/usr/bin/env python3
"""
Test Analysis Pipeline End-to-End
Generates mock data and runs complete analysis
"""

import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_mock_results import generate_mock_results
from scripts.analyze_results import load_results
from src.evaluation.metrics import DisparityMetrics, StereotypeCongruenceScore
from src.analysis.statistical import StatisticalAnalyzer, VLMCalibration
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.visualization import ResultsVisualizer


def test_pipeline():
    """Run end-to-end test of analysis pipeline."""

    print("="*60)
    print("TESTING I2I REFUSAL BIAS ANALYSIS PIPELINE")
    print("="*60)

    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp())
    results_dir = temp_dir / "results"
    analysis_dir = temp_dir / "analysis"

    try:
        # 1. Generate mock data
        print("\n1. GENERATING MOCK DATA")
        print("-"*40)
        results_file = generate_mock_results(
            model_name="test_model",
            output_dir=results_dir,
            bias_strength=0.2,
            seed=42
        )
        print(f"✓ Mock data generated: {results_file}")

        # 2. Load results
        print("\n2. LOADING RESULTS")
        print("-"*40)
        df = load_results(str(results_dir), "test_model")
        print(f"✓ Loaded {len(df)} results")
        print(f"  Columns: {list(df.columns)}")

        # 3. Initialize analyzers
        print("\n3. INITIALIZING ANALYZERS")
        print("-"*40)
        analyzer = StatisticalAnalyzer()
        sensitivity = SensitivityAnalyzer()
        disparity = DisparityMetrics()
        print("✓ All analyzers initialized")

        # 4. Basic disparity calculation
        print("\n4. DISPARITY CALCULATION")
        print("-"*40)
        disp_result = disparity.calculate_disparity(df.to_dict("records"))
        print(f"  Max rate: {disp_result.max_rate:.2%} ({disp_result.max_race})")
        print(f"  Min rate: {disp_result.min_rate:.2%} ({disp_result.min_race})")
        print(f"  Disparity: {disp_result.disparity:.2%}")
        print(f"  Significant: {disp_result.is_significant} (p={disp_result.p_value:.4f})")

        # 5. Mixed-effects logistic regression
        print("\n5. MIXED-EFFECTS LOGISTIC REGRESSION")
        print("-"*40)
        if "image_id" in df.columns:
            me_result = analyzer.mixed_effects_logistic(
                df,
                formula="is_refused ~ C(race) + C(category)",
                random_effects=["image_id"]
            )
            if "error" not in me_result:
                print(f"  ✓ Model converged: {me_result['converged']}")
                print(f"  AIC: {me_result['model_fit']['aic']:.2f}")
                print(f"  Number of groups: {me_result['n_groups']}")
            else:
                print(f"  Error: {me_result['error']}")
        else:
            print("  Skipped (no image_id column)")

        # 6. Threshold sensitivity analysis
        print("\n6. THRESHOLD SENSITIVITY ANALYSIS")
        print("-"*40)
        if "clip_similarity" in df.columns:
            threshold_analysis = sensitivity.threshold_ablation(
                df.to_dict("records"),
                thresholds=[0.90, 0.92, 0.95, 0.98]
            )
            print(f"  Disparity range: {threshold_analysis['disparity_range']['min']:.3f} - {threshold_analysis['disparity_range']['max']:.3f}")
            print(f"  Disparity std: {threshold_analysis['disparity_range']['std']:.3f}")
            print(f"  Ranking correlation: {threshold_analysis['avg_ranking_correlation']:.3f}")
            print(f"  ✓ {threshold_analysis['interpretation']}")
        else:
            print("  Skipped (no clip_similarity column)")

        # 7. Bootstrap analysis (small sample for speed)
        print("\n7. BOOTSTRAP ANALYSIS")
        print("-"*40)
        if "image_id" in df.columns:
            bootstrap_result = sensitivity.bootstrap_disparity(
                df,
                n_bootstrap=100,  # Reduced for testing
                sample_unit="image"
            )
            print(f"  Disparity: {bootstrap_result['disparity'].mean:.3f}")
            print(f"  95% CI: [{bootstrap_result['disparity'].ci_lower:.3f}, {bootstrap_result['disparity'].ci_upper:.3f}]")
            print(f"  SE: {bootstrap_result['disparity'].std_error:.3f}")
            print(f"  ✓ Bootstrap completed (n={bootstrap_result['n_bootstrap']})")
        else:
            print("  Skipped (no image_id column)")

        # 8. ANOVA tests
        print("\n8. ANOVA TESTS")
        print("-"*40)
        race_anova = analyzer.one_way_anova_race(df, "is_refused")
        print(f"  Race effect: F={race_anova.f_statistic:.2f}, p={race_anova.p_value:.4f}, η²={race_anova.effect_size:.3f}")

        cat_anova = analyzer.one_way_anova_category(df, "is_refused")
        print(f"  Category effect: F={cat_anova.f_statistic:.2f}, p={cat_anova.p_value:.4f}, η²={cat_anova.effect_size:.3f}")

        # 9. Odds ratios
        print("\n9. ODDS RATIOS")
        print("-"*40)
        odds_ratios = analyzer.calculate_all_odds_ratios(df, reference_race="White")
        for or_result in odds_ratios[:3]:  # Show first 3
            if or_result["odds_ratio"] is not None:
                print(f"  {or_result['race1']}: OR={or_result['odds_ratio']:.2f} (95% CI: [{or_result['ci_lower']:.2f}, {or_result['ci_upper']:.2f}])")

        # 10. Test VLM calibration (with simulated data)
        print("\n10. VLM CALIBRATION TEST")
        print("-"*40)
        import numpy as np
        n_samples = 100
        human_labels = [bool(x) for x in np.random.randint(0, 2, n_samples)]
        vlm_predictions = [bool(x) for x in np.random.randint(0, 2, n_samples)]
        races_sample = [np.random.choice(["White", "Black", "East Asian"]) for _ in range(n_samples)]

        calibrator = VLMCalibration()
        calibration_result = calibrator.calculate_confusion_matrix(
            human_labels, vlm_predictions, races_sample
        )
        print(f"  Overall accuracy: {calibration_result['overall']['accuracy']:.2f}")
        print(f"  Accuracy disparity: {calibration_result['summary']['accuracy_disparity']:.2f}")
        print(f"  ✓ {calibration_result['summary']['calibration_quality']}")

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print(f"\nTest data location: {temp_dir}")
        print("The temporary directory will be cleaned up automatically.")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n✓ Cleaned up temporary directory")


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
