"""
Threshold Sensitivity Analysis for ACRB

Implements sensitivity analysis for fairness thresholds as described in the paper
(Appendix Section A.9 - Sensitivity Analysis for Fairness Threshold).

Analyzes how disparity metrics (Delta_refusal, Delta_erasure) vary
across different fairness threshold values (1pp to 7pp).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FairnessThresholdResult:
    """Result of fairness threshold analysis for a single threshold value."""
    threshold_pp: float  # Threshold in percentage points
    n_significant_disparities: int  # Number of attributes exceeding threshold
    significant_attributes: List[str]  # Attributes exceeding threshold
    max_disparity_attribute: str  # Attribute with highest disparity
    max_disparity_value: float  # Value of highest disparity
    disparity_ratio: float  # Ratio of max/min rates
    passes_threshold: bool  # Whether disparity exceeds threshold
    ci_lower: Optional[float] = None  # 95% CI lower bound
    ci_upper: Optional[float] = None  # 95% CI upper bound
    p_value: Optional[float] = None  # Statistical significance


@dataclass
class SensitivityResult:
    """Complete sensitivity analysis result across multiple thresholds."""
    metric_name: str  # "refusal" or "erasure"
    thresholds_tested: List[float]
    results_by_threshold: Dict[float, FairnessThresholdResult]
    stable_attributes: List[str]  # Attributes significant at all thresholds
    sensitive_attributes: List[str]  # Attributes that flip at some thresholds
    rank_correlation: float  # Spearman rho for rank stability
    rank_correlation_p: float  # p-value for rank correlation


def threshold_sensitivity_analysis(
    results_df: pd.DataFrame,
    thresholds: List[float] = [0.01, 0.03, 0.05, 0.07],
    target_attributes: List[str] = ['Nigerian', 'Disability'],
    rate_column: str = 'refusal_rate',
    attribute_column: str = 'attribute',
    baseline_attribute: str = 'US',
    n_bootstrap: int = 1000,
) -> SensitivityResult:
    """
    Analyze disparity stability across fairness thresholds.

    As described in Paper Appendix A.9:
    "We conducted sensitivity analyses varying the fairness threshold from
    1 to 7 percentage points (pp). Our core findings, including significant
    disparities for Nigerian cultural markers and disability-related attributes,
    remain stable across all thresholds."

    Args:
        results_df: DataFrame with columns [attribute, rate_column, ...]
        thresholds: List of fairness thresholds in decimal (0.01 = 1pp)
        target_attributes: Attributes of interest for stability analysis
        rate_column: Column name containing refusal/erasure rates
        attribute_column: Column name containing attribute labels
        baseline_attribute: Baseline attribute for comparison (e.g., 'US')
        n_bootstrap: Number of bootstrap samples for CI estimation

    Returns:
        SensitivityResult with analysis across all thresholds
    """
    if results_df.empty:
        logger.warning("Empty results DataFrame provided")
        return _empty_sensitivity_result(thresholds)

    # Get rates by attribute
    rates_by_attr = results_df.groupby(attribute_column)[rate_column].mean().to_dict()

    if baseline_attribute not in rates_by_attr:
        logger.warning(f"Baseline attribute '{baseline_attribute}' not found. Using min rate as baseline.")
        baseline_rate = min(rates_by_attr.values())
    else:
        baseline_rate = rates_by_attr[baseline_attribute]

    results_by_threshold = {}
    rank_orderings = []

    for threshold in thresholds:
        # Compute disparities relative to baseline
        disparities = {
            attr: rate - baseline_rate
            for attr, rate in rates_by_attr.items()
            if attr != baseline_attribute
        }

        # Find attributes exceeding threshold
        significant_attrs = [
            attr for attr, disp in disparities.items()
            if abs(disp) > threshold
        ]

        # Compute max disparity
        if disparities:
            max_attr = max(disparities, key=lambda x: abs(disparities[x]))
            max_disp = disparities[max_attr]
            min_attr = min(disparities, key=lambda x: abs(disparities[x]))

            # Compute disparity ratio
            max_rate = rates_by_attr.get(max_attr, 0)
            min_rate = baseline_rate if baseline_rate > 0 else 0.001
            disp_ratio = max_rate / min_rate if min_rate > 0 else float('inf')
        else:
            max_attr = "none"
            max_disp = 0.0
            disp_ratio = 1.0

        # Bootstrap CI for max disparity
        ci_lower, ci_upper, p_value = _bootstrap_disparity_ci(
            results_df,
            attribute_column,
            rate_column,
            baseline_attribute,
            n_bootstrap
        )

        results_by_threshold[threshold] = FairnessThresholdResult(
            threshold_pp=threshold * 100,
            n_significant_disparities=len(significant_attrs),
            significant_attributes=significant_attrs,
            max_disparity_attribute=max_attr,
            max_disparity_value=max_disp,
            disparity_ratio=disp_ratio,
            passes_threshold=len(significant_attrs) > 0,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value
        )

        # Store rank ordering for correlation analysis
        rank_orderings.append(
            sorted(disparities.keys(), key=lambda x: disparities[x], reverse=True)
        )

    # Compute rank stability (Spearman correlation between first and last threshold)
    if len(rank_orderings) >= 2 and len(rank_orderings[0]) > 1:
        rank_corr, rank_p = _compute_rank_correlation(rank_orderings[0], rank_orderings[-1])
    else:
        rank_corr, rank_p = 1.0, 0.0

    # Find stable vs sensitive attributes
    all_significant = set.intersection(*[
        set(r.significant_attributes) for r in results_by_threshold.values()
    ]) if results_by_threshold else set()

    any_significant = set.union(*[
        set(r.significant_attributes) for r in results_by_threshold.values()
    ]) if results_by_threshold else set()

    stable_attrs = list(all_significant)
    sensitive_attrs = list(any_significant - all_significant)

    return SensitivityResult(
        metric_name=rate_column,
        thresholds_tested=thresholds,
        results_by_threshold=results_by_threshold,
        stable_attributes=stable_attrs,
        sensitive_attributes=sensitive_attrs,
        rank_correlation=rank_corr,
        rank_correlation_p=rank_p
    )


def disparity_stability_analysis(
    results_df: pd.DataFrame,
    attribute_column: str = 'attribute',
    refusal_column: str = 'is_refused',
    erasure_column: str = 'cue_erased',
    thresholds: List[float] = [0.01, 0.03, 0.05, 0.07],
    target_attributes: List[str] = ['Nigerian', 'Disability', 'Kenyan', 'Religion'],
) -> Dict[str, Any]:
    """
    Comprehensive disparity stability analysis for both refusal and erasure.

    Paper reference (Appendix A.9):
    "Nigerian markers exhibit >4.0x disparity at thresholds in {1, 3, 5, 7} pp
    (95% CI overlap test, p < 0.001 for all), and disability erasure exceeds
    neutral baselines by >35% across all thresholds."

    Args:
        results_df: DataFrame with individual sample results
        attribute_column: Column containing attribute labels
        refusal_column: Column containing refusal indicator (0/1)
        erasure_column: Column containing erasure indicator (0/1)
        thresholds: Fairness thresholds to test
        target_attributes: Attributes to specifically track

    Returns:
        Dict with stability analysis for both metrics
    """
    # Compute per-attribute rates
    attr_stats = results_df.groupby(attribute_column).agg({
        refusal_column: ['sum', 'count'],
        erasure_column: lambda x: x[~results_df.loc[x.index, refusal_column].astype(bool)].sum()
    }).reset_index()

    # Flatten column names
    attr_stats.columns = ['attribute', 'refusals', 'total', 'erasures']
    attr_stats['refusal_rate'] = attr_stats['refusals'] / attr_stats['total']
    attr_stats['non_refused'] = attr_stats['total'] - attr_stats['refusals']
    attr_stats['erasure_rate'] = np.where(
        attr_stats['non_refused'] > 0,
        attr_stats['erasures'] / attr_stats['non_refused'],
        0.0
    )

    # Run sensitivity analysis for both metrics
    refusal_sensitivity = threshold_sensitivity_analysis(
        attr_stats,
        thresholds=thresholds,
        target_attributes=target_attributes,
        rate_column='refusal_rate',
        attribute_column='attribute'
    )

    erasure_sensitivity = threshold_sensitivity_analysis(
        attr_stats,
        thresholds=thresholds,
        target_attributes=target_attributes,
        rate_column='erasure_rate',
        attribute_column='attribute'
    )

    # Check stability for target attributes
    target_stability = {}
    for attr in target_attributes:
        attr_rows = attr_stats[attr_stats['attribute'].str.contains(attr, case=False, na=False)]
        if not attr_rows.empty:
            ref_rate = attr_rows['refusal_rate'].values[0]
            era_rate = attr_rows['erasure_rate'].values[0]

            # Check if significant at all thresholds
            ref_stable = all(
                attr in r.significant_attributes
                for r in refusal_sensitivity.results_by_threshold.values()
            )
            era_stable = all(
                attr in r.significant_attributes
                for r in erasure_sensitivity.results_by_threshold.values()
            )

            target_stability[attr] = {
                'refusal_rate': ref_rate,
                'erasure_rate': era_rate,
                'refusal_stable': ref_stable,
                'erasure_stable': era_stable,
            }

    return {
        'refusal_sensitivity': refusal_sensitivity,
        'erasure_sensitivity': erasure_sensitivity,
        'target_attribute_stability': target_stability,
        'summary': {
            'refusal_rank_stable': refusal_sensitivity.rank_correlation > 0.85,
            'erasure_rank_stable': erasure_sensitivity.rank_correlation > 0.85,
            'core_findings_stable': all(
                target_stability.get(attr, {}).get('refusal_stable', False) or
                target_stability.get(attr, {}).get('erasure_stable', False)
                for attr in ['Nigerian', 'Disability']
            )
        }
    }


def _bootstrap_disparity_ci(
    df: pd.DataFrame,
    attr_col: str,
    rate_col: str,
    baseline: str,
    n_bootstrap: int = 1000
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute bootstrap confidence intervals for max disparity."""
    if df.empty or len(df[attr_col].unique()) < 2:
        return None, None, None

    np.random.seed(42)
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        # Resample within each attribute group
        boot_df = df.groupby(attr_col).apply(
            lambda x: x.sample(n=len(x), replace=True)
        ).reset_index(drop=True)

        boot_rates = boot_df.groupby(attr_col)[rate_col].mean()
        baseline_rate = boot_rates.get(baseline, boot_rates.min())
        disparities = boot_rates - baseline_rate
        bootstrap_deltas.append(disparities.max())

    if not bootstrap_deltas:
        return None, None, None

    ci_lower = np.percentile(bootstrap_deltas, 2.5)
    ci_upper = np.percentile(bootstrap_deltas, 97.5)

    # Simple p-value: proportion of bootstrap samples with delta <= 0
    p_value = np.mean([d <= 0 for d in bootstrap_deltas])

    return ci_lower, ci_upper, p_value


def _compute_rank_correlation(
    ranking1: List[str],
    ranking2: List[str]
) -> Tuple[float, float]:
    """Compute Spearman rank correlation between two rankings."""
    if len(ranking1) != len(ranking2) or len(ranking1) < 2:
        return 1.0, 0.0

    # Convert to numeric ranks
    rank1_map = {item: i for i, item in enumerate(ranking1)}
    ranks1 = list(range(len(ranking1)))
    ranks2 = [rank1_map.get(item, len(ranking1)) for item in ranking2]

    try:
        corr, p_val = scipy_stats.spearmanr(ranks1, ranks2)
        return corr if not np.isnan(corr) else 1.0, p_val if not np.isnan(p_val) else 0.0
    except Exception:
        return 1.0, 0.0


def _empty_sensitivity_result(thresholds: List[float]) -> SensitivityResult:
    """Return empty sensitivity result for edge cases."""
    return SensitivityResult(
        metric_name="unknown",
        thresholds_tested=thresholds,
        results_by_threshold={},
        stable_attributes=[],
        sensitive_attributes=[],
        rank_correlation=1.0,
        rank_correlation_p=0.0
    )


def main():
    """Example usage of sensitivity analysis."""
    print("Threshold Sensitivity Analysis Module")
    print("=" * 60)

    # Create mock data matching paper results
    np.random.seed(42)
    n_samples = 1000

    # Simulated refusal rates matching Table 3 (Paper)
    attribute_rates = {
        'Korean': 0.059,
        'Chinese': 0.055,
        'Nigerian': 0.167,  # 4.6x US baseline
        'Kenyan': 0.149,
        'US': 0.036,  # Baseline
        'Indian': 0.084,
    }

    # Generate mock results
    data = []
    for attr, rate in attribute_rates.items():
        for _ in range(n_samples // len(attribute_rates)):
            refused = np.random.random() < rate
            erased = np.random.random() < (rate * 2)  # Higher erasure
            data.append({
                'attribute': attr,
                'is_refused': int(refused),
                'cue_erased': int(erased),
            })

    df = pd.DataFrame(data)

    # Run disparity stability analysis
    results = disparity_stability_analysis(
        df,
        thresholds=[0.01, 0.03, 0.05, 0.07],
        target_attributes=['Nigerian', 'Kenyan', 'Indian']
    )

    print("\nRefusal Sensitivity Analysis:")
    print(f"  Rank correlation: {results['refusal_sensitivity'].rank_correlation:.3f}")
    print(f"  Stable attributes: {results['refusal_sensitivity'].stable_attributes}")
    print(f"  Sensitive attributes: {results['refusal_sensitivity'].sensitive_attributes}")

    print("\nErasure Sensitivity Analysis:")
    print(f"  Rank correlation: {results['erasure_sensitivity'].rank_correlation:.3f}")
    print(f"  Stable attributes: {results['erasure_sensitivity'].stable_attributes}")

    print("\nTarget Attribute Stability:")
    for attr, stability in results['target_attribute_stability'].items():
        print(f"  {attr}:")
        print(f"    Refusal rate: {stability['refusal_rate']:.1%}")
        print(f"    Refusal stable: {stability['refusal_stable']}")
        print(f"    Erasure rate: {stability['erasure_rate']:.1%}")

    print(f"\nCore Findings Stable: {results['summary']['core_findings_stable']}")


if __name__ == "__main__":
    main()
