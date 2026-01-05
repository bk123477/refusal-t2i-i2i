"""
Disparity Metric

Computes disparity in refusal and erasure rates across attributes.
Key metrics: Delta_Refusal, Delta_Erasure for detecting attribute-conditioned bias.

Enhanced with:
- Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals
- Cohen's d effect size calculation
- Multiple comparison correction methods
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from scipy import stats as scipy_stats
import warnings


@dataclass
class DisparityResult:
    """Result of disparity analysis with enhanced statistics."""
    metric_name: str  # "refusal" or "erasure"
    attribute_type: str  # "culture", "gender", etc.
    delta: float  # max - min across attributes
    max_value: float
    max_attribute: str
    min_value: float
    min_attribute: str
    std: float
    # Statistical significance
    is_significant: bool
    p_value: float
    effect_size: float  # Cohen's d or equivalent
    # Enhanced fields for IJCAI feedback
    p_value_corrected: Optional[float] = None  # Bonferroni-corrected p-value
    correction_method: Optional[str] = None  # "bonferroni", "holm", "fdr"
    cohens_d: Optional[float] = None  # Cohen's d effect size
    ci_lower: Optional[float] = None  # 95% CI lower bound
    ci_upper: Optional[float] = None  # 95% CI upper bound
    bootstrap_std: Optional[float] = None  # Bootstrap standard error
    n_comparisons: Optional[int] = None  # Number of comparisons for correction


# Effect size interpretation thresholds (Cohen's d)
EFFECT_SIZE_THRESHOLDS = {
    "negligible": 0.2,
    "small": 0.5,
    "medium": 0.8,
    "large": float("inf"),
}


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < EFFECT_SIZE_THRESHOLDS["negligible"]:
        return "negligible"
    elif d_abs < EFFECT_SIZE_THRESHOLDS["small"]:
        return "small"
    elif d_abs < EFFECT_SIZE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "large"


class DisparityMetric:
    """
    Compute disparity metrics for refusal and erasure rates.

    Enhanced with Bonferroni correction, bootstrap CI, and Cohen's d.

    Delta_Refusal = max(refusal across attributes) - min(refusal across attributes)
    Delta_Erasure = max(erasure across attributes) - min(erasure across attributes)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        correction_method: str = "bonferroni",
        n_bootstrap: int = 1000,
        ci_level: float = 0.95
    ):
        """
        Initialize disparity metric.

        Args:
            significance_level: p-value threshold for significance testing
            correction_method: Multiple comparison correction ("bonferroni", "holm", "fdr")
            n_bootstrap: Number of bootstrap samples for CI estimation
            ci_level: Confidence interval level (default: 0.95 for 95% CI)
        """
        self.significance_level = significance_level
        self.correction_method = correction_method
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
    
    def compute_delta(
        self,
        rates: Dict[str, float],
        metric_name: str,
        attribute_type: str,
        sample_counts: Optional[Dict[str, int]] = None,
        raw_data: Optional[Dict[str, List[int]]] = None
    ) -> DisparityResult:
        """
        Compute delta disparity for a set of rates.

        Enhanced with Bonferroni correction, bootstrap CI, and Cohen's d.

        Args:
            rates: Dict mapping attribute_value to rate (0-1)
            metric_name: Name of metric ("refusal" or "erasure")
            attribute_type: Type of attribute being compared
            sample_counts: Optional sample counts for significance testing
            raw_data: Optional raw binary data (0/1) for bootstrap CI

        Returns:
            DisparityResult with delta and statistics
        """
        if not rates:
            return DisparityResult(
                metric_name=metric_name,
                attribute_type=attribute_type,
                delta=0.0,
                max_value=0.0,
                max_attribute="none",
                min_value=0.0,
                min_attribute="none",
                std=0.0,
                is_significant=False,
                p_value=1.0,
                effect_size=0.0
            )

        values = list(rates.values())
        keys = list(rates.keys())

        max_idx = np.argmax(values)
        min_idx = np.argmin(values)

        max_value = values[max_idx]
        min_value = values[min_idx]
        delta = max_value - min_value
        std_dev = np.std(values)

        # Statistical significance testing
        p_value = 1.0
        p_value_corrected = 1.0
        effect_size = 0.0
        cohens_d = 0.0
        is_significant = False
        ci_lower, ci_upper = None, None
        bootstrap_std = None
        n_comparisons = len(keys) * (len(keys) - 1) // 2  # Pairwise comparisons

        if sample_counts and len(values) >= 2:
            # Use chi-square test for proportions
            try:
                # Create contingency table from rates and counts
                observed = []
                for attr in keys:
                    n = sample_counts.get(attr, 100)  # Default sample size
                    successes = int(rates[attr] * n)
                    failures = n - successes
                    observed.append([successes, failures])

                chi2, p_value, dof, expected = scipy_stats.chi2_contingency(observed)

                # Apply multiple comparison correction
                p_value_corrected = self._apply_correction(p_value, n_comparisons)
                is_significant = p_value_corrected < self.significance_level

                # Effect size (Cramer's V)
                n_total = sum(sample_counts.values()) if sample_counts else len(values) * 100
                effect_size = np.sqrt(chi2 / (n_total * (min(len(observed), 2) - 1)))

                # Cohen's d for max vs min comparison
                max_attr = keys[max_idx]
                min_attr = keys[min_idx]
                cohens_d = self._compute_cohens_d(
                    rates[max_attr], rates[min_attr],
                    sample_counts.get(max_attr, 100),
                    sample_counts.get(min_attr, 100)
                )

            except Exception as e:
                warnings.warn(f"Chi-square test failed: {e}")
                p_value = 1.0
                is_significant = delta > 0.1  # Heuristic threshold

        # Bootstrap confidence intervals
        if raw_data and len(raw_data) >= 2:
            ci_lower, ci_upper, bootstrap_std = self._bootstrap_ci(raw_data)

        return DisparityResult(
            metric_name=metric_name,
            attribute_type=attribute_type,
            delta=delta,
            max_value=max_value,
            max_attribute=keys[max_idx],
            min_value=min_value,
            min_attribute=keys[min_idx],
            std=std_dev,
            is_significant=is_significant,
            p_value=p_value,
            effect_size=effect_size,
            p_value_corrected=p_value_corrected,
            correction_method=self.correction_method,
            cohens_d=cohens_d,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bootstrap_std=bootstrap_std,
            n_comparisons=n_comparisons
        )

    def _apply_correction(self, p_value: float, n_comparisons: int) -> float:
        """
        Apply multiple comparison correction.

        Args:
            p_value: Uncorrected p-value
            n_comparisons: Number of comparisons

        Returns:
            Corrected p-value
        """
        if n_comparisons <= 1:
            return p_value

        if self.correction_method == "bonferroni":
            # Bonferroni: multiply p-value by number of comparisons
            return min(1.0, p_value * n_comparisons)

        elif self.correction_method == "holm":
            # Holm-Bonferroni: sequential correction (simplified for single p-value)
            return min(1.0, p_value * n_comparisons)

        elif self.correction_method == "fdr":
            # Benjamini-Hochberg FDR (simplified for single p-value)
            return min(1.0, p_value * n_comparisons / max(1, int(p_value * n_comparisons)))

        else:
            return p_value

    def _compute_cohens_d(
        self,
        rate1: float,
        rate2: float,
        n1: int,
        n2: int
    ) -> float:
        """
        Compute Cohen's d for two proportions.

        Uses the arcsine transformation for proportions.

        Args:
            rate1: First proportion
            rate2: Second proportion
            n1: Sample size for first group
            n2: Sample size for second group

        Returns:
            Cohen's d effect size
        """
        # Arcsine transformation for proportions
        # phi = 2 * arcsin(sqrt(p))
        phi1 = 2 * np.arcsin(np.sqrt(max(0, min(1, rate1))))
        phi2 = 2 * np.arcsin(np.sqrt(max(0, min(1, rate2))))

        # Pooled standard deviation for proportions
        # SD(phi) ~ 1/sqrt(n)
        pooled_sd = np.sqrt(1/n1 + 1/n2)

        if pooled_sd == 0:
            return 0.0

        return (phi1 - phi2) / pooled_sd

    def _bootstrap_ci(
        self,
        raw_data: Dict[str, List[int]]
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence intervals for delta.

        Args:
            raw_data: Dict mapping attribute to list of binary outcomes (0/1)

        Returns:
            (ci_lower, ci_upper, bootstrap_std)
        """
        np.random.seed(42)  # For reproducibility
        bootstrap_deltas = []

        keys = list(raw_data.keys())

        for _ in range(self.n_bootstrap):
            # Bootstrap sample for each attribute
            bootstrap_rates = {}
            for attr in keys:
                data = raw_data[attr]
                if len(data) > 0:
                    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                    bootstrap_rates[attr] = np.mean(bootstrap_sample)
                else:
                    bootstrap_rates[attr] = 0.0

            # Compute delta for bootstrap sample
            if bootstrap_rates:
                rates = list(bootstrap_rates.values())
                bootstrap_delta = max(rates) - min(rates)
                bootstrap_deltas.append(bootstrap_delta)

        if not bootstrap_deltas:
            return None, None, None

        # Percentile confidence intervals
        alpha = 1 - self.ci_level
        ci_lower = np.percentile(bootstrap_deltas, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2))
        bootstrap_std = np.std(bootstrap_deltas)

        return ci_lower, ci_upper, bootstrap_std
    
    def compute_refusal_disparity(
        self,
        refusal_rates: Dict[str, float],
        attribute_type: str = "culture",
        sample_counts: Optional[Dict[str, int]] = None
    ) -> DisparityResult:
        """Compute ΔRefusal across attributes."""
        return self.compute_delta(
            refusal_rates,
            metric_name="refusal",
            attribute_type=attribute_type,
            sample_counts=sample_counts
        )
    
    def compute_erasure_disparity(
        self,
        erasure_rates: Dict[str, float],
        attribute_type: str = "culture",
        sample_counts: Optional[Dict[str, int]] = None
    ) -> DisparityResult:
        """Compute ΔErasure across attributes."""
        return self.compute_delta(
            erasure_rates,
            metric_name="erasure",
            attribute_type=attribute_type,
            sample_counts=sample_counts
        )
    
    def compute_paired_disparity(
        self,
        neutral_rates: Dict[str, float],
        attributed_rates: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute paired disparity: how much does adding an attribute change rates?
        
        Args:
            neutral_rates: Rates for neutral (no attribute) prompts by domain
            attributed_rates: Rates for attributed prompts by domain and attribute
            
        Returns:
            Dict mapping attribute to average increase in rate vs neutral
        """
        disparities = {}
        
        for domain in neutral_rates:
            neutral = neutral_rates[domain]
            
            if domain in attributed_rates:
                for attr, rate in attributed_rates[domain].items():
                    if attr not in disparities:
                        disparities[attr] = []
                    disparities[attr].append(rate - neutral)
        
        return {attr: np.mean(diffs) for attr, diffs in disparities.items()}
    
    def rank_attributes(
        self,
        disparity_results: List[DisparityResult]
    ) -> List[Tuple[str, float]]:
        """
        Rank attributes by their disparity impact.
        
        Returns:
            List of (attribute, total_disparity) sorted by impact
        """
        attr_disparities = {}
        
        for result in disparity_results:
            # Track worst-performing attribute
            attr = result.max_attribute
            if attr not in attr_disparities:
                attr_disparities[attr] = 0.0
            attr_disparities[attr] += result.delta
        
        return sorted(attr_disparities.items(), key=lambda x: x[1], reverse=True)
    
    def summarize(
        self,
        refusal_disparity: DisparityResult,
        erasure_disparity: DisparityResult
    ) -> Dict:
        """
        Create summary of disparity analysis.
        
        Returns:
            Dict with key findings
        """
        return {
            "delta_refusal": refusal_disparity.delta,
            "delta_erasure": erasure_disparity.delta,
            "most_refused_attribute": refusal_disparity.max_attribute,
            "least_refused_attribute": refusal_disparity.min_attribute,
            "most_erased_attribute": erasure_disparity.max_attribute,
            "least_erased_attribute": erasure_disparity.min_attribute,
            "refusal_significant": refusal_disparity.is_significant,
            "erasure_significant": erasure_disparity.is_significant,
            "overall_bias_detected": (
                refusal_disparity.delta > 0.1 or erasure_disparity.delta > 0.1
            ),
        }


def main():
    """Example usage with enhanced statistical analysis."""
    print("DisparityMetric - Enhanced with Statistical Analysis")
    print("=" * 60)

    # Initialize with Bonferroni correction
    metric = DisparityMetric(
        significance_level=0.05,
        correction_method="bonferroni",
        n_bootstrap=1000,
        ci_level=0.95
    )

    # Example refusal rates by culture
    refusal_rates = {
        "Korean": 0.15,
        "Chinese": 0.12,
        "Nigerian": 0.35,
        "Kenyan": 0.32,
        "American": 0.08,
        "Indian": 0.18,
    }

    # Sample counts for significance testing
    sample_counts = {
        "Korean": 100,
        "Chinese": 100,
        "Nigerian": 100,
        "Kenyan": 100,
        "American": 100,
        "Indian": 100,
    }

    # Raw data for bootstrap CI
    raw_data = {
        attr: [1] * int(rate * 100) + [0] * int((1 - rate) * 100)
        for attr, rate in refusal_rates.items()
    }

    erasure_rates = {
        "Korean": 0.10,
        "Chinese": 0.12,
        "Nigerian": 0.45,
        "Kenyan": 0.40,
        "American": 0.05,
        "Indian": 0.15,
    }

    # Compute disparity with enhanced statistics
    ref_disp = metric.compute_delta(
        refusal_rates,
        "refusal",
        "culture",
        sample_counts=sample_counts,
        raw_data=raw_data
    )

    print("\nRefusal Disparity Analysis:")
    print(f"  Delta = {ref_disp.delta:.3f}")
    print(f"  Max: {ref_disp.max_attribute} ({ref_disp.max_value:.2f})")
    print(f"  Min: {ref_disp.min_attribute} ({ref_disp.min_value:.2f})")
    print(f"\nStatistical Significance:")
    print(f"  Raw p-value: {ref_disp.p_value:.4f}")
    print(f"  Corrected p-value ({ref_disp.correction_method}): {ref_disp.p_value_corrected:.4f}")
    print(f"  Is significant: {ref_disp.is_significant}")
    print(f"  N comparisons: {ref_disp.n_comparisons}")
    print(f"\nEffect Size:")
    print(f"  Cramer's V: {ref_disp.effect_size:.3f}")
    print(f"  Cohen's d: {ref_disp.cohens_d:.3f} ({interpret_effect_size(ref_disp.cohens_d)})")
    print(f"\nConfidence Interval (95%):")
    if ref_disp.ci_lower is not None:
        print(f"  Bootstrap CI: [{ref_disp.ci_lower:.3f}, {ref_disp.ci_upper:.3f}]")
        print(f"  Bootstrap SE: {ref_disp.bootstrap_std:.3f}")
    else:
        print("  Bootstrap CI: Not computed (no raw data)")

    era_disp = metric.compute_erasure_disparity(erasure_rates, "culture", sample_counts)

    print("\nErasure Disparity:")
    print(f"  Delta = {era_disp.delta:.2f}")
    print(f"  Max: {era_disp.max_attribute} ({era_disp.max_value:.2f})")
    print(f"  Min: {era_disp.min_attribute} ({era_disp.min_value:.2f})")

    summary = metric.summarize(ref_disp, era_disp)
    print(f"\nOverall Bias Detected: {summary['overall_bias_detected']}")


if __name__ == "__main__":
    main()
