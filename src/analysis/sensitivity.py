"""
Sensitivity and Robustness Analysis
- Threshold sensitivity analysis
- Seed variability analysis
- Bootstrap confidence intervals
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class ThresholdSensitivityResult:
    """Result of threshold sensitivity analysis."""
    threshold: float
    disparity: float
    max_race: str
    min_race: str
    n_refused: int
    refusal_rate: float


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    std_error: float


class SensitivityAnalyzer:
    """Analyze robustness and sensitivity of metrics."""

    RACES = ["White", "Black", "East Asian", "Southeast Asian",
             "Indian", "Middle Eastern", "Latino_Hispanic"]

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def threshold_ablation(
        self,
        results: list[dict],
        thresholds: list[float] = [0.90, 0.92, 0.94, 0.95, 0.96, 0.98],
        similarity_field: str = "clip_similarity"
    ) -> dict:
        """
        Analyze disparity ranking stability across CLIP similarity thresholds.

        Args:
            results: List of result dictionaries with similarity scores
            thresholds: List of CLIP thresholds to test
            similarity_field: Field name for similarity score

        Returns:
            Dict with disparity by threshold and ranking correlation
        """
        threshold_results = []

        for threshold in thresholds:
            # Reclassify refusals based on threshold
            results_copy = []
            for r in results:
                r_copy = r.copy()
                # If similarity < threshold, consider it refused
                if similarity_field in r_copy:
                    r_copy["is_refused_at_threshold"] = r_copy[similarity_field] < threshold
                else:
                    r_copy["is_refused_at_threshold"] = r_copy.get("is_refused", False)
                results_copy.append(r_copy)

            # Calculate disparity
            rates = {}
            for race in self.RACES:
                race_results = [r for r in results_copy if r.get("race") == race]
                if race_results:
                    refused = sum(1 for r in race_results if r["is_refused_at_threshold"])
                    rates[race] = refused / len(race_results)
                else:
                    rates[race] = 0.0

            max_race = max(rates, key=rates.get)
            min_race = min(rates, key=rates.get)
            disparity = rates[max_race] - rates[min_race]

            # Overall refusal rate
            total_refused = sum(1 for r in results_copy if r["is_refused_at_threshold"])
            refusal_rate = total_refused / len(results_copy)

            threshold_results.append(ThresholdSensitivityResult(
                threshold=threshold,
                disparity=disparity,
                max_race=max_race,
                min_race=min_race,
                n_refused=total_refused,
                refusal_rate=refusal_rate
            ))

        # Calculate ranking correlation
        disparities_by_threshold = [r.disparity for r in threshold_results]

        # Rank correlation (Spearman) between race rankings at different thresholds
        race_rankings = []
        for tr in threshold_results:
            race_results_dict = {}
            for race in self.RACES:
                race_results = [r for r in results if r.get("race") == race]
                if race_results:
                    # Use original similarity scores
                    avg_sim = np.mean([r.get(similarity_field, 1.0) for r in race_results])
                    race_results_dict[race] = avg_sim

            # Rank races by similarity (lower = more likely refused)
            sorted_races = sorted(race_results_dict.items(), key=lambda x: x[1])
            ranking = [r[0] for r in sorted_races]
            race_rankings.append(ranking)

        # Compute pairwise Kendall's tau between rankings
        ranking_correlations = []
        for i in range(len(race_rankings)):
            for j in range(i + 1, len(race_rankings)):
                tau, _ = stats.kendalltau(
                    [self.RACES.index(r) for r in race_rankings[i]],
                    [self.RACES.index(r) for r in race_rankings[j]]
                )
                ranking_correlations.append(tau)

        avg_rank_correlation = np.mean(ranking_correlations) if ranking_correlations else 1.0

        return {
            "threshold_results": [
                {
                    "threshold": r.threshold,
                    "disparity": r.disparity,
                    "max_race": r.max_race,
                    "min_race": r.min_race,
                    "refusal_rate": r.refusal_rate,
                    "n_refused": r.n_refused
                }
                for r in threshold_results
            ],
            "disparity_range": {
                "min": min(disparities_by_threshold),
                "max": max(disparities_by_threshold),
                "std": np.std(disparities_by_threshold)
            },
            "avg_ranking_correlation": avg_rank_correlation,
            "interpretation": self._interpret_threshold_sensitivity(
                disparities_by_threshold,
                avg_rank_correlation
            )
        }

    def _interpret_threshold_sensitivity(
        self,
        disparities: list[float],
        rank_corr: float
    ) -> str:
        """Interpret threshold sensitivity results."""
        disparity_std = np.std(disparities)

        if disparity_std < 0.02 and rank_corr > 0.8:
            return "Highly robust: Disparity rankings stable across thresholds"
        elif disparity_std < 0.05 and rank_corr > 0.6:
            return "Moderately robust: Some variation but consistent trends"
        else:
            return "WARNING: Disparity sensitive to threshold choice"

    def bootstrap_disparity(
        self,
        df: pd.DataFrame,
        n_bootstrap: int = 1000,
        sample_unit: str = "image",
        metric: str = "is_refused"
    ) -> dict:
        """
        Bootstrap confidence intervals for disparity by resampling images.

        Args:
            df: DataFrame with results
            n_bootstrap: Number of bootstrap iterations
            sample_unit: "image" or "prompt" (unit to resample)
            metric: Metric to analyze

        Returns:
            Dict with bootstrap results for each race and overall disparity
        """
        if sample_unit == "image":
            if "image_id" not in df.columns:
                raise ValueError("image_id column required for image-based bootstrapping")
            sample_ids = df["image_id"].unique()
        elif sample_unit == "prompt":
            if "prompt_id" not in df.columns:
                raise ValueError("prompt_id column required for prompt-based bootstrapping")
            sample_ids = df["prompt_id"].unique()
        else:
            # Row-level resampling
            sample_ids = None

        bootstrap_disparities = []
        bootstrap_rates_by_race = {race: [] for race in self.RACES}

        for i in range(n_bootstrap):
            # Resample
            if sample_ids is not None:
                sampled_ids = np.random.choice(sample_ids, size=len(sample_ids), replace=True)
                if sample_unit == "image":
                    bootstrap_df = df[df["image_id"].isin(sampled_ids)]
                else:
                    bootstrap_df = df[df["prompt_id"].isin(sampled_ids)]
            else:
                bootstrap_df = df.sample(n=len(df), replace=True)

            # Calculate rates
            rates = {}
            for race in self.RACES:
                race_data = bootstrap_df[bootstrap_df["race"] == race][metric]
                if len(race_data) > 0:
                    rates[race] = race_data.mean()
                    bootstrap_rates_by_race[race].append(race_data.mean())
                else:
                    rates[race] = 0.0
                    bootstrap_rates_by_race[race].append(0.0)

            # Disparity
            if rates:
                disparity = max(rates.values()) - min(rates.values())
                bootstrap_disparities.append(disparity)

        # Compute confidence intervals
        disparity_ci = self._percentile_ci(bootstrap_disparities)

        race_cis = {}
        for race in self.RACES:
            if bootstrap_rates_by_race[race]:
                race_cis[race] = self._percentile_ci(bootstrap_rates_by_race[race])

        return {
            "disparity": BootstrapResult(
                mean=np.mean(bootstrap_disparities),
                ci_lower=disparity_ci[0],
                ci_upper=disparity_ci[1],
                std_error=np.std(bootstrap_disparities)
            ),
            "rates_by_race": {
                race: BootstrapResult(
                    mean=np.mean(bootstrap_rates_by_race[race]),
                    ci_lower=race_cis[race][0],
                    ci_upper=race_cis[race][1],
                    std_error=np.std(bootstrap_rates_by_race[race])
                )
                for race in self.RACES if race in race_cis
            },
            "n_bootstrap": n_bootstrap,
            "sample_unit": sample_unit
        }

    def _percentile_ci(
        self,
        values: list[float],
        alpha: float = 0.05
    ) -> tuple[float, float]:
        """Calculate percentile-based confidence interval."""
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        return (lower, upper)

    def seed_variability_analysis(
        self,
        results_by_seed: dict[int, list[dict]]
    ) -> dict:
        """
        Analyze disparity stability across random seeds.

        Args:
            results_by_seed: Dict mapping seed -> results list

        Returns:
            Seed variability metrics
        """
        disparities_by_seed = []
        max_races = []
        min_races = []

        for seed, results in results_by_seed.items():
            rates = {}
            for race in self.RACES:
                race_results = [r for r in results if r.get("race") == race]
                if race_results:
                    refused = sum(1 for r in race_results if r.get("is_refused", False))
                    rates[race] = refused / len(race_results)
                else:
                    rates[race] = 0.0

            max_race = max(rates, key=rates.get)
            min_race = min(rates, key=rates.get)
            disparity = rates[max_race] - rates[min_race]

            disparities_by_seed.append(disparity)
            max_races.append(max_race)
            min_races.append(min_race)

        # Check consistency
        most_common_max = max(set(max_races), key=max_races.count) if max_races else None
        most_common_min = min(set(min_races), key=min_races.count) if min_races else None

        max_race_consistency = max_races.count(most_common_max) / len(max_races) if max_races else 0
        min_race_consistency = min_races.count(most_common_min) / len(min_races) if min_races else 0

        return {
            "disparities": disparities_by_seed,
            "mean_disparity": np.mean(disparities_by_seed),
            "std_disparity": np.std(disparities_by_seed),
            "cv_disparity": np.std(disparities_by_seed) / np.mean(disparities_by_seed) if np.mean(disparities_by_seed) > 0 else 0,
            "most_common_max_race": most_common_max,
            "most_common_min_race": most_common_min,
            "max_race_consistency": max_race_consistency,
            "min_race_consistency": min_race_consistency,
            "interpretation": self._interpret_seed_variability(
                np.std(disparities_by_seed),
                max_race_consistency,
                min_race_consistency
            )
        }

    def _interpret_seed_variability(
        self,
        std_disparity: float,
        max_consistency: float,
        min_consistency: float
    ) -> str:
        """Interpret seed variability results."""
        avg_consistency = (max_consistency + min_consistency) / 2

        if std_disparity < 0.02 and avg_consistency > 0.8:
            return "Highly consistent across seeds"
        elif std_disparity < 0.05 and avg_consistency > 0.6:
            return "Moderately consistent across seeds"
        else:
            return "WARNING: High variability across seeds"

    def jackknife_disparity(
        self,
        df: pd.DataFrame,
        leave_out_field: str = "prompt_id"
    ) -> dict:
        """
        Jackknife analysis: leave-one-out by prompt/category.

        Args:
            df: DataFrame
            leave_out_field: Field to leave out (e.g., "prompt_id", "category")

        Returns:
            Jackknife disparity estimates
        """
        unique_values = df[leave_out_field].unique()

        jackknife_disparities = []

        for val in unique_values:
            # Leave out this value
            jackknife_df = df[df[leave_out_field] != val]

            # Calculate disparity
            rates = {}
            for race in self.RACES:
                race_data = jackknife_df[jackknife_df["race"] == race]["is_refused"]
                if len(race_data) > 0:
                    rates[race] = race_data.mean()
                else:
                    rates[race] = 0.0

            if rates:
                disparity = max(rates.values()) - min(rates.values())
                jackknife_disparities.append(disparity)

        # Jackknife statistics
        n = len(jackknife_disparities)
        jackknife_mean = np.mean(jackknife_disparities)
        jackknife_var = ((n - 1) / n) * sum((d - jackknife_mean)**2 for d in jackknife_disparities)
        jackknife_se = np.sqrt(jackknife_var)

        return {
            "jackknife_disparities": jackknife_disparities,
            "mean": jackknife_mean,
            "std_error": jackknife_se,
            "ci_lower": jackknife_mean - 1.96 * jackknife_se,
            "ci_upper": jackknife_mean + 1.96 * jackknife_se,
            "leave_out_field": leave_out_field,
            "n_iterations": n
        }
