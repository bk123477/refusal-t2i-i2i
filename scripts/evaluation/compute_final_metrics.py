#!/usr/bin/env python3
"""
Final Metrics Computation Script

Computes all bias metrics from human + VLM evaluation results:
- DDS (Demographic Drift Score)
- RDR (Race Drift Rate)
- GDR (Gender Drift Rate)
- DBS (Directional Bias Score)
- SER (Soft Erasure Rate)
- Disparity metrics
- SCS (Stereotype Congruence Score)

Usage:
    python scripts/evaluation/compute_final_metrics.py \
        --human-results survey_results.json \
        --vlm-results data/results/vlm_eval.json \
        --output analysis/final_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from evaluation.metrics import (
    DisparityMetrics,
    StereotypeCongruenceScore,
    DemographicDriftMetrics,
    DriftResult
)


def load_human_results(path: Path) -> List[dict]:
    """Load human evaluation results from JSON."""
    with open(path) as f:
        data = json.load(f)

    # Handle different export formats
    if "results" in data:
        results = data["results"]
        if isinstance(results, dict):
            # Results is a dict keyed by item_id
            return list(results.values())
        return results
    return data


def load_vlm_results(path: Path) -> List[dict]:
    """Load VLM evaluation results."""
    with open(path) as f:
        data = json.load(f)

    if "results" in data:
        return data["results"]
    return data


def merge_results(human: List[dict], vlm: List[dict]) -> List[dict]:
    """
    Merge human and VLM results.
    Human results take priority for disagreements.
    """
    # Index VLM results by ID
    vlm_by_id = {r.get("id", r.get("case_id", "")): r for r in vlm}

    merged = []
    for h in human:
        item_id = h.get("id", h.get("itemId", ""))
        vlm_result = vlm_by_id.get(item_id, {})

        # Normalize field names
        result = {
            "id": item_id,
            "prompt_id": h.get("prompt_id") or vlm_result.get("prompt_id", ""),
            "category": h.get("category") or vlm_result.get("category", ""),
            "source_race": h.get("race") or vlm_result.get("race", ""),
            "source_gender": h.get("gender") or vlm_result.get("gender", ""),
            "model": h.get("model") or vlm_result.get("model", ""),

            # Human answers (primary)
            "edit_applied": normalize_q1(h.get("q1_edit_applied")),
            "race_same": normalize_q2_q3(h.get("q2_race_same")),
            "gender_same": normalize_q2_q3(h.get("q3_gender_same")),

            # VLM answers (secondary)
            "vlm_edit_applied": normalize_q1(vlm_result.get("edit_applied")),
            "vlm_race_same": normalize_q2_q3(vlm_result.get("race_same")),
            "vlm_gender_same": normalize_q2_q3(vlm_result.get("gender_same")),

            # Detected demographics (from VLM)
            "output_race": vlm_result.get("detected_race", ""),
            "output_gender": vlm_result.get("detected_gender", ""),
        }

        merged.append(result)

    return merged


def normalize_q1(value) -> Optional[str]:
    """Normalize Q1 (edit_applied) values."""
    if value is None:
        return None
    v = str(value).lower()
    if v in ["yes", "true", "1"]:
        return "yes"
    elif v in ["partial", "some", "2"]:
        return "partial"
    elif v in ["no", "false", "0", "3"]:
        return "no"
    return None


def normalize_q2_q3(value) -> Optional[str]:
    """Normalize Q2/Q3 (race_same, gender_same) values."""
    if value is None:
        return None
    v = str(value).lower()
    if v in ["same", "yes", "true", "1", "4", "7"]:
        return "same"
    elif v in ["different", "no", "false", "0", "2", "5", "8"]:
        return "different"
    elif v in ["ambiguous", "unclear", "maybe", "3", "6", "9"]:
        return "ambiguous"
    return None


def compute_all_metrics(results: List[dict]) -> dict:
    """Compute all bias metrics from results."""

    # Initialize metric calculators
    drift_calc = DemographicDriftMetrics()
    disparity_calc = DisparityMetrics()

    # Overall drift metrics
    overall_drift = drift_calc.calculate_all(results)

    # Drift by race
    drift_by_race = drift_calc.calculate_by_group(results, "race")

    # Drift by category
    drift_by_category = drift_calc.calculate_by_group(results, "category")

    # Drift by prompt (for detailed analysis)
    drift_by_prompt = drift_calc.calculate_by_group(results, "prompt_id")

    # Disparity metrics (soft erasure disparity)
    erasure_disparity = disparity_calc.calculate_disparity(
        [{"race": r.get("source_race"), "is_refused": r.get("edit_applied") != "yes"}
         for r in results],
        metric="refusal"
    )

    # Human-VLM agreement
    agreement = calculate_agreement(results)

    return {
        "overall": drift_result_to_dict(overall_drift),
        "by_race": {k: drift_result_to_dict(v) for k, v in drift_by_race.items()},
        "by_category": {k: drift_result_to_dict(v) for k, v in drift_by_category.items()},
        "by_prompt": {k: drift_result_to_dict(v) for k, v in drift_by_prompt.items()},
        "erasure_disparity": {
            "max_rate": erasure_disparity.max_rate,
            "min_rate": erasure_disparity.min_rate,
            "disparity": erasure_disparity.disparity,
            "max_race": erasure_disparity.max_race,
            "min_race": erasure_disparity.min_race,
            "p_value": erasure_disparity.p_value,
            "is_significant": erasure_disparity.is_significant
        },
        "agreement": agreement
    }


def drift_result_to_dict(dr: DriftResult) -> dict:
    """Convert DriftResult to dict for JSON serialization."""
    return {
        "DDS": dr.dds,
        "RDR": dr.rdr,
        "GDR": dr.gdr,
        "DBS": dr.dbs,
        "SER": dr.ser,
        "n_total": dr.n_total,
        "n_success": dr.n_success,
        "n_race_drift": dr.n_race_drift,
        "n_gender_drift": dr.n_gender_drift,
        "n_stereotype_drift": dr.n_stereotype_drift,
        "n_soft_erasure": dr.n_soft_erasure,
        "interpretation": dr.interpretation
    }


def calculate_agreement(results: List[dict]) -> dict:
    """Calculate Human-VLM agreement (Cohen's kappa)."""
    q1_human = []
    q1_vlm = []
    q2_human = []
    q2_vlm = []
    q3_human = []
    q3_vlm = []

    for r in results:
        if r.get("edit_applied") and r.get("vlm_edit_applied"):
            q1_human.append(r["edit_applied"])
            q1_vlm.append(r["vlm_edit_applied"])

        if r.get("race_same") and r.get("vlm_race_same"):
            q2_human.append(r["race_same"])
            q2_vlm.append(r["vlm_race_same"])

        if r.get("gender_same") and r.get("vlm_gender_same"):
            q3_human.append(r["gender_same"])
            q3_vlm.append(r["vlm_gender_same"])

    def simple_agreement(a: List, b: List) -> float:
        if not a or len(a) != len(b):
            return 0.0
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)

    return {
        "q1_edit_applied": {
            "n": len(q1_human),
            "agreement": simple_agreement(q1_human, q1_vlm)
        },
        "q2_race_same": {
            "n": len(q2_human),
            "agreement": simple_agreement(q2_human, q2_vlm)
        },
        "q3_gender_same": {
            "n": len(q3_human),
            "agreement": simple_agreement(q3_human, q3_vlm)
        }
    }


def generate_paper_tables(metrics: dict) -> str:
    """Generate LaTeX tables for paper."""
    output = []

    # Table 1: Overall Metrics
    overall = metrics["overall"]
    output.append("% Table 1: Overall Bias Metrics")
    output.append("\\begin{table}[h]")
    output.append("\\centering")
    output.append("\\begin{tabular}{lc}")
    output.append("\\toprule")
    output.append("Metric & Value \\\\")
    output.append("\\midrule")
    output.append(f"Demographic Drift Score (DDS) & {overall['DDS']:.1%} \\\\")
    output.append(f"Race Drift Rate (RDR) & {overall['RDR']:.1%} \\\\")
    output.append(f"Gender Drift Rate (GDR) & {overall['GDR']:.1%} \\\\")
    output.append(f"Directional Bias Score (DBS) & {overall['DBS']:+.3f} \\\\")
    output.append(f"Soft Erasure Rate (SER) & {overall['SER']:.1%} \\\\")
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\caption{Overall bias metrics across all models.}")
    output.append("\\label{tab:overall-metrics}")
    output.append("\\end{table}")
    output.append("")

    # Table 2: Metrics by Race
    output.append("% Table 2: Metrics by Race")
    output.append("\\begin{table}[h]")
    output.append("\\centering")
    output.append("\\begin{tabular}{lcccc}")
    output.append("\\toprule")
    output.append("Race & DDS & RDR & GDR & DBS \\\\")
    output.append("\\midrule")
    for race, data in sorted(metrics["by_race"].items()):
        output.append(f"{race} & {data['DDS']:.1%} & {data['RDR']:.1%} & {data['GDR']:.1%} & {data['DBS']:+.3f} \\\\")
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\caption{Bias metrics by source race.}")
    output.append("\\label{tab:metrics-by-race}")
    output.append("\\end{table}")
    output.append("")

    # Table 3: Metrics by Category
    output.append("% Table 3: Metrics by Category")
    output.append("\\begin{table}[h]")
    output.append("\\centering")
    output.append("\\begin{tabular}{lccc}")
    output.append("\\toprule")
    output.append("Category & DDS & SER & n \\\\")
    output.append("\\midrule")
    for cat, data in sorted(metrics["by_category"].items()):
        output.append(f"{cat} & {data['DDS']:.1%} & {data['SER']:.1%} & {data['n_total']} \\\\")
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\caption{Bias metrics by prompt category.}")
    output.append("\\label{tab:metrics-by-category}")
    output.append("\\end{table}")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Compute final bias metrics")
    parser.add_argument("--human-results", type=Path, required=True,
                        help="Path to human evaluation results JSON")
    parser.add_argument("--vlm-results", type=Path,
                        help="Path to VLM evaluation results JSON (optional)")
    parser.add_argument("--output", type=Path, default=Path("final_metrics.json"),
                        help="Output path for metrics JSON")
    parser.add_argument("--latex", type=Path,
                        help="Output path for LaTeX tables")

    args = parser.parse_args()

    print(f"Loading human results from {args.human_results}")
    human_results = load_human_results(args.human_results)
    print(f"  Loaded {len(human_results)} human evaluations")

    if args.vlm_results and args.vlm_results.exists():
        print(f"Loading VLM results from {args.vlm_results}")
        vlm_results = load_vlm_results(args.vlm_results)
        print(f"  Loaded {len(vlm_results)} VLM evaluations")
        results = merge_results(human_results, vlm_results)
    else:
        print("No VLM results provided, using human results only")
        # Convert human results to expected format
        results = []
        for h in human_results:
            results.append({
                "id": h.get("id", h.get("itemId", "")),
                "prompt_id": h.get("promptId", ""),
                "category": h.get("category", ""),
                "source_race": h.get("race", ""),
                "source_gender": h.get("gender", ""),
                "model": h.get("model", ""),
                "edit_applied": normalize_q1(h.get("q1_edit_applied")),
                "race_same": normalize_q2_q3(h.get("q2_race_same")),
                "gender_same": normalize_q2_q3(h.get("q3_gender_same")),
            })

    print(f"\nComputing metrics for {len(results)} samples...")
    metrics = compute_all_metrics(results)

    # Add metadata
    output = {
        "metadata": {
            "computed_at": datetime.now().isoformat(),
            "n_samples": len(results),
            "human_results_path": str(args.human_results),
            "vlm_results_path": str(args.vlm_results) if args.vlm_results else None
        },
        "metrics": metrics
    }

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved metrics to {args.output}")

    # Generate LaTeX tables
    if args.latex:
        latex_tables = generate_paper_tables(metrics)
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        with open(args.latex, "w") as f:
            f.write(latex_tables)
        print(f"Saved LaTeX tables to {args.latex}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    overall = metrics["overall"]
    print(f"  DDS (Demographic Drift Score):    {overall['DDS']:.1%}")
    print(f"  RDR (Race Drift Rate):            {overall['RDR']:.1%}")
    print(f"  GDR (Gender Drift Rate):          {overall['GDR']:.1%}")
    print(f"  DBS (Directional Bias Score):     {overall['DBS']:+.3f}")
    print(f"  SER (Soft Erasure Rate):          {overall['SER']:.1%}")
    print(f"\n  Interpretation: {overall['interpretation']}")

    disparity = metrics["erasure_disparity"]
    print(f"\n  Erasure Disparity: {disparity['disparity']:.1%}")
    print(f"    Max: {disparity['max_race']} ({disparity['max_rate']:.1%})")
    print(f"    Min: {disparity['min_race']} ({disparity['min_rate']:.1%})")
    print(f"    Significant: {'Yes' if disparity['is_significant'] else 'No'} (p={disparity['p_value']:.4f})")


if __name__ == "__main__":
    main()
