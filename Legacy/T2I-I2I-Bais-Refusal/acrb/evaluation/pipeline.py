"""
ACRB Evaluation Pipeline (I2I Focus)

End-to-end pipeline for I2I Attribute-Conditioned Refusal Bias evaluation.
Orchestrates prompt loading, model inference, and metric computation.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

from ..models import I2IModelWrapper
from ..metrics import RefusalDetector, CueRetentionScorer, ErasureCalculator, DisparityMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result with refined metrics."""
    model_name: str
    total_samples: int
    total_refused: int
    total_failed: int
    refusal_rate: float
    failure_rate: float
    # Per-attribute metrics (Keys: "type:value")
    refusal_by_attribute: Dict[str, float]
    erasure_by_attribute: Dict[str, float]
    substitution_by_attribute: Dict[str, float]
    # Disparity metrics
    delta_refusal: float
    delta_erasure: float
    # Metadata
    timestamp: str
    run_id: str
    config: Dict


class ACRBEvaluationPipeline:
    """
    ACRB I2I Evaluation Pipeline with robustness fixes.
    """

    def __init__(
        self,
        model_name: str = "flux-2-dev",
        output_dir: str = "experiments/results",
        attribute_types: Optional[List[str]] = None,
        seed: int = 42,
        run_id: Optional[str] = None,
        i2i_dataset_dir: Optional[str] = None  # Path to FFHQ/COCO etc.
    ):
        self.model_name = model_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.i2i_dataset_dir = Path(i2i_dataset_dir) if i2i_dataset_dir else None
        self.output_dir = Path(output_dir) / model_name / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)

        self.attribute_types = attribute_types or ["culture", "gender", "disability", "religion", "age"]
        self.seed = seed

        self._init_components()

    def _init_components(self):
        # I2I Model
        self.model = I2IModelWrapper(
            model_name=self.model_name,
            output_dir=str(self.output_dir / "images")
        )

        self.refusal_detector = RefusalDetector(use_clip=True)
        self.cue_scorer = CueRetentionScorer(use_local=False)
        self.erasure_calc = ErasureCalculator()
        self.disparity_metric = DisparityMetric()

    def load_prompts(self, prompts_file: Optional[str] = None) -> List[Dict]:
        """Load prompts from file."""
        if prompts_file and Path(prompts_file).exists():
            with open(prompts_file, 'r') as f:
                return json.load(f)

        # Default location
        default_path = Path("data/prompts/acrb_expanded/acrb_minimal_pairs.json")
        if default_path.exists():
            with open(default_path, 'r') as f:
                return json.load(f)

        logger.error("No prompts file found")
        return []

    def run_inference(self, prompts: List[Dict]) -> List[Dict]:
        """Run I2I inference on prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            current_seed = self.seed + i

            try:
                # Prepare source image
                source_image_path = self._prepare_i2i_source(prompt, current_seed)
                if not source_image_path:
                    results.append({
                        **prompt,
                        "is_failed": True,
                        "failure_reason": "I2I_source_not_found"
                    })
                    continue

                # Apply attribute-conditioned edit
                edit_result = self.model.edit(
                    image_path=source_image_path,
                    instruction=prompt.get("expanded_text", prompt.get("text", "")),
                    seed=current_seed
                )

                results.append({
                    **prompt,
                    "source_image": source_image_path,
                    "image_path": edit_result.output_path,
                    "generation_success": edit_result.success,
                    "api_refused": edit_result.refused,
                    "is_failed": not edit_result.success and not edit_result.refused,
                    "failure_reason": edit_result.refusal_message if not edit_result.success else None,
                    "used_seed": current_seed
                })
            except Exception as e:
                logger.error(f"I2I Inference failed for prompt {i}: {e}")
                results.append({**prompt, "is_failed": True, "failure_reason": str(e)})

        return results

    def _prepare_i2i_source(self, prompt: Dict, seed: int) -> Optional[str]:
        """Get source image for I2I editing."""
        # Check dataset directory
        if self.i2i_dataset_dir and self.i2i_dataset_dir.exists():
            base_prompt_id = prompt.get("base_prompt_id", prompt.get("id", ""))

            # Try to find matching images
            matches = list(self.i2i_dataset_dir.glob(f"{base_prompt_id}*"))
            if not matches:
                domain = prompt.get('domain', '')
                matches = list(self.i2i_dataset_dir.glob(f"{domain}*"))

            if not matches:
                # Fall back to any available image
                matches = list(self.i2i_dataset_dir.glob("*.jpg")) + \
                          list(self.i2i_dataset_dir.glob("*.png"))

            if matches:
                selected = matches[seed % len(matches)]
                return str(selected)

        logger.error("No source images available. Set i2i_dataset_dir.")
        return None

    def evaluate_refusals(self, results: List[Dict]) -> List[Dict]:
        """Detect hard refusals."""
        evaluated = []
        for result in results:
            if result.get("is_failed"):
                evaluated.append({**result, "is_refused": False})
                continue

            refusal_result = self.refusal_detector.detect(
                image_path=result.get("image_path"),
                text_output=result.get("refusal_message"),
            )

            is_refused = refusal_result.is_refused or result.get("api_refused", False)

            evaluated.append({
                **result,
                "is_refused": is_refused,
                "refusal_type": refusal_result.refusal_type.value if not result.get("api_refused") else "API_BLOCK",
                "refusal_confidence": refusal_result.confidence,
            })
        return evaluated

    def evaluate_cue_retention(self, results: List[Dict]) -> List[Dict]:
        """Score cue retention for non-refused results."""
        evaluated = []
        for result in results:
            if result.get("is_refused") or result.get("is_failed") or not result.get("image_path"):
                evaluated.append({**result, "cue_retention_score": None})
                continue

            retention_result = self.cue_scorer.score(
                image_path=result["image_path"],
                attribute_type=result.get("attribute_type", ""),
                attribute_value=result.get("attribute_value", ""),
                attribute_marker=result.get("attribute_marker", ""),
            )
            evaluated.append({
                **result,
                "cue_retention_score": retention_result.retention_score,
                "cue_present": retention_result.is_present,
                "cue_substituted": retention_result.is_substituted,
            })
        return evaluated

    def compute_metrics(self, results: List[Dict]) -> EvaluationResult:
        """Compute evaluation metrics."""
        total = len(results)
        refused = sum(1 for r in results if r.get("is_refused"))
        failed = sum(1 for r in results if r.get("is_failed"))

        metrics_by_attr = {}

        for r in results:
            if r.get("is_failed"):
                continue

            attr_type = r.get("attribute_type", "unknown")
            attr_val = r.get("attribute_value", "unknown")
            key = f"{attr_type}:{attr_val}"

            if key not in metrics_by_attr:
                metrics_by_attr[key] = {
                    "refused": 0, "total": 0, "erased": 0,
                    "substituted": 0, "retained": 0
                }

            metrics_by_attr[key]["total"] += 1
            if r.get("is_refused"):
                metrics_by_attr[key]["refused"] += 1
            elif r.get("cue_retention_score") is not None:
                score = r["cue_retention_score"]
                if score < self.erasure_calc.retention_threshold:
                    metrics_by_attr[key]["erased"] += 1
                elif r.get("cue_substituted"):
                    metrics_by_attr[key]["substituted"] += 1
                else:
                    metrics_by_attr[key]["retained"] += 1

        # Calculate Rates
        refusal_rates = {
            k: v["refused"]/v["total"] if v["total"] > 0 else 0
            for k, v in metrics_by_attr.items()
        }
        erasure_rates = {
            k: v["erased"]/v["total"] if v["total"] > 0 else 0
            for k, v in metrics_by_attr.items()
        }
        substitution_rates = {
            k: v["substituted"]/v["total"] if v["total"] > 0 else 0
            for k, v in metrics_by_attr.items()
        }

        ref_disp = self.disparity_metric.compute_refusal_disparity(refusal_rates)
        era_disp = self.disparity_metric.compute_erasure_disparity(erasure_rates)

        return EvaluationResult(
            model_name=self.model_name,
            total_samples=total,
            total_refused=refused,
            total_failed=failed,
            refusal_rate=refused / (total - failed) if (total - failed) > 0 else 0,
            failure_rate=failed / total if total > 0 else 0,
            refusal_by_attribute=refusal_rates,
            erasure_by_attribute=erasure_rates,
            substitution_by_attribute=substitution_rates,
            delta_refusal=ref_disp.delta,
            delta_erasure=era_disp.delta,
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            config={"attribute_types": self.attribute_types, "seed": self.seed}
        )

    def run(self, prompts_file: Optional[str] = None) -> EvaluationResult:
        """Run complete evaluation pipeline."""
        logger.info(f"Starting ACRB I2I Run [{self.run_id}] for {self.model_name}")

        prompts = self.load_prompts(prompts_file)
        if not prompts:
            raise ValueError("No prompts available")

        results = self.run_inference(prompts)
        results = self.evaluate_refusals(results)
        results = self.evaluate_cue_retention(results)
        evaluation = self.compute_metrics(results)
        self._save_results(results, evaluation)

        return evaluation

    def _save_results(self, results: List[Dict], evaluation: EvaluationResult):
        """Save results to disk."""
        with open(self.output_dir / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(asdict(evaluation), f, indent=2, default=str)
        logger.info(f"Run completed. Results stored in {self.output_dir}")


def main():
    """Example usage."""
    pipeline = ACRBEvaluationPipeline(
        model_name="flux-2-dev",
        attribute_types=["culture", "gender"],
        i2i_dataset_dir="data/raw/ffhq"
    )

    print("ACRB I2I Evaluation Pipeline initialized")
    print(f"  Model: {pipeline.model_name}")
    print(f"  Attributes: {pipeline.attribute_types}")


if __name__ == "__main__":
    main()
