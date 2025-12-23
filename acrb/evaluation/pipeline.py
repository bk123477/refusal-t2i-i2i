"""
ACRB Evaluation Pipeline

End-to-end pipeline for Attribute-Conditioned Refusal Bias evaluation.
Orchestrates prompt generation, model inference, and metric computation.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

from ..prompt_generation import BasePromptGenerator, AttributeExpander
from ..models import T2IModelWrapper, I2IModelWrapper
from ..metrics import RefusalDetector, CueRetentionScorer, ErasureCalculator, DisparityMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result with refined metrics."""
    model_name: str
    mode: str
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


class ACRBPipeline:
    """
    Refined ACRB Evaluation Pipeline with robustness fixes.
    """
    
    def __init__(
        self,
        model_name: str = "seedream-4.5",
        mode: str = "t2i",
        output_dir: str = "experiments/results",
        attribute_types: Optional[List[str]] = None,
        seed: int = 42,
        run_id: Optional[str] = None,
        llm_model: Optional[str] = None,  # e.g., "gpt-oss-20b"
        llm_api_base: str = "http://localhost:8000/v1",
        i2i_dataset_dir: Optional[str] = None  # Path to FFHQ/COCO etc.
    ):
        self.model_name = model_name
        self.mode = mode
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.i2i_dataset_dir = Path(i2i_dataset_dir) if i2i_dataset_dir else None
        self.output_dir = Path(output_dir) / model_name / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        
        self.attribute_types = attribute_types or ["culture", "gender", "disability", "religion", "age"]
        self.seed = seed
        
        self._init_components()
    
    def _init_components(self):
        self.prompt_generator = BasePromptGenerator(seed=self.seed)
        self.attribute_expander = AttributeExpander()
        
        if self.llm_model:
            self.prompt_generator.enable_llm(model_name=self.llm_model, api_base=self.llm_api_base)
            self.attribute_expander.enable_llm(model_name=self.llm_model, api_base=self.llm_api_base)
        
        # Models
        if self.mode == "t2i":
            self.model = T2IModelWrapper(model_name=self.model_name, output_dir=str(self.output_dir / "images"))
            self.t2i_aux_model = None
        else:
            self.model = I2IModelWrapper(model_name=self.model_name, output_dir=str(self.output_dir / "images"))
            # Auxiliary T2I model for generating neutral source images (using FLUX.2 as a strong neutral baseline)
            self.t2i_aux_model = T2IModelWrapper(model_name="flux-2-dev", output_dir=str(self.output_dir / "images/source"))
            
        self.refusal_detector = RefusalDetector(use_clip=True)
        self.cue_scorer = CueRetentionScorer(use_local=False)
        self.erasure_calc = ErasureCalculator()
        self.disparity_metric = DisparityMetric()

    def generate_prompts(self, max_base: int = 100) -> List[Dict]:
        base_prompts = self.prompt_generator.sample_prompts(max_base)
        
        if self.llm_model:
            expanded = []
            for bp in base_prompts:
                expanded.extend(self.attribute_expander.expand_prompt_llm(bp, attribute_types=self.attribute_types))
        else:
            expanded = self.attribute_expander.expand_all(base_prompts, attribute_types=self.attribute_types)
            
        return self.attribute_expander.export_to_dict(expanded)

    def run_inference(self, prompts: List[Dict]) -> List[Dict]:
        results = []
        for i, prompt in enumerate(prompts):
            # Seed diversity: base seed + prompt index
            current_seed = self.seed + i
            
            if self.mode == "t2i":
                try:
                    gen_results = self.model.generate(
                        prompt=prompt["expanded_text"],
                        num_images=1,
                        seed=current_seed
                    )
                    result = gen_results[0] if gen_results else None
                    results.append(self._process_inference_result(prompt, result, current_seed))
                except Exception as e:
                    logger.error(f"T2I Inference failed for prompt {i}: {e}")
                    results.append({**prompt, "generation_success": False, "is_failed": True, "failure_reason": str(e)})
            else:
                # I2I Mode: Two-step minimal-pair generation
                try:
                    # 1. Prepare neutral source image
                    source_image_path = self._prepare_i2i_source(prompt, current_seed)
                    if not source_image_path:
                        results.append({**prompt, "is_failed": True, "failure_reason": "I2I_source_generation_failed"})
                        continue
                    
                    # 2. Apply attribute-conditioned edit
                    edit_result = self.model.edit(
                        image_path=source_image_path,
                        instruction=prompt["expanded_text"],
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
        """Generate or retrieve a neutral source image for the base prompt."""
        base_prompt_id = prompt["base_prompt_id"]
        
        # 1. Check if we have a provided dataset directory (Priority 1)
        if self.i2i_dataset_dir and self.i2i_dataset_dir.exists():
            # Strategy: Find images mapped to this prompt or domain
            # We look for files named [prompt_id].png or [domain]_*.png
            matches = list(self.i2i_dataset_dir.glob(f"{base_prompt_id}*"))
            if not matches:
                matches = list(self.i2i_dataset_dir.glob(f"{prompt['domain']}*"))
            
            if matches:
                # Use deterministic selection based on seed
                selected = matches[seed % len(matches)]
                return str(selected)

        # 2. Check internal cache (Priority 2)
        source_dir = Path(self.output_dir) / "images/source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_filename = f"source_{base_prompt_id}.png"
        source_path = source_dir / source_filename
        
        if source_path.exists():
            return str(source_path)
            
        # 3. Fallback: Generate via auxiliary T2I (Priority 3)
        if not self.t2i_aux_model:
            return None
            
        try:
            logger.info(f"Generating synthetic source for {base_prompt_id} via T2I fallback.")
            gen_results = self.t2i_aux_model.generate(
                prompt=prompt["base_text"],
                num_images=1,
                seed=self.seed
            )
            if gen_results and gen_results[0].success:
                import shutil
                shutil.copy(gen_results[0].image_path, source_path)
                return str(source_path)
        except Exception as e:
            logger.error(f"Failed to generate I2I source image: {e}")
            
        return None

    def _process_inference_result(self, prompt: Dict, result: Any, seed: int) -> Dict:
        """Helper to format inference results."""
        return {
            **prompt,
            "image_path": result.image_path if result else None,
            "generation_success": bool(result and result.success),
            "api_refused": bool(result and result.refused),
            "is_failed": result is None or (not result.success and not result.refused),
            "failure_reason": None if (result and result.success) else ("empty" if not result else "api_error"),
            "refusal_message": result.refusal_message if result else None,
            "used_seed": seed
        }

    def evaluate_refusals(self, results: List[Dict]) -> List[Dict]:
        evaluated = []
        for result in results:
            if result.get("is_failed"):
                evaluated.append({**result, "is_refused": False})
                continue
                
            refusal_result = self.refusal_detector.detect(
                image_path=result.get("image_path"),
                text_output=result.get("refusal_message"),
            )
            
            # Prioritize API refusal flag if present
            is_refused = refusal_result.is_refused or result.get("api_refused", False)
            
            evaluated.append({
                **result,
                "is_refused": is_refused,
                "refusal_type": refusal_result.refusal_type.value if not result.get("api_refused") else "API_BLOCK",
                "refusal_confidence": refusal_result.confidence,
            })
        return evaluated

    def evaluate_cue_retention(self, results: List[Dict]) -> List[Dict]:
        evaluated = []
        for result in results:
            if result.get("is_refused") or result.get("is_failed") or not result.get("image_path"):
                evaluated.append({**result, "cue_retention_score": None})
                continue
            
            retention_result = self.cue_scorer.score(
                image_path=result["image_path"],
                attribute_type=result["attribute_type"],
                attribute_value=result["attribute_value"],
                attribute_marker=result["attribute_marker"],
            )
            evaluated.append({
                **result,
                "cue_retention_score": retention_result.retention_score,
                "cue_present": retention_result.is_present,
                "cue_substituted": retention_result.is_substituted,
            })
        return evaluated

    def compute_metrics(self, results: List[Dict]) -> EvaluationResult:
        total = len(results)
        refused = sum(1 for r in results if r.get("is_refused"))
        failed = sum(1 for r in results if r.get("is_failed"))
        
        # Nested Grouping: { "type:value": { "refused": X, "total": Y, "erased": Z, "substituted": W } }
        metrics_by_attr = {}
        
        for r in results:
            if r.get("is_failed"): continue
            
            attr_type = r.get("attribute_type", "unknown")
            attr_val = r.get("attribute_value", "unknown")
            key = f"{attr_type}:{attr_val}"
            
            if key not in metrics_by_attr:
                metrics_by_attr[key] = {"refused": 0, "total": 0, "erased": 0, "substituted": 0, "retained": 0}
            
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
        refusal_rates = {k: v["refused"]/v["total"] if v["total"] > 0 else 0 for k, v in metrics_by_attr.items()}
        erasure_rates = {k: v["erased"]/v["total"] if v["total"] > 0 else 0 for k, v in metrics_by_attr.items()}
        substitution_rates = {k: v["substituted"]/v["total"] if v["total"] > 0 else 0 for k, v in metrics_by_attr.items()}
        
        ref_disp = self.disparity_metric.compute_refusal_disparity(refusal_rates)
        era_disp = self.disparity_metric.compute_erasure_disparity(erasure_rates)
        
        return EvaluationResult(
            model_name=self.model_name,
            mode=self.mode,
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

    def run(self, max_base: int = 100) -> EvaluationResult:
        logger.info(f"Starting ACRB Run [{self.run_id}] for {self.model_name}")
        prompts = self.generate_prompts(max_base)
        results = self.run_inference(prompts)
        results = self.evaluate_refusals(results)
        results = self.evaluate_cue_retention(results)
        evaluation = self.compute_metrics(results)
        self._save_results(results, evaluation)
        return evaluation

    def _save_results(self, results: List[Dict], evaluation: EvaluationResult):
        with open(self.output_dir / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(asdict(evaluation), f, indent=2, default=str)
        logger.info(f"Run completed. Results stored in {self.output_dir}")


def main():
    """Example usage."""
    pipeline = ACRBPipeline(
        model_name="seedream-4.5",
        mode="t2i",
        attribute_types=["culture", "gender"],
    )
    
    print("ACRB Pipeline initialized")
    print(f"  Model: {pipeline.model_name}")
    print(f"  Mode: {pipeline.mode}")
    print(f"  Attributes: {pipeline.attribute_types}")
    
    # Dry run - generate prompts only
    prompts = pipeline.generate_prompts(max_base=5)
    print(f"\nGenerated {len(prompts)} prompts")
    for p in prompts[:3]:
        print(f"  [{p['attribute_type']}:{p['attribute_value']}] {p['expanded_text'][:60]}...")


if __name__ == "__main__":
    main()
