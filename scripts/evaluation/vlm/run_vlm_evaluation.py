#!/usr/bin/env python3
"""
Run VLM Evaluation on Generated Images
Evaluates soft erasure AND identity drift using Qwen3-VL + Gemini ensemble

Usage:
    # Production (30B model - default) with identity drift detection
    python scripts/evaluation/run_vlm_evaluation.py --results-dir data/results/flux/20260109_120000

    # Local testing (8B model)
    python scripts/evaluation/run_vlm_evaluation.py --results-dir data/results/flux/20260109_120000 --qwen-model 8B

    # Skip identity drift (faster, erasure-only)
    python scripts/evaluation/run_vlm_evaluation.py --results-dir data/results/flux/20260109_120000 --skip-identity-drift

"""

import argparse
import json
from pathlib import Path
from PIL import Image
import sys
from typing import Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.vlm_evaluator import VLMEvaluator
from src.data.prompt_loader import PromptLoader
from src.config import PathConfig


def _resolve_output_image_path(
    output_image_path: Optional[str],
    results_dir: Path,
    race_code: Optional[str],
    images_root: Optional[Path] = None
) -> Optional[Path]:
    """Resolve output image path, handling stale absolute paths."""
    if not output_image_path:
        return None

    candidate = Path(output_image_path)
    if candidate.exists():
        return candidate

    images_root = images_root or (results_dir / "images")
    parts = candidate.parts
    if "images" in parts:
        idx = parts.index("images")
        rel = Path(*parts[idx + 1 :])
        candidate = images_root / rel
        if candidate.exists():
            return candidate

    if race_code:
        candidate = images_root / race_code / Path(output_image_path).name
        if candidate.exists():
            return candidate

    if images_root.exists():
        matches = list(images_root.rglob(Path(output_image_path).name))
        if len(matches) == 1:
            return matches[0]

    return None


def _resolve_source_image_path(
    source_image_path: Optional[str],
    race_code: Optional[str],
    source_images_root: Optional[Path] = None
) -> Optional[Path]:
    """Resolve source image path, handling stale absolute paths."""
    if not source_image_path:
        return None

    candidate = Path(source_image_path)
    if candidate.exists():
        return candidate

    source_images_root = source_images_root or PathConfig().source_images_base
    parts = candidate.parts
    if "source_images" in parts:
        idx = parts.index("source_images")
        rel = Path(*parts[idx + 1 :])
        candidate = source_images_root / rel
        if candidate.exists():
            return candidate

    if race_code:
        candidate = (source_images_root / "final") / race_code / Path(source_image_path).name
        if candidate.exists():
            return candidate

    return None


def evaluate_results(
    results_dir: Path,
    evaluator: VLMEvaluator,
    prompts_file: Path,
    images_root: Optional[Path],
    source_images_root: Optional[Path],
    include_identity_drift: bool = True
) -> dict:
    """Run VLM evaluation on experiment results (erasure + identity drift)."""

    # Load experiment results
    results_file = results_dir / "results.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return {}

    with open(results_file) as f:
        results = json.load(f)

    # Load prompts for attribute extraction
    prompt_loader = PromptLoader(str(prompts_file))

    print(f"Evaluating {len(results)} results...")

    evaluated_results = []
    for i, result in enumerate(results):
        # Skip if already refused or errored
        if result.get("is_refused") or result.get("error"):
            result["vlm_evaluation"] = None
            evaluated_results.append(result)
            continue

        # Load generated image
        image_path = _resolve_output_image_path(
            output_image_path=result.get("output_image"),
            results_dir=results_dir,
            race_code=result.get("race_code"),
            images_root=images_root
        )

        if not image_path or not image_path.exists():
            print(f"  [{i+1}] Image not found: {result.get('output_image')}")
            result["vlm_evaluation"] = None
            evaluated_results.append(result)
            continue

        try:
            image = Image.open(image_path)

            # Load source image for edit comparison
            source_image = None
            source_path = _resolve_source_image_path(
                source_image_path=result.get("source_image"),
                race_code=result.get("race_code"),
                source_images_root=source_images_root
            )
            if source_path and source_path.exists():
                source_image = Image.open(source_path)

            # Use the original prompt text as instruction
            prompt_id = result.get("prompt_id", "")
            prompt_obj = prompt_loader.get_by_id(prompt_id)
            instruction = prompt_obj.prompt if prompt_obj else result.get("prompt_text", "")

            if source_image is None:
                # Fallback to attribute check when source image is unavailable
                vlm_result = evaluator.check_attribute(
                    image=image,
                    attribute=instruction,
                    return_detailed=True
                )
            else:
                vlm_result = evaluator.check_edit(
                    source_image=source_image,
                    edited_image=image,
                    instruction=instruction,
                    return_detailed=True
                )

            result["vlm_evaluation"] = vlm_result
            result["is_erased"] = vlm_result["final_result"] in ["NO", "PARTIAL"]

            status = "ERASED" if result["is_erased"] else "PRESENT"
            consensus = "consensus" if vlm_result["consensus"] else "DISAGREE"

            # Identity Drift Detection (if enabled and source image available)
            if include_identity_drift and source_image is not None:
                source_race = result.get("race_code", "")
                source_gender = result.get("gender", "")
                category = result.get("category", "")

                identity_result = evaluator.evaluate_identity_preservation(
                    source_image=source_image,
                    edited_image=image,
                    source_race=source_race,
                    source_gender=source_gender,
                    prompt=instruction
                )

                result["identity_preservation"] = identity_result
                result["stereotype_replacement"] = evaluator.is_stereotype_replacement(
                    drift_result=identity_result,
                    prompt=instruction,
                    source_race=source_race,
                    source_gender=source_gender,
                    category=category
                )

                drift_status = ""
                if not identity_result.get("identity_preserved", True):
                    drift_status = f" | DRIFT: {identity_result.get('racial_drift', '')} {identity_result.get('gender_drift', '')}"
                print(f"  [{i+1}/{len(results)}] {prompt_id}: {status} ({consensus}){drift_status}")
            else:
                print(f"  [{i+1}/{len(results)}] {prompt_id}: {status} ({consensus})")

        except Exception as e:
            print(f"  [{i+1}] Error evaluating {image_path}: {e}")
            result["vlm_evaluation"] = {"error": str(e)}

        evaluated_results.append(result)

    return evaluated_results


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM evaluation on generated images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production (30B model)
  python scripts/evaluation/run_vlm_evaluation.py --results-dir data/results/flux/20260109_120000

  # Local testing (8B model)
  python scripts/evaluation/run_vlm_evaluation.py --results-dir data/results/flux/20260109_120000 --qwen-model 8B

  # All results for a model
  python scripts/evaluation/run_vlm_evaluation.py --model flux
        """
    )
    parser.add_argument("--results-dir", type=str,
                       help="Specific experiment results directory")
    parser.add_argument("--model", type=str,
                       help="I2I model name (flux, step1x, qwen) - evaluates latest experiment")
    parser.add_argument("--qwen-model", type=str, default="30B", choices=["30B", "8B"],
                       help="Qwen VLM model size: 30B (default) or 8B")
    parser.add_argument("--prompts", type=str,
                       default="data/prompts/i2i_prompts.json",
                       help="Path to prompts JSON")
    parser.add_argument("--no-ensemble", action="store_true",
                       help="Use Qwen only (no Gemini ensemble)")
    parser.add_argument("--skip-identity-drift", action="store_true",
                       help="Skip identity drift detection (faster, erasure-only)")
    parser.add_argument("--images-root", type=str,
                       help="Override images root directory (default: results_dir/images)")
    parser.add_argument("--source-images-root", type=str,
                       help="Override source images root (default: data/source_images)")
    parser.add_argument("--output", type=str,
                       help="Output file path (default: vlm_results.json in results dir)")

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.model:
        base_dir = PathConfig().results_dir / args.model
        if not base_dir.exists():
            print(f"Model directory not found: {base_dir}")
            sys.exit(1)
        # Get latest experiment
        exp_dirs = sorted(base_dir.glob("*/"), key=lambda x: x.name, reverse=True)
        if not exp_dirs:
            print(f"No experiments found in {base_dir}")
            sys.exit(1)
        results_dir = exp_dirs[0]
    else:
        print("Error: Either --results-dir or --model must be specified")
        sys.exit(1)

    print(f"Results directory: {results_dir}")
    print(f"Qwen model: {args.qwen_model}")
    print(f"Ensemble: {not args.no_ensemble}")
    print(f"Identity Drift: {not args.skip_identity_drift}")

    images_root = Path(args.images_root) if args.images_root else None
    source_images_root = Path(args.source_images_root) if args.source_images_root else None

    if images_root:
        print(f"Images root: {images_root}")
    if source_images_root:
        print(f"Source images root: {source_images_root}")
    print()

    # Initialize evaluator
    evaluator = VLMEvaluator(
        use_ensemble=not args.no_ensemble,
        qwen_model_size=args.qwen_model
    )

    # Run evaluation
    evaluated_results = evaluate_results(
        results_dir=results_dir,
        evaluator=evaluator,
        prompts_file=Path(args.prompts),
        images_root=images_root,
        source_images_root=source_images_root,
        include_identity_drift=not args.skip_identity_drift
    )

    # Save results
    output_path = Path(args.output) if args.output else results_dir / "vlm_results.json"
    with open(output_path, "w") as f:
        json.dump(evaluated_results, f, indent=2, default=str)

    # Summary
    total = len(evaluated_results)
    erased = sum(1 for r in evaluated_results if r.get("is_erased"))
    refused = sum(1 for r in evaluated_results if r.get("is_refused"))
    needs_review = sum(1 for r in evaluated_results
                      if r.get("vlm_evaluation") and not r["vlm_evaluation"].get("consensus", True))

    # Identity drift stats
    has_identity = [r for r in evaluated_results if r.get("identity_preservation")]
    racial_drift = sum(1 for r in has_identity
                      if r["identity_preservation"].get("racial_drift", "SAME") != "SAME")
    gender_drift = sum(1 for r in has_identity
                      if r["identity_preservation"].get("gender_drift", "SAME") != "SAME")
    stereotype_repl = sum(1 for r in evaluated_results if r.get("stereotype_replacement"))

    print()
    print("=" * 50)
    print("VLM EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Total:              {total}")
    print(f"Hard Refused:       {refused} ({refused/total*100:.1f}%)")
    print(f"Soft Erased:        {erased} ({erased/total*100:.1f}%)")
    print(f"Needs Review:       {needs_review} ({needs_review/total*100:.1f}%)")

    if has_identity:
        print()
        print("--- Identity Drift ---")
        print(f"Evaluated:          {len(has_identity)}")
        print(f"Racial Drift:       {racial_drift} ({racial_drift/len(has_identity)*100:.1f}%)")
        print(f"Gender Drift:       {gender_drift} ({gender_drift/len(has_identity)*100:.1f}%)")
        print(f"Stereotype Repl:    {stereotype_repl} ({stereotype_repl/len(has_identity)*100:.1f}%)")

    print()
    print(f"Results saved:  {output_path}")


if __name__ == "__main__":
    main()
