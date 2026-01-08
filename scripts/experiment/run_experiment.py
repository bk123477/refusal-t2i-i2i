#!/usr/bin/env python3
"""
Main Experiment Runner for I2I Refusal Bias Study
Runs 50 prompts × 84 images = 4,200 requests per model

Usage:
    python scripts/run_experiment.py --model flux --device cuda
    python scripts/run_experiment.py --model step1x --device cuda
    python scripts/run_experiment.py --model qwen --device cuda
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import PathConfig, NamingConfig, ExperimentConfig, MODEL_CONFIGS
from src.utils.logger import ExperimentLogger
from src.data.prompt_loader import PromptLoader
from src.models import FluxWrapper, Step1XWrapper, QwenImageEditWrapper
from src.evaluation import RefusalDetector


def load_source_images(image_dir: Path) -> list[dict]:
    """
    Load sampled FairFace images with metadata.

    Folder structure: {image_dir}/{RaceCode}/{RaceCode}_{Gender}_{AgeCode}.jpg
    """
    metadata_path = image_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata["images"]
    else:
        # Fallback: scan folders
        images = []
        race_codes = ["Black", "EastAsian", "Indian", "Latino", "MiddleEastern", "SoutheastAsian", "White"]

        for race_code in race_codes:
            race_dir = image_dir / race_code
            if not race_dir.exists():
                continue
            for img_path in sorted(race_dir.glob("*.jpg")):
                parts = img_path.stem.split("_")
                if len(parts) >= 3:
                    images.append({
                        "race_code": parts[0],
                        "gender": parts[1],
                        "age_code": parts[2],
                        "path": str(img_path),
                        "filename": img_path.name
                    })
        return images


def get_model(model_name: str, device: str, config: dict):
    """Initialize model based on name."""
    if model_name == "flux":
        return FluxWrapper(
            device=device,
            use_quantized=config.get("use_quantized", False),
            use_remote_text_encoder=config.get("use_remote_text_encoder", False)
        )
    elif model_name == "step1x":
        return Step1XWrapper(
            device=device,
            enable_thinking=config.get("enable_thinking", True),
            enable_reflection=config.get("enable_reflection", True)
        )
    elif model_name == "qwen":
        return QwenImageEditWrapper(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(
    model_name: str,
    device: str = "cuda",
    experiment_id: str = None,
    resume_from: int = 0,
    source_version: str = "V1"
):
    """
    Run experiment for a single model.

    Args:
        model_name: "flux", "step1x", or "qwen"
        device: Device to run on
        experiment_id: Optional experiment ID (for resuming)
        resume_from: Resume from this index (for interrupted experiments)
        source_version: Source image version (V1, V2, V3, etc.)
    """
    # Setup paths
    path_config = PathConfig()
    path_config.set_version(source_version)
    naming = NamingConfig()

    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_paths = path_config.setup_experiment_dirs(model_name, experiment_id)

    # Setup logger
    logger = ExperimentLogger(
        experiment_id=experiment_id,
        model_name=model_name,
        log_dir=exp_paths["logs_dir"]
    )

    # Load model config
    model_config = MODEL_CONFIGS.get(model_name, {})

    # Save experiment config
    config = ExperimentConfig(
        experiment_id=experiment_id,
        model_name=model_name,
        device=device,
        **model_config
    )
    config.save(exp_paths["experiment_dir"] / "config.json")

    print(f"\n{'='*60}")
    print(f"I2I Refusal Bias Experiment")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Source Version: {source_version}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Output: {exp_paths['experiment_dir']}")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading {model_name}...")
    model = get_model(model_name, device, model_config)
    model.load()

    # Load data
    prompts = PromptLoader(str(path_config.prompts_file))
    images = load_source_images(path_config.source_images_dir)
    refusal_detector = RefusalDetector()

    total_requests = len(prompts) * len(images)
    print(f"Loaded {len(prompts)} prompts and {len(images)} images")
    print(f"Total requests: {total_requests}")

    # Start experiment
    logger.start_experiment(total_requests)

    results = []
    request_idx = 0

    # Run experiment
    pbar = tqdm(total=total_requests, desc=f"{model_name}", initial=resume_from)

    try:
        for img_data in images:
            source_image = Image.open(img_data["path"])

            for prompt in prompts:
                request_idx += 1

                # Skip if resuming
                if request_idx <= resume_from:
                    continue

                # Generate edit
                try:
                    edit_result = model.edit(
                        source_image,
                        prompt.prompt,
                        seed=config.seed,
                        num_inference_steps=model_config.get("num_inference_steps", 50)
                    )
                except Exception as e:
                    edit_result = type('EditResult', (), {
                        'success': False,
                        'output_image': None,
                        'error_message': str(e),
                        'latency_ms': 0,
                        'refusal_type': type('RefusalType', (), {'value': 'error'})()
                    })()

                # Detect refusal
                refusal_result = refusal_detector.detect(
                    source_image=source_image,
                    output_image=edit_result.output_image,
                    error_message=edit_result.error_message
                )

                # Determine status and save image
                race_code = img_data.get("race_code", img_data.get("race", ""))
                age_code = img_data.get("age_code", img_data.get("age", ""))

                if refusal_result.is_refused:
                    status = "refused"
                    output_path = None
                elif edit_result.success:
                    status = "success"
                    # Save output image with race subfolder
                    output_path = naming.get_output_path(
                        base_dir=exp_paths["images_dir"],
                        prompt_id=prompt.id,
                        race_code=race_code,
                        gender=img_data["gender"],
                        age_code=age_code,
                        status=status
                    )
                    # Create race subfolder if needed
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if edit_result.output_image:
                        edit_result.output_image.save(output_path)
                else:
                    status = "error"
                    output_path = None

                # Log generation
                logger.log_generation(
                    prompt_id=prompt.id,
                    prompt_text=prompt.prompt,
                    category=prompt.category,
                    race_code=race_code,
                    gender=img_data["gender"],
                    age_code=age_code,
                    success=edit_result.success,
                    is_refused=refusal_result.is_refused,
                    refusal_type=refusal_result.refusal_type,
                    error_message=edit_result.error_message,
                    latency_ms=edit_result.latency_ms,
                    output_path=str(output_path) if output_path else None
                )

                # Save result
                result = {
                    "request_idx": request_idx,
                    "model": model_name,
                    "prompt_id": prompt.id,
                    "prompt_text": prompt.prompt,
                    "category": prompt.category,
                    "hypothesis": prompt.hypothesis,
                    "race_code": race_code,
                    "gender": img_data["gender"],
                    "age_code": age_code,
                    "source_image": img_data["path"],
                    "output_image": str(output_path) if output_path else None,
                    "success": edit_result.success,
                    "is_refused": refusal_result.is_refused,
                    "refusal_type": refusal_result.refusal_type,
                    "refusal_confidence": refusal_result.confidence,
                    "latency_ms": edit_result.latency_ms,
                    "error_message": edit_result.error_message,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

                pbar.update(1)

                # Checkpoint save
                if request_idx % config.save_interval == 0:
                    _save_results(results, exp_paths["results_file"])
                    logger.log_checkpoint(f"Saved at {request_idx}")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at request {request_idx}")
        print(f"To resume: --resume-from {request_idx}")

    finally:
        pbar.close()

        # Save final results
        _save_results(results, exp_paths["results_file"])

        # End experiment and get summary
        summary = logger.end_experiment()

        # Save summary
        with open(exp_paths["summary_file"], "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Results saved to {exp_paths['results_file']}")
        print(f"✓ Summary saved to {exp_paths['summary_file']}")
        print(f"✓ Logs saved to {exp_paths['logs_dir']}")

    return results


def _save_results(results: list, path: Path):
    """Save results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run I2I Refusal Bias Experiment")
    parser.add_argument("--model", type=str, required=True,
                       choices=["flux", "step1x", "qwen"],
                       help="Model to run")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--version", type=str, default="final",
                       help="Source image version (final, V1-V7)")
    parser.add_argument("--experiment-id", type=str, default=None,
                       help="Experiment ID (for resuming)")
    parser.add_argument("--resume-from", type=int, default=0,
                       help="Resume from this request index")

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        device=args.device,
        experiment_id=args.experiment_id,
        resume_from=args.resume_from,
        source_version=args.version
    )


if __name__ == "__main__":
    main()
