#!/usr/bin/env python3
"""
Main Experiment Runner for I2I Refusal Bias Study
Executes 50 prompts × 84 images = 4,200 requests per model with comprehensive logging

Usage:
    python scripts/experiment/run_experiment.py --model flux --device cuda
    python scripts/experiment/run_experiment.py --model step1x --device cuda
    python scripts/experiment/run_experiment.py --model qwen --device cuda
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
    Load final selected FairFace images with comprehensive metadata.

    Expected structure: {image_dir}/{RaceCode}/{RaceCode}_{Gender}_{AgeCode}.jpg
    Metadata includes selection logs and verification data.
    """
    metadata_path = image_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        images = metadata["images"]

        # Fix paths to use current image_dir instead of stored absolute paths
        for img in images:
            # Build correct path: image_dir / race_code / filename
            img["path"] = str(image_dir / img["race_code"] / img["filename"])

        print(f"✓ Loaded {len(images)} images from metadata.json")
        print(f"  Source versions: {set(img['source_version'] for img in images)}")
        print(f"  Expected total: {metadata.get('expected_total', 'N/A')}")

        # Validate completeness
        if len(images) != 84:
            print(f"⚠️ Warning: Expected 84 images, but loaded {len(images)}")
            print("   This may indicate incomplete finalization")

        return images
    else:
        # Fallback: scan folders with validation
        print("⚠️ metadata.json not found, scanning directories...")
        images = []
        race_codes = ["Black", "EastAsian", "Indian", "Latino", "MiddleEastern", "SoutheastAsian", "White"]

        for race_code in race_codes:
            race_dir = image_dir / race_code
            if not race_dir.exists():
                print(f"⚠️ Race directory missing: {race_code}")
                continue

            for img_path in sorted(race_dir.glob("*.jpg")):
                parts = img_path.stem.split("_")
                if len(parts) >= 3:
                    images.append({
                        "race_code": parts[0],
                        "gender": parts[1],
                        "age_code": parts[2],
                        "path": str(img_path),
                        "filename": img_path.name,
                        "source_version": "unknown",  # Fallback when no metadata
                        "final_path": str(img_path)
                    })

        print(f"✓ Scanned {len(images)} images from directories")
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
    source_version: str = "final",
    categories: list = None
):
    """
    Execute complete I2I experiment for a single model with comprehensive logging.

    Args:
        model_name: Model to test ("flux", "step1x", or "qwen")
        device: Compute device ("cuda", "cpu", etc.)
        experiment_id: Unique experiment identifier (auto-generated if None)
        resume_from: Request index to resume from (for interrupted experiments)
        source_version: Source image version ("final" for selected images, or "V1-V7")
        categories: List of categories to filter prompts (None for all categories)
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
        source_version=source_version,
        categories_filter=",".join(categories) if categories else None,
        **model_config
    )
    config.save(exp_paths["experiment_dir"] / "config.json")

    print(f"\n{'='*70}")
    print(f"I2I REFUSAL BIAS EXPERIMENT - {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Source Version: {source_version}")
    print(f"Categories: {', '.join(categories) if categories else 'All (A-E)'}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Output Directory: {exp_paths['experiment_dir']}")
    print(f"Resume From: {resume_from}" if resume_from > 0 else "Resume From: Beginning")
    print(f"{'='*70}\n")

    # Load and validate data first
    print("📋 Loading experiment data...")

    # Load prompts
    try:
        prompts = PromptLoader(str(path_config.prompts_file))
        print(f"✓ Loaded {len(prompts)} prompts")

        # Filter by categories if specified
        if categories:
            prompts = [p for p in prompts if p.category in categories]
            print(f"✓ Filtered to {len(prompts)} prompts (categories: {', '.join(categories)})")
    except Exception as e:
        print(f"❌ Failed to load prompts: {e}")
        return []

    # Load images
    try:
        images = load_source_images(path_config.source_images_dir)
        if len(images) != 84:
            print(f"⚠️ Warning: Expected 84 images, got {len(images)}")
            if len(images) == 0:
                print("❌ No images found! Check source_images directory")
                return []
        else:
            print(f"✓ Loaded {len(images)} source images")
    except Exception as e:
        print(f"❌ Failed to load images: {e}")
        return []

    # Initialize refusal detector
    try:
        refusal_detector = RefusalDetector()
        print("✓ Initialized refusal detection")
    except Exception as e:
        print(f"❌ Failed to initialize refusal detector: {e}")
        return []

    total_requests = len(prompts) * len(images)
    print(f"📊 Total requests: {total_requests} ({len(prompts)} prompts × {len(images)} images)")
    print()

    # Load model
    print(f"🤖 Loading {model_name} model...")
    try:
        model = get_model(model_name, device, model_config)
        model.load()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return []

    print()

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
        print(f"\n\n⏸️ Experiment interrupted at request {request_idx}")
        print(f"📊 Progress saved. To resume: --resume-from {request_idx}")
        print(f"💾 Current results: {len(results)} requests completed")

    except Exception as e:
        print(f"\n\n❌ Unexpected error at request {request_idx}: {e}")
        print("🔍 Check logs for details")

    finally:
        pbar.close()

        # Save results even on failure
        if results:
            try:
                _save_results(results, exp_paths["results_file"])
                print(f"\n✓ Results saved: {len(results)} requests to {exp_paths['results_file']}")
            except Exception as e:
                print(f"❌ Failed to save results: {e}")

        # Generate and save summary
        try:
            summary = logger.end_experiment()
            with open(exp_paths["summary_file"], "w") as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Summary saved to {exp_paths['summary_file']}")

            # Print key metrics
            if summary.get('total_requests', 0) > 0:
                success_rate = summary.get('success_rate', 0)
                refusal_rate = summary.get('refusal_rate', 0)
                print(f"📈 Success Rate: {success_rate:.1f}%")
                print(f"🚫 Refusal Rate: {refusal_rate:.1f}%")

        except Exception as e:
            print(f"❌ Failed to generate summary: {e}")

        print(f"📁 Logs directory: {exp_paths['logs_dir']}")
        print(f"🖼️ Images directory: {exp_paths['images_dir']}")

    return results


def _save_results(results: list, path: Path):
    """Save results to JSON with validation and backup."""
    if not results:
        print("⚠️ No results to save")
        return

    try:
        # Create backup if file exists
        if path.exists():
            backup_path = path.with_suffix('.backup.json')
            path.rename(backup_path)
            print(f"💾 Created backup: {backup_path.name}")

        # Save new results
        with open(path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(results)} results to {path.name}")

    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        # Restore backup if save failed
        backup_path = path.with_suffix('.backup.json')
        if backup_path.exists():
            backup_path.rename(path)
            print("🔄 Restored backup file")


def main():
    parser = argparse.ArgumentParser(
        description="Execute I2I Refusal Bias Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FLUX experiment
  python scripts/experiment/run_experiment.py --model flux --device cuda

  # Run with custom experiment ID
  python scripts/experiment/run_experiment.py --model step1x --experiment-id test_001

  # Resume interrupted experiment
  python scripts/experiment/run_experiment.py --model qwen --resume-from 1000

  # Use specific source version
  python scripts/experiment/run_experiment.py --model flux --version V7
        """
    )

    parser.add_argument("--model", type=str, required=True,
                       choices=["flux", "step1x", "qwen"],
                       help="I2I model to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Compute device (cuda/cpu/mps)")
    parser.add_argument("--version", type=str, default="final",
                       help="Source image version (final=selected images, V1-V7=versions)")
    parser.add_argument("--experiment-id", type=str, default=None,
                       help="Unique experiment identifier (auto-generated if not provided)")
    parser.add_argument("--resume-from", type=int, default=0,
                       help="Resume from specific request index (for interrupted experiments)")
    parser.add_argument("--categories", type=str, default=None,
                       help="Filter by prompt categories (comma-separated, e.g., 'A,B,C' or 'E')")

    args = parser.parse_args()

    # Validate arguments
    if args.resume_from < 0:
        parser.error("--resume-from must be non-negative")

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip().upper() for c in args.categories.split(",")]
        valid_categories = {"A", "B", "C", "D", "E"}
        invalid = set(categories) - valid_categories
        if invalid:
            parser.error(f"Invalid categories: {invalid}. Valid: A, B, C, D, E")

    print(f"🚀 Starting I2I Refusal Bias Experiment")
    print(f"📋 Model: {args.model}")
    print(f"💻 Device: {args.device}")
    print(f"🖼️ Source: {args.version}")
    if categories:
        print(f"📁 Categories: {', '.join(categories)}")
    print(f"{'-'*50}")

    try:
        results = run_experiment(
            model_name=args.model,
            device=args.device,
            experiment_id=args.experiment_id,
            resume_from=args.resume_from,
            source_version=args.version,
            categories=categories
        )

        if results:
            print(f"\n✅ Experiment completed successfully!")
            print(f"📊 Total results: {len(results)}")
        else:
            print(f"\n❌ Experiment failed or produced no results")
            exit(1)

    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
