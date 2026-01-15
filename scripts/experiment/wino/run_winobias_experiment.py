#!/usr/bin/env python3
"""
Run WinoBias Gender Stereotype Experiment with Multi-Image I2I Models

Usage:
  python scripts/experiment/run_winobias_experiment.py \
    --model flux2 \
    --output-dir data/results/winobias/flux2 \
    --start-id 1 --end-id 50
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import time
from PIL import Image
import logging

# Set Hugging Face cache directory (for server environment)
os.environ["HF_HOME"] = "/scratch/chans/models/huggingface"
os.environ["HF_HUB_CACHE"] = "/scratch/chans/models/huggingface/hub"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.flux_wrapper import FluxWrapper
from src.models.qwen_wrapper import QwenImageEditWrapper
from src.models.step1x_wrapper import Step1XWrapper


def setup_logging(output_dir: Path, model_name: str) -> logging.Logger:
    """Setup logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"experiment_{model_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger('winobias_experiment')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def image_filename_to_path(filename: str, base_dir: Path) -> Path:
    """Convert image filename to full path."""
    # filename format: Race_Gender_Age.jpg (e.g., Black_Male_40s.jpg)
    parts = filename.replace('.jpg', '').split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid image filename: {filename}")
    race = parts[0]
    return base_dir / race / filename


def load_prompts(prompts_file: Path) -> list:
    """Load WinoBias prompts from JSON."""
    with open(prompts_file, 'r') as f:
        return json.load(f)


def run_experiment(args):
    """Run WinoBias experiment with specified model."""
    
    # Setup output directory first
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.model)
    
    # Log experiment start
    logger.info("="*70)
    logger.info("WinoBias Gender Stereotype Experiment")
    logger.info("="*70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Prompts File: {args.prompts_file}")
    logger.info(f"Base Directory: {args.base_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Prompt Range: {args.start_id} to {args.end_id}")
    logger.info(f"Inference Steps: {args.steps}")
    logger.info(f"Random Seed: {args.seed}")
    
    # Load prompts (new format: list of dicts with id, prompt, input_image_1, input_image_2)
    logger.info("\nLoading prompts...")
    try:
        prompts = load_prompts(args.prompts_file)
        logger.info(f"✓ Loaded {len(prompts)} prompts from {args.prompts_file.name}")
    except Exception as e:
        logger.error(f"✗ Failed to load prompts: {e}")
        return
    
    # Filter prompts by ID range
    prompts = [p for p in prompts if args.start_id <= p['id'] <= args.end_id]
    logger.info(f"✓ Filtered to {len(prompts)} prompts (ID {args.start_id}-{args.end_id})")
    logger.info("="*70 + "\n")
    
    # Model-specific configuration
    if args.model == "flux2":
        logger.info("Model Configuration:")
        logger.info(f"  - Quantized: {args.quantized}")
        logger.info(f"  - Remote Text Encoder: {args.remote_encoder}")
        model = FluxWrapper(
            device=args.device,
            use_quantized=args.quantized,
            use_remote_text_encoder=args.remote_encoder
        )
    elif args.model == "qwen":
        logger.info("Model Configuration: Qwen-Image-Edit-2511")
        model = QwenImageEditWrapper(device=args.device)
    elif args.model == "step1x":
        logger.info("Model Configuration:")
        logger.info(f"  - Thinking Mode: {args.thinking}")
        logger.info(f"  - Reflection Mode: {args.reflection}")
        model = Step1XWrapper(
            device=args.device,
            enable_thinking=args.thinking,
            enable_reflection=args.reflection
        )
    else:
        logger.error(f"Unknown model: {args.model}")
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load model
    logger.info("\nLoading model...")
    model_load_start = time.time()
    try:
        model.load()
        model_load_time = time.time() - model_load_start
        logger.info(f"✓ Model loaded successfully ({model_load_time:.2f}s)\n")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run experiments
    logger.info("="*70)
    logger.info("Starting Experiment Loop")
    logger.info("="*70)
    
    results = []
    success_count = 0
    error_count = 0
    total_inference_time = 0
    
    for idx, prompt_entry in enumerate(prompts, 1):
        prompt_id = prompt_entry['id']
        prompt_text = prompt_entry['prompt']
        image_1_filename = prompt_entry['input_image_1']
        image_2_filename = prompt_entry['input_image_2']
        
        logger.info(f"\n{'─'*70}")
        logger.info(f"[{idx}/{len(prompts)}] Prompt ID: {prompt_id}")
        logger.info(f"{'─'*70}")
        logger.info(f"Prompt: {prompt_text}")
        logger.info(f"Input Image 1: {image_1_filename}")
        logger.info(f"Input Image 2: {image_2_filename}")
        
        try:
            # Load source images
            logger.debug(f"Resolving image paths...")
            img_a_path = image_filename_to_path(image_1_filename, args.base_dir)
            img_b_path = image_filename_to_path(image_2_filename, args.base_dir)
            logger.debug(f"  Image 1 path: {img_a_path}")
            logger.debug(f"  Image 2 path: {img_b_path}")
            
            if not img_a_path.exists():
                raise FileNotFoundError(f"Image 1 not found: {img_a_path}")
            if not img_b_path.exists():
                raise FileNotFoundError(f"Image 2 not found: {img_b_path}")
            
            logger.info("Loading images...")
            img_a = Image.open(img_a_path)
            img_b = Image.open(img_b_path)
            logger.info(f"✓ Images loaded: {img_a.size}, {img_b.size}")
            
            # Run inference
            logger.info(f"Generating image with {args.model}...")
            start_time = time.time()
            
            result = model.edit(
                source_image=[img_a, img_b],
                prompt=prompt_text,
                seed=args.seed,
                num_inference_steps=args.steps,
            )
            
            elapsed = time.time() - start_time
            total_inference_time += elapsed
            
            if result.success:
                # Save output image
                output_filename = f"prompt_{prompt_id:03d}_{timestamp}.png"
                output_path = output_dir / output_filename
                
                logger.info("Saving output image...")
                result.output_image.save(output_path)
                
                logger.info(f"✓ SUCCESS")
                logger.info(f"  Inference Time: {elapsed:.2f}s")
                logger.info(f"  Model Latency: {result.latency_ms:.0f}ms")
                logger.info(f"  Output File: {output_filename}")
                logger.info(f"  Full Path: {output_path}")
                
                results.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "input_image_1": image_1_filename,
                    "input_image_2": image_2_filename,
                    "success": True,
                    "output_file": output_filename,
                    "output_path": str(output_path),
                    "latency_ms": result.latency_ms,
                    "elapsed_sec": elapsed,
                    "timestamp": datetime.now().isoformat()
                })
                success_count += 1
            else:
                logger.warning(f"✗ GENERATION FAILED")
                logger.warning(f"  Refusal Type: {result.refusal_type.value}")
                logger.warning(f"  Error: {result.error_message}")
                logger.warning(f"  Latency: {result.latency_ms:.0f}ms")
                
                results.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "input_image_1": image_1_filename,
                    "input_image_2": image_2_filename,
                    "success": False,
                    "error": result.error_message,
                    "refusal_type": result.refusal_type.value,
                    "latency_ms": result.latency_ms,
                    "timestamp": datetime.now().isoformat()
                })
                error_count += 1
        
        except Exception as e:
            logger.error(f"✗ EXCEPTION OCCURRED")
            logger.error(f"  Exception Type: {type(e).__name__}")
            logger.error(f"  Exception Message: {str(e)}")
            import traceback
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            
            results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "input_image_1": image_1_filename,
                "input_image_2": image_2_filename,
                "success": False,
                "error": str(e),
                "exception_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            })
            error_count += 1
    
    # Calculate statistics
    avg_inference_time = total_inference_time / len(prompts) if prompts else 0
    success_rate = (success_count / len(prompts) * 100) if prompts else 0
    
    # Save results
    results_file = output_dir / f"results_{args.model}_{timestamp}.json"
    logger.info(f"\nSaving results to {results_file}...")
    
    results_data = {
        "experiment": {
            "model": args.model,
            "start_time": timestamp,
            "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_prompts": len(prompts),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate_percent": round(success_rate, 2),
            "total_inference_time_sec": round(total_inference_time, 2),
            "avg_inference_time_sec": round(avg_inference_time, 2)
        },
        "config": {
            "device": args.device,
            "steps": args.steps,
            "seed": args.seed,
            "prompts_file": str(args.prompts_file),
            "base_dir": str(args.base_dir),
            "output_dir": str(output_dir),
            "prompt_range": f"{args.start_id}-{args.end_id}",
            "quantized": args.quantized if args.model == "flux2" else None,
            "remote_encoder": args.remote_encoder if args.model == "flux2" else None,
            "thinking": args.thinking if args.model == "step1x" else None,
            "reflection": args.reflection if args.model == "step1x" else None,
        },
        "results": results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Results saved successfully")
    
    # Print final summary
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Total Prompts: {len(prompts)}")
    logger.info(f"Successful: {success_count} ({success_rate:.1f}%)")
    logger.info(f"Failed: {error_count}")
    logger.info(f"Total Inference Time: {total_inference_time:.2f}s")
    logger.info(f"Average Time per Prompt: {avg_inference_time:.2f}s")
    logger.info(f"Results File: {results_file}")
    logger.info(f"Log File: {[h.baseFilename for h in logger.handlers if isinstance(h, logging.FileHandler)][0]}")
    logger.info(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run WinoBias experiment with multi-image I2I models"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        required=True,
        choices=["flux2", "qwen", "step1x"],
        help="Model to use for experiment"
    )
    
    # Input/Output
    parser.add_argument(
        "--prompts-file",
        default="data/prompts/winobias_prompts.json",
        help="Path to WinoBias prompts JSON"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/source_images/final"),
        help="Base directory for identity images"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for results"
    )
    
    # Experiment parameters
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="First prompt ID to run"
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=50,
        help="Last prompt ID to run"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run model on"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Flux2-specific options
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use 4-bit quantized Flux2 model"
    )
    parser.add_argument(
        "--remote-encoder",
        action="store_true",
        help="Use remote text encoder for Flux2"
    )
    
    # Step1X-specific options
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode for Step1X"
    )
    parser.add_argument(
        "--reflection",
        action="store_true",
        help="Enable reflection mode for Step1X"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    args.prompts_file = Path(args.prompts_file).resolve()
    args.base_dir = args.base_dir.resolve()
    args.output_dir = Path(args.output_dir).resolve()
    
    # Check files exist
    if not args.prompts_file.exists():
        print(f"Error: Prompts file not found: {args.prompts_file}")
        sys.exit(1)
    
    if not args.base_dir.exists():
        print(f"Error: Base directory not found: {args.base_dir}")
        sys.exit(1)
    
    run_experiment(args)


if __name__ == "__main__":
    main()
