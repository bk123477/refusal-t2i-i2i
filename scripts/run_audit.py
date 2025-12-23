#!/usr/bin/env python3
"""
ACRB Audit CLI
Professional entry point for running Attribute-Conditioned Refusal Bias audits.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path so we can import acrb
sys.path.append(str(Path(__file__).parent.parent))

from acrb.evaluation.pipeline import ACRBPipeline

def main():
    parser = argparse.ArgumentParser(description="ACRB: Attribute-Conditioned Refusal Bias Auditor")
    
    # Core settings
    parser.add_argument("--model", type=str, default="qwen-image-edit-2511", help="Model to audit")
    parser.add_argument("--mode", type=str, choices=["t2i", "i2i"], default="t2i", help="Audit mode")
    parser.add_argument("--samples", type=int, default=10, help="Number of base prompts to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # LLM Settings
    parser.add_argument("--llm", type=str, help="LLM for dynamic prompt expansion (e.g., gpt-oss-20b)")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="LLM API base URL")
    
    # I2I Specific
    parser.add_argument("--dataset", type=str, help="Directory for real-world images (FFHQ/COCO)")
    
    # Output
    parser.add_argument("--output", type=str, default="experiments/results", help="Directory to save results")
    parser.add_argument("--run-id", type=str, help="Custom run identifier")

    args = parser.parse_args()

    print("="*60)
    print("      🛡️  ACRB: Attribute-Conditioned Refusal Bias Audit      ")
    print("="*60)
    print(f"  Model:    {args.model}")
    print(f"  Mode:     {args.mode.upper()}")
    print(f"  Samples:  {args.samples}")
    print(f"  LLM:      {args.llm or 'None (Template-based)'}")
    print(f"  Dataset:  {args.dataset or 'SyntheticFallback'}")
    print("-" * 60)

    try:
        pipeline = ACRBPipeline(
            model_name=args.model,
            mode=args.mode,
            output_dir=args.output,
            seed=args.seed,
            run_id=args.run_id,
            llm_model=args.llm,
            llm_api_base=args.api_base,
            i2i_dataset_dir=args.dataset
        )
        
        evaluation = pipeline.run(max_base=args.samples)
        
        print("\n" + "="*60)
        print("                📊 AUDIT RESULTS SUMMARY                ")
        print("="*60)
        print(f"  Refusal Rate:    {evaluation.refusal_rate:.2%}")
        print(f"  Failure Rate:    {evaluation.failure_rate:.2%}")
        print(f"  Delta Refusal:   {evaluation.delta_refusal:.2%}")
        print(f"  Delta Erasure:   {evaluation.delta_erasure:.2%}")
        print("-" * 60)
        print(f"  Detailed results: {pipeline.output_dir}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error during audit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
