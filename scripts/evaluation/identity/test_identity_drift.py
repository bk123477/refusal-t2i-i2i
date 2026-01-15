#!/usr/bin/env python3
"""
Test script for Identity Drift Detection in VLM Evaluator

Usage:
    python scripts/evaluation/test_identity_drift.py --source <source_image> --edited <edited_image>
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.vlm_evaluator import VLMEvaluator


def main():
    parser = argparse.ArgumentParser(description="Test Identity Drift Detection")
    parser.add_argument("--source", type=str, required=True,
                       help="Path to source image")
    parser.add_argument("--edited", type=str, required=True,
                       help="Path to edited image")
    parser.add_argument("--race", type=str, default="Black",
                       help="Source person's race (e.g., 'Black', 'White', 'Asian')")
    parser.add_argument("--gender", type=str, default="Male",
                       help="Source person's gender ('Male' or 'Female')")
    parser.add_argument("--prompt", type=str, default="Make this person look professional",
                       help="Edit instruction that was used")
    parser.add_argument("--category", type=str, default=None,
                       help="Prompt category (A, B, C, D, E)")
    parser.add_argument("--qwen-model", type=str, default="30B",
                       choices=["30B", "8B"],
                       help="Qwen model size to use")
    parser.add_argument("--no-ensemble", action="store_true",
                       help="Use only Qwen (no Gemini ensemble)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed VLM responses")

    args = parser.parse_args()

    # Load images
    print(f"Loading images...")
    source_image = Image.open(args.source)
    edited_image = Image.open(args.edited)

    # Initialize evaluator
    print(f"Initializing VLM Evaluator (Qwen {args.qwen_model})...")
    evaluator = VLMEvaluator(
        use_ensemble=not args.no_ensemble,
        qwen_model_size=args.qwen_model
    )

    # Evaluate identity preservation
    print(f"\nEvaluating identity preservation...")
    print(f"  Source race: {args.race}")
    print(f"  Source gender: {args.gender}")
    print(f"  Edit prompt: {args.prompt}")
    print()

    result = evaluator.evaluate_identity_preservation(
        source_image=source_image,
        edited_image=edited_image,
        source_race=args.race,
        source_gender=args.gender,
        prompt=args.prompt,
        return_detailed=args.detailed
    )

    # Display results
    print("="*60)
    print("IDENTITY DRIFT RESULTS")
    print("="*60)
    print(f"Racial Drift: {result['racial_drift']}")
    print(f"Gender Drift: {result['gender_drift']}")
    print(f"Body Transformation: {result['body_transform']}")
    print(f"Identity Preserved: {result['identity_preserved']}")
    print(f"\nDetails: {result['drift_details']}")

    # Check for stereotype replacement
    is_replacement = evaluator.is_stereotype_replacement(
        drift_result=result,
        prompt=args.prompt,
        source_race=args.race,
        source_gender=args.gender,
        category=args.category
    )

    print(f"\nStereotype Replacement Detected: {is_replacement}")

    if is_replacement:
        print("\n⚠️  WARNING: This edit shows evidence of stereotype replacement!")
        print("   The model may have changed the person's identity to match")
        print("   stereotypical associations with the edit instruction.")

    # Show detailed analysis if requested
    if args.detailed and "detailed_analysis" in result:
        print("\n" + "="*60)
        print("DETAILED VLM ANALYSIS")
        print("="*60)

        for analysis_type, analysis_data in result["detailed_analysis"].items():
            print(f"\n{analysis_type.upper()}:")
            print(f"  Result: {analysis_data['result']}")
            print(f"  Consensus: {analysis_data['consensus']}")
            print(f"  Qwen response: {analysis_data['raw_responses']['qwen']}")
            if analysis_data['raw_responses']['gemini']:
                print(f"  Gemini response: {analysis_data['raw_responses']['gemini']}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
