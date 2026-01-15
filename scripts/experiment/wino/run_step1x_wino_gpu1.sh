#!/bin/bash
#
# Run Step1X WinoBias experiment on GPU 1
# Prompts: 26-50
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Configuration
MODEL="step1x"
PROMPTS_FILE="$PROJECT_ROOT/data/prompts/winobias_prompts.json"
BASE_DIR="$PROJECT_ROOT/data/source_images/final"
OUTPUT_DIR="$PROJECT_ROOT/data/results/experiments/winobias_step1x/gpu1"

# Prompt range for GPU 1
START_ID=26
END_ID=50

# Inference parameters
STEPS=50
SEED=42

echo "========================================================================"
echo "Step1X WinoBias Experiment - GPU 1"
echo "========================================================================"
echo "Model: $MODEL"
echo "Prompts: $START_ID to $END_ID (25 prompts)"
echo "Output: $OUTPUT_DIR"
echo "Device: GPU 1 (CUDA_VISIBLE_DEVICES=1)"
echo "========================================================================"
echo ""

# Run experiment on GPU 1
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 "$SCRIPT_DIR/run_winobias_experiment.py" \
    --model "$MODEL" \
    --prompts-file "$PROMPTS_FILE" \
    --base-dir "$BASE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --start-id "$START_ID" \
    --end-id "$END_ID" \
    --steps "$STEPS" \
    --seed "$SEED"

echo ""
echo "========================================================================"
echo "GPU 1 Experiment Complete!"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================================"
