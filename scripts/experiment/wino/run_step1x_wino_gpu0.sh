#!/bin/bash
#
# Run Step1X WinoBias experiment on GPU 0
# Prompts: 1-25
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Configuration
MODEL="step1x"
PROMPTS_FILE="$PROJECT_ROOT/data/prompts/winobias_prompts.json"
BASE_DIR="$PROJECT_ROOT/data/source_images/final"
OUTPUT_DIR="$PROJECT_ROOT/data/results/experiments/winobias_step1x/gpu0"

# Prompt range for GPU 0
START_ID=1
END_ID=25

# Inference parameters
STEPS=50
SEED=42

echo "========================================================================"
echo "Step1X WinoBias Experiment - GPU 0"
echo "========================================================================"
echo "Model: $MODEL"
echo "Prompts: $START_ID to $END_ID (25 prompts)"
echo "Output: $OUTPUT_DIR"
echo "Device: GPU 0 (CUDA_VISIBLE_DEVICES=0)"
echo "========================================================================"
echo ""

# Run experiment on GPU 0
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 "$SCRIPT_DIR/run_winobias_experiment.py" \
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
echo "GPU 0 Experiment Complete!"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================================"
