#!/bin/bash
#============================================================
# Qwen-Image-Edit-2511 Experiment Runner
# I2I Refusal Bias Study - IJCAI 2026
#============================================================
# Model: Qwen-Image-Edit-2511 (Alibaba)
# URL: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
#
# Features:
#   - Improved character consistency
#   - Mitigated image drift
#   - Integrated LoRA capabilities
#   - Multi-image input support
#
# Total Requests: 50 prompts × 84 images = 4,200
#
# Output Structure:
#   data/results/qwen/{experiment_id}/
#   ├── images/           # Generated images
#   ├── logs/             # Experiment logs
#   │   ├── experiment.log
#   │   ├── refusals.jsonl
#   │   ├── errors.jsonl
#   │   └── timings.jsonl
#   ├── config.json       # Experiment config
#   ├── results.json      # Full results
#   └── summary.json      # Summary statistics
#
# Prerequisites:
#   pip install git+https://github.com/huggingface/diffusers
#
# Usage:
#   bash scripts/run_qwen.sh
#   bash scripts/run_qwen.sh --resume <experiment_id> <request_idx>
#============================================================

set -e

# Configuration
MODEL="qwen"
DEVICE="cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "============================================================"
echo "  Qwen-Image-Edit-2511 Experiment Runner"
echo "  I2I Refusal Bias Study"
echo "============================================================"
echo -e "${NC}"

# Check if resuming
if [ "$1" == "--resume" ]; then
    EXPERIMENT_ID="$2"
    RESUME_FROM="$3"
    echo -e "${YELLOW}Resuming experiment: ${EXPERIMENT_ID} from request ${RESUME_FROM}${NC}"

    cd "$PROJECT_ROOT"
    python scripts/experiment/run_experiment.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --experiment-id "$EXPERIMENT_ID" \
        --resume-from "$RESUME_FROM"
else
    # New experiment
    EXPERIMENT_ID=$(date +"%Y%m%d_%H%M%S")

    echo "Model: $MODEL"
    echo "Device: $DEVICE"
    echo "Experiment ID: $EXPERIMENT_ID"
    echo ""

    # Check prerequisites
    echo "Checking prerequisites..."

    # Check if QwenImageEditPlusPipeline is available
    python -c "from diffusers import QwenImageEditPlusPipeline" 2>/dev/null || {
        echo -e "${RED}ERROR: QwenImageEditPlusPipeline not found!${NC}"
        echo ""
        echo "Please install the latest diffusers:"
        echo "  pip install git+https://github.com/huggingface/diffusers"
        exit 1
    }

    # Check if source images exist
    if [ ! -d "$PROJECT_ROOT/data/source_images/fairface" ] || [ -z "$(ls -A $PROJECT_ROOT/data/source_images/fairface 2>/dev/null)" ]; then
        echo -e "${RED}ERROR: Source images not found!${NC}"
        echo "Please run: python scripts/sample_fairface.py"
        exit 1
    fi

    # Check if prompts exist
    if [ ! -f "$PROJECT_ROOT/data/prompts/i2i_prompts.json" ]; then
        echo -e "${RED}ERROR: Prompts file not found!${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Prerequisites check passed${NC}"
    echo ""

    # Run experiment
    echo "Starting experiment..."
    echo "Output: data/results/qwen/$EXPERIMENT_ID/"
    echo ""

    cd "$PROJECT_ROOT"
    python scripts/experiment/run_experiment.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --experiment-id "$EXPERIMENT_ID"
fi

echo ""
echo -e "${GREEN}============================================================"
echo "  Experiment Complete!"
echo "============================================================${NC}"
