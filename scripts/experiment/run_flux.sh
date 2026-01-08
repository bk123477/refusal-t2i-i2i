#!/bin/bash
#============================================================
# FLUX.2-dev Experiment Runner
# I2I Refusal Bias Study - IJCAI 2026
#============================================================
# Model: FLUX.2-dev (Black Forest Labs)
# URL: https://huggingface.co/black-forest-labs/FLUX.2-dev
#
# Total Requests: 50 prompts × 84 images = 4,200
#
# Output Structure:
#   data/results/flux/{experiment_id}/
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
# Usage:
#   bash scripts/run_flux.sh
#   bash scripts/run_flux.sh --resume <experiment_id> <request_idx>
#============================================================

set -e

# Configuration
MODEL="flux"
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
echo "  FLUX.2-dev Experiment Runner"
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
    echo "Output: data/results/flux/$EXPERIMENT_ID/"
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
