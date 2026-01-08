#!/bin/bash
#============================================================
# Step1X-Edit-v1p2 Experiment Runner
# I2I Refusal Bias Study - IJCAI 2026
#============================================================
# Model: Step1X-Edit-v1p2 (StepFun)
# URL: https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2
#
# Features:
#   - Native reasoning edit
#   - Thinking mode + Reflection mode
#   - RegionE acceleration (optional)
#
# Total Requests: 50 prompts × 84 images = 4,200
#
# Output Structure:
#   data/results/step1x/{experiment_id}/
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
#   git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git
#   pip install -e diffusers
#   pip install RegionE  # optional
#
# Usage:
#   bash scripts/run_step1x.sh
#   bash scripts/run_step1x.sh --resume <experiment_id> <request_idx>
#============================================================

set -e

# Configuration
MODEL="step1x"
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
echo "  Step1X-Edit-v1p2 Experiment Runner"
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

    # Check if Step1X diffusers is installed
    python -c "from diffusers import Step1XEditPipelineV1P2" 2>/dev/null || {
        echo -e "${RED}ERROR: Step1XEditPipelineV1P2 not found!${NC}"
        echo ""
        echo "Please install the custom diffusers branch:"
        echo "  git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git"
        echo "  cd diffusers && pip install -e ."
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

    # Check RegionE (optional)
    python -c "from RegionE import RegionEHelper" 2>/dev/null && \
        echo -e "${GREEN}✓ RegionE found - faster inference enabled${NC}" || \
        echo -e "${YELLOW}⚠ RegionE not found - using standard inference${NC}"

    echo -e "${GREEN}✓ Prerequisites check passed${NC}"
    echo ""

    # Run experiment
    echo "Starting experiment..."
    echo "Output: data/results/step1x/$EXPERIMENT_ID/"
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
