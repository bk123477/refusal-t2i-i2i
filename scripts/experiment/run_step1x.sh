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
# Total Requests: 54 prompts × 84 images = 4,536
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
#   bash scripts/experiment/run_step1x.sh                    # Interactive category selection
#   bash scripts/experiment/run_step1x.sh --all              # All categories (no prompt)
#   bash scripts/experiment/run_step1x.sh --categories A,B   # Run specific categories
#   bash scripts/experiment/run_step1x.sh --resume ID IDX    # Resume experiment
#============================================================

set -e

# Configuration
MODEL="step1x"
DEVICE="cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

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

# Parse arguments
CATEGORIES=""
RESUME_MODE=false
EXPERIMENT_ID=""
RESUME_FROM=""
INTERACTIVE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_MODE=true
            EXPERIMENT_ID="$2"
            RESUME_FROM="$3"
            shift 3
            ;;
        --categories)
            CATEGORIES="$2"
            INTERACTIVE=false
            shift 2
            ;;
        --all)
            CATEGORIES="A,B,C,D,E"
            INTERACTIVE=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (no options)           Interactive category selection"
            echo "  --all                  Run all categories (A-E)"
            echo "  --categories A,B,C     Run specific categories"
            echo "  --resume <id> <idx>    Resume experiment"
            echo ""
            echo "Categories: A(Neutral), B(Occupational), C(Cultural), D(Vulnerability), E(Harmful)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution logic
if [ "$RESUME_MODE" = true ]; then
    if [ -z "$EXPERIMENT_ID" ] || [ -z "$RESUME_FROM" ]; then
        echo -e "${RED}Usage: $0 --resume <experiment_id> <request_index>${NC}"
        exit 1
    fi

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

    # Interactive category selection if no categories specified
    if [ "$INTERACTIVE" = true ] && [ -z "$CATEGORIES" ]; then
        echo -e "${YELLOW}No categories specified. Starting interactive selection...${NC}"
        echo ""
        CATEGORIES=$(python "$SCRIPT_DIR/category_selector.py" --quick) || {
            echo -e "${RED}Category selection cancelled${NC}"
            exit 1
        }
    fi

    # Default to all if still empty
    if [ -z "$CATEGORIES" ]; then
        CATEGORIES="A,B,C,D,E"
    fi

    echo "Model: $MODEL"
    echo "Device: $DEVICE"
    echo "Experiment ID: $EXPERIMENT_ID"
    echo "Categories: $CATEGORIES"
    num_cats=$(echo "$CATEGORIES" | tr ',' '\n' | wc -l | tr -d ' ')
    estimated=$((num_cats * 10 * 84))
    echo "Estimated requests: $estimated"
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
    if [ ! -d "$PROJECT_ROOT/data/source_images/final" ]; then
        echo -e "${RED}ERROR: Source images not found!${NC}"
        echo "Please complete image selection first"
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
        --experiment-id "$EXPERIMENT_ID" \
        --categories "$CATEGORIES"
fi

echo ""
echo -e "${GREEN}============================================================"
echo "  Experiment Complete!"
echo "============================================================${NC}"
