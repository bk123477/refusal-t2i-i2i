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
# Total Requests: 54 prompts × 84 images = 4,536
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
#   bash scripts/experiment/run_qwen.sh                    # Interactive category selection
#   bash scripts/experiment/run_qwen.sh --all              # All categories (no prompt)
#   bash scripts/experiment/run_qwen.sh --categories A,B   # Run specific categories
#   bash scripts/experiment/run_qwen.sh --resume ID IDX    # Resume experiment
#============================================================

set -e

# Configuration
MODEL="qwen"
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
echo "  Qwen-Image-Edit-2511 Experiment Runner"
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

    # Check if QwenImageEditPlusPipeline is available
    python -c "from diffusers import QwenImageEditPlusPipeline" 2>/dev/null || {
        echo -e "${RED}ERROR: QwenImageEditPlusPipeline not found!${NC}"
        echo ""
        echo "Please install the latest diffusers:"
        echo "  pip install git+https://github.com/huggingface/diffusers"
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
        --experiment-id "$EXPERIMENT_ID" \
        --categories "$CATEGORIES"
fi

echo ""
echo -e "${GREEN}============================================================"
echo "  Experiment Complete!"
echo "============================================================${NC}"
