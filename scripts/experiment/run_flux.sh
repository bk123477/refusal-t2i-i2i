#!/bin/bash
#===============================================================================
# FLUX.2-dev Experiment Runner - I2I Refusal Bias Study
#===============================================================================
# Model: FLUX.2-dev (Black Forest Labs)
# URL: https://huggingface.co/black-forest-labs/FLUX.2-dev
# License: Apache 2.0
#
# Total Requests: 50 prompts x 84 images = 4,200
# Expected Duration: ~2-3 hours (on A100 GPU)
#
# Output Structure:
#   data/results/flux/{experiment_id}/
#   ├── images/           # Generated images by race/gender/age
#   ├── logs/             # Comprehensive experiment logs
#   │   ├── experiment.log    # Main experiment log
#   │   ├── refusals.jsonl    # Refusal detection results
#   │   ├── errors.jsonl      # Error tracking
#   │   └── timings.jsonl     # Performance metrics
#   ├── config.json       # Complete experiment configuration
#   ├── results.json      # Full results dataset
#   ├── results.backup.json   # Automatic backup
#   └── summary.json      # Statistical summary & metrics
#
# Usage:
#   bash scripts/experiment/run_flux.sh                    # New experiment
#   bash scripts/experiment/run_flux.sh --resume ID IDX   # Resume experiment
#===============================================================================

set -e

# Configuration
MODEL="flux"
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
echo "  FLUX.2-dev Experiment Runner"
echo "  I2I Refusal Bias Study"
echo "============================================================"
echo -e "${NC}"

# Function to check system prerequisites
check_prerequisites() {
    echo "Checking system prerequisites..."

    # Check Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Python not found!${NC}"
        exit 1
    fi

    # Check CUDA (if using GPU)
    if [ "$DEVICE" = "cuda" ]; then
        if ! python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            echo -e "${YELLOW}CUDA not available, falling back to CPU${NC}"
            DEVICE="cpu"
        else
            echo "CUDA available"
        fi
    fi

    # Check source images
    if [ ! -d "$PROJECT_ROOT/data/source_images/final" ]; then
        echo -e "${RED}Final source images not found!${NC}"
        echo "Please run image selector and finalize selection first:"
        echo "  python tools/image_selector/app.py"
        exit 1
    fi

    # Validate final images
    image_count=$(find "$PROJECT_ROOT/data/source_images/final" -name "*.jpg" | wc -l)
    if [ "$image_count" -ne 84 ]; then
        echo -e "${RED}Invalid image count: $image_count (expected: 84)${NC}"
        echo "Please complete image selection in the web interface"
        exit 1
    fi
    echo "Found $image_count final images"

    # Check prompts
    if [ ! -f "$PROJECT_ROOT/data/prompts/i2i_prompts.json" ]; then
        echo -e "${RED}Prompts file not found!${NC}"
        echo "Please ensure prompts are available"
        exit 1
    fi

    # Check available disk space (rough estimate: 4K images x ~500KB = ~2GB)
    available_space=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 2000000 ]; then  # 2GB in KB
        echo -e "${YELLOW}Low disk space detected${NC}"
    fi

    echo -e "${GREEN}All prerequisites verified${NC}"
    echo ""
}

# Main execution logic
if [ "$1" == "--resume" ]; then
    if [ $# -ne 3 ]; then
        echo -e "${RED}Usage: $0 --resume <experiment_id> <request_index>${NC}"
        exit 1
    fi

    EXPERIMENT_ID="$2"
    RESUME_FROM="$3"

    echo -e "${YELLOW}Resuming FLUX experiment${NC}"
    echo "   Experiment ID: $EXPERIMENT_ID"
    echo "   Resume from: request $RESUME_FROM"
    echo ""

    cd "$PROJECT_ROOT"
    python scripts/experiment/run_experiment.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --experiment-id "$EXPERIMENT_ID" \
        --resume-from "$RESUME_FROM"

else
    # New experiment
    EXPERIMENT_ID=$(date +"%Y%m%d_%H%M%S")

    echo -e "${GREEN}Starting New FLUX Experiment${NC}"
    echo "   Model: $MODEL"
    echo "   Device: $DEVICE"
    echo "   Experiment ID: $EXPERIMENT_ID"
    echo "   Estimated requests: 4,200"
    echo ""

    # Run prerequisite checks
    check_prerequisites

    # Create output directory preview
    output_dir="$PROJECT_ROOT/data/results/flux/$EXPERIMENT_ID"
    echo "Output directory: $output_dir"
    echo ""

    # Confirm before starting
    echo "Press Enter to start experiment, or Ctrl+C to cancel..."
    read -r || exit 1

    echo ""
    echo -e "${GREEN}Starting FLUX experiment...${NC}"

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
