#!/bin/bash
#===============================================================================
# Experiment 2: Step1X-Edit Identity Preservation
#===============================================================================
# Model: Step1X-Edit (StepFun)
#
# Selected Prompts (from team annotation intersection analysis):
# - B01: CEO/Executive → Gender (f→m) + Race drift
# - B05: Housekeeper → Gender (m→f) reverse stereotype
# - B09: Fashion Model → Race + Identity change
# - D03: 30 years older → Over-aging + Race change
#
# Total Requests: 4 prompts × 84 images = 336
# Conditions: edited (with identity prompt), preserved (baseline)
#
# Usage:
#   bash scripts/experiment/exp2_preservation/run_step1x_preservation.sh
#   bash scripts/experiment/exp2_preservation/run_step1x_preservation.sh --edited
#   bash scripts/experiment/exp2_preservation/run_step1x_preservation.sh --preserved
#   bash scripts/experiment/exp2_preservation/run_step1x_preservation.sh --resume 100
#===============================================================================

set -e

# HuggingFace cache path (change to your path)
export HF_HOME="/scratch/chans/models/huggingface"
export HF_HUB_CACHE="/scratch/chans/models/huggingface/hub"

MODEL="step1x"
DEVICE="cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}"
echo "============================================================"
echo "  Experiment 2: Step1X-Edit Identity Preservation"
echo "============================================================"
echo -e "${NC}"

RUN_EDITED=true
RUN_PRESERVED=true
RESUME_FROM=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --edited)
            RUN_EDITED=true
            RUN_PRESERVED=false
            shift
            ;;
        --preserved)
            RUN_EDITED=false
            RUN_PRESERVED=true
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "  --edited       Only edited condition"
            echo "  --preserved    Only preserved condition"
            echo "  --resume N     Resume from task index N"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

check_prerequisites() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Python not found!${NC}"
        exit 1
    fi

    if [ "$DEVICE" = "cuda" ]; then
        if ! python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            echo -e "${YELLOW}CUDA not available, using CPU${NC}"
            DEVICE="cpu"
        else
            gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            echo -e "${GREEN}✓ CUDA: $gpu_name${NC}"
        fi
    fi

    if [ ! -d "$PROJECT_ROOT/data/source_images/final" ]; then
        echo -e "${RED}Source images not found!${NC}"
        exit 1
    fi
    echo "✓ Source images found"
    echo ""
}

check_prerequisites

EXPERIMENT_ID=$(date +"%Y%m%d_%H%M%S")
echo -e "${BLUE}Experiment ID: $EXPERIMENT_ID${NC}"
echo ""

if [ "$RUN_EDITED" = true ]; then
    echo -e "${GREEN}▶ Running EDITED condition${NC}"
    cd "$PROJECT_ROOT"
    python scripts/experiment/exp2_preservation/run_preservation_experiment.py \
        --model "$MODEL" --device "$DEVICE" --condition edited \
        --experiment-id "${EXPERIMENT_ID}_edited" --resume-from "$RESUME_FROM"
    echo ""
fi

if [ "$RUN_PRESERVED" = true ]; then
    echo -e "${GREEN}▶ Running PRESERVED condition${NC}"
    cd "$PROJECT_ROOT"
    python scripts/experiment/exp2_preservation/run_preservation_experiment.py \
        --model "$MODEL" --device "$DEVICE" --condition preserved \
        --experiment-id "${EXPERIMENT_ID}_preserved" --resume-from "$RESUME_FROM"
    echo ""
fi

echo -e "${GREEN}============================================================"
echo "  Step1X-Edit Complete!"
echo "============================================================${NC}"
echo "Output: $PROJECT_ROOT/data/results/exp2_pairwise/step1x/"
