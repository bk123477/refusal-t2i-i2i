#!/bin/bash
#===============================================================================
# Experiment 2-2: Step1X Identity Preservation (Remaining Prompts)
#===============================================================================
# Model: Step1X-Edit
#
# Remaining Prompts (excluding B01, B05, B09, D03 from exp2-1):
# 16 prompts total (7 B + 9 D)
#
# Total Requests: 16 prompts × 75 sampled sources (step1x-specific) = 1,200
#
# Usage:
#   bash scripts/experiment/exp2_preservation/dexp2-2_all\(exclude_B01,B05,B09,D03\)/run_step1x_preservation.sh
#   bash ... --resume 100
#===============================================================================

set -e

MODEL="step1x"
DEVICE="cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}"
echo "============================================================"
echo "  Experiment 2-2: Step1X Identity Preservation"
echo "  (Remaining 16 Prompts - excluding B01,B05,B09,D03)"
echo "============================================================"
echo -e "${NC}"

RESUME_FROM=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume|--resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
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

echo -e "${GREEN}▶ Running PRESERVED condition (with identity prompts)${NC}"
echo "Prompts: B02,B03,B04,B06,B07,B08,B10,D01,D02,D04,D05,D06,D07,D08,D09,D10"
echo ""
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/run_preservation_experiment.py" \
    --model "$MODEL" --device "$DEVICE" \
    --experiment-id "${EXPERIMENT_ID}" --resume-from "$RESUME_FROM"
echo ""

echo -e "${GREEN}============================================================"
echo "  Step1X Exp2-2 Complete!"
echo "============================================================${NC}"
echo "Output: $PROJECT_ROOT/data/results/exp2_pairwise/step1x/preserved/"
