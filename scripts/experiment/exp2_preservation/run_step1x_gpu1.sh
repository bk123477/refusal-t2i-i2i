#!/bin/bash
#===============================================================================
# Experiment 2: Step1X-Edit - GPU 1 (MALE)
#===============================================================================
# Handles: All MALE images (7 races × 6 ages = 42 images × 4 prompts = 168 tasks)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/run_step1x_gpu1.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/run_step1x_gpu1.sh --edited
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/run_step1x_gpu1.sh --preserved
#===============================================================================

set -e

# HuggingFace cache path (change to your path)
export HF_HOME="/scratch/chans/models/huggingface"
export HF_HUB_CACHE="/scratch/chans/models/huggingface/hub"

MODEL="step1x"
DEVICE="cuda"
GENDER="Male"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}"
echo "============================================================"
echo "  Exp2: Step1X-Edit - GPU 1 (MALE)"
echo "  42 images × 4 prompts = 168 tasks per condition"
echo "============================================================"
echo -e "${NC}"

RUN_EDITED=true
RUN_PRESERVED=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --edited) RUN_EDITED=true; RUN_PRESERVED=false; shift ;;
        --preserved) RUN_EDITED=false; RUN_PRESERVED=true; shift ;;
        --help|-h) echo "Usage: $0 [--edited|--preserved]"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

EXPERIMENT_ID=$(date +"%Y%m%d_%H%M%S")
echo -e "${BLUE}Experiment ID: ${EXPERIMENT_ID}_gpu1${NC}"
echo ""

if [ "$RUN_EDITED" = true ]; then
    echo -e "${GREEN}▶ EDITED condition (Male)${NC}"
    cd "$PROJECT_ROOT"
    python scripts/experiment/exp2_preservation/run_preservation_experiment.py \
        --model "$MODEL" --device "$DEVICE" --condition edited \
        --experiment-id "${EXPERIMENT_ID}_gpu1_edited" \
        --gender "$GENDER"
    echo ""
fi

if [ "$RUN_PRESERVED" = true ]; then
    echo -e "${GREEN}▶ PRESERVED condition (Male)${NC}"
    cd "$PROJECT_ROOT"
    python scripts/experiment/exp2_preservation/run_preservation_experiment.py \
        --model "$MODEL" --device "$DEVICE" --condition preserved \
        --experiment-id "${EXPERIMENT_ID}_gpu1_preserved" \
        --gender "$GENDER"
    echo ""
fi

echo -e "${GREEN}GPU 1 (Male) Complete!${NC}"
