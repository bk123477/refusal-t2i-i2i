#!/bin/bash
#===============================================================================
# Experiment 2: Qwen-Image-Edit - GPU 0 (FEMALE)
#===============================================================================
# Handles: All FEMALE images (7 races × 6 ages = 42 images × 4 prompts = 168 tasks)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/run_qwen_gpu0.sh
#===============================================================================

set -e

# HuggingFace cache path (change to your path)
export HF_HOME="/scratch/chans/models/huggingface"
export HF_HUB_CACHE="/scratch/chans/models/huggingface/hub"

MODEL="qwen"
DEVICE="cuda"
GENDER="Female"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}"
echo "============================================================"
echo "  Exp2: Qwen-Image-Edit - GPU 0 (FEMALE)"
echo "  42 images × 4 prompts = 168 tasks (preserved)"
echo "============================================================"
echo -e "${NC}"

EXPERIMENT_ID=$(date +"%Y%m%d_%H%M%S")
echo -e "${BLUE}Experiment ID: ${EXPERIMENT_ID}_gpu0${NC}"
echo ""

echo -e "${GREEN}▶ PRESERVED condition (Female)${NC}"
cd "$PROJECT_ROOT"
python scripts/experiment/exp2_preservation/run_preservation_experiment.py \
    --model "$MODEL" --device "$DEVICE" \
    --experiment-id "${EXPERIMENT_ID}_gpu0" \
    --gender "$GENDER"
echo ""

echo -e "${GREEN}GPU 0 (Female) Complete!${NC}"
