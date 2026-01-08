#!/bin/bash
# Setup script for I2I Refusal Bias Study
# Installs all required dependencies and models

set -e

echo "=========================================="
echo "I2I Refusal Bias Study - Environment Setup"
echo "=========================================="

# 1. Install base requirements
echo ""
echo "[1/5] Installing base requirements..."
pip install -r requirements.txt

# 2. Install diffusers from custom branch for Step1X-Edit-v1p2
echo ""
echo "[2/5] Installing custom diffusers for Step1X-Edit-v1p2..."
if [ ! -d "external/diffusers-step1x" ]; then
    mkdir -p external
    git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git external/diffusers-step1x
fi
pip install -e external/diffusers-step1x

# 3. Install RegionE for faster inference (optional)
echo ""
echo "[3/5] Installing RegionE (optional, for faster Step1X inference)..."
pip install RegionE || echo "RegionE installation failed - continuing without it"

# 4. Install latest diffusers for Qwen and FLUX.2
echo ""
echo "[4/5] Installing latest diffusers from GitHub..."
pip install git+https://github.com/huggingface/diffusers

# 5. Download FairFace dataset
echo ""
echo "[5/5] FairFace dataset will be downloaded on first run"
echo "      Run: python scripts/sample_fairface.py"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Model URLs:"
echo "  - FairFace: https://huggingface.co/datasets/HuggingFaceM4/FairFace"
echo "  - FLUX.2-dev: https://huggingface.co/black-forest-labs/FLUX.2-dev"
echo "  - Step1X-Edit-v1p2: https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2"
echo "  - Qwen-Image-Edit-2511: https://huggingface.co/Qwen/Qwen-Image-Edit-2511"
echo "=========================================="
