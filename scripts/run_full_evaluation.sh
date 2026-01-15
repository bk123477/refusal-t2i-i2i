#!/bin/bash
#
# Full Evaluation Pipeline
#
# After image generation is complete, run this script to:
# 1. Run VLM evaluation on generated images
# 2. Launch human evaluation survey (TURBO mode)
# 3. Compute final metrics from human + VLM results
#
# Usage:
#   ./scripts/run_full_evaluation.sh [--model flux|step1x|qwen|all]
#

set -e

MODEL="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  I2I Bias Evaluation Pipeline"
echo "  Model: $MODEL"
echo "=============================================="
echo ""

# Step 1: Find latest experiment directories
echo "[1/4] Finding experiment results..."

if [ "$MODEL" = "all" ]; then
    MODELS="flux step1x qwen"
else
    MODELS="$MODEL"
fi

for m in $MODELS; do
    RESULTS_DIR="$PROJECT_ROOT/data/results/$m"
    if [ -d "$RESULTS_DIR" ]; then
        LATEST=$(ls -t "$RESULTS_DIR" | head -1)
        echo "  $m: $LATEST"
    else
        echo "  $m: No results found"
    fi
done
echo ""

# Step 2: Run VLM Evaluation
echo "[2/4] Running VLM evaluation..."
echo "  (This may take a while for 4536+ images per model)"
echo ""

for m in $MODELS; do
    RESULTS_DIR="$PROJECT_ROOT/data/results/$m"
    if [ -d "$RESULTS_DIR" ]; then
        LATEST=$(ls -t "$RESULTS_DIR" | head -1)
        LATEST_DIR="$RESULTS_DIR/$LATEST"

        if [ ! -f "$LATEST_DIR/vlm_results.json" ]; then
            echo "  Running VLM evaluation for $m ($LATEST)..."
            python "$PROJECT_ROOT/scripts/evaluation/run_vlm_evaluation.py" \
                --results-dir "$LATEST_DIR" \
                --qwen-model 30B \
                --skip-identity-drift
        else
            echo "  $m: VLM results already exist, skipping"
        fi
    fi
done
echo ""

# Step 3: Launch Human Evaluation Survey
echo "[3/4] Human Evaluation Survey"
echo ""
echo "  The TURBO survey is available at:"
echo "    http://localhost:3000/turbo"
echo ""
echo "  To start the survey server:"
echo "    cd $PROJECT_ROOT/survey && npm run dev"
echo ""
echo "  Evaluator assignments:"
echo "    - 희찬 (Heechan): Step1X-Edit (4,536 samples)"
echo "    - 시은 (Sieun): FLUX.2-dev (4,536 samples)"
echo "    - 민기 (Mingi): Qwen-Image-Edit (4,536 samples)"
echo ""
echo "  After evaluation, export results from the survey UI."
echo ""

read -p "Press Enter when human evaluation is complete..."

# Step 4: Compute Final Metrics
echo "[4/4] Computing final metrics..."
echo ""

# Look for exported results
HUMAN_RESULTS=$(ls -t "$PROJECT_ROOT"/survey/turbo_eval_*.json 2>/dev/null | head -1 || true)

if [ -z "$HUMAN_RESULTS" ]; then
    HUMAN_RESULTS=$(ls -t "$PROJECT_ROOT"/turbo_eval_*.json 2>/dev/null | head -1 || true)
fi

if [ -z "$HUMAN_RESULTS" ]; then
    echo "  Error: No human evaluation results found."
    echo "  Please export results from the survey UI first."
    exit 1
fi

echo "  Using human results: $HUMAN_RESULTS"

# Find VLM results
VLM_RESULTS=""
for m in $MODELS; do
    RESULTS_DIR="$PROJECT_ROOT/data/results/$m"
    if [ -d "$RESULTS_DIR" ]; then
        LATEST=$(ls -t "$RESULTS_DIR" | head -1)
        VLM_FILE="$RESULTS_DIR/$LATEST/vlm_results.json"
        if [ -f "$VLM_FILE" ]; then
            VLM_RESULTS="$VLM_FILE"
            break
        fi
    fi
done

# Compute metrics
OUTPUT_DIR="$PROJECT_ROOT/analysis"
mkdir -p "$OUTPUT_DIR"

python "$PROJECT_ROOT/scripts/evaluation/compute_final_metrics.py" \
    --human-results "$HUMAN_RESULTS" \
    ${VLM_RESULTS:+--vlm-results "$VLM_RESULTS"} \
    --output "$OUTPUT_DIR/final_metrics.json" \
    --latex "$OUTPUT_DIR/paper_tables.tex"

echo ""
echo "=============================================="
echo "  Pipeline Complete!"
echo "=============================================="
echo ""
echo "  Results saved to:"
echo "    - $OUTPUT_DIR/final_metrics.json"
echo "    - $OUTPUT_DIR/paper_tables.tex"
echo ""
