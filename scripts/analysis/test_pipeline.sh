#!/bin/bash
# Test the complete analysis pipeline with mock data

set -e  # Exit on error

echo "======================================"
echo "I2I Refusal Bias - Pipeline Test"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Step 1: Generate mock data
echo -e "\n${BLUE}Step 1: Generating mock results...${NC}"
python scripts/generate_mock_results.py \
    --models flux step1x qwen \
    --bias-strength 0.2 \
    --seed 42 \
    --output-dir data/results

echo -e "${GREEN}✓ Mock results generated${NC}"

# Step 2: Analyze individual models
echo -e "\n${BLUE}Step 2: Analyzing individual models...${NC}"

for model in flux step1x qwen; do
    echo -e "\n  Analyzing ${model}..."
    python scripts/analyze_results.py \
        --results-dir data/results \
        --model "$model" \
        --output-dir "results/test_analysis/${model}" \
        --prompts data/prompts/i2i_prompts.json

    echo -e "${GREEN}  ✓ ${model} analysis complete${NC}"
done

# Step 3: Compare models
echo -e "\n${BLUE}Step 3: Comparing models...${NC}"
python scripts/compare_models.py \
    --results-dir data/results \
    --models flux step1x qwen \
    --output-dir results/test_comparison

echo -e "${GREEN}✓ Model comparison complete${NC}"

# Step 4: Verify outputs
echo -e "\n${BLUE}Step 4: Verifying outputs...${NC}"

ERRORS=0

# Check if results were generated
for model in flux step1x qwen; do
    RESULT_FILE="data/results/${model}/mock_*/results.json"
    if ls $RESULT_FILE 1> /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ ${model} results.json exists${NC}"
    else
        echo -e "${RED}  ✗ ${model} results.json missing${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check if analysis outputs exist
for model in flux step1x qwen; do
    REPORT_FILE="results/test_analysis/${model}/analysis_report.json"
    if [ -f "$REPORT_FILE" ]; then
        echo -e "${GREEN}  ✓ ${model} analysis report exists${NC}"
    else
        echo -e "${RED}  ✗ ${model} analysis report missing${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check if comparison output exists
if [ -f "results/test_comparison/comparison_report.json" ]; then
    echo -e "${GREEN}  ✓ Comparison report exists${NC}"
else
    echo -e "${RED}  ✗ Comparison report missing${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check if figures were generated
FIGURE_COUNT=$(find results/test_analysis -name "*.png" | wc -l)
if [ "$FIGURE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}  ✓ Generated ${FIGURE_COUNT} figures${NC}"
else
    echo -e "${RED}  ✗ No figures generated${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo -e "\n======================================"
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Results saved to:"
    echo "  - Mock data: data/results/"
    echo "  - Analysis: results/test_analysis/"
    echo "  - Comparison: results/test_comparison/"
    exit 0
else
    echo -e "${RED}✗ ${ERRORS} test(s) failed${NC}"
    exit 1
fi
