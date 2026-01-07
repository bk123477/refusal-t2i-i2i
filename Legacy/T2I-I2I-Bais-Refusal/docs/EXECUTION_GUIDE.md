# ACRB Prompt Generation Pipeline - Execution Guide

This guide provides step-by-step instructions for running the ACRB prompt generation pipeline.

## Prerequisites

### 1. Environment Setup

```bash
# Activate virtual environment (if using)
source venv/bin/activate  # or your venv path

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys

Set your Google API key for Gemini:

```bash
export GOOGLE_API_KEY="your-api-key-here"
# OR
export ACRB_LLM_API_KEY="your-api-key-here"
```

Verify API key is set:
```bash
echo $GOOGLE_API_KEY
```

## Quick Start (Minimal Example)

### Step 1: Generate Prompts (Small Test)

```bash
python3 scripts/design_prompts.py \
  --num-base 10 \
  --llm gemini-3-flash-preview \
  --output data/prompts/expanded_prompts.json
```

This generates 10 base prompts with all attribute expansions (~250 prompts total).

### Step 2: Validate and Curate

```bash
python3 scripts/validate_prompt_constraints.py \
  --expanded data/prompts/expanded_prompts.json \
  --benign-validation \
  --benign-llm gemini-3-flash-preview \
  --dedup \
  --write-clean data/prompts/expanded_prompts.curated.json
```

## Full Pipeline (Production)

### Step 1: Generate Prompts with Validation

```bash
python3 scripts/design_prompts.py \
  --num-base 100 \
  --domains all \
  --attributes culture gender disability religion age \
  --llm gemini-3-flash-preview \
  --benign-validation \
  --benign-llm gemini-3-flash-preview \
  --max-llm-attempts 1 \
  --output data/prompts/expanded_prompts.json
```

**Expected output:**
- `data/prompts/expanded_prompts.json`: ~2,500 prompts (100 base × 25 attributes)

**Time estimate:** ~30-60 minutes (depends on API rate limits)

### Step 2: Validate Constraints

```bash
python3 scripts/validate_prompt_constraints.py \
  --expanded data/prompts/expanded_prompts.json \
  --benign-validation \
  --benign-llm gemini-3-flash-preview \
  --dedup \
  --max-base-prompts 100 \
  --write-clean data/prompts/expanded_prompts.curated.json \
  --details data/prompts/validation_details.json
```

**Outputs:**
- `data/prompts/expanded_prompts.curated.json`: Curated prompt set
- `data/prompts/validation_details.json`: Per-prompt validation scores
- `data/prompts/validation_report.json`: Aggregate statistics

### Step 3: Pilot Test (Optional but Recommended)

Test refusal rates on a small sample before full experiment:

```bash
python3 scripts/pilot_boundary.py \
  --model flux-2-dev \
  --prompts data/prompts/expanded_prompts.curated.json \
  --samples 25
```

**Expected output:**
- `experiments/pilot/pilot_boundary.json`: Refusal rate report

## Using Makefile (Alternative)

If you prefer using Make:

```bash
# Set environment variables
export GOOGLE_API_KEY="your-key"
export LLM_MODEL="gemini-3-flash-preview"
export NUM_BASE_PROMPTS=100

# Generate prompts
make generate-prompts
```

## Troubleshooting

### Error: "No module named 'tqdm'"

```bash
pip install tqdm
# Or reinstall all dependencies
pip install -r requirements.txt
```

### Error: "API key not found"

```bash
# Check if key is set
echo $GOOGLE_API_KEY

# Set it if missing
export GOOGLE_API_KEY="your-key-here"
```

### Error: "Rate limit exceeded"

- Reduce `--num-base` value
- Add delays between API calls (modify script if needed)
- Use `--max-llm-attempts 1` to minimize retries

### LLM Not Working? Use Template Mode

```bash
python3 scripts/design_prompts.py \
  --num-base 10 \
  --no-llm \
  --output data/prompts/expanded_prompts.json
```

This uses template-based expansion (faster, but less natural).

## Output Files

After running the pipeline, you'll have:

1. **`data/prompts/expanded_prompts.json`**: Raw generated prompts
2. **`data/prompts/expanded_prompts.curated.json`**: Validated, deduplicated prompts (use this for experiments)
3. **`data/prompts/validation_details.json`**: Per-prompt validation scores
4. **`data/prompts/validation_report.json`**: Summary statistics

## Next Steps

Once you have curated prompts:

1. **Run T2I experiments:**
   ```bash
   ./run_experiment.sh --model flux2 --samples 100
   ```

2. **Run I2I experiments:**
   ```bash
   python3 scripts/generate_all.py --models qwen-edit --mode i2i
   ```

3. **Evaluate results:**
   ```bash
   python3 scripts/evaluate_all.py --results-dir experiments/images
   ```

## Notes

- **LLM Model**: Always use `gemini-3-flash-preview` (not gemini-2.0-flash)
- **Validation**: Enable `--benign-validation` for paper-grade quality
- **Deduplication**: Always run `--dedup` to avoid near-duplicate prompts
- **Pilot Testing**: Recommended before large-scale experiments to verify refusal rates

For detailed pipeline documentation, see `docs/PROMPT_PIPELINE.md`.



