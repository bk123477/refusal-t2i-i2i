# ACRB Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd I2I-T2I-Bias-Refusal

# Install dependencies
pip install torch torchvision
pip install transformers diffusers
pip install pillow requests tqdm
pip install openai clip
pip install scipy numpy pandas
```

## Quick Test (5 minutes)

```bash
# Test the implementation without LLM or actual generation
python scripts/test_pipeline.py
```

## Basic Usage (10-15 minutes)

### Test with minimal samples (no LLM, template-based)
```bash
python scripts/run_audit.py \
  --model flux-2-dev \
  --mode t2i \
  --samples 5 \
  --attributes culture gender
```

**Output**: `experiments/results/flux-2-dev/{timestamp}/`
- 50 prompts generated (5 base × 10 variations)
- Runtime: ~10-15 minutes
- Cost: ~$5-10

## Production Usage

### Full T2I Audit with LLM Expansion
```bash
export ACRB_LLM_API_KEY=your_gemini_api_key
export OPENAI_API_KEY=your_openai_api_key

python scripts/run_audit.py \
  --model flux-2-dev \
  --mode t2i \
  --samples 100 \
  --llm gemini-3-flash-preview \
  --llm-api-key $ACRB_LLM_API_KEY \
  --vlm gpt-4o-mini
```

**Output**:
- 2,500 prompts (100 base × 25 variations)
- Runtime: ~1-2 hours
- Cost: ~$60-120

### I2I Audit with Real Images
```bash
python scripts/run_audit.py \
  --model qwen-image-edit-2511 \
  --mode i2i \
  --dataset /path/to/ffhq \
  --samples 50 \
  --llm gemini-3-flash-preview
```

## Programmatic Usage

```python
from acrb.pipeline import ACRBPipeline, ACRBConfig

# Configure
config = ACRBConfig(
    model_name="flux-2-dev",
    mode="t2i",
    max_base_prompts=10,
    attribute_types=["culture", "gender"],
    llm_model="gemini-3-flash-preview",
    seed=42
)

# Run
pipeline = ACRBPipeline(config)
result = pipeline.run()

# Results
print(f"Refusal Rate: {result.refusal_rate:.2%}")
print(f"Δ_refusal: {result.delta_refusal:.4f}")
print(f"Δ_erasure: {result.delta_erasure:.4f}")

# Top 5 most refused attributes
import sorted
top_refused = sorted(
    result.refusal_by_attribute.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]
for attr, rate in top_refused:
    print(f"{attr}: {rate:.2%}")
```

## Understanding Results

### Output Files
```
experiments/results/{model}/{run_id}/
├── stage1_prompts.json          # All generated prompts
├── stage2_generation.json        # Generation results
├── stage3_evaluation.json        # Detailed metrics
├── acrb_summary.json            # Final Δ metrics
└── images/                       # All generated images
```

### Key Metrics

1. **Refusal Rate**: Overall % of prompts refused
2. **Δ_refusal**: Disparity in refusal across attributes (max - min)
3. **Δ_erasure**: Disparity in cue erasure across attributes
4. **Refusal by Attribute**: Per-attribute refusal rates (e.g., `culture:Nigerian: 0.164`)
5. **Erasure by Attribute**: Per-attribute erasure rates (e.g., `disability:wheelchair_user: 0.371`)

### Interpreting Δ Metrics

- **Δ < 0.05**: Low disparity (fair)
- **Δ = 0.05-0.10**: Moderate disparity
- **Δ = 0.10-0.20**: High disparity (concerning)
- **Δ > 0.20**: Severe disparity (critical bias)

### Example Results
```json
{
  "delta_refusal": 0.129,        // Nigerian 16.4% vs American 3.5%
  "delta_erasure": 0.191,         // Wheelchair 37.1% vs Neutral 25.6%
  "most_refused": "culture:Nigerian",
  "most_erased": "disability:wheelchair_user"
}
```

## Common Options

### Model Selection
```bash
# T2I Models
--model flux-2-dev              # FLUX.2 (open-source)
--model gpt-image-1.5           # DALL-E 3 (OpenAI)
--model nano-banana-pro         # Imagen 3 (Google)

# I2I Models
--model qwen-image-edit-2511    # Qwen Edit (Alibaba)
--model flux-kontext            # FLUX.1 Kontext (BFL)
--model step1x-edit             # Step1X Edit (StepFun)
```

### Attribute Selection
```bash
# All attributes (default)
--attributes culture gender disability religion age

# Subset for faster testing
--attributes culture
--attributes culture gender
--attributes disability religion
```

### LLM Selection
```bash
# Google Gemini (recommended)
--llm gemini-3-flash-preview
--llm-api-base https://generativelanguage.googleapis.com/v1beta/openai

# OpenAI GPT
--llm gpt-4o-mini
--llm-api-base https://api.openai.com/v1

# Local LLM (vLLM)
--llm llama-3-70b
--llm-api-base http://localhost:8000/v1
```

### VLM Selection
```bash
# OpenAI GPT-4o (recommended)
--vlm gpt-4o-mini

# Local Qwen (requires GPU)
--vlm qwen2.5-vl-7b
```

## Troubleshooting

### Issue: "CLIP not available"
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Issue: "Rate limit (429)"
- Reduce `--samples` to lower API call volume
- Use `--llm None` to skip LLM expansion (template-based)

### Issue: "No source images for I2I"
- Provide `--dataset /path/to/images` with real images
- Pipeline will auto-generate synthetic sources if not provided

### Issue: "VLM API error"
- Check API key: `echo $OPENAI_API_KEY`
- Verify model availability: `--vlm gpt-4o-mini`

## Performance Tips

1. **Start small**: Use `--samples 5` to test
2. **Limit attributes**: Use `--attributes culture` for faster runs
3. **Skip LLM**: Omit `--llm` for template-based (faster, cheaper)
4. **Batch mode**: Process multiple models sequentially

## Example Workflow

```bash
# Step 1: Test implementation
python scripts/test_pipeline.py

# Step 2: Small test audit
python scripts/run_audit.py --samples 5

# Step 3: Medium audit with LLM
python scripts/run_audit.py --samples 20 --llm gemini-3-flash-preview

# Step 4: Full production audit
python scripts/run_audit.py --samples 100 --llm gemini-3-flash-preview

# Step 5: Analyze results
python -c "
import json
with open('experiments/results/flux-2-dev/{run_id}/acrb_summary.json') as f:
    r = json.load(f)
    print(f'Δ_refusal: {r[\"delta_refusal\"]:.4f}')
    print(f'Δ_erasure: {r[\"delta_erasure\"]:.4f}')
"
```

## Next Steps

1. **Run first audit**: `python scripts/run_audit.py --samples 5`
2. **Review results**: Check `experiments/results/`
3. **Read full docs**: See `ALGORITHM_IMPLEMENTATION.md`
4. **Scale up**: Increase `--samples` to 50-100
5. **Compare models**: Run multiple models and compare Δ metrics

## Support

- **Implementation Guide**: `ALGORITHM_IMPLEMENTATION.md`
- **Full Documentation**: `IMPLEMENTATION_SUMMARY.md`
- **Paper**: `paper/main.tex` (Algorithm 1)
- **Code**: `acrb/pipeline.py` (main implementation)
