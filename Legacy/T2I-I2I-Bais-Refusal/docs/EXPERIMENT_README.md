# ACRB Experiment Guide - Server-Ready Scripts

Complete experiment pipeline for IJCAI-ECAI 2026 submission.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate images (T2I)
python scripts/generate_all.py --models flux sd35 sdxl --samples 100

# 3. Evaluate (CLIP + VLM)
python scripts/evaluate_all.py --results-dir experiments/generation

# 4. Compute metrics and generate tables
python scripts/compute_results.py --input experiments/generation --output paper/tables
```

## Detailed Usage

### 1. Image Generation (`generate_all.py`)

Generates images using open-source diffusion models with GPU memory management.

#### T2I Generation

```bash
# FLUX.1-dev only (recommended for testing)
python scripts/generate_all.py --models flux --samples 100

# All T2I models
python scripts/generate_all.py --models flux sd35 sdxl --samples 1000

# With custom prompts
python scripts/generate_all.py \
  --models flux \
  --prompts-file data/prompts/custom_prompts.json \
  --samples 500

# Resume interrupted run
python scripts/generate_all.py --models flux --samples 1000 --resume
```

#### I2I Generation

```bash
# InstructPix2Pix with source images
python scripts/generate_all.py \
  --models instruct-pix2pix \
  --mode i2i \
  --source-images-dir data/external/ffhq \
  --samples 500
```

#### Output Structure

```
experiments/generation/
├── images/                          # Generated images
│   ├── flux_P0000_00.png
│   ├── sd35_P0001_01.png
│   └── ...
├── generated_prompts.json           # Auto-generated prompts
├── results_flux.json                # Per-model results
├── results_sd35.json
└── checkpoint_flux.json             # Resume checkpoints
```

#### Supported Models

| Model Key | Name | Type | Memory | Speed |
|-----------|------|------|--------|-------|
| `flux` | FLUX.1-dev | T2I | ~12GB | ~28s/img |
| `sd35` | SD 3.5 Large | T2I | ~14GB | ~35s/img |
| `sdxl` | SDXL | T2I | ~8GB | ~20s/img |
| `instruct-pix2pix` | InstructPix2Pix | I2I | ~6GB | ~15s/img |

### 2. Evaluation (`evaluate_all.py`)

Evaluates generated images for hard and soft refusal.

```bash
# Full evaluation (CLIP + VLM)
python scripts/evaluate_all.py \
  --results-dir experiments/generation \
  --vlm gpt-4o-mini

# CLIP refusal detection only
python scripts/evaluate_all.py \
  --results-dir experiments/generation \
  --skip-cue

# Test run (limit samples)
python scripts/evaluate_all.py \
  --results-dir experiments/generation \
  --max-samples 50

# Custom refusal threshold
python scripts/evaluate_all.py \
  --results-dir experiments/generation \
  --refusal-threshold 0.3
```

#### Requirements

- **CLIP**: Automatic (no API key needed)
- **VLM (GPT-4o-mini)**: Set `OPENAI_API_KEY` environment variable

```bash
export OPENAI_API_KEY="sk-..."
```

#### Output Structure

```
experiments/generation/
├── evaluation_checkpoint_refusal.json   # After CLIP detection
├── evaluation_results.json              # Final results
└── ...
```

#### Evaluation Metrics

**Hard Refusal (CLIP-based)**
- Detects explicit blocking/safety screens
- Threshold: 0.25 (adjustable)
- No API cost

**Soft Refusal (VLM-based)**
- Cue retention scoring (0-1)
- Detects attribute erasure and substitution
- Cost: ~$0.001 per image with GPT-4o-mini

### 3. Results Computation (`compute_results.py`)

Computes disparity metrics and generates publication-ready tables.

```bash
# Full pipeline
python scripts/compute_results.py \
  --input experiments/generation/evaluation_results.json \
  --output paper/tables

# CSV only
python scripts/compute_results.py \
  --input experiments/generation \
  --output experiments/tables \
  --format csv

# LaTeX only
python scripts/compute_results.py \
  --input experiments/generation \
  --output paper/tables \
  --format latex
```

#### Output Structure

```
paper/tables/
├── disparity_metrics.json           # Full metrics
├── summary_table.csv                # Table 1
├── summary_table.tex
├── attribute_table.csv              # Table 2
├── attribute_table.tex
├── refusal_rates.csv                # Raw data
├── erasure_rates.csv
└── substitution_rates.csv
```

#### Disparity Metrics

```
Δ_refusal = max_a(R(a)) - min_a(R(a))
Δ_erasure = max_a(E(a)) - min_a(E(a))
```

Where:
- R(a) = refusal rate for attribute a
- E(a) = erasure rate for attribute a

## Complete Pipeline Example

```bash
# Step 0: Setup
export OPENAI_API_KEY="sk-..."
cd /path/to/I2I-T2I-Bias-Refusal

# Step 1: Generate (FLUX only, 100 samples)
python scripts/generate_all.py \
  --models flux \
  --samples 100 \
  --seed 42 \
  --output-dir experiments/flux_run1

# Step 2: Evaluate
python scripts/evaluate_all.py \
  --results-dir experiments/flux_run1 \
  --vlm gpt-4o-mini

# Step 3: Compute results
python scripts/compute_results.py \
  --input experiments/flux_run1/evaluation_results.json \
  --output paper/tables/flux_run1

# Check results
cat paper/tables/flux_run1/disparity_metrics.json
```

## Memory Management

The scripts implement automatic GPU memory management:

1. **Sequential model loading**: Only one model in memory at a time
2. **CPU offloading**: Automatic for large models
3. **VAE/Attention slicing**: Enabled for memory efficiency
4. **Checkpointing**: Save progress every 10 samples

Recommended GPU: 16GB+ VRAM (RTX 4080/A4000 or better)

## Troubleshooting

### Out of Memory

```bash
# Use smaller batch of prompts
python scripts/generate_all.py --models flux --samples 10

# Use SDXL instead of FLUX
python scripts/generate_all.py --models sdxl --samples 100
```

### VLM Rate Limiting

```bash
# Skip VLM scoring (CLIP only)
python scripts/evaluate_all.py --skip-cue --results-dir experiments/generation

# Or use smaller sample
python scripts/evaluate_all.py --max-samples 50 --results-dir experiments/generation
```

### Resume Interrupted Run

```bash
# Generation will auto-save checkpoints
python scripts/generate_all.py --models flux --samples 1000 --resume
```

## Cost Estimation

### Compute Costs (Local GPU)

- **FLUX.1-dev**: ~28s/image → 1000 images = ~8 hours
- **SD 3.5**: ~35s/image → 1000 images = ~10 hours
- **SDXL**: ~20s/image → 1000 images = ~5.5 hours

### API Costs (VLM Evaluation)

- **GPT-4o-mini**: ~$0.001/image
- 1000 images = ~$1
- 5000 images = ~$5

### Total Budget (Full Experiment)

- Compute: Free (local GPU)
- VLM evaluation: ~$5-10
- **Total: $5-10**

## Citation

```bibtex
@inproceedings{acrb2026,
  title={ACRB: A Unified Framework for Auditing Attribute-Conditioned Refusal Bias via Dynamic LLM-Driven Red-Teaming},
  author={Your Name},
  booktitle={IJCAI-ECAI 2026},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details.
