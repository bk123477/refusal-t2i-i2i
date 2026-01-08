# Silent Discrimination: Race-Conditioned Refusal Bias in I2I Editing Models

**IJCAI 2026 Submission**

## Overview

This repository contains the code and data for studying race-conditioned refusal bias in Image-to-Image (I2I) editing models. We investigate whether I2I models discriminatorily refuse or modify edit requests based on the race of the person in the source image.

## Research Questions

- **RQ1**: Do neutral edit requests show racial bias in refusal rates?
- **RQ2**: Which edit types (occupational, cultural, disability) show the highest racial disparity?
- **RQ3**: Do models exhibit asymmetric gatekeeping for stereotype-congruent vs incongruent edits?

## Experiment Scale

| Metric | Count |
|--------|-------|
| Source Images | 84 (6 ages × 2 genders × 7 races) |
| Prompts per Image | 50 |
| Models | 3 (FLUX.2, Step1X-Edit, Qwen-Image-Edit) |
| **Total I2I Requests** | **12,600** |

## Models & Datasets

| Resource | Link |
|----------|------|
| **FairFace Dataset** | https://huggingface.co/datasets/HuggingFaceM4/FairFace |
| **FLUX.2-dev** | https://huggingface.co/black-forest-labs/FLUX.2-dev |
| **Step1X-Edit-v1p2** | https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2 |
| **Qwen-Image-Edit-2511** | https://huggingface.co/Qwen/Qwen-Image-Edit-2511 |

## Project Structure

```
├── data/
│   ├── prompts/
│   │   └── i2i_prompts.json       # 50 edit prompts (5 categories × 10)
│   ├── config/
│   │   └── fairface_sampling.json # Sampling configuration
│   ├── source_images/
│   │   └── fairface/              # 84 sampled images
│   └── results/                   # Experiment outputs
│
├── src/
│   ├── data/
│   │   ├── fairface_sampler.py    # FairFace dataset sampler
│   │   └── prompt_loader.py       # Prompt management
│   ├── models/
│   │   ├── base.py                # I2I model interface
│   │   ├── flux_wrapper.py        # FLUX.2-dev wrapper
│   │   ├── step1x_wrapper.py      # Step1X-Edit-v1p2 wrapper
│   │   └── qwen_wrapper.py        # Qwen-Image-Edit-2511 wrapper
│   ├── evaluation/
│   │   ├── refusal_detector.py    # Hard refusal detection
│   │   ├── erasure_detector.py    # Soft erasure detection
│   │   ├── vlm_evaluator.py       # VLM-based attribute checking
│   │   └── metrics.py             # Disparity & SCS metrics
│   └── analysis/
│       ├── statistical.py         # ANOVA, Chi-square, etc.
│       └── visualization.py       # Figures & tables
│
├── scripts/
│   ├── setup_environment.sh       # Environment setup
│   ├── download_fairface.py       # Download FairFace dataset
│   ├── sample_fairface.py         # Sample 84 images
│   ├── run_experiment.py          # Main experiment runner
│   └── analyze_results.py         # Statistical analysis
│
├── RESEARCH_PROPOSAL.md           # Full research proposal
└── requirements.txt
```

## Prompt Categories

| Category | Name | Count | Description |
|----------|------|-------|-------------|
| A | Neutral Baseline | 10 | Should show no racial bias |
| B | Occupational Stereotype | 10 | Doctor, CEO, Judge, etc. |
| C | Cultural/Religious | 10 | Hijab, Kente, Sari, etc. |
| D | Vulnerability | 10 | Wheelchair, Prosthetic, etc. |
| E | Harmful/Safety | 10 | Weapons, Criminal imagery, etc. |

## Quick Start

### 1. Setup Environment

```bash
# Install base requirements
pip install -r requirements.txt

# Install latest diffusers for Qwen and FLUX.2
pip install git+https://github.com/huggingface/diffusers

# Install custom diffusers for Step1X-Edit-v1p2
git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git external/diffusers-step1x
pip install -e external/diffusers-step1x

# Optional: RegionE for faster Step1X inference
pip install RegionE
```

Or use the setup script:
```bash
bash scripts/setup_environment.sh
```

### 2. Download FairFace Dataset

```bash
python scripts/download_fairface.py
```

### 3. Sample 84 Images (Factorial Design)

```bash
python scripts/sample_fairface.py --output-dir data/source_images/fairface
```

### 4. Run Experiment

```bash
# Single model
python scripts/experiment/run_experiment.py --model flux --device cuda
python scripts/experiment/run_experiment.py --model step1x --device cuda
python scripts/experiment/run_experiment.py --model qwen --device cuda

# All models
python scripts/experiment/run_experiment.py --model all --device cuda
```

### 5. Analyze Results

```bash
python scripts/analyze_results.py --results-dir data/results
```

## Model Configuration

### FLUX.2-dev
```python
from src.models import FluxWrapper

# Full precision (high VRAM required)
model = FluxWrapper(device="cuda")

# 4-bit quantization for RTX 4090/5090
model = FluxWrapper(device="cuda", use_quantized=True, use_remote_text_encoder=True)
```

### Step1X-Edit-v1p2
```python
from src.models import Step1XWrapper

# With thinking and reflection (best quality)
model = Step1XWrapper(device="cuda", enable_thinking=True, enable_reflection=True)

# Base mode (faster)
model = Step1XWrapper(device="cuda", enable_thinking=False, enable_reflection=False)
```

### Qwen-Image-Edit-2511
```python
from src.models import QwenImageEditWrapper

model = QwenImageEditWrapper(device="cuda")
```

## Key Metrics

### Hard Refusal Rate
$$R_{r,c} = \frac{\text{refused}_{r,c}}{\text{total}_{r,c}}$$

### Soft Erasure Rate
$$E_{r,c} = \frac{\text{erased}_{r,c}}{\text{generated}_{r,c}}$$

### Disparity
$$\Delta_{\text{refusal}}^{(c)} = \max_r R_{r,c} - \min_r R_{r,c}$$

### Stereotype Congruence Score (Novel)
$$SCS = \frac{R_{\text{incongruent}} - R_{\text{congruent}}}{R_{\text{baseline}}}$$

## Hypotheses

| ID | Hypothesis | Expected Result |
|----|------------|-----------------|
| H1 | Neutral prompts show no racial bias | Δ < 3pp, p > 0.05 |
| H2 | Occupational prompts show stereotype bias | Higher refusal for Black/Latino in professional roles |
| H3 | Cultural prompts show gatekeeping | SCS > 0 for cross-cultural edits |
| H4 | Disability + Race interaction | Black + disability = highest erasure |
| H5 | Models differ in bias patterns | Significant model × race interaction |
| H6 | Harmful prompts show asymmetry | "Threatening" either over- or under-refused by race |

## Citation

```bibtex
@inproceedings{anonymous2026silent,
  title={Silent Discrimination: Race-Conditioned Refusal Bias in Image-to-Image Editing Models},
  author={Anonymous},
  booktitle={Proceedings of IJCAI 2026},
  year={2026}
}
```

## License

MIT License
