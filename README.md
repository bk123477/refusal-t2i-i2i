# Silent Discrimination: Race-Conditioned Refusal Bias in I2I Editing Models

**IJCAI 2026 Submission**

## Overview

This repository contains the code and data for studying race-conditioned refusal bias in Image-to-Image (I2I) editing models. We investigate whether I2I models discriminatorily refuse or modify edit requests based on the race of the person in the source image.

**Key Innovation**: Rigorous dataset curation with complete audit trails ensures research reproducibility and eliminates confounding factors from image quality variations.

## Research Questions

- **RQ1**: Do neutral edit requests show racial bias in refusal rates?
- **RQ2**: Which edit types (occupational, cultural, disability) show the highest racial disparity?
- **RQ3**: Do models exhibit asymmetric gatekeeping for stereotype-congruent vs incongruent edits?

## Dataset: Curated FairFace Subset

### Selection Process
1. **Initial Sampling**: Extract candidates from FairFace using demographic queries (7 races × 2 genders × 6 ages = 84 combinations)
2. **Quality Curation**: Manual selection using `tools/image_selector/` with strict criteria:
   - Frontal face pose (exclude side profiles)
   - Clear focus and proper lighting
   - Neutral expression, upright posture
   - Facial features clearly identifiable
3. **Audit Trail**: Complete logging of selection rationale and criteria compliance
4. **Final Dataset**: 84 high-quality images with full reproducibility documentation

### Experiment Scale

| Metric | Count |
|--------|-------|
| Source Images | 84 (curated subset with audit trail) |
| Prompts per Image | 50 (5 categories × 10 prompts each) |
| Models | 3 (FLUX.2-dev, Step1X-Edit-v1p2, Qwen-Image-Edit-2511) |
| **Total I2I Requests** | **12,600** |

## Resources

### Datasets & Models

| Resource | Link | Notes |
|----------|------|-------|
| **FairFace Dataset** | https://huggingface.co/datasets/HuggingFaceM4/FairFace | Source dataset for demographic sampling |
| **FLUX.2-dev** | https://huggingface.co/black-forest-labs/FLUX.2-dev | Open-source I2I model |
| **Step1X-Edit-v1p2** | https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2 | Advanced reasoning-based editing |
| **Qwen-Image-Edit-2511** | https://huggingface.co/Qwen/Qwen-Image-Edit-2511 | Character-consistent editing |

### Dataset Curation Tools

- **`tools/image_selector/`**: Interactive web application for dataset curation
  - Manual quality assessment with strict criteria
  - Complete audit logging of selection rationale
  - Automatic dataset finalization with metadata
  - Full reproducibility documentation

### Key Features
- **Audit Trail**: Every image selection logged with timestamp, criteria compliance, and rationale
- **Quality Assurance**: Systematic exclusion of low-quality images (blur, poor lighting, etc.)
- **Demographic Balance**: Perfect factorial design (7×2×6 = 84 combinations)
- **Reproducibility**: Complete documentation for research transparency

## Project Structure

```
├── data/
│   ├── prompts/
│   │   └── i2i_prompts.json       # 50 edit prompts (5 categories × 10)
│   ├── config/
│   │   └── fairface_sampling.json # Sampling configuration
│   ├── source_images/
│   │   ├── final/                 # Final 84 images (512×512 JPG)
│   │   │   ├── Black/             # 12 images per race
│   │   │   ├── White/
│   │   │   └── .../
│   │   └── fairface/              # Curation process files
│   │       ├── V1-V7/             # Version folders
│   │       ├── selections.json    # Selection records
│   │       └── selection_logs.json # Complete audit trail
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
├── tools/
│   └── image_selector/            # Interactive dataset curation tool
│       ├── app.py                 # Flask web application
│       ├── templates/
│       │   └── index.html         # Selection interface with audit logging
│       └── requirements.txt       # Tool dependencies
│
├── scripts/
│   ├── analysis/
│   │   ├── analyze_results.py     # Statistical analysis & visualization
│   │   ├── compare_models.py      # Cross-model comparison
│   │   └── test_analysis_pipeline.py # Analysis pipeline testing
│   ├── data/
│   │   ├── download_fairface.py   # Download FairFace dataset
│   │   ├── sample_fairface.py     # Initial demographic sampling
│   │   └── sample_fairface_parallel.py # Parallel sampling
│   └── experiment/
│       ├── run_experiment.py      # Main experiment runner
│       ├── run_flux.sh            # FLUX experiment (interactive categories)
│       ├── run_step1x.sh          # Step1X experiment
│       ├── run_qwen.sh            # Qwen experiment
│       ├── category_selector.py   # Interactive category selection CLI
│       └── test/                   # Quick test scripts (1 image × 5 prompts)
│           ├── run_test.sh
│           └── test_single_prompt.py
│
├── docs/
│   ├── RESEARCH_PROPOSAL.md       # Full research proposal
│   ├── RESEARCH_PROPOSAL_DETAIL.md # Detailed methodology
│   └── RESEARCH_PROPOSAL_EN.md    # English version
│
├── README.md                      # This file
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
python scripts/data/download_fairface.py
```

### 3. Initial Demographic Sampling

Extract candidate images for each demographic combination:

```bash
python scripts/data/sample_fairface.py --output-dir data/source_images/fairface
```

### 4. Curate Final Dataset (Interactive)

Use the web-based curation tool to select 84 high-quality images with audit logging:

```bash
cd tools/image_selector
pip install -r requirements.txt
python app.py
# Open http://localhost:5050 in browser
```

**Curation Process:**
- Review selection criteria (facial pose, image quality, expression)
- Select best version for each demographic combination
- Document selection rationale and criteria compliance
- Finalize dataset with complete audit trail

### 5. Run Experiment

```bash
# Interactive category selection (recommended)
bash scripts/experiment/run_flux.sh
bash scripts/experiment/run_step1x.sh
bash scripts/experiment/run_qwen.sh

# Run all categories directly
bash scripts/experiment/run_flux.sh --all

# Run specific categories only
bash scripts/experiment/run_flux.sh --categories A,B,C

# Show help
bash scripts/experiment/run_flux.sh --help
```

**Output Location:**
```
data/results/{model}/{experiment_id}/
├── config.json       # Experiment settings (seed, categories, etc.)
├── results.json      # Full results (4,200 requests)
├── summary.json      # Statistics by race/gender/age
├── images/           # Generated images by race
└── logs/             # Detailed logs (refusals, errors, timings)
```

### 6. Analyze Results

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

## Dataset Quality Assurance

### Selection Criteria
All 84 final images meet strict quality standards:

**Facial Pose & Expression:**
- Frontal face orientation (exclude side/partial profiles)
- Neutral expression with upright head posture

**Image Quality:**
- Sharp focus with clear facial features
- Proper lighting conditions
- No obstructions or artifacts

**Audit Documentation:**
- Selection rationale logged for each image
- Criteria compliance verified and recorded
- Timestamp and version tracking for reproducibility

### Curation Process
1. **Automated Sampling**: Extract candidates using demographic metadata queries
2. **Manual Curation**: Interactive selection with quality assessment
3. **Audit Logging**: Document selection criteria and rationale
4. **Final Verification**: Complete dataset with full traceability

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
