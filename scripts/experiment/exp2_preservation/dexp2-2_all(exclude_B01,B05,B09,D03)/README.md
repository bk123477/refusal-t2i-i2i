# Experiment 2-2: Identity Preservation (Remaining Prompts)

## Overview

This experiment runs the remaining 16 prompts that were NOT included in exp2-1.

## Prompts

| ID | Category | Prompt Description |
|----|----------|-------------------|
| **B02** | B | Military Strategist |
| **B03** | B | Medical Specialist |
| **B04** | B | Kindergarten Teacher |
| **B06** | B | Airport Security |
| **B07** | B | Politician (Crisis) |
| **B08** | B | Professional Athlete |
| **B10** | B | Cleaner/Janitor |
| **D01** | D | Congenital Disability |
| **D02** | D | Wheelchair User (Conflict) |
| **D04** | D | Age-related Physical Strain |
| **D05** | D | Facial Marks (Past Event) |
| **D06** | D | Priority Seating |
| **D07** | D | Pregnancy Priority Area |
| **D08** | D | Ongoing Medical Care |
| **D09** | D | Economic Hardship |
| **D10** | D | Digital Exclusion |

**Excluded (already done in exp2-1):** B01, B05, B09, D03

## Scale

- **Per Model**: 16 prompts × 84 images = 1,344 images
- **Total**: 1,344 × 3 models = **4,032 images**

## Usage

### Single GPU

```bash
# FLUX
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/dexp2-2_all\(exclude_B01,B05,B09,D03\)/run_flux_preservation.sh

# Qwen
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/dexp2-2_all\(exclude_B01,B05,B09,D03\)/run_qwen_preservation.sh

# Step1X
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/dexp2-2_all\(exclude_B01,B05,B09,D03\)/run_step1x_preservation.sh
```

### Dual GPU (Gender Split)

```bash
# Step1X: Terminal 1 - GPU 0 (Female, 672 tasks)
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/dexp2-2_all\(exclude_B01,B05,B09,D03\)/run_step1x_gpu0.sh

# Step1X: Terminal 2 - GPU 1 (Male, 672 tasks)
CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/dexp2-2_all\(exclude_B01,B05,B09,D03\)/run_step1x_gpu1.sh

# Qwen: Similar for GPU split
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/dexp2-2_all\(exclude_B01,B05,B09,D03\)/run_qwen_gpu0.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/dexp2-2_all\(exclude_B01,B05,B09,D03\)/run_qwen_gpu1.sh
```

### Resume from Checkpoint

```bash
# Resume from task 500
bash run_flux_preservation.sh --resume-from 500
```

## Output

```
data/results/exp2_pairwise/
├── flux/preserved/
│   ├── B02_Black_Female_20s.png
│   ├── ...
│   └── results_exp2-2_*.json
├── qwen/preserved/
└── step1x/preserved/
```
