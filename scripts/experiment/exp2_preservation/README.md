# Experiment 2: Identity Preservation

## Selected Prompts

| ID | Prompt | Failure Mode |
|----|--------|--------------|
| **B01** | CEO/Executive | Gender (f→m) + Race drift |
| **B05** | Housekeeper | Gender (m→f) reverse stereotype |
| **B09** | Fashion Model | Race + Identity change |
| **D03** | 30 Years Older | Over-aging + Race change |

## Scale

- **Per Condition**: 4 prompts × 84 images = 336 images
- **Per Model**: 336 × 2 conditions = 672 images
- **Total**: 672 × 3 models = **2,016 images**

## Usage

```bash
# GPU별 실행
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/run_step1x_preservation.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/run_flux_preservation.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/experiment/exp2_preservation/run_qwen_preservation.sh

# 특정 조건만
bash scripts/experiment/exp2_preservation/run_step1x_preservation.sh --edited
bash scripts/experiment/exp2_preservation/run_step1x_preservation.sh --preserved

# 재개
bash scripts/experiment/exp2_preservation/run_step1x_preservation.sh --resume 100
```

## Output

```
data/results/exp2_pairwise/
├── step1x/
│   ├── edited/
│   └── preserved/
├── flux/
└── qwen/
```
