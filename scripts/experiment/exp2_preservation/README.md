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

### Step1X (GPU 0/1 분할)

```bash
# 터미널 1 - GPU 0 (Female, 168 tasks)
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/run_step1x_gpu0.sh

# 터미널 2 - GPU 1 (Male, 168 tasks)
CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/run_step1x_gpu1.sh
```

### Qwen (GPU 0/1 분할, 옵션)

```bash
# 터미널 1 - GPU 0 (Female, 168 tasks)
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/run_qwen_gpu0.sh

# 터미널 2 - GPU 1 (Male, 168 tasks)
CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/run_qwen_gpu1.sh
```

### FLUX / Qwen (단일 GPU)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/experiment/exp2_preservation/run_flux_preservation.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/experiment/exp2_preservation/run_qwen_preservation.sh
```

### 특정 조건만 실행

```bash
bash scripts/experiment/exp2_preservation/run_step1x_gpu0.sh --edited     # identity prompt 포함
bash scripts/experiment/exp2_preservation/run_step1x_gpu0.sh --preserved  # baseline만
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
