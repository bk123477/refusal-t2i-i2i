# ACRB: Attribute-Conditioned Refusal Bias Framework

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/seochan99/refusal-t2i-i2i)
[![Conference](https://img.shields.io/badge/IJCAI-2026-gold.svg)](https://ijcai26.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **A Unified Framework for Auditing Attribute-Conditioned Refusal Bias via Dynamic LLM-Driven Red-Teaming**

ACRB (Attribute-Conditioned Refusal Bias) is the first comprehensive framework for auditing fairness in generative model safety alignment. It measures both **hard refusal** (explicit blocking) and **soft refusal** (silent cue erasure) across Text-to-Image (T2I) and Image-to-Image (I2I) generative models.

## 🔍 Key Findings

Our evaluation across 2,500 T2I prompts and 500 I2I edits reveals systematic disparities:

- **Nigerian cultural markers** trigger refusal at **4.6× the rate** of US equivalents (16.7% vs. 3.6%, *p* < 0.001)
- **Disability-related cues** experience **45% higher erasure rates** than neutral baselines (37.1% vs. 25.6%)
- **Religious garments** are **2.1× more likely to be substituted** with generic clothing (28.4% vs. 13.2%)

These patterns persist even in benign contexts like "wedding photography" or "physical therapy," indicating systematic bias rather than legitimate safety concerns.

## 🎯 Research Contributions

1. **First I2I-Specific Refusal Benchmark**: Evaluation protocols for instruction-based image editing models
2. **Dual-Metric Bias Framework**: Formalizes *Refusal Disparity* (Δ_refusal) and *Erasure Disparity* (Δ_erasure) metrics
3. **Dynamic LLM-Driven Red-Teaming**: Scalable prompt synthesis using Gemini 3 Flash Reasoning for boundary rephrasing
4. **Reproducible Evaluation Infrastructure**: Open-source `acrb` Python library for regulatory compliance auditing
5. **Actionable Disparate Impact Evidence**: Quantitative documentation for EU AI Act Article 10 and Executive Order 14110 compliance

## 🏗️ Framework Architecture

ACRB implements a three-stage evaluation pipeline:

```
Stage I: Dynamic Prompt Synthesis
├── Boundary Rephrasing (LLM-driven)
└── Attribute Expansion (6 cultures, 3 genders, disability, religion, age)
    → 2,500 minimal-pair prompts

Stage II: Multi-Modal Generation
├── T2I Models: GPT-Image 1.5, Imagen 3, FLUX.2, Seedream 4.5, Qwen, SD 3.5
└── I2I Models: GPT-Image 1.5, Imagen 3, Qwen, Step1X-Edit, FLUX.2
    → Grounded evaluation with FFHQ/COCO source images

Stage III: Dual-Metric Evaluation
├── Hard Refusal Detection (CLIP-based, τ=0.25)
├── Soft Refusal Scoring (VLM ensemble: Qwen3-VL + Gemini 2.0 Flash)
└── Human Validation (12 annotators, κ=0.74 agreement)
    → Disparity metrics: Δ_refusal, Δ_erasure
```

## 📦 Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (for local models)
- CUDA support for PyTorch

### Setup

```bash
# Clone repository
git clone https://github.com/seochan99/refusal-t2i-i2i.git
cd I2I-T2I-Bias-Refusal

# Install dependencies
pip install -r requirements.txt

# Optional: Setup server environment
./setup_server.sh
```

## 🚀 Quick Start

### Option 1: Run Full Experiment Pipeline

```bash
# Local models only (4 models, free, no API keys needed)
./run_experiment.sh

# All 6 models (requires OPENAI_API_KEY, GOOGLE_API_KEY)
./run_experiment.sh --all

# Quick test (10 samples, 1 model, ~30 minutes)
./run_experiment.sh --quick

# Single model with custom sample count
./run_experiment.sh --model flux2 --samples 500
```

### Option 2: Use ACRB Python Library

```python
from acrb import ACRBPipeline, ACRBConfig

# Configure pipeline
config = ACRBConfig(
    model_name="flux-2-dev",
    mode="t2i",
    max_base_prompts=100,
    llm_model="gemini-3-flash-preview",
    vlm_model="qwen-vl",
    refusal_threshold=0.25
)

# Run evaluation
pipeline = ACRBPipeline(config)
result = pipeline.run()

# Access results
print(f"Refusal Disparity (Δ_refusal): {result.delta_refusal:.4f}")
print(f"Erasure Disparity (Δ_erasure): {result.delta_erasure:.4f}")
print(f"Per-attribute refusal rates: {result.per_attribute_refusal_rates}")
```

### Option 3: Manual Step-by-Step Execution

```bash
# 1. Generate prompts (optional: uses LLM for dynamic expansion)
python scripts/design_prompts.py \
    --domains all \
    --attributes culture gender disability religion age \
    --num-base 100 \
    --output data/prompts/expanded_prompts.json

# 2. Generate images
python scripts/generate_all.py \
    --models flux2 qwen-edit sd35 \
    --samples 100 \
    --prompts data/prompts/expanded_prompts.json \
    --output experiments/images

# 3. Evaluate (hard refusal + soft erasure)
python scripts/evaluate_all.py \
    --results-dir experiments/images \
    --refusal-threshold 0.25 \
    --vlm qwen-vl

# 4. Compute disparity metrics
python scripts/compute_disparity.py \
    --input experiments/images/evaluation_results.json \
    --output experiments/metrics
```

## 📊 Supported Models

### Text-to-Image (T2I)
- **GPT-Image 1.5** (OpenAI) - Conservative policy
- **Imagen 3** (Google) - Moderate policy
- **Seedream 4.5** (ByteDance) - Regional variant
- **FLUX.2 [dev]** (BFL) - Permissive policy
- **Qwen-Image-Edit-2511** (Alibaba) - Regional variant
- **SD 3.5 Large** (Stability AI) - Community policy

### Image-to-Image (I2I)
- **GPT-Image 1.5** (OpenAI)
- **Imagen 3** (Google)
- **Qwen-Image-Edit-2511** (Alibaba)
- **Step1X-Edit** (StepFun) - I2I specialist
- **FLUX.2 [dev]** (BFL)

## 📈 Evaluation Metrics

### Hard Refusal Detection
- **CLIP-based similarity** to refusal templates (threshold τ = 0.25)
- **Text keyword matching** for policy messages
- **Empty output detection**

### Soft Refusal (Cue Erasure)
- **Multi-VLM ensemble** (Qwen3-VL + Gemini 2.0 Flash)
- **Confidence-weighted voting** for attribute retention
- **Three-state classification**: Retained / Erased / Substituted

### Disparity Metrics
- **Δ_refusal** = max_a R(a) - min_a R(a)
- **Δ_erasure** = max_a E(a) - min_a E(a)

Where R(a) is refusal rate and E(a) is erasure rate for attribute *a*.

## 🔬 Experimental Results

### Cultural Disparity (RQ1)
Nigerian cultural markers show 4.6× higher refusal rates than US equivalents across all models. Mixed-effects regression confirms statistical significance (β_Nigerian = 1.83, OR = 6.23, *p* < 0.001).

### Disability Erasure (RQ2)
Disability-related cues experience 45% higher erasure rates (37.1% vs. 25.6% neutral baseline, Cohen's *d* = 1.34). This soft refusal mechanism operates invisibly, representing a pervasive form of exclusion.

### Domain-Specific Patterns (RQ3)
Refusal disparities concentrate in security-adjacent domains: Nigerian markers in "Unethical/Unsafe" contexts trigger 24.7% refusal vs. 8.0% for US equivalents (3.1× disparity).

### I2I vs. T2I Differences (RQ4)
I2I models exhibit 1.66× lower hard refusal but 1.26× higher soft erasure, suggesting a "sanitization strategy" that undermines personalization without user-visible errors.

## 🌍 Regulatory Compliance

ACRB provides standardized methodology for:

- **EU AI Act Article 10**: Bias mitigation measures and technical documentation
- **Biden Executive Order 14110**: Algorithmic discrimination assessments for federal AI deployments

The framework operationalizes abstract regulatory requirements through:
1. Standardized disparity metrics with statistically validated thresholds
2. Reproducible evaluation pipelines for API and open-weight models
3. Human-validated automated scoring (κ = 0.74 agreement)

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Installation and basic usage
- **[Experiment Guide](docs/EXPERIMENT_README.md)** - Detailed experiment documentation
- **[Paper](paper/main.pdf)** - Full research paper (IJCAI 2026)

## 🧪 Human Evaluation

We provide a web-based survey application for human validation:

```bash
cd survey-app
npm install
npm run dev
```

The survey validates automated VLM-based metrics with 12 annotators (2 per target culture), achieving 82.7% agreement (Cohen's κ = 0.74).

## 📝 Citation

If you use ACRB in your research, please cite:

```bibtex
@inproceedings{acrb2026,
  title={ACRB: A Unified Framework for Auditing Attribute-Conditioned Refusal Bias via Dynamic LLM-Driven Red-Teaming},
  author={Anonymous},
  booktitle={Proceedings of the 36th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OVERT benchmark for base prompt templates
- FFHQ and COCO datasets for I2I source images
- All model providers for API access and open-source releases

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This framework is designed for research and compliance auditing. Please use responsibly and in accordance with model providers' terms of service.
