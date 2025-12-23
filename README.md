# 🛡️ ACRB: Attribute-Conditioned Refusal Bias Framework

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/acrb-audit)
[![Conference](https://img.shields.io/badge/IJCAI--ECAI-2026-gold.svg)](https://ijcai26.org/)
[![Status](https://img.shields.io/badge/Research-Ready-brightgreen.svg)]()

> **The Next-Generation Unified Audit for Generative AI Safety & Fairness.**

ACRB (Attribute-Conditioned Refusal Bias) is a unified, high-fidelity framework designed to audit structural biases within the safety-alignment mechanisms of Text-to-Image (T2I) and Image-to-Image (I2I) generative models. 

By measuring both **Hard Refusal** (explicit blocking) and **Soft Refusal** (silent attribute scrubbing), ACRB provides a granular view of how identity factors—culture, gender, disability, religion, and age—intersect with safety-triggered over-refusal.

---

## ✨ Key Technical Pillars

- **💎 High-Fidelity Multi-Modal Audit**: Unified evaluation for T2I generation and I2I editing (FFHQ/COCO-grounded).
- **🧠 Dynamic Red-Teaming**: Leveraging `gpt-oss-20b` to generate linguistically complex boundary prompts.
- **📊 Granular Metrics**: Quantification of cue erasure and refusal disparity across 9 safety domains.
- **📦 Professional core library**: Namespaced `acrb` package ready for integration into large-scale production auditing.

---

## 🏗️ Project Architecture

```bash
.
├── acrb/                # Core namespaced library
│   ├── evaluation/      # Pipeline orchestration
│   ├── metrics/         # Refusal & Erasure scorers
│   ├── models/          # SOTA T2I/I2I model wrappers
│   └── prompt_generation/ # LLM-driven expansion logic
├── scripts/             # Professional CLI & Plotting 
│   ├── run_audit.py     # Main Entry Point
│   ├── setup_datasets.sh # Dataset hook utility
│   └── survey_app/      # Premium Human Survey UI
├── data/
│   ├── raw/             # Base prompts (OVERT-aligned)
│   └── external/        # FFHQ & COCO hooks
├── figs/                # Publication-ready assets
└── experiments/         # Evaluation result cache
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run a Unified Audit
Run a bias audit on FLUX.2 using gpt-oss-20B for dynamic expansion:
```bash
python scripts/run_audit.py \
    --model flux-2-dev \
    --mode t2i \
    --samples 100 \
    --llm gpt-oss-20b
```

---

## 📝 Citation

```bibtex
@inproceedings{acrb2026,
  title={ACRB: Evaluating Attribute-Conditioned Refusal Bias in Unified Generative Pipelines},
  author={Anonymous},
  booktitle={IJCAI-ECAI},
  year={2026}
}
```

## 📜 License
Research Use Only. See individual model licenses for generation outputs.
