# OVERT Benchmark (Reference)

This directory contains the OVERT (Over-Refusal Evaluation on Text-to-Image Models) benchmark, included as a reference for ACRB research.

## Purpose

OVERT provides base prompt templates across 9 safety domains that ACRB uses for boundary rephrasing. This is a reference implementation, not a direct dependency.

## Contents

- `data/`: OVERT prompt datasets (OVERT-full, OVERT-mini, OVERT-unsafe)
- `main.py`: OVERT evaluation script
- `utils/`: Utility functions for OVERT evaluation
- `figs/`: OVERT paper figures

## Usage in ACRB

ACRB adapts OVERT's base prompts through:
1. LLM-driven boundary rephrasing
2. Attribute expansion across demographic dimensions
3. I2I evaluation protocol extension

## Original Source

- Paper: [OVERT: Over-Refusal Evaluation on Text-to-Image Models](https://arxiv.org/abs/2505.21347)
- Dataset: [HuggingFace](https://huggingface.co/datasets/yixiaoh/OVERT)

## Note

This is included for reference only. For full OVERT functionality, please refer to the original repository.



