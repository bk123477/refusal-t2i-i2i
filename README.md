# ACRB: Attribute-Conditioned Refusal Bias Framework

**WildGuardMix 기반 버전 - 2025년 1월 업데이트**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/seochan99/refusal-t2i-i2i)
[![Conference](https://img.shields.io/badge/IJCAI-2026-gold.svg)](https://ijcai26.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-WildGuardMix-orange.svg)](https://huggingface.co/datasets/allenai/wildguardmix)

WildGuardMix 데이터셋을 기반으로 하는 통합 프레임워크로, 생성 모델 안전 정렬에서의 속성-조건화된 refusal bias를 감사합니다. ACRB는 Text-to-Image (T2I) 및 Image-to-Image (I2I) 생성 모델에서 하드 refusal (명시적 차단)과 소프트 refusal (조용한 cue erasure)을 모두 측정합니다.

## Key Findings

Evaluation across 2,500 T2I prompts and 500 I2I edits reveals systematic disparities:

- Nigerian cultural markers trigger refusal at 4.6× the rate of US equivalents (16.7% vs. 3.6%, p < 0.001)
- Disability-related cues experience 45% higher erasure rates than neutral baselines (37.1% vs. 25.6%)
- Religious garments are 2.1× more likely to be substituted with generic clothing (28.4% vs. 13.2%)

These patterns persist in benign contexts such as "wedding photography" or "physical therapy," indicating systematic bias rather than legitimate safety concerns.

## Research Contributions

1. I2I-specific refusal benchmark: Evaluation protocols for instruction-based image editing models
2. Dual-metric bias framework: Formalizes Refusal Disparity (Δ_refusal) and Erasure Disparity (Δ_erasure) metrics
3. Dynamic LLM-driven red-teaming: Scalable prompt synthesis using Gemini 3 Flash Reasoning for boundary rephrasing
4. Reproducible evaluation infrastructure: Open-source `acrb` Python library for regulatory compliance auditing
5. Disparate impact evidence: Quantitative documentation for EU AI Act Article 10 and Executive Order 14110 compliance

## Framework Architecture (WildGuardMix 기반)

ACRB는 WildGuardMix 데이터셋을 기반으로 하는 3단계 평가 파이프라인을 구현합니다:

```
Stage I: WildGuardMix 기반 프롬프트 준비
├── WildGuardMix 다운로드 (86K+ benign prompts)
├── LLM → T2I context 변환 ("How to X?" → "Person doing X")
└── ACRB 속성 확장 (culture, gender, disability, religion, age)
    → 확장 가능한 boundary prompts

Stage II: 멀티-모달 생성
├── T2I 모델: FLUX.2, SD 3.5, GPT-Image, Imagen 3, Qwen, 등
└── I2I 모델: GPT-Image, Imagen 3, Qwen, Step1X-Edit, FLUX.2
    → FFHQ/COCO 소스 이미지 기반 평가

Stage III: 이중 메트릭 평가
├── 하드 refusal 검출 (CLIP 기반, τ=0.25)
├── 소프트 refusal 스코어링 (VLM 앙상블)
└── 인간 검증 (κ=0.74 agreement)
    → Disparity 메트릭: Δ_refusal, Δ_erasure
```

## 설치 (WildGuardMix 기반)

### 사전 요구사항

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (로컬 모델용)
- CUDA 지원 PyTorch
- HuggingFace 접근 권한 (WildGuardMix 다운로드용)

### 설정

```bash
# 저장소 클론
git clone https://github.com/seochan99/refusal-t2i-i2i.git
cd I2I-T2I-Bias-Refusal

# 의존성 설치 (WildGuardMix 포함)
pip install -r requirements.txt
pip install datasets huggingface-hub  # WildGuardMix 다운로드용

# 선택: 서버 환경 설정
./setup_server.sh

# WildGuardMix 접근 설정 (필요시)
huggingface-cli login  # API 토큰으로 로그인
```

### 새로운 메인 파이프라인 사용법

```bash
# WildGuardMix 기반 전체 실험 실행
python acrb_pipeline.py --phase all --models flux2 sd35 --num-prompts 500

# 단계별 실행
python acrb_pipeline.py --phase prepare --num-prompts 1000    # 프롬프트 준비
python acrb_pipeline.py --phase experiment --models flux2     # 모델 테스트
python acrb_pipeline.py --phase analyze                       # Bias 분석
```
```

## 빠른 시작 (WildGuardMix 기반)

### 옵션 1: 새로운 메인 파이프라인 실행

```bash
# WildGuardMix 기반 전체 실험 (권장)
python acrb_pipeline.py --phase all --models flux2 sd35 --num-prompts 100

# 빠른 테스트 (50개 프롬프트, 2개 모델, ~1시간)
python acrb_pipeline.py --phase all --models flux2 --num-prompts 50

# 단계별 실행으로 디버깅
python acrb_pipeline.py --phase prepare --num-prompts 1000  # WildGuardMix 준비
python acrb_pipeline.py --phase experiment --models flux2   # 모델 테스트
python acrb_pipeline.py --phase analyze                     # Bias 분석

# 레거시 파이프라인 (호환성 유지)
./run_experiment.sh --model flux2 --samples 100
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

## Supported Models

### Text-to-Image (T2I)
- GPT-Image 1.5 (OpenAI) - Conservative policy
- Imagen 3 (Google) - Moderate policy
- Seedream 4.5 (ByteDance) - Regional variant
- FLUX.2 [dev] (BFL) - Permissive policy
- Qwen-Image-2512 (Alibaba) - Regional variant
- SD 3.5 Large (Stability AI) - Community policy

### Image-to-Image (I2I)
- GPT-Image 1.5 (OpenAI)
- Imagen 3 (Google)
- Qwen-Image-Edit-2511 (Alibaba)
- Step1X-Edit (StepFun) - I2I specialist
- FLUX.2 [dev] (BFL)

## Evaluation Metrics

### Hard Refusal Detection
- CLIP-based similarity to refusal templates (threshold τ = 0.25)
- Text keyword matching for policy messages
- Empty output detection

### Soft Refusal (Cue Erasure)
- Multi-VLM ensemble (Qwen3-VL + Gemini 3 Flash Preview)
- Confidence-weighted voting for attribute retention
- Three-state classification: Retained / Erased / Substituted

### Disparity Metrics
- Δ_refusal = max_a R(a) - min_a R(a)
- Δ_erasure = max_a E(a) - min_a E(a)

Where R(a) is refusal rate and E(a) is erasure rate for attribute a.

## Experimental Results

### Cultural Disparity (RQ1)
Nigerian cultural markers show 4.6× higher refusal rates than US equivalents across all models. Mixed-effects regression confirms statistical significance (β_Nigerian = 1.83, OR = 6.23, p < 0.001).

### Disability Erasure (RQ2)
Disability-related cues experience 45% higher erasure rates (37.1% vs. 25.6% neutral baseline, Cohen's d = 1.34). This soft refusal mechanism operates invisibly, representing a pervasive form of exclusion.

### Domain-Specific Patterns (RQ3)
Refusal disparities concentrate in security-adjacent domains: Nigerian markers in "Unethical/Unsafe" contexts trigger 24.7% refusal vs. 8.0% for US equivalents (3.1× disparity).

### I2I vs. T2I Differences (RQ4)
I2I models exhibit 1.66× lower hard refusal but 1.26× higher soft erasure, suggesting a sanitization strategy that undermines personalization without user-visible errors.

## Regulatory Compliance

ACRB provides standardized methodology for:

- EU AI Act Article 10: Bias mitigation measures and technical documentation
- Biden Executive Order 14110: Algorithmic discrimination assessments for federal AI deployments

The framework operationalizes abstract regulatory requirements through:
1. Standardized disparity metrics with statistically validated thresholds
2. Reproducible evaluation pipelines for API and open-weight models
3. Human-validated automated scoring (κ = 0.74 agreement)

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Installation and basic usage
- [Experiment Guide](docs/EXPERIMENT_README.md) - Detailed experiment documentation
- [Paper](paper/main.pdf) - Full research paper (IJCAI 2026)

## Human Evaluation

A web-based survey application is provided for human validation:

```bash
cd survey-app
npm install
npm run dev
```

The survey validates automated VLM-based metrics with 12 annotators (2 per target culture), achieving 82.7% agreement (Cohen's κ = 0.74).

## Citation

If you use ACRB in your research, please cite:

```bibtex
@inproceedings{acrb2026,
  title={ACRB: A Unified Framework for Auditing Attribute-Conditioned Refusal Bias via Dynamic LLM-Driven Red-Teaming},
  author={Anonymous},
  booktitle={Proceedings of the 36th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```

## Contributing

We welcome contributions. Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OVERT benchmark for base prompt templates
- FFHQ and COCO datasets for I2I source images
- All model providers for API access and open-source releases

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This framework is designed for research and compliance auditing. Please use responsibly and in accordance with model providers' terms of service.
