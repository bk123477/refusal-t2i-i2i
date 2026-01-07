# ACRB: I2I Refusal Bias Framework

**I2I-Focused Version - January 2025**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/seochan99/refusal-t2i-i2i)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Image-to-Image (I2I) 생성 모델에서의 속성-조건화된 refusal bias를 감사하는 프레임워크입니다. ACRB는 하드 refusal (명시적 차단)과 소프트 refusal (조용한 cue erasure)을 모두 측정합니다.

## Key Research Questions

- I2I 모델에서 특정 문화/인종/장애 관련 편집 요청이 더 자주 거부되는가?
- 거부되지 않은 경우에도 요청된 속성이 조용히 삭제되거나 대체되는가?
- 이러한 패턴이 모델/공급업체에 따라 어떻게 다른가?

## Framework Architecture

```
Stage I: 프롬프트 준비
├── OVERT/WildGuardMix 기반 base prompts
└── ACRB 속성 확장 (culture, gender, disability, religion, age)

Stage II: I2I 생성
├── 소스 이미지: FFHQ/COCO 데이터셋
└── I2I 모델: GPT-Image, Imagen 3, FLUX.2, Step1X-Edit, Qwen-Edit

Stage III: 이중 메트릭 평가
├── 하드 refusal 검출 (CLIP 기반)
├── 소프트 refusal 스코어링 (VLM 앙상블)
└── Disparity 메트릭: Δ_refusal, Δ_erasure
```

## 설치

### 사전 요구사항

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (로컬 모델용)
- CUDA 지원 PyTorch

### 설정

```bash
# 저장소 클론
git clone https://github.com/seochan99/refusal-t2i-i2i.git
cd refusal-t2i-i2i

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력
```

## 빠른 시작

### Python Library 사용

```python
from acrb import ACRBPipeline, ACRBConfig

# Configure pipeline (I2I only)
config = ACRBConfig(
    model_name="flux-2-dev",
    max_base_prompts=100,
    i2i_source_images_dir="data/raw/ffhq",
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
```

## Supported I2I Models

| Model | Provider | Type | Notes |
|-------|----------|------|-------|
| GPT-Image 1.5 | OpenAI | Closed | Conservative policy |
| Imagen 3 | Google | Closed | Moderate policy |
| Seedream 4.5 | ByteDance | Closed | Regional variant |
| FLUX.2 [dev] | Black Forest Labs | Open | Permissive policy |
| Step1X-Edit | StepFun | Open | I2I specialist |
| Qwen-Image-Edit-2511 | Alibaba | Open | Regional variant |

## Evaluation Metrics

### Hard Refusal Detection
- CLIP-based similarity to refusal templates
- Text keyword matching for policy messages
- Empty output detection

### Soft Refusal (Cue Erasure)
- Multi-VLM ensemble (Qwen3-VL + Gemini 3 Flash)
- Confidence-weighted voting for attribute retention
- Three-state classification: Retained / Erased / Substituted

### Disparity Metrics
- Δ_refusal = max_a R(a) - min_a R(a)
- Δ_erasure = max_a E(a) - min_a E(a)

## 프로젝트 구조

```
acrb/                    # Core library
├── pipeline.py          # Main I2I pipeline
├── config.py            # Configuration
├── models/              # I2I model wrappers
├── metrics/             # Refusal detection + cue retention
├── analysis/            # Statistical analysis
├── preprocessing/       # Visibility controls
└── vlm/                 # VLM backends

data/
├── prompts/             # ACRB prompts
│   ├── acrb_expanded/   # Expanded prompts
│   └── overt_base/      # OVERT base prompts
└── raw/                 # Source images (FFHQ/COCO)

scripts/                 # Utility scripts
docs/                    # Documentation
```

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Experiment Guide](docs/EXPERIMENT_README.md)

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub.
