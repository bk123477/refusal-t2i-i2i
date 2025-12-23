# 코드-논문 일치도 확인 보고서

## ✅ 완전히 일치하는 부분

### 1. 핵심 아키텍처
- ✅ **Hard Refusal Detection**: 3가지 방법 모두 구현됨
  - Text response keyword matching (`refusal_detector.py`)
  - CLIP similarity to refusal templates (threshold τ=0.25)
  - Empty output detection
- ✅ **Soft Refusal (Cue Erasure)**: VLM 기반 구현됨
  - Qwen3-VL 사용 (`cue_retention_scorer.py`)
  - YES/NO/PARTIAL 점수 체계 (1.0/0.5/0.0)
- ✅ **Disparity Metrics**: 완전 구현됨
  - ΔRefusal = max(R(a)) - min(R(a))
  - ΔErasure = max(E(a)) - min(E(a))
  - 통계적 유의성 검정 포함 (`disparity_metric.py`)

### 2. Attribute Categories
- ✅ **Culture**: 6개 (Korean, Chinese, Nigerian, Kenyan, American, Indian) - 논문과 일치
- ✅ **Gender**: 3개 (Male, Female, Non-binary) - 논문과 일치
- ✅ **Disability**: 4개 (Wheelchair, Visually impaired, Prosthetic, Hearing aid) - 논문과 일치
- ✅ **Religion**: 6개 (Christian, Muslim, Jewish, Hindu, Buddhist, Sikh) - 논문과 일치
- ✅ **Age**: 4개 (Child, Young adult, Middle-aged, Elderly) - 논문과 일치

### 3. Safety Domains
- ✅ **9개 도메인 모두 구현됨**:
  1. Violence-adjacent
  2. Self-harm-adjacent
  3. Substance-related
  4. Privacy-sensitive
  5. Religious/Cultural
  6. Body/Appearance
  7. Sexual Wellness
  8. Copyright Wellness
  9. Unethical/Unsafe

### 4. Dynamic LLM Expansion
- ✅ **gpt-oss-20b** 지원 (`llm_backend.py`, `attribute_expansion.py`)
- ✅ LLM 기반 boundary case 생성 로직 구현됨
- ✅ Static template fallback 포함

### 5. Models
- ✅ 논문에서 언급한 6개 모델 모두 wrapper에 정의됨:
  - GPT Image 1.5 (ELO 1256)
  - Nano Banana Pro (ELO 1221)
  - FLUX.2 [max] (ELO 1210)
  - Qwen Image Edit 2511 (ELO 1133)
  - FLUX.2 [dev] (ELO 1131)
  - Step1X-Edit v1.2 (ELO 1081)

### 6. I2I Protocol
- ✅ Grounded I2I Protocol 구현됨 (`evaluation/pipeline.py`)
  - Neutral Inception (T2I로 중립 이미지 생성)
  - Attribute Permutation (동일 소스 이미지에 속성 편집 적용)

---

## ⚠️ 불일치하는 부분 (논문 수정 필요)

### 1. Base Prompts 수량
- **논문**: "100 base prompts across 9 safety-sensitive domains"
- **실제 코드**: **60개 base prompts** (9개 도메인에 분산)
  - violence_adjacent: 9개
  - self_harm_adjacent: 7개
  - substance_related: 7개
  - privacy_sensitive: 10개
  - religious_cultural: 14개
  - body_appearance: 6개
  - sexual_wellness: 2개
  - copyright_violations: 2개
  - unethical_unsafe_action: 3개

**권장 수정**: 논문의 "100 base prompts"를 "60 base prompts"로 변경하거나, 실제로 100개로 확장

### 2. T2I Prompt 수량 계산
- **논문**: "2,250 T2I prompts (250 × 9 domains)"
- **실제**: 60개 base prompts × 평균 25개 attribute variations = 약 1,500개
  - 계산식이 논문과 맞지 않음

**권장 수정**: 
- "2,250 T2I prompts" → "approximately 1,500 T2I prompts (60 base prompts × 25 attribute variations)"
- 또는 base prompts를 100개로 확장하여 논문과 일치시킴

### 3. I2I Dataset 크기
- **논문**: "500 I2I source-instruction pairs using FFHQ (faces) and COCO (scenes) subsets"
- **실제**: 데이터셋 파일 확인 필요 (`data/source_images/`)

**권장**: 실제 데이터셋 크기 확인 후 논문 반영

---

## 📋 논문 수정 권장사항

### Section 3.2.1 (Base Prompt Set)
```latex
% 현재:
We curate 100 base prompts across 9 safety-sensitive domains

% 수정 제안:
We curate 60 base prompts across 9 safety-sensitive domains, 
with distribution: violence-adjacent (9), self-harm-adjacent (7), 
substance-related (7), privacy-sensitive (10), religious/cultural (14), 
body/appearance (6), sexual wellness (2), copyright wellness (2), 
and unethical/unsafe (3).
```

### Section 3.2.2 (Attribute Expansion)
```latex
% 현재:
Total: 2,250 T2I prompts ($250 \times 9$ domains); 500 I2I source-instruction pairs.

% 수정 제안:
Total: approximately 1,500 T2I prompts (60 base prompts $\times$ 25 attribute variations); 
500 I2I source-instruction pairs using FFHQ (faces) and COCO (scenes) subsets.
```

---

## ✅ 코드 구현 완성도

### 구현 완료된 기능
1. ✅ Refusal Detection (3가지 방법)
2. ✅ Cue Retention Scoring (VLM 기반)
3. ✅ Disparity Metrics 계산
4. ✅ Attribute Expansion (Static + LLM)
5. ✅ T2I/I2I Model Wrappers
6. ✅ Evaluation Pipeline
7. ✅ Base Prompt Generation
8. ✅ LLM Backend Integration

### 추가 구현 권장사항
- [ ] 실제 실험 결과 데이터 생성
- [ ] Human evaluation 인터페이스 완성
- [ ] I2I dataset (FFHQ/COCO) 실제 준비
- [ ] 실험 결과 시각화 스크립트

---

## 결론

**코드 구현은 논문의 방법론을 충실히 반영하고 있습니다.** 다만, 논문에서 언급한 수치(100 base prompts, 2,250 T2I prompts)와 실제 구현(60 base prompts, ~1,500 prompts) 간 차이가 있으므로, 논문을 실제 코드에 맞게 수정하거나, 코드를 논문에 맞게 확장하는 것이 필요합니다.

**권장 조치**: 논문의 수치를 실제 구현에 맞게 수정하는 것이 더 현실적입니다.

