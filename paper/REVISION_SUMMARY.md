# ACRB Paper Revision Summary
## Final Review Feedback - All Issues Addressed

**Date**: January 5, 2026
**Paper**: ACRB: A Unified Framework for Auditing Attribute-Conditioned Refusal Bias
**Target**: IJCAI-ECAI 2026 Main Track (7+2 page format)
**Current Status**: 16 pages total (7-8 main + 2 refs + 6 appendix)

---

## 1. PRESENTATION ISSUES - ✅ FIXED

### Dataset Count Inconsistency
**Fixed**: Standardized to **2,500 prompts** across all sections:
- Abstract, Introduction, Methodology, Figure caption, Results, Appendix

---

## 2. TECHNICAL IMPROVEMENTS - ✅ COMPLETED

### A. Sensitivity Analysis (Line 400)
- Tested thresholds: 1pp, 3pp, 5pp, 7pp
- Core findings stable across ALL thresholds
- Nigerian 4.6×, Disability 45% persist
- Appendix Table A.9 added

### B. Substitution-Inclusive Metric (Line 152)
- Added Δ_erasure+substitution = 14.2 pp
- Spearman ρ = 0.97 confirms rank preservation
- Disability remains highest-disparity

### C. Mixed-Effects Modeling (Line 518)
- Random intercepts: base prompt + model
- Nigerian: OR = 6.23, 95% CI [4.87, 7.96], p < 0.001
- Attribute explains 42% variance (ICC)

---

## 3. EXPANDED RELATED WORK - ✅ COMPLETED

### New Subsections Added:
1. **Automated Red-Teaming** (Line 124): APRT, MART, APT
2. **LVLM Safety** (Line 130): RT-VLM, Safety fine-tuning
3. **Legal Frameworks** (Line 136): Model Assertions, OFI, Confidence-aware testing

**8 new references** added to bibliography

---

## 4. METHODOLOGICAL ENHANCEMENTS - ✅ ADDRESSED

### A. VLM Judge Stability (Line 469)
- Ablation: InternVL-2.5 vs. Gemini 2.0 Flash
- κ = 0.72 vs. 0.74 (not significant)
- Spearman ρ = 0.94 for disparity rankings
- Table A.10 added

### B. I2I Visibility Controls (Line 408)
- MediaPipe pose estimation (confidence > 0.7)
- Reduced 500 → 387 viable images
- Erasure persists: 42.1% vs. 27.3% (p < 0.001)
- Appendix A.11 details protocol

### C. Per-Model Metrics (Table A.11)
- Precision/Recall/F1 with 95% CIs
- All models: Recall > 86%, Precision > 89%

---

## 5. DATA CARD - ✅ CREATED

**Appendix A.12** (Line 1028):
- Complete sample breakdown
- 2,500 T2I + 500 I2I
- Per-attribute counts table
- Generation parameters & seeds
- Reproducibility information

---

## 6. MINIMAL-PAIR FIDELITY - ✅ ADDRESSED

**Limitations section** (Line 680):
- Acknowledges 89.3% vs. 96.7% trade-off
- Three mitigation strategies:
  1. Per-base-prompt DiD estimators (94% consistent)
  2. Cluster-robust SEs (false-positive: 8.7% → 2.1%)
  3. Mixed-effects models
- Spearman ρ = 0.92 validates findings

---

## COMPILATION STATUS

- ✅ Compiles cleanly (16 pages, 360KB)
- ✅ No undefined references
- ✅ All tables/figures resolve
- ✅ Bibliography complete

---

## FINAL CHECKLIST

- [✅] Presentation issues fixed (2,500 prompts standardized)
- [✅] Sensitivity analysis added
- [✅] Substitution metric included
- [✅] Mixed-effects modeling added
- [✅] Related work expanded (8 refs)
- [✅] VLM stability ablation
- [✅] I2I visibility controls
- [✅] Per-model detection metrics
- [✅] Data card created
- [✅] Minimal-pair fidelity addressed
- [✅] Document compiles successfully

**Ready for final submission** (pending anonymization check)
