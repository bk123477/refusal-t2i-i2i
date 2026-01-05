# Quick Reference: Where to Find Each Change

## Main Paper Changes

### Presentation Fixes (2,400 → 2,500)
- Line 57: Abstract
- Line 68: Introduction (stage I description)
- Line 70: Introduction (evaluation results)
- Line 253: Figure 1 caption
- Line 280: Methodology (equation)
- Line 377: Prompt design section
- Line 630: Discussion section
- Line 666: Conclusion
- Line 769, 776: Appendix summary table

### New Technical Content

**Substitution-Inclusive Metric**
- Line 152-153: After disparity metrics definition
- Introduces Δ_erasure+substitution with Spearman validation

**Sensitivity Analysis for Thresholds**
- Line 400-401: After I2I policy normalization
- References Appendix Table A.9

**Mixed-Effects Regression**
- Line 518-519: Within RQ1 results section
- OR, 95% CI, ICC statistics

**VLM Judge Stability**
- Line 469-470: After cue retention scoring
- References Appendix Table A.10

**I2I Visibility Controls**
- Line 408-409: After grounded I2I protocol
- References Appendix A.11

**Minimal-Pair Fidelity Discussion**
- Line 680-681: In Limitations section
- DiD estimators, cluster-robust SEs

### Expanded Related Work

**Automated Red-Teaming**
- Line 124-128: New subsection
- Cites: samvelyan2024aprt, yu2024mart, chao2024apt

**LVLM Safety Frameworks**
- Line 130-134: New subsection
- Cites: gou2024rtvlm, wang2024lvlm

**Legal/Auditing Frameworks**
- Line 136-140: New subsection
- Cites: black2022fairness, raji2020ofi, oakden2024confidence

---

## Appendix Additions

**Table A.9: Sensitivity Analysis**
- Line 929-962
- Disparity stability across 1pp, 3pp, 5pp, 7pp thresholds

**Table A.10: VLM Judge Stability**
- Line 964-988
- Ensemble configuration comparison

**Table A.11: Per-Model Detection Metrics**
- Line 990-1013
- Precision/Recall/F1 with 95% CIs

**Section A.11: I2I Visibility Protocol**
- Line 1015-1026
- MediaPipe methodology, exclusion criteria

**Section A.12: Data Card**
- Line 1028-1079
- Complete dataset documentation

---

## References Added (references.bib)

Line 219-275: Eight new citations
1. samvelyan2024aprt (APRT)
2. yu2024mart (MART)
3. chao2024apt (APT)
4. wang2024lvlm (VLM safety)
5. gou2024rtvlm (RT-VLM)
6. black2022fairness (Model Assertions)
7. raji2020ofi (OFI)
8. oakden2024confidence (Confidence-aware testing)

---

## File Locations

- **Main paper**: /Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/paper/main.tex
- **References**: /Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/paper/references.bib
- **Compiled PDF**: /Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/paper/main.pdf
- **This summary**: /Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/paper/REVISION_SUMMARY.md
- **Quick reference**: /Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/paper/CHANGES_REFERENCE.md
