Prompt Generation Pipeline (ACRB)
=================================

Purpose
-------
This is the single source of truth for how we generate and validate boundary prompts.
It follows the OVERT-style workflow: generate, filter, deduplicate, sample, and then
pilot-test refusal rates before any large API run.

Sources
-------
- OVERT pipeline summary: `docs/OVERT.txt` (see "OVERT dataset construction pipeline")
- OVERT reference repo: `external/OVERT-main/README.md`

Inputs
------
- Base prompts: `data/raw/base_prompts.json` (OVERT-style benign-but-triggering)
- Attribute taxonomy: `data/raw/attribute_categories.json`
- LLM for boundary expansion: default `gemini-3-flash-preview`
- Sentence model (optional): set `ACRB_SENTENCE_MODEL` or pass `--sentence-model`
- Benign classifier (optional): set `ACRB_BENIGN_MODEL` or pass `--benign-model`

Pipeline Steps (End-to-End)
---------------------------
1) Boundary rephrasing (B):
   - Use LLM to add domain-specific trigger vocabulary while staying benign.
   - Validation gate: domain trigger presence, no cross-domain contamination.

2) Attribute expansion (E):
   - Insert culture/gender/age/etc. cues while preserving the core scenario.
   - Validation gate: attribute marker present, semantic similarity threshold,
     structural similarity (token length drift), benign intent (optional).

3) Constraint validation + logging:
   - Store per-prompt scores (semantic similarity, token diff, benign score).
   - Output a validation report + per-prompt details JSON.

4) Deduplication:
   - Simhash-based base-prompt dedup to avoid semantic near-duplicates.

5) Base-level sampling:
   - Sample balanced base prompts across domains (keeps all expansions).

6) Boundary pilot check:
   - Run a small T2I batch to estimate refusal rate.
   - Target refusal rate band: adjust prompts if too low or too high.

7) Freeze curated prompt set:
   - Only curated prompts are used for full experiments.

Recommended Commands
--------------------
1) Generate prompts (boundary + attributes, validation on):
```
python3 scripts/design_prompts.py \
  --num-base 50 \
  --llm gemini-3-flash-preview \
  --sentence-model /path/to/sbert \
  --benign-validation \
  --benign-llm gemini-3-flash-preview \
  --max-llm-attempts 1
```

2) Validate + dedup + sample + export curated set:
```
python3 scripts/validate_prompt_constraints.py \
  --expanded data/prompts/expanded_prompts.json \
  --sentence-model /path/to/sbert \
  --benign-validation \
  --benign-llm gemini-3-flash-preview \
  --dedup \
  --max-base-prompts 100 \
  --write-clean data/prompts/expanded_prompts.curated.json \
  --details data/prompts/validation_details.json
```

3) Pilot boundary refusal rate (small sample):
```
python3 scripts/pilot_boundary.py \
  --model flux-2-dev \
  --prompts data/prompts/expanded_prompts.curated.json \
  --samples 25
```

Artifacts Produced
------------------
- `data/prompts/expanded_prompts.json`: raw expansions
- `data/prompts/validation_report.json`: aggregate validation stats
- `data/prompts/validation_details.json`: per-prompt validation scores
- `data/prompts/expanded_prompts.curated.json`: curated prompt set
- `experiments/pilot/pilot_boundary.json`: pilot refusal-rate report

Notes
-----
- sBERT similarity is required for paper-grade semantic invariance claims.
- Benign validation is required for paper-grade "boundary but safe" claims.
- The curated prompt file is the only input for full T2I/I2I evaluations.
