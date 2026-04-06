# Session Postmortem — 2026-04-05

> **Document ID**: PM-2026-04-05
> **ISO Reference**: ISO/IEC 42001:2023 (A.9.3), ISO/IEC 29119:2021
> **Statut**: Final
> **Classification**: Interne
> **Auteur**: Claude Opus 4.6
> **Severity**: CRITICAL — project architecture assumption invalidated

---

## Executive Summary

This session uncovered four findings that fundamentally change the project's
trajectory. The most critical: **Gemma 3 1B IT cannot perform RAG grounding
on French regulatory text**. All prior generation metrics (cited_pct) are
invalidated — the regex was catching document/page patterns in hallucinated
text, not genuine citations. The retrieval pipeline is functional; the
generation model is the bottleneck.

Additionally, the recall@5 metric was inflated by 8pp due to a max_k
mismatch between the measurement script and the production search code.

---

## Timeline

| Time | Event | Commit |
|------|-------|--------|
| T+0h | Discovered max_k=10 vs max_k=5 mismatch between recall.py and precompute_retrieval.py | — |
| T+0.5h | Aligned recall.py to max_k=5, new baseline: 55.4% (was 63.4%) | 9c17a52 |
| T+1h | Prepared SFT v5 eval kernel v3 with new retrieval contexts (targeted rows DB, max_k=5) | 7ac4285 |
| T+1.5h | SFT v5 kernel v3 completed: cited_pct=57.4% (was 60.1%), gate still PASS | — |
| T+2h | First HHEM kernel push — transformers 5.0 incompatibility | c0addf9 |
| T+3h | Second HHEM push — patched all_tied_weights_keys | fdbe732 |
| T+3.5h | Third HHEM push — downgraded transformers to 4.46.3 | 8aa4256 |
| T+4h | Fourth HHEM push — patched None dict copy | f427927 |
| T+5h | HHEM results: sanity check PASS but real data mean=0.071, 99% red, no hit/miss discrimination | — |
| T+5.5h | HHEM evaluated and REJECTED for FR regulatory text | 3530c02 |
| T+6h | Base model diagnostic kernel launched to isolate SFT vs model capability | aa2ba50 |
| T+7h | Base model results: 57.7% cited_pct — nearly identical to SFT v5 (57.4%) | — |
| T+8h | Human inspection of 34 responses: ALL garbage — hallucinations, fake URLs, football references | — |
| T+8.5h | Critical finding confirmed: 1B model cannot do RAG grounding on FR regulatory prose | — |

---

## Finding 1: max_k=10 Inflated Recall by 8pp

### What happened

`recall.py` used `max_k=10` while `precompute_retrieval.py` (production search)
used `max_k=5`. The recall measurement was operating in a more generous regime
than the actual production pipeline.

### Root cause

When `max_k=10`, the search retrieves 10 results. `_inject_neighbor_tables()`
and parent-children regrouping operated on contexts at ranks 6-10, pulling in
additional table and prose matches that inflated the page coverage metric.
The production pipeline only ever sees 5 results.

### Impact

| Metric | Old (max_k=10) | New (max_k=5) | Delta |
|--------|----------------|---------------|-------|
| recall@5 | 63.4% (189/298) | **55.4% (165/298)** | **-8.0pp** |

The 63.4% figure reported in all prior sessions, CLAUDE.md, and design specs
was **not production-realistic**. The true baseline is 55.4%.

### Action taken

- `recall.py` aligned to `max_k=5` (commit 9c17a52)
- New baseline committed and verified

### Lesson

Measurement code and production code MUST use identical parameters. Any
discrepancy between the eval harness and the deployed search produces
misleading metrics. This is a basic ISO 29119 requirement (test environment
= production environment).

---

## Finding 2: SFT v5 Eval on New Retrieval Contexts

### What happened

Kernel v3 evaluated SFT v5 using the latest retrieval contexts (targeted rows
DB, max_k=5 alignment). The cited_pct dropped from 60.1% (old contexts) to
57.4% (new contexts).

### Results

| Metric | v2 (old contexts) | v3 (new contexts) | Delta |
|--------|-------------------|-------------------|-------|
| SFT v5 cited_pct | 60.1% | 57.4% | -2.7pp |
| Gate threshold | >56.7% | >56.7% | — |
| Gate status | PASS | PASS | — |
| Hit cited_pct | 51.9% | 51.2% | -0.7pp |
| Miss cited_pct | 72.6% | 65.2% | -7.4pp |
| Abstain_pct | 18.5% | 18.1% | -0.4pp |

Per-question responses saved for the first time (eval_1b_sft_v5_responses.jsonl),
enabling downstream HHEM analysis.

### Impact

The gate remains PASS but the margin is thin (57.4% vs 56.7% threshold). More
importantly, this metric is later invalidated by Finding 4 (regex catches
patterns in hallucinated text).

---

## Finding 3: HHEM-2.1-Open Unusable for FR Regulatory Text

### What happened

HHEM-2.1-Open (Vectara, T5-base) was identified as the P0 offline faithfulness
metric in `GENERATION_EVAL_METHODOLOGY.md`. Four kernel pushes were required to
get it running due to transformers 5.0 incompatibility on Kaggle.

### Kernel iteration log

| Push | Issue | Fix |
|------|-------|-----|
| 1 (c0addf9) | Clean implementation, CPU mode | transformers 5.0 breaks HHEM internals |
| 2 (fdbe732) | `all_tied_weights_keys` attribute missing | Monkey-patch T5 class |
| 3 (8aa4256) | Deeper incompatibility | Downgrade to transformers==4.46.3 |
| 4 (f427927) | `_tied_weights_keys` returns None (not dict) | Patch to return empty dict |

### Results

**Sanity check** (5 short FR premise/hypothesis pairs):

| Pair | Expected | HHEM score | Verdict |
|------|----------|------------|---------|
| Factual (premise entails hypothesis) | High | 0.54 | Correct |
| Contradiction (premise contradicts) | Low | 0.03 | Correct |
| Partial support | Medium | 0.19 | Correct |
| Unrelated | Low | 0.05 | Correct |
| Synonym equivalence | High | 0.42 | Correct |

Sanity check PASS — HHEM discriminates correctly on short, simple FR text.

**Real data** (298 questions, SFT v5 responses + retrieval contexts):

| Segment | Count | Mean | Median | Green (>0.5) | Red (<0.2) |
|---------|-------|------|--------|-------------|-----------|
| Global | 298 | 0.071 | 0.048 | 1 (0.3%) | 295 (99.0%) |
| Hit | 166 | 0.069 | 0.048 | 0 (0.0%) | 164 (98.8%) |
| Miss | 132 | 0.073 | 0.050 | 1 (0.8%) | 131 (99.2%) |

- Mean 0.071 (gate target was >= 0.85)
- 99% red — effectively a flat-line score
- Hit mean (0.069) is approximately equal to miss mean (0.073) — **no discrimination**
- Runtime: 2669 seconds (~44 min) for 298 questions on CPU

### Root cause

HHEM-2.1-Open is trained on English summarization tasks (NLI-style
premise/hypothesis classification). French regulatory prose with legal
terminology, nested article references, and domain-specific vocabulary falls
outside its training distribution. The model cannot parse the semantic
relationship between a regulatory context paragraph and a generated answer
in French.

### Decision

**HHEM evaluated and REJECTED** (ISO 42001 A.9.3 compliant: the metric was
evaluated, documented, and rejected with evidence — not skipped).

HHEM is **not** a viable faithfulness metric for this project. The gap
identified in `GENERATION_EVAL_METHODOLOGY.md` (TB-04 RAGAS Faithfulness
>= 0.85) remains open.

### Lesson

Offline evaluation models advertised as "multilingual" (FR/EN/DE) may only
support surface-level language understanding. Domain-specific evaluation on
regulatory text requires domain-specific evaluation models or human judges.
Always run a sanity check AND a real-data check before trusting any metric.

---

## Finding 4 (CRITICAL): Gemma 3 1B Cannot Do RAG Grounding

### What happened

After HHEM failed to discriminate, human inspection of the 34 human-authored
questions revealed that **ALL responses are garbage**. Both SFT v5 and the
base Gemma 3 1B IT produce hallucinated text that bears no relationship to
the retrieval context provided.

### Evidence

**Base model diagnostic** (kernel aa2ba50, same contexts as SFT v5):

| Metric | SFT v5 | Base 1B | Delta |
|--------|--------|---------|-------|
| cited_pct | 57.4% | 57.7% | +0.3pp (within noise) |
| hit cited_pct | 51.2% | 50.6% | -0.6pp |
| miss cited_pct | 65.2% | 66.7% | +1.5pp |
| abstain_pct | 18.1% | 11.1% | -7.0pp |

SFT v5 and base produce **nearly identical** cited_pct despite months of
training work. The SFT made the model abstain more (+7pp) but did not
improve actual citation behavior.

**Human inspection of 34 responses** (human_eval_34q.csv):

Observed pathologies in response previews:

1. **Football references**: "la Federation Francaise de Football (FFP)" in a
   chess regulation response
2. **Fake URLs**: "https://www.federation-des-federations-de-jeu-en-france.fr/
   actualites/championat-franc_rapide-et-blitz" — completely invented
3. **Invented organizational structures**: "La direction de la Federation
   internationale des Echecks et la direction de la France Francaise des
   Eschecs" — garbled entity names
4. **Generic bureaucratic filler**: "Cette decision est prise en fonction de
   la situation generale de la Federation" — means nothing, answers nothing
5. **Wrong domain content**: table formatting from unrelated documents mixed
   into answers about Elo calculations
6. **Repetitive template text**: "Les joueurs doivent etre informes de toute
   modification apportee aux regles" — generic statement unrelated to question

Of 34 human questions, **0 received a useful, faithful, cited response**.

**But the retrieval contexts are correct.** Verified: the correct documents,
correct pages, and relevant content ARE in the top-5 retrieval results. The
model receives the right information and ignores it.

### Root cause

**The 57.4% cited_pct is a mirage.** The regex-based citation detection
(`SOURCE_PATTERNS` + page patterns) catches document names and page numbers
that appear in the hallucinated text by coincidence or post-rationalization.

This is precisely the finding from Wallat et al. (ICTIR 2025, Best Paper
Honorable Mention): **up to 57% of citations lack faithfulness** — the model
responds from parametric memory then post-rationalizes by finding
superficially matching citation patterns.

Our empirical confirmation:
- cited_pct (regex) = 57.4% — appears healthy
- Human inspection = 0/34 useful responses — completely broken
- The gap between these two numbers IS the post-rationalization phenomenon

**Why 1B fails at RAG grounding on FR regulatory text:**

1. **Model capacity**: 1B parameters is insufficient for instruction-following
   on domain-specific French text. The model cannot reliably extract and
   reformulate information from a provided context.

2. **Parametric memory dominance**: The model's parametric knowledge (from
   pretraining) overrides the context window. When asked about "FFE" it
   generates plausible-sounding but incorrect text from memory instead of
   grounding in the provided passages.

3. **French regulatory vocabulary**: Legal/administrative French with nested
   references (articles, sous-articles, alineas) requires precise parsing
   that a 1B model cannot reliably perform.

4. **Context length vs attention**: With 5 retrieval contexts + prompt
   instructions, the effective context is 1000-2000 tokens. The 1B model
   cannot maintain attention over this span to extract relevant information.

### Impact

| What is invalidated | Why |
|---------------------|-----|
| SFT v5 gate PASS (60.1%, 57.4%) | Regex metric measures noise, not faithfulness |
| All generation eval baselines (24.8% to 60.1%) | Same regex metric, same problem |
| TAPT sweep finding (ep1 = 46.2% best) | Same metric — TAPT may not have helped at all |
| SFT v1-v4 comparisons | Same metric — all noise |
| Prompt v2 improvements | Cannot validate without working generation |
| ADR-001 gate rollback logic | Gate triggered on a non-functional metric |

**What is NOT invalidated:**

| What remains valid | Why |
|-------------------|-----|
| Retrieval pipeline (recall@5 = 55.4%) | Measured by page-level GS matching, independent of generation |
| Targeted rows, gradient intent, RRF | Search infrastructure works correctly |
| SFT v5 training methodology (RAFT) | Training data is sound; model capacity is the issue |
| 270M postmortem (model too small) | Confirmed — and 1B is ALSO too small |
| Corpus quality (28 PDFs, chunks, tables) | Independent of generation model |

### The bottleneck

```
BEFORE this session:
  Believed bottleneck: Retrieval (55-63% recall, gate R1 FAIL at 70%)
  Believed status: Generation OK (57-60% citations, gate PASS)

AFTER this session:
  Actual bottleneck: GENERATION MODEL (1B too small for FR regulatory RAG)
  Actual status: Retrieval works (55.4% is honest, contexts are correct)
                 Generation is broken (0/34 useful responses despite correct contexts)
```

---

## Actions Taken This Session

| # | Action | Commit | Status |
|---|--------|--------|--------|
| 1 | Aligned recall.py max_k=5 | 9c17a52 | DONE |
| 2 | SFT v5 eval v3 with new contexts + per-question responses | 7ac4285 | DONE |
| 3 | HHEM kernel (4 iterations) | c0addf9, fdbe732, 8aa4256, f427927 | DONE |
| 4 | HHEM evaluated and rejected | 3530c02 | DONE |
| 5 | Base model diagnostic kernel | aa2ba50 | DONE |
| 6 | Human inspection of 34 responses | — | DONE |

---

## Lessons Learned

### 1. Never trust a proxy metric without human validation

cited_pct (regex) was treated as the primary generation metric for 6 sessions.
It was never cross-validated against human judgment until this session. The
ICTIR 2025 finding (57% post-rationalized) was documented in
`GENERATION_EVAL_METHODOLOGY.md` but treated as a theoretical caveat rather
than an actionable risk.

**Rule**: Any metric used as a gate MUST be validated against human judgment
on >= 20 samples before being trusted for decisions.

### 2. Measurement code must mirror production code exactly

The max_k=10 vs max_k=5 mismatch persisted for multiple sessions. A simple
assertion or shared constant would have prevented 8pp of inflated recall.

**Rule**: Eval harness parameters MUST be imported from the same constants
as the production search code. No independent parameter definitions.

### 3. "Multilingual support" does not mean "domain support"

HHEM-2.1 claims FR/EN/DE support. It discriminates correctly on simple French
pairs. But regulatory French with nested legal references defeats it entirely.
The same likely applies to other "multilingual" evaluation models.

**Rule**: Always test evaluation models on representative real data, not just
sanity checks, before integrating into the pipeline.

### 4. Small models cannot reliably do RAG grounding

The project hypothesis was that a 1B model with SFT (RAFT-style citations)
could learn to ground its responses in retrieved context. This is falsified.
Both base and SFT v5 produce garbage. The SFT training taught the model to
abstain more (+7pp) but not to actually read and cite the context.

This aligns with the broader finding from the 270M postmortem: sub-3B models
lack the instruction-following capacity for complex RAG with French
regulatory text.

### 5. Generation quality should have been validated FIRST

The project spent 3+ months optimizing retrieval (chantiers 3, 5) and
training generation models (chantiers 4, 4a) without ever performing a
serious human quality check on the end-to-end output. A 30-minute inspection
of 10 responses at any point would have revealed the generation problem.

**Rule**: Before optimizing any component, verify the end-to-end output
with human inspection. Optimize the actual bottleneck, not the measurable one.

---

## Revised Project State

### What works

- **Corpus**: 28 PDFs, 1117 children, 117 tables, 45 targeted rows — sound
- **Retrieval**: recall@5 = 55.4% (honest, max_k=5) — functional, improvement possible
- **Search infrastructure**: 5-way RRF, gradient intent, adaptive-k — correct
- **SFT v5 training data**: 2142 RAFT answers, 95.1% valid citations — reusable for larger models

### What does not work

- **Generation (Gemma 3 1B)**: Cannot ground responses in FR regulatory context
- **Generation (Gemma 3 270M)**: Previously confirmed non-functional (postmortem)
- **cited_pct metric**: Measures noise, not faithfulness
- **HHEM-2.1**: Cannot evaluate FR regulatory text

### Corrected metrics

| Metric | Old (inflated) | New (honest) | Notes |
|--------|---------------|-------------|-------|
| recall@5 | 63.4% | **55.4%** | max_k=5 alignment |
| SFT v5 cited_pct | 60.1% | **57.4%** (but invalid) | New contexts + metric invalidated |
| Generation quality | "Gate PASS" | **0/34 useful** | Human inspection |
| Faithfulness | Not measured | **Not measurable offline** | HHEM rejected |

---

## Next Steps

### Immediate (before next development session)

1. **Update CLAUDE.md** with corrected metrics and generation finding
2. **Update model_card.json** to reflect generation model status
3. **Document ADR-002**: Gemma 3 1B generation gate FAIL, same as 270M

### Strategic options (requires decision)

| Option | Model | RAM | Feasibility | Risk |
|--------|-------|-----|-------------|------|
| A | Gemma 3n E2B (2B effective) | ~2 GB | LiteRT-LM native, Kaggle T4 | Exceeds 500MB spec, may still be too small |
| B | Ministral 3B | ~3 GB | Apache 2.0, FR native, LLaMA.cpp | No LiteRT, larger RAM |
| C | Qwen3 1.7B | ~1.5 GB | Apache 2.0, strong benchmarks | LiteRT unconfirmed |
| D | Cloud API fallback | 0 MB local | Works immediately | Violates offline requirement |
| E | Retrieval-only mode (no generation) | 0 MB | Ship now | Reduced functionality |

**Recommendation**: Option A (Gemma 3n E2B) is the most promising — it is
mobile-first, LiteRT-LM native, and 2B effective parameters may cross the
threshold for FR regulatory RAG. The 500MB RAM spec (VISION.md) must be
relaxed to ~2GB. This should be validated with a quick human eval (10
questions, manual inspection) BEFORE any training investment.

### Eval methodology

The project needs a generation eval that actually works:

1. **Human evaluation** remains the only reliable method (34 questions, 3
   binary scores: useful/faithful/cited)
2. **Larger LLM-as-judge** (if cloud budget available): Gemini Flash or
   Claude Haiku for automated faithfulness scoring
3. **BERTScore** (CamemBERT): French-specific, ~400MB, may work better than
   HHEM for FR text similarity — worth testing
4. **Drop cited_pct as a gate metric** — it has been empirically shown to not
   correlate with actual quality

---

## Artifacts

| Artifact | Path |
|----------|------|
| SFT v5 v3 summary | data/benchmarks/eval_1b_sft_v5_v3/eval_1b_sft_v5_summary.json |
| SFT v5 v3 responses | data/benchmarks/eval_1b_sft_v5_v3/eval_1b_sft_v5_responses.jsonl |
| HHEM results | data/benchmarks/eval_1b_sft_v5_v3/hhem_faithfulness.json |
| Human eval (34q) | data/benchmarks/eval_1b_sft_v5_v3/human_eval_34q.csv |
| Base model v3 summary | data/benchmarks/eval_1b_base_v3/eval_1b_base_summary.json |
| Base model v3 responses | data/benchmarks/eval_1b_base_v3/eval_1b_base_responses.jsonl |
| Recall baseline (honest) | data/benchmarks/recall_baseline.json |

---

## References

- Wallat et al. (ICTIR 2025) — "Correctness is not Faithfulness in RAG Attributions", Best Paper HM
- Vectara HHEM-2.1-Open — https://huggingface.co/vectara/hallucination_evaluation_model
- docs/GENERATION_EVAL_METHODOLOGY.md — eval methodology (gaps now empirically confirmed)
- data/benchmarks/eval_1b_sft_v5_v3/ — all session artifacts
- memory/project_postmortem_270m.md — 270M postmortem (same conclusion, smaller model)

---

*Postmortem ISO 42001/29119 — Pocket Arbiter Project*
*Session 2026-04-05: Generation model capability gap confirmed empirically.*
