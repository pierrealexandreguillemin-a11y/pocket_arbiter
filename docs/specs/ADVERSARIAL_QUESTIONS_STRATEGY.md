# SPEC-ADV-V1: Adversarial Questions Strategy

> **Version**: 1.0.0
> **Date**: 2026-01-25
> **Conformance**: ISO 42001 (AI Quality), arXiv:1806.03822 (SQuAD 2.0), arXiv:2412.12300 (UAEval4RAG)

## 1. Motivation

Robust RAG systems must distinguish **answerable** from **unanswerable** questions. Per SQuAD 2.0 research, models that achieve 86% F1 on answerable-only benchmarks drop to 66% when unanswerable questions are introduced—a 20-point gap. Training and evaluation with adversarial questions is essential for production-grade systems.

## 2. Industry Standards

### 2.1 SQuAD 2.0 Categories (Rajpurkar et al., 2018)

| Category | % | Description | Example |
|----------|---|-------------|---------|
| **NEGATION** | 9% | Inserting/removing negation | "What types of pharmacy functions have *never* been outsourced?" |
| **ANTONYM** | 20% | Using opposite meanings | Ask about "decline" when text says "expansion" |
| **ENTITY_SWAP** | 21% | Replace entities/numbers | "4th assessment" when only "3rd assessment" exists |
| **MUTUAL_EXCLUSION** | 15% | Request mutually exclusive info | Ask for "free unconditionally" when conditions exist |
| **IMPOSSIBLE_CONDITION** | 4% | Conditions not satisfied | "After what battle did forces leave for good?" (they returned) |
| **OTHER_NEUTRAL** | 24% | Paragraph doesn't imply answer | Topic-relevant but unanswerable |

### 2.2 UAEval4RAG Categories (2024)

| Category | Description |
|----------|-------------|
| **OUT_OF_SCOPE** | Question outside corpus domain |
| **PARTIAL_INFO** | Information exists but incomplete |
| **FALSE_PREMISE** | Question based on incorrect assumption |
| **MULTI_HOP_IMPOSSIBLE** | Required inference chain unavailable |
| **UNDERSPECIFIED** | Question lacks necessary specificity |

### 2.3 SQuAD2-CR Extensions (Contrast Reasoning)

| Category | Description |
|----------|-------------|
| **VOCABULARY_MISMATCH** | Synonyms/paraphrases not in corpus |
| **FALSE_PRESUPPOSITION** | Presupposes non-existent facts |
| **NUMBER_SWAP** | Different numerical values |

### 2.4 Domain-Specific (Chess Arbitration)

| Category | Description |
|----------|-------------|
| **SAFETY_CONCERNED** | Questions about harmful/illegal actions |

## 3. Pocket Arbiter Adversarial Taxonomy

Unified taxonomy combining industry standards:

```
adversarial:{source}:{category}:{seq}:{hash}

Categories (13 total):
- OUT_OF_SCOPE         # UAEval4RAG - hors domaine
- VOCABULARY_MISMATCH  # SQuAD2-CR - synonymes
- FALSE_PRESUPPOSITION # SQuAD2-CR - presuppose faux
- ENTITY_SWAP          # SQuAD 2.0 - entites modifiees
- ANTONYM              # SQuAD 2.0 - sens oppose
- NEGATION             # SQuAD 2.0 - negation ajoutee
- NUMBER_SWAP          # SQuAD2-CR - nombres modifies
- PARTIAL_INFO         # UAEval4RAG - info partielle
- SAFETY_CONCERNED     # Custom - securite
- MULTI_HOP_IMPOSSIBLE # UAEval4RAG - inference impossible
- FALSE_PREMISE        # UAEval4RAG - premisse fausse
- UNDERSPECIFIED       # UAEval4RAG - sous-specifie
- MUTUAL_EXCLUSION     # SQuAD 2.0 - exclusion mutuelle
```

## 4. Current Adversarial Inventory

### 4.1 Distribution (105 questions from gold_standard_fr.json v5.30)

| hard_type | Count | % |
|-----------|-------|---|
| OUT_OF_SCOPE | 22 | 21.0% |
| VOCABULARY_MISMATCH | 16 | 15.2% |
| FALSE_PRESUPPOSITION | 12 | 11.4% |
| ENTITY_SWAP | 10 | 9.5% |
| ANTONYM | 10 | 9.5% |
| NEGATION | 8 | 7.6% |
| NUMBER_SWAP | 7 | 6.7% |
| PARTIAL_INFO | 4 | 3.8% |
| SAFETY_CONCERNED | 4 | 3.8% |
| MULTI_HOP_IMPOSSIBLE | 3 | 2.9% |
| FALSE_PREMISE | 3 | 2.9% |
| UNDERSPECIFIED | 3 | 2.9% |
| MUTUAL_EXCLUSION | 3 | 2.9% |

### 4.2 Comparison with SQuAD 2.0 Distribution

| Category | SQuAD 2.0 | Pocket Arbiter | Gap |
|----------|-----------|----------------|-----|
| Entity Swap | 21% | 9.5% | -11.5% |
| Antonym | 20% | 9.5% | -10.5% |
| Mutual Exclusion | 15% | 2.9% | -12.1% |
| Negation | 9% | 7.6% | -1.4% |
| Out of Scope | 24%* | 21.0% | -3.0% |

*SQuAD "Other Neutral" mapped to OUT_OF_SCOPE

**Action**: Increase ENTITY_SWAP, ANTONYM, and MUTUAL_EXCLUSION to match SQuAD 2.0 proportions.

## 5. Recommended Ratios

### 5.1 Training Set Ratio

Per SQuAD 2.0 methodology:
- **2:1** answerable-to-unanswerable (training)
- **1:1** answerable-to-unanswerable (evaluation/test)

### 5.2 Target for Pocket Arbiter

Current Gold Standard: 420 questions (v7.3.0)

| Split | Answerable | Unanswerable | Ratio |
|-------|------------|--------------|-------|
| Training | 280 | 140 | 2:1 |
| Evaluation | 70 | 70 | 1:1 |

**Target unanswerable**: ~140 questions (current: 105 pending review)

## 6. Question Creation Guidelines

### 6.1 SQuAD 2.0 Crowdworker Methodology

1. Questions must be **relevant to corpus domain** (chess arbitration)
2. A **plausible answer must exist** in the text (same type as question asks)
3. The question **cannot be answered** based on corpus alone
4. Each question requires **corpus_truth** field documenting why unanswerable

### 6.2 Quality Criteria

Each adversarial question must include:

```json
{
  "id": "adversarial:human:ENTITY_SWAP:001:a1b2c3d4",
  "question": "Selon l'article 7.3 du reglement A01...",
  "is_impossible": true,
  "hard_type": "ENTITY_SWAP",
  "corpus_truth": "A01 n'a pas d'article 7.3 - Art 1.1 existe",
  "plausible_answer": "Les joueurs etrangers peuvent...",
  "expected_docs": ["A01_2025_26_Championnat_de_France.pdf"],
  "validation": {
    "status": "VERIFIED",
    "method": "manual_verification",
    "reviewer": "human"
  }
}
```

### 6.3 Plausible Answer Requirement

Per SQuAD 2.0 findings, ~50% of false positives match the plausible answer. Every unanswerable question MUST include a `plausible_answer` field containing text that:
- Exists in the corpus
- Has the same semantic type as what the question asks
- Would be a reasonable (but wrong) response

## 7. Import/Migration Strategy

### 7.1 Phase 1: Review Existing Questions

Location: `tests/data/adversarial_review.json` (105 questions)

Tasks:
1. Verify each question's `hard_type` classification
2. Add missing `plausible_answer` fields
3. Verify `corpus_truth` against source PDFs
4. Assign new URN-like IDs

### 7.2 Phase 2: Balance Distribution

Create additional questions to match SQuAD 2.0 proportions:
- +15 ENTITY_SWAP questions
- +14 ANTONYM questions
- +17 MUTUAL_EXCLUSION questions
- -10 OUT_OF_SCOPE (over-represented)

Target: ~140 questions with balanced distribution.

### 7.3 Phase 3: Integrate into Gold Standard

Migrate to `gold_standard_annales_fr_v7.json`:
- Set `is_impossible: true`
- Use `adversarial:{source}:{category}:{seq}:{hash}` ID format
- Add to `questions` array
- Update `coverage.adversarial_questions` count

## 8. Evaluation Metrics

### 8.1 SQuAD 2.0 Metrics

- **Exact Match (EM)**: Strict string match
- **F1**: Token-level overlap
- **No-Answer F1**: F1 specifically for unanswerable detection

### 8.2 Extended Metrics for RAG

- **Abstention Rate**: % of unanswerable questions correctly refused
- **Hallucination Rate**: % of unanswerable questions answered confidently
- **Plausible Answer Trap Rate**: % falling for plausible answer

## 9. References

### 9.1 Academic Sources

- [SQuAD 2.0](https://arxiv.org/abs/1806.03822) - Rajpurkar, Jia & Liang (ACL 2018)
- [UAEval4RAG](https://arxiv.org/abs/2412.12300) - Unanswerable Question Evaluation (2024)
- [SQuAD2-CR](https://aclanthology.org) - Contrast Reasoning Extensions

### 9.2 Project Documents

- `docs/GOLD_STANDARD_SPECIFICATION.md` - Main specification
- `docs/specs/GOLD_STANDARD_V6_ANNALES.md` - Schema details
- `tests/data/adversarial_review.json` - Questions pending review

## 10. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-25 | Initial specification |
