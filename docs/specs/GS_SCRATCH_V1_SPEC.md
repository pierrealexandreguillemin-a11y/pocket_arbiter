# Gold Standard Scratch v1.0 - Specification

> **Document ID**: SPEC-GS-SCRATCH-001
> **ISO Reference**: ISO 42001 A.6.2.2 (Provenance), ISO 29119-3 (Test Data), ISO 25010
> **Version**: 1.0
> **Date**: 2026-02-05
> **Statut**: Approuve
> **Classification**: Qualite
> **Auteur**: Claude Opus 4.5
> **Mots-cles**: gold standard, BY DESIGN, evaluation, RAG, echecs, arbitre, FFE

---

## 1. Objet et Portee

Ce document specifie le Gold Standard "Scratch v1" genere BY DESIGN depuis les 1,857 chunks du corpus FR.

### 1.1 Methode BY DESIGN

```
PRINCIPE FONDAMENTAL:
  chunk (INPUT) --> generation --> question (OUTPUT)

  PAS: question --> matching --> chunk (post-hoc)
```

| Attribut | Valeur |
|----------|--------|
| chunk_match_score | 100% (BY DESIGN) |
| chunk_match_method | by_design_input |
| Hallucination possible | 0% (reponse extraite du chunk) |

### 1.2 Fichiers Associes

| Fichier | Description |
|---------|-------------|
| `tests/data/gs_scratch_v1.json` | Gold Standard 584 questions |
| `data/gs_generation/chunk_strata.json` | Stratification corpus |
| `data/gs_generation/validation_report.json` | Rapport validation |
| `scripts/evaluation/annales/generate_real_questions.py` [SUPPRIME 2026-02-27] | Script generation |

---

## 2. Metriques et Distribution

### 2.1 Vue d'Ensemble

| Metrique | Valeur | Cible | Status |
|----------|--------|-------|--------|
| Total questions | 584 | 600-800 | PROCHE |
| Unanswerable | 32.0% | 25-33% | PASS |
| fact_single | 53.9% | <60% | PASS |
| summary | 30.2% | 15-25% | WARNING |
| reasoning | 15.9% | 10-20% | PASS |
| hard difficulty | 32.0% | >=10% | PASS |

### 2.2 Distribution UAEval4RAG (hard_type)

| Categorie | Count | % |
|-----------|-------|---|
| ANSWERABLE | 397 | 68.0% |
| INSUFFICIENT_INFO | 55 | 9.4% |
| AMBIGUOUS | 45 | 7.7% |
| TEMPORAL_MISMATCH | 35 | 6.0% |
| OUT_OF_SCOPE | 24 | 4.1% |
| FALSE_PREMISE | 15 | 2.6% |
| COUNTERFACTUAL | 13 | 2.2% |

### 2.3 Conformite Standards

| Standard | Exigence | Status |
|----------|----------|--------|
| SQuAD 2.0 | 25-33% unanswerable | PASS (32%) |
| UAEval4RAG | 6 categories | PASS (6) |
| Know Your RAG | fact_single <60% | PASS (53.9%) |
| ISO 42001 A.6.2.2 | chunk_id 100% | PASS |
| TRIPLET_GENERATION_SPEC | BY DESIGN | PASS |

---

## 3. Schema JSON

Conforme a `docs/specs/GS_SCHEMA_V2.md` (SPEC-GS-SCH-002).

### 3.1 Groupes (8)

1. **Racine**: id, legacy_id
2. **content**: question, expected_answer, is_impossible
3. **mcq**: original_question, choices, mcq_answer, correct_answer, original_answer
4. **provenance**: chunk_id, docs, pages, article_reference, answer_explanation, annales_source
5. **classification**: category, keywords, difficulty, question_type, cognitive_level, reasoning_type, reasoning_class, answer_type, hard_type
6. **validation**: status, method, reviewer, answer_current, verified_date, pages_verified, batch
7. **processing**: chunk_match_score, chunk_match_method, reasoning_class_method, triplet_ready, extraction_flags, answer_source, quality_score
8. **audit**: history, qat_revalidation, requires_inference

### 3.2 Champs Specifiques BY DESIGN

| Champ | Valeur | Justification |
|-------|--------|---------------|
| `processing.chunk_match_score` | 100 | BY DESIGN = chunk est INPUT |
| `processing.chunk_match_method` | "by_design_input" | Pas de matching post-hoc |
| `validation.method` | "by_design_generation" | Generation depuis chunk |
| `processing.extraction_flags` | ["by_design"] | Flag tracabilite |

---

## 4. Processus de Generation

### 4.1 Pipeline

```
Phase 0: Stratification corpus (1,857 chunks -> strates)
    |
Phase 1: Generation answerable BY DESIGN
    |
Phase 2: Generation unanswerable (6 categories UAEval4RAG)
    |
Phase 3: Validation anti-hallucination (embeddings)
    |
Phase 4: Enrichissement Schema v2.0
    |
Phase 5: Deduplication et equilibrage
    |
Output: gs_scratch_v1.json (584 questions)
```

### 4.2 Quality Gates

| Gate | Seuil | Status |
|------|-------|--------|
| G1-1 chunk_match_score=100 | 100% | PASS |
| G2-1 unanswerable.is_impossible=true | 100% | PASS |
| G3-1 validation_passed | 100% | PASS |
| G4-1 schema_fields>=42 | 43 | PASS |
| G5-2 anchor_independence<0.90 | PASS | PASS |
| G5-3 fact_single<60% | 53.9% | PASS |
| G5-5 unanswerable 25-33% | 32% | PASS |

---

## 5. Validation

### 5.1 Methode

- **BY DESIGN**: Reponse extraite directement du chunk source
- **Deduplication**: Seuil 0.95 (SemHash-style)
- **Anchor independence**: Seuil <0.90

### 5.2 Rapport

Voir `data/gs_generation/validation_report.json`

---

## 6. Limitations Connues

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Questions generees mecaniquement | Style moins naturel | Reformulation future |
| answer_explanation non rempli | Tracabilite reduite | A completer |
| Keywords generiques | Indexation limitee | Enrichissement prevu |

---

## 7. References

| Document | ID |
|----------|-----|
| Gold Standard Specification | SPEC-GS-001 |
| Schema v2.0 | SPEC-GS-SCH-002 |
| Triplet Generation Spec | SPEC-TRIP-001 |
| Plan BY DESIGN | PLAN-GS-SCRATCH-001 |

---

## 8. Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-02-05 | Claude Opus 4.5 | Creation initiale - 584 questions BY DESIGN |

---

*Document ISO 42001/29119/25010 - Pocket Arbiter Project*
