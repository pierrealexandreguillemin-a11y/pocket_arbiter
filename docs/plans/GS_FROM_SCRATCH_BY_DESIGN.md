# Plan: Gold Standard FROM SCRATCH BY DESIGN

> **Document ID**: PLAN-GS-SCRATCH-001
> **Date**: 2026-02-05
> **Objectif**: Generer un GS conforme aux standards industrie depuis les 1,857 chunks
> **Methode**: BY DESIGN (chunk visible lors generation = INPUT, pas post-hoc)

---

## 1. Vue d'Ensemble

### 1.1 Objectifs Quantitatifs

| Metrique | Cible | Justification |
|----------|-------|---------------|
| Total questions | ~600 | Seuil projet pour significativite statistique |
| Unanswerable (is_impossible=true) | 25-33% | Inspire SQuAD 2.0 train split (~33.4%) |
| fact_single | <60% | Seuil projet pour eviter dominance (cf. Know Your RAG) |
| summary | 15-25% | Seuil projet |
| reasoning | 10-20% | Seuil projet |
| hard difficulty | >=10% | Seuil projet |
| Schema v2.0 fields | 46/46 | GS_SCHEMA_V2.md |

### 1.2 Principes BY DESIGN

```
CRITIQUE: chunk_id = INPUT (pas OUTPUT)

FAUX: question -> matching -> chunk_id (post-hoc)
VRAI: chunk -> generation -> question (BY DESIGN)
```

### 1.3 Quality Gates Bloquantes

Toutes verifiables via script Python avec exit code != 0 si echec.

---

## 2. Pipeline de Generation

### PHASE 0: Stratification du Corpus (PREREQUIS)

**Acteur**: Python script
**Input**: `corpus/processed/chunks_mode_b_fr.json` (1,857 chunks)
**Output**: `data/gs_generation/chunk_strata.json`

**Objectif**: Repartir les chunks en strates pour garantir la diversite.

```python
# Stratification par source/page/type
strata = {
    "LA-octobre2025": [],    # Lois de l'arbitrage
    "R01_*": [],             # Reglements generaux
    "J01_*": [],             # Jeunes
    "Interclubs_*": [],      # Competitions
    "other": []
}

# Quotas par strate (proportionnel)
# Objectif: coverage >= 80% documents
```

**GATE G0-1**: `len(strata) >= 5 and sum(quotas) == target_total`
**GATE G0-2**: `coverage_documents >= 0.80`

---

### PHASE 1: Generation Answerable BY DESIGN

**Acteur**: Claude (via Claude Code)
**Input**: Chunks individuels + context
**Output**: Questions avec chunk_id = INPUT chunk

**Methode**: Pour chaque chunk selectionne, generer 0-3 questions.

```python
# Prompt BY DESIGN
PROMPT = """
CHUNK (ID: {chunk_id}):
{chunk_text}

TACHE: Generer 0 a 3 questions DONT LA REPONSE EST DANS CE CHUNK.

CONTRAINTES:
1. La reponse DOIT etre extractible du chunk (verbatim ou paraphrase proche)
2. Varier le type: factual, procedural, scenario, comparative
3. Varier le niveau cognitif: Remember, Understand, Apply, Analyze
4. Varier la classe de raisonnement: fact_single, summary, reasoning

OUTPUT FORMAT (JSON):
[
  {
    "question": "...",
    "expected_answer": "...",  // Copie exacte du passage chunk
    "reasoning_class": "fact_single|summary|reasoning",
    "cognitive_level": "Remember|Understand|Apply|Analyze",
    "question_type": "factual|procedural|scenario|comparative",
    "difficulty": 0.0-1.0
  }
]

Si le chunk n'est pas propice a des questions (table des matieres, liste vide), retourner [].
"""
```

**Quotas de distribution** (enforces a chaque batch):
- fact_single: 40-50% (pas plus)
- summary: 15-25%
- reasoning: 10-20%
- unanswerable: genere en Phase 2

**GATE G1-1 (BLOQUANT)**: `chunk_match_score == 100` pour chaque question
**GATE G1-2 (BLOQUANT)**: `expected_answer in chunk_text OR semantic_sim >= 0.95`
**GATE G1-3**: `fact_single_ratio < 0.60`
**GATE G1-4**: `questions.endswith('?') == 100%`

---

### PHASE 2: Generation Unanswerable BY DESIGN

**Acteur**: Claude (via Claude Code)
**Input**: Chunks + corpus knowledge
**Output**: Questions impossibles (6 categories UAEval4RAG)

**Methode**: Generer des questions qui SEMBLENT liees au chunk mais NE PEUVENT PAS etre repondues.

```python
# 6 categories UAEval4RAG
UAEval_CATEGORIES = [
    "OUT_OF_SCOPE",           # Sujet hors corpus
    "INSUFFICIENT_INFO",      # Info partielle dans corpus
    "FALSE_PREMISE",          # Question basee sur premisse fausse
    "TEMPORAL_MISMATCH",      # Demande info d'une autre epoque
    "AMBIGUOUS",              # Question ambigue sans contexte
    "COUNTERFACTUAL"          # Scenario hypothetique non documente
]

PROMPT_UNANSWERABLE = """
CHUNK CONTEXT (ID: {chunk_id}):
{chunk_text}

TACHE: Generer 1 question IMPOSSIBLE A REPONDRE avec ce corpus.

CATEGORIES (choisir une):
- OUT_OF_SCOPE: Sujet non couvert (ex: "Quelles sont les regles FIBA?")
- INSUFFICIENT_INFO: Info partielle (ex: "Quel est le salaire d'un arbitre?")
- FALSE_PREMISE: Premisse fausse (ex: "Pourquoi le roque est-il interdit?")
- TEMPORAL_MISMATCH: Autre epoque (ex: "Quelles etaient les regles en 1950?")
- AMBIGUOUS: Question floue (ex: "Comment ca marche?")
- COUNTERFACTUAL: Hypothetique (ex: "Que se passerait-il si le roi pouvait sauter?")

OUTPUT FORMAT (JSON):
{
  "question": "...",
  "hard_type": "OUT_OF_SCOPE|INSUFFICIENT_INFO|...",
  "corpus_truth": "...",  // Ce que dit vraiment le corpus
  "is_impossible": true
}
"""
```

**Quota**: 25-33% du total (cible: 30%)

**GATE G2-1 (BLOQUANT)**: `is_impossible == true` pour 100% questions Phase 2
**GATE G2-2**: `unanswerable_ratio >= 0.25 and <= 0.33`
**GATE G2-3**: `len(set(hard_types)) >= 4` (diversite categories)

---

### PHASE 3: Validation Anti-Hallucination

**Acteur**: Python script + embeddings
**Input**: Questions generees
**Output**: Questions validees ou rejetees

**Methode**: Double verification pour chaque question answerable.

```python
def validate_question(q, chunk_text):
    """
    GATE G3-1: Answer-in-chunk verification
    """
    answer = q["expected_answer"]

    # Test 1: Presence directe
    if answer.lower() in chunk_text.lower():
        return True, "verbatim_match"

    # Test 2: Keyword coverage >= 80%
    keywords = extract_keywords(answer)
    coverage = sum(1 for k in keywords if k in chunk_text) / len(keywords)
    if coverage >= 0.80:
        return True, f"keyword_coverage_{coverage:.2f}"

    # Test 3: Semantic similarity >= 0.90
    sim = cosine_similarity(embed(answer), embed(chunk_text))
    if sim >= 0.90:
        return True, f"semantic_{sim:.2f}"

    return False, "REJECTED_HALLUCINATION"
```

**GATE G3-1 (BLOQUANT)**: `validation_passed == 100%`
**GATE G3-2 (BLOQUANT)**: `0 questions rejected for hallucination`

---

### PHASE 4: Enrichissement Schema v2.0

**Acteur**: Python script
**Input**: Questions validees
**Output**: Questions avec 46 champs complets

**Transformation** vers Schema v2.0 (8 groupes):

```python
def to_schema_v2(q, chunk_id, chunk_meta):
    return {
        "id": generate_id(q),
        "legacy_id": "",

        # GROUP 1: content
        "content": {
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "is_impossible": q.get("is_impossible", False)
        },

        # GROUP 2: mcq (vide pour questions naturelles)
        "mcq": {
            "original_question": q["question"],
            "choices": {},
            "mcq_answer": "",
            "correct_answer": q["expected_answer"],
            "original_answer": q["expected_answer"]
        },

        # GROUP 3: provenance (ISO 42001)
        "provenance": {
            "chunk_id": chunk_id,  # BY DESIGN INPUT
            "docs": [chunk_meta["source"]],
            "pages": chunk_meta.get("pages", []),
            "article_reference": extract_article_ref(chunk_meta),
            "answer_explanation": "",
            "annales_source": None
        },

        # GROUP 4: classification
        "classification": {
            "category": infer_category(chunk_meta),
            "keywords": extract_keywords(q["question"]),
            "difficulty": q["difficulty"],
            "question_type": q["question_type"],
            "cognitive_level": q["cognitive_level"],
            "reasoning_type": infer_reasoning_type(q),
            "reasoning_class": q["reasoning_class"],
            "answer_type": "extractive",
            "hard_type": q.get("hard_type", "ANSWERABLE")
        },

        # GROUP 5: validation (ISO 29119)
        "validation": {
            "status": "VALIDATED",
            "method": "by_design_generation",
            "reviewer": "claude_code",
            "answer_current": True,
            "verified_date": "2026-02-05",
            "pages_verified": True,
            "batch": "gs_scratch_v1"
        },

        # GROUP 6: processing
        "processing": {
            "chunk_match_score": 100,  # BY DESIGN = 100%
            "chunk_match_method": "by_design_input",
            "reasoning_class_method": "generation_prompt",
            "triplet_ready": True,
            "extraction_flags": ["by_design"],
            "answer_source": "chunk_extraction",
            "quality_score": compute_quality_score(q)
        },

        # GROUP 7: audit
        "audit": {
            "history": f"[BY DESIGN] Generated from {chunk_id} on 2026-02-05",
            "qat_revalidation": None,
            "requires_inference": False
        }
    }
```

**GATE G4-1 (BLOQUANT)**: `fields_populated == 46/46` pour chaque question
**GATE G4-2**: `chunk_match_method == 'by_design_input'` pour 100%

---

### PHASE 5: Deduplication et Equilibrage

**Acteur**: Python script + embeddings
**Input**: Questions enrichies
**Output**: Questions dedupliquees avec distributions cibles

**Methode**:

```python
# 1. Deduplication SemHash
from semhash import SemHash

hasher = SemHash(threshold=0.95)
unique_questions = hasher.deduplicate(questions)

# 2. Anchor independence check
for q in unique_questions:
    chunk_text = get_chunk(q["provenance"]["chunk_id"])["text"]
    sim = cosine_similarity(embed(q["content"]["question"]), embed(chunk_text))
    assert sim < 0.90, f"Anchor too similar to positive: {q['id']}"

# 3. Distribution equilibrage
final_questions = balance_distribution(
    unique_questions,
    targets={
        "fact_single": (0.40, 0.50),
        "summary": (0.15, 0.25),
        "reasoning": (0.10, 0.20),
        "unanswerable": (0.25, 0.33),
        "hard_difficulty": (0.10, 1.00)
    }
)
```

**GATE G5-1**: `inter_question_similarity < 0.95` (0 duplicates)
**GATE G5-2 (BLOQUANT)**: `anchor_positive_similarity < 0.90`
**GATE G5-3**: `fact_single_ratio < 0.60`
**GATE G5-4**: `hard_ratio >= 0.10`
**GATE G5-5**: `unanswerable_ratio in [0.25, 0.33]`

---

## 3. Quality Gates Consolidees

### 3.1 Gates BLOQUANTES (exit code 1 si echec)

| ID | Gate | Seuil | Verification |
|----|------|-------|--------------|
| **G1-1** | chunk_match_score | = 100% | `assert score == 100` |
| **G1-2** | answer_in_chunk | verbatim OR sim >= 0.95 | Script validation |
| **G2-1** | unanswerable.is_impossible | = true | `assert all(is_impossible)` |
| **G3-1** | validation_passed | = 100% | Script validate_question |
| **G3-2** | hallucination_count | = 0 | `assert rejected == 0` |
| **G4-1** | schema_fields | = 46/46 | Schema validation |
| **G5-2** | anchor_independence | < 0.90 | Embedding check |

### 3.2 Gates Qualite (WARNING si echec)

| ID | Gate | Seuil | Impact |
|----|------|-------|--------|
| G0-2 | corpus_coverage | >= 80% | Diversite |
| G1-3 | fact_single_ratio | < 60% | Distribution |
| G2-2 | unanswerable_ratio | 25-33% | SQuAD 2.0 |
| G5-3 | fact_single_ratio | < 60% | Final check |
| G5-4 | hard_ratio | >= 10% | Difficulte |
| G5-5 | unanswerable_ratio | 25-33% | Balance |

---

## 4. Scripts a Creer/Modifier

### 4.1 Nouveaux Scripts

| Script | Role | Localisation |
|--------|------|--------------|
| `generate_gs_by_design.py` | Orchestrateur principal | `scripts/evaluation/annales/` |
| `stratify_corpus.py` | Phase 0 - Stratification | `scripts/evaluation/annales/` |
| `validate_anti_hallucination.py` | Phase 3 - Validation | `scripts/evaluation/annales/` |
| `enrich_schema_v2.py` | Phase 4 - Enrichissement | `scripts/evaluation/annales/` |
| `balance_distribution.py` | Phase 5 - Equilibrage | `scripts/evaluation/annales/` |
| `quality_gates.py` | Verification gates | `scripts/evaluation/annales/` |

### 4.2 Scripts Existants a Reutiliser

| Script | Usage | Localisation |
|--------|-------|--------------|
| `reformulate_by_design.py` | Validation semantique | `scripts/evaluation/annales/:45-91` |
| `validate_gs_quality.py` | Keyword/semantic scores | `scripts/evaluation/annales/:109-134` |
| `link_gs_to_chunks.py` | Article extraction | `scripts/evaluation/annales/:19-25` |
| `merge_adversarial_to_gs.py` | Schema v2 conversion | `scripts/evaluation/annales/:25-90` |

---

## 5. Fichiers de Sortie

| Fichier | Contenu | Format |
|---------|---------|--------|
| `tests/data/gs_scratch_v1.json` | GS complet Schema v2.0 | JSON |
| `tests/data/gs_scratch_v1_validation.json` | Rapport validation | JSON |
| `data/gs_generation/chunk_strata.json` | Stratification corpus | JSON |
| `data/gs_generation/generation_log.jsonl` | Log generation | JSONL |
| `data/gs_generation/rejected_questions.json` | Questions rejetees | JSON |

---

## 6. Estimation Effort

| Phase | Questions | Tokens IN | Tokens OUT | Duree estimee |
|-------|-----------|-----------|------------|---------------|
| Phase 0 | - | - | - | Python seul |
| Phase 1 | ~500-600 | ~600K | ~200K | Batches 20-30 Q |
| Phase 2 | ~150-200 | ~200K | ~80K | Batches 20-30 Q |
| Phase 3 | - | - | - | Python seul |
| Phase 4 | - | - | - | Python seul |
| Phase 5 | - | - | - | Python seul |
| **Total** | **~600-800** | **~800K** | **~280K** | - |

---

## 7. Verification Finale

```bash
# Script de verification finale
python -c "
import json
import sys

with open('tests/data/gs_scratch_v1.json', encoding='utf-8') as f:
    gs = json.load(f)

questions = gs['questions']
errors = []

# BLOCKING GATES
for q in questions:
    # G4-1: 46 fields
    if count_fields(q) != 46:
        errors.append(f'G4-1: {q[\"id\"]} missing fields')

    # G1-1: chunk_match_score = 100
    if q['processing']['chunk_match_score'] != 100:
        errors.append(f'G1-1: {q[\"id\"]} score != 100')

    # G5-2: anchor independence (requires embeddings)
    # Verified in Phase 5

# G2-2: unanswerable ratio
unanswerable = sum(1 for q in questions if q['content']['is_impossible'])
ratio = unanswerable / len(questions)
if not (0.25 <= ratio <= 0.33):
    errors.append(f'G2-2: unanswerable ratio {ratio:.2%} not in [25%, 33%]')

# G5-3: fact_single ratio
fact_single = sum(1 for q in questions
    if q['classification']['reasoning_class'] == 'fact_single'
    and not q['content']['is_impossible'])
answerable = len(questions) - unanswerable
fs_ratio = fact_single / answerable if answerable > 0 else 0
if fs_ratio >= 0.60:
    errors.append(f'G5-3: fact_single ratio {fs_ratio:.2%} >= 60%')

# G5-4: hard ratio
hard = sum(1 for q in questions if q['classification']['difficulty'] >= 0.7)
if hard / len(questions) < 0.10:
    errors.append(f'G5-4: hard ratio {hard/len(questions):.2%} < 10%')

if errors:
    print(f'VERIFICATION FAILED: {len(errors)} errors')
    for e in errors[:20]:
        print(f'  {e}')
    sys.exit(1)
else:
    print(f'VERIFICATION PASSED')
    print(f'  Total: {len(questions)}')
    print(f'  Unanswerable: {unanswerable} ({ratio:.1%})')
    print(f'  fact_single: {fact_single} ({fs_ratio:.1%})')
    print(f'  hard: {hard} ({hard/len(questions):.1%})')
    sys.exit(0)
"
```

---

## 8. Standards de Reference

| Standard | Application |
|----------|-------------|
| SQuAD 2.0 | 25-33% unanswerable |
| UAEval4RAG | 6 categories hard_type |
| Know Your RAG (COLING 2025) | reasoning_class distribution |
| RAGen/Source2Synth | Context-grounded generation |
| NV-Embed/E5 | Anchor independence < 0.90 |
| SemHash/SoftDedup | Deduplication < 0.95 |
| ISO 42001 A.6.2.2 | Provenance tracking |
| ISO 29119 | Test data validation |
| ISO 25010 | Quality metrics |

---

## 9. Prochaine Etape

1. **Creer `scripts/evaluation/annales/generate_gs_by_design.py`** - Orchestrateur principal
2. **Executer Phase 0** - Stratification corpus
3. **Executer Phases 1-5** par batches avec verification gates a chaque etape
4. **Validation finale** avec script de verification

---

*Plan ISO 42001/25010/29119 - Pocket Arbiter Project*
*Methode: BY DESIGN (chunk = INPUT, pas OUTPUT)*
