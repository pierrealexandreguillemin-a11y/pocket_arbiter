# Gold Standard Conformity Checklist

> **Document ID**: SPEC-GS-CONF-001
> **ISO Reference**: ISO 42001, ISO 25010, ISO 29119
> **Version**: 1.0
> **Date**: 2026-01-26
> **Statut**: Approuve
> **Classification**: Critique
> **Usage**: One-shot - Mise en conformite GS avant generation triplets
> **Mots-cles**: conformite, checklist, gold standard, triplets, validation, ISO

---

## 0. Objectif

Ce document consolide TOUTES les exigences de conformite pour le Gold Standard avant generation de triplets pour fine-tuning EmbeddingGemma QAT.

**Contexte critique**: Les triplets sont generes ainsi:
```
anchor   = question (du GS)
positive = chunk (via expected_chunk_id du GS)
negative = hard negative (du corpus)
```

**Si expected_chunk_id pointe vers un chunk qui NE CONTIENT PAS la reponse, le triplet est FAUX et on entraine le modele sur des donnees incorrectes.**

---

## 1. Criteres Bloquants (MUST PASS)

### 1.1 Alignement Chunk-Reponse

| ID | Critere | Seuil | Justification | Source |
|----|---------|-------|---------------|--------|
| **CB-01** | chunk_match_score | **= 100%** | positive = chunk contenant reponse | TRIPLET_GENERATION_SPEC S4.5 |
| CB-02 | expected_chunk_id valide | 100% existent dans corpus | Pas de references cassees | ISO 42001 A.6.2.2 |
| CB-03 | expected_chunk_id non-null | 100% (sauf requires_context) | Mapping complet | UNIFIED_TRAINING_DATA_SPEC S3.1 |

**Verification CB-01:**
```python
# La reponse DOIT etre dans le chunk
for question in gs['questions']:
    chunk = get_chunk(question['expected_chunk_id'])
    assert question['expected_answer'] in chunk['text'] or semantic_match(question['expected_answer'], chunk['text']) >= 0.95
```

### 1.2 Validation BY DESIGN

| ID | Critere | Seuil | Justification | Source |
|----|---------|-------|---------------|--------|
| **CB-04** | Context-grounded generation | 100% | Chunk visible lors creation question | RAGen, Source2Synth |
| CB-05 | Zero generation recursive | 0% | Prevention model collapse | Liu et al. 2024 |
| CB-06 | Lineage documentee | 100% | Tracabilite complete | ISO 42001 A.6.2.3 |

### 1.3 Provenance ISO 42001

| ID | Critere | Seuil | Justification | Source |
|----|---------|-------|---------------|--------|
| CB-07 | expected_docs present | 100% | Source document tracable | ISO 42001 A.6.2.2 |
| CB-08 | expected_pages present | >= 80% | Localisation precise | ISO 42001 A.6.2.2 |
| CB-09 | requires_context_reason | 100% si requires_context=true | Justification exclusion | ISO 42001 A.7.3 |

---

## 2. Criteres Qualite (SHOULD PASS)

### 2.1 Distribution Questions

| ID | Critere | Seuil | Cible Industrie | Source |
|----|---------|-------|-----------------|--------|
| CQ-01 | reasoning_class present | 100% | Classification complete | Know Your RAG |
| CQ-02 | fact_single | 40-50% | Ancrage factuel | Know Your RAG |
| CQ-03 | summary | 15-25% | Synthese semantique | Know Your RAG |
| CQ-04 | reasoning | 10-20% | Multi-hop inference | Know Your RAG |
| CQ-05 | Diversity categories | >= 80% corpus | Couverture domaine | BEIR |

**Note v7.6**: Distribution actuelle (57% summary) est un AVANTAGE DELIBERE pour fine-tuning on-device (LREM +19.2% Q&A).

### 2.2 Deduplication

| ID | Critere | Seuil | Methode | Source |
|----|---------|-------|---------|--------|
| CQ-06 | Similarite inter-questions | < 5% | SemHash cosine < 0.95 | SoftDedup |
| CQ-07 | Anchor independence | cosine(anchor,positive) < 0.9 | Embedding check | E5 training |

### 2.3 Qualite Reponses

| ID | Critere | Seuil | Justification | Source |
|----|---------|-------|---------------|--------|
| CQ-08 | expected_answer non-vide | 100% | Reponse de reference | ISO 25010 |
| CQ-09 | Validation humaine | 100% GS | Labels officiels | Industrie standard |
| CQ-10 | quality_score | >= 0.7 | Seuil qualite acceptable | Interne |

---

## 3. Criteres Triplets (PRE-GENERATION)

### 3.1 Hard Negatives

| ID | Critere | Seuil | Justification | Source |
|----|---------|-------|---------------|--------|
| CT-01 | same_doc_diff_page | >= 40% | Hardest negatives | NV-Embed-v2 |
| CT-02 | Pas de duplicate negatives | 0% | Diversite | Best practices |
| CT-03 | Negative != Positive | 100% | Coherence logique | Evident |

### 3.2 Format Export

| ID | Critere | Seuil | Format | Source |
|----|---------|-------|--------|--------|
| CT-04 | Schema JSON valide | 100% | Draft-07 | ISO 29119 |
| CT-05 | Split train/val | 80/20 | Seed fixe | ML best practices |
| CT-06 | DVC tracked | 100% | Reproductibilite | ISO 12207 |

---

## 4. Checklist Actionnable

### 4.1 Phase 1: Audit Initial

```bash
# Executer audit complet
python -c "
import json

with open('tests/data/gold_standard_annales_fr_v7.json', encoding='utf-8') as f:
    gs = json.load(f)

questions = gs['questions']
print(f'=== AUDIT GS v{gs[\"version\"]} ===')
print(f'Total questions: {len(questions)}')

# CB-01: chunk_match_score = 100%
score_100 = sum(1 for q in questions if q.get('metadata', {}).get('chunk_match_score') == 100)
print(f'CB-01 chunk_match_score=100: {score_100}/{len(questions)} ({100*score_100/len(questions):.1f}%)')

# CB-02: expected_chunk_id valide
has_chunk_id = sum(1 for q in questions if q.get('expected_chunk_id'))
print(f'CB-02 expected_chunk_id present: {has_chunk_id}/{len(questions)}')

# CB-09: requires_context_reason
rc = [q for q in questions if q.get('metadata', {}).get('requires_context')]
rc_no_reason = [q for q in rc if not q.get('metadata', {}).get('requires_context_reason')]
print(f'CB-09 requires_context sans reason: {len(rc_no_reason)}/{len(rc)}')

# CQ-01: reasoning_class
has_rc = sum(1 for q in questions if q.get('metadata', {}).get('reasoning_class'))
print(f'CQ-01 reasoning_class present: {has_rc}/{len(questions)}')

# CQ-08: expected_answer
has_answer = sum(1 for q in questions if q.get('expected_answer', '').strip())
print(f'CQ-08 expected_answer non-vide: {has_answer}/{len(questions)}')
"
```

### 4.2 Phase 2: Correction CB-01 (BLOQUANT)

**Probleme**: chunk_match_score < 100% signifie que la reponse n'est PAS dans le chunk.

**Solutions par ordre de priorite:**

| Priorite | Solution | Effort | Impact |
|----------|----------|--------|--------|
| 1 | Realigner expected_chunk_id vers chunk contenant reponse | Eleve | 100% fix |
| 2 | Reformuler expected_answer pour matcher chunk | Moyen | Partiel |
| 3 | Marquer requires_context=true si reponse non-extractible | Faible | Exclusion |

**Script realignement:**
```python
# Pour chaque question avec score < 100:
# 1. Chercher chunk contenant expected_answer (exact ou semantique)
# 2. Si trouve: mettre a jour expected_chunk_id
# 3. Si non trouve: marquer requires_context=true avec reason
```

### 4.3 Phase 3: Correction CB-09 (BLOQUANT)

**Probleme**: requires_context=true sans raison = violation ISO 42001 tracabilite.

**Actions:**
```python
VALID_REASONS = [
    "answer_requires_calculation",      # Reponse = calcul (dates, Elo)
    "answer_requires_context_position", # Reponse depend position echiquier
    "answer_requires_external_data",    # Donnees hors corpus (Elo joueur)
    "answer_is_reformulation",          # Reponse = synthese non-extractible
    "chunk_not_in_corpus"               # Article reference hors corpus
]

for q in questions:
    if q.get('metadata', {}).get('requires_context') and not q.get('metadata', {}).get('requires_context_reason'):
        # Analyser et assigner raison
        q['metadata']['requires_context_reason'] = determine_reason(q)
```

### 4.4 Phase 4: Validation Finale

```bash
# Toutes les conditions MUST PASS
python -c "
import json
import sys

with open('tests/data/gold_standard_annales_fr_v7.json', encoding='utf-8') as f:
    gs = json.load(f)

questions = gs['questions']
testable = [q for q in questions if not q.get('metadata', {}).get('requires_context')]

errors = []

# CB-01: chunk_match_score = 100% pour testables
for q in testable:
    score = q.get('metadata', {}).get('chunk_match_score', 0)
    if score != 100:
        errors.append(f'CB-01 FAIL: {q[\"id\"]} score={score}')

# CB-09: requires_context_reason
rc = [q for q in questions if q.get('metadata', {}).get('requires_context')]
for q in rc:
    if not q.get('metadata', {}).get('requires_context_reason'):
        errors.append(f'CB-09 FAIL: {q[\"id\"]} missing reason')

if errors:
    print(f'CONFORMITY FAILED: {len(errors)} errors')
    for e in errors[:10]:
        print(f'  {e}')
    sys.exit(1)
else:
    print('CONFORMITY PASSED: Ready for triplet generation')
    sys.exit(0)
"
```

---

## 5. Etat Actuel GS v7.6

| Critere | Seuil | Actuel | Status |
|---------|-------|--------|--------|
| CB-01 chunk_match_score=100 | 100% | 2.9% (12/420) | **FAIL** |
| CB-02 expected_chunk_id | 100% | 100% | PASS |
| CB-03 expected_chunk_id non-null | 100% testables | 100% | PASS |
| CB-09 requires_context_reason | 100% | 54% (50/92) | **FAIL** |
| CQ-01 reasoning_class | 100% | 100% | PASS |
| CQ-08 expected_answer | 100% | 100% | PASS |

**Verdict: GS v7.6 NON CONFORME pour generation triplets.**

---

## 6. References

### 6.1 Documents Projet (Approfondissement)

| Document | Section | Contenu | Quand consulter |
|----------|---------|---------|-----------------|
| [QUALITY_REQUIREMENTS.md](../QUALITY_REQUIREMENTS.md) | S4 | Minima Training Data (TD-01 a TV-04) | Seuils qualite |
| [TRIPLET_GENERATION_SPEC.md](TRIPLET_GENERATION_SPEC.md) | S4.5 | BY DESIGN, positive = chunk contenant reponse | Definition triplets |
| [TRIPLET_GENERATION_SPEC.md](TRIPLET_GENERATION_SPEC.md) | S11 | References academiques (NV-Embed, E5, GTE) | Hard negatives |
| [UNIFIED_TRAINING_DATA_SPEC.md](UNIFIED_TRAINING_DATA_SPEC.md) | S1.3-1.4 | Validation BY DESIGN, context-grounded | Methode validation |
| [UNIFIED_TRAINING_DATA_SPEC.md](UNIFIED_TRAINING_DATA_SPEC.md) | S2.4 | Quality Gates par etape | Pipeline qualite |
| [UNIFIED_TRAINING_DATA_SPEC.md](UNIFIED_TRAINING_DATA_SPEC.md) | S3 | Specs detaillees par etape | Implementation |
| [GOLD_STANDARD_SPECIFICATION.md](../GOLD_STANDARD_SPECIFICATION.md) | S1.3 | Metriques GS v7.6 | Etat actuel |
| [GOLD_STANDARD_SPECIFICATION.md](../GOLD_STANDARD_SPECIFICATION.md) | S10 | Standards industrie (MTEB, BEIR) | Benchmarks |
| [AI_POLICY.md](../AI_POLICY.md) | S3-4 | Controles ISO 42001, anti-hallucination | Gouvernance IA |
| [ISO_STANDARDS_REFERENCE.md](../ISO_STANDARDS_REFERENCE.md) | S1.2 | ISO 42001 details | Conformite ISO |
| [research/FINETUNING_RESOURCES.md](../research/FINETUNING_RESOURCES.md) | S2-4 | Sources Google, hyperparametres | Fine-tuning |
| [GS_ANNALES_V7_OPTIMIZATION_SPEC.md](GS_ANNALES_V7_OPTIMIZATION_SPEC.md) | - | Optimisation GS pour triplets | Ameliorations |

### 6.2 Standards Industrie

| Standard | Reference | Exigence |
|----------|-----------|----------|
| Know Your RAG | arXiv:2411.19710 | Distribution reasoning_class |
| LREM | arXiv:2510.14321 | Reasoning training +19.2% |
| NV-Embed-v2 | arXiv:2405.17428 | Hard negative mining |
| SoftDedup | arXiv:2407.06564 | Deduplication < 0.95 |
| RAGen | arXiv:2411.14831 | Context-grounded generation |

### 6.3 Normes ISO

| Norme | Controle | Application |
|-------|----------|-------------|
| ISO 42001 A.6.2.2 | Provenance | expected_chunk_id, expected_docs |
| ISO 42001 A.6.2.3 | Lineage | Methodology documentee |
| ISO 42001 A.7.3 | Documentation | requires_context_reason |
| ISO 25010 | Exactitude | expected_answer dans chunk |
| ISO 29119 | Test data | Schema validation |

### 6.4 Sources Web de Confiance (2026-01-26)

#### ArXiv (Papers)
| Paper | ID | Contribution |
|-------|-----|--------------|
| Know Your RAG | [arXiv:2411.19710](https://arxiv.org/abs/2411.19710) | Taxonomie reasoning_class |
| LREM | [arXiv:2510.14321](https://arxiv.org/abs/2510.14321) | Reasoning training +19.2% |
| NV-Embed-v2 | [arXiv:2405.17428](https://arxiv.org/abs/2405.17428) | Hard negative mining |
| NV-Retriever | [arXiv:2407.15831](https://arxiv.org/abs/2407.15831) | TopK-PercPos method |
| SoftDedup | [arXiv:2407.06564](https://arxiv.org/abs/2407.06564) | Deduplication < 0.95 |
| RAGen | [arXiv:2411.14831](https://arxiv.org/abs/2411.14831) | Context-grounded generation |
| Source2Synth | [arXiv:2409.08239](https://arxiv.org/abs/2409.08239) | Grounded synthetic data |
| SimCSE | [arXiv:2104.08821](https://arxiv.org/abs/2104.08821) | Contrastive learning |
| E5 | [arXiv:2212.03533](https://arxiv.org/abs/2212.03533) | Anchor independence |

#### Google AI (Official)
| Resource | URL | Contenu |
|----------|-----|---------|
| EmbeddingGemma Fine-tuning | [ai.google.dev](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers) | Guide officiel |
| EmbeddingGemma Model Card | [ai.google.dev](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) | Specs techniques |
| AI Edge RAG SDK | [ai.google.dev](https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android) | Deploiement Android |
| Gemma Cookbook | [github.com/google-gemini](https://github.com/google-gemini/gemma-cookbook) | Notebooks officiels |
| AI Edge Torch | [github.com/google-ai-edge](https://github.com/google-ai-edge/ai-edge-torch) | Conversion TFLite |

#### HuggingFace
| Resource | URL | Contenu |
|----------|-----|---------|
| EmbeddingGemma Blog | [huggingface.co/blog](https://huggingface.co/blog/embeddinggemma) | Architecture, fine-tuning |
| Train Sentence Transformers | [huggingface.co/blog](https://huggingface.co/blog/train-sentence-transformers) | Guide ST v3 |
| MTEB Leaderboard | [huggingface.co/spaces](https://huggingface.co/spaces/mteb/leaderboard) | Benchmark embeddings |
| EmbeddingGemma QAT | [huggingface.co/google](https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized) | Modele recommande |

#### GitHub
| Repository | URL | Usage |
|------------|-----|-------|
| BEIR Benchmark | [github.com/beir-cellar](https://github.com/beir-cellar/beir) | Evaluation retrieval |
| Sentence Transformers | [github.com/UKPLab](https://github.com/UKPLab/sentence-transformers) | Training triplets |
| SemHash | [github.com/MinishLab](https://github.com/MinishLab/semhash) | Fuzzy deduplication |
| LlamaIndex Finetune | [github.com/run-llama](https://github.com/run-llama/finetune-embedding) | Synthetic triplets |

#### Kaggle
| Notebook | URL | Auteur |
|----------|-----|--------|
| Fine-tune EmbeddingGemma | [kaggle.com](https://www.kaggle.com/code/nilaychauhan/fine-tune-embeddinggemma) | Nilay Chauhan (Google) |
| Fine-tune Gemma LoRA | [kaggle.com](https://www.kaggle.com/code/nilaychauhan/fine-tune-gemma-models-in-keras-using-lora) | Nilay Chauhan (Google) |

#### Autres Sources
| Source | URL | Contenu |
|--------|-----|---------|
| Microsoft Data Science | [medium.com](https://medium.com/data-science-at-microsoft/the-path-to-a-golden-dataset-or-how-to-evaluate-your-rag-045e23d1f13f) | Golden Dataset RAG |
| Statsig | [statsig.com](https://www.statsig.com/perspectives/golden-datasets-evaluation-standards) | Evaluation standards |
| SBERT Training | [sbert.net](https://www.sbert.net/docs/sentence_transformer/training_overview.html) | Training overview |
| Pinecone MNR Loss | [pinecone.io](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/) | MultipleNegativesRankingLoss |

---

## 7. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-26 | Creation - Consolidation requirements ISO + industrie |

---

*Document ISO 42001/25010/29119 - Pocket Arbiter Project*
*Usage: Checklist one-shot avant generation triplets*
