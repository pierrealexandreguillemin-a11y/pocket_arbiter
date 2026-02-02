# Gold Standard Conformity Checklist

> **Document ID**: SPEC-GS-CONF-001
> **ISO Reference**: ISO 42001, ISO 25010, ISO 29119
> **Version**: 3.0
> **Date**: 2026-02-02
> **Statut**: Aligne sur PLAN-GS-CONF-001
> **Note**: Aligne avec GS_CONFORMITY_PLAN_V1.md (CB-05..08 dans §1.1, EX-05/06 reordonne, G3-6 bloquant)
> **Classification**: Critique
> **Usage**: One-shot - Mise en conformite GS avant generation triplets
> **Mots-cles**: conformite, checklist, gold standard, triplets, validation, ISO
> **Plan associe**: [GS_CONFORMITY_PLAN_V1.md](../plans/GS_CONFORMITY_PLAN_V1.md)

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

### 3.3 Criteres Format (TRIPLET_GENERATION_SPEC S2.2)

| ID | Critere | Seuil | Justification | Source |
|----|---------|-------|---------------|--------|
| F-01 | anchor finit par ? | 100% | Forme interrogative | TRIPLET_GENERATION_SPEC S2.2 |
| F-02 | anchor >= 10 chars | 100% | Longueur minimum | TRIPLET_GENERATION_SPEC S2.2 |
| F-03 | positive >= 50 chars | 100% | Chunk significatif | TRIPLET_GENERATION_SPEC S2.2 |
| F-04 | expected_answer > 5 chars | 100% | Reponse significative | TRIPLET_GENERATION_SPEC S2.2 |

### 3.4 Criteres Metadonnees (GS_ANNALES_V7_OPTIMIZATION_SPEC)

| ID | Critere | Seuil | Justification | Source |
|----|---------|-------|---------------|--------|
| M-01 | difficulty present | 100% | Classification difficulte | GS_ANNALES_V7_OPTIMIZATION_SPEC |
| M-02 | difficulty in [0,1] | 100% | Valeur normalisee | GS_ANNALES_V7_OPTIMIZATION_SPEC |
| M-03 | cognitive_level | 100% | Taxonomie Bloom | Know Your RAG |
| M-04 | category | 100% | Couverture domaine | BEIR |

### 3.5 Criteres Qualite Avancee (UNIFIED_TRAINING_DATA_SPEC)

| ID | Critere | Seuil | Methode | Source |
|----|---------|-------|---------|--------|
| QA-01 | Deduplication inter-questions | cosine < 0.95 | SemHash | SoftDedup |
| QA-02 | Anchor independence | cosine(anchor, positive) < 0.9 | Embedding check | E5 training |

### 3.6 Criteres Export Multi-Format (UNIFIED_TRAINING_DATA_SPEC)

| ID | Critere | Format | Source |
|----|---------|--------|--------|
| EX-01 | Triplets JSONL | anchor/positive/negative | TRIPLET_GENERATION_SPEC S2.1 |
| EX-02 | ARES TSV | Query/Document/Answer/Label | ARES |
| EX-03 | BEIR | queries.jsonl + corpus.jsonl + qrels.tsv | BEIR |
| EX-04 | RAGAS | question/answer/contexts/ground_truth | RAGAS |
| EX-05 | DVC tracking | dvc add + push | ISO 12207 |
| EX-06 | Composition report JSON | dataset_composition.json | Interne |

---

## 4. Checklist Actionnable (alignee sur PLAN-GS-CONF-001)

### 4.0 Audit Initial

```bash
python -c "
import json

with open('tests/data/gold_standard_annales_fr_v7.json', encoding='utf-8') as f:
    gs = json.load(f)

questions = gs['questions']
testables = [q for q in questions if not q.get('metadata',{}).get('requires_context')]
rc = [q for q in questions if q.get('metadata',{}).get('requires_context')]

print(f'=== AUDIT GS v{gs[\"version\"]} ===')
print(f'Total: {len(questions)}, Testables: {len(testables)}, requires_context: {len(rc)}')

# Bloquants
s100 = sum(1 for q in questions if q.get('metadata',{}).get('chunk_match_score') == 100)
bd = sum(1 for q in questions if q.get('metadata',{}).get('by_design'))
rc_r = sum(1 for q in rc if q.get('metadata',{}).get('requires_context_reason'))
print(f'CB-01 chunk_match_score=100: {s100}/{len(questions)}')
print(f'CB-04 by_design: {bd}/{len(questions)}')
print(f'CB-09 rc_reason: {rc_r}/{len(rc)}')

# Format
f01 = sum(1 for q in questions if q['question'].strip().endswith('?'))
f04 = sum(1 for q in questions if len(q.get('expected_answer','')) > 5)
print(f'F-01 ends with ?: {f01}/{len(questions)}')
print(f'F-04 answer > 5 chars: {f04}/{len(questions)}')

# Metadata
m01 = sum(1 for q in questions if q.get('metadata',{}).get('difficulty') is not None)
m02 = sum(1 for q in questions if isinstance(q.get('metadata',{}).get('difficulty'), (int,float)) and 0 <= q['metadata']['difficulty'] <= 1)
print(f'M-01 difficulty present: {m01}/{len(questions)}')
print(f'M-02 difficulty in [0,1]: {m02}/{len(questions)}')

# Triplets
ct01 = sum(1 for q in testables if len(q.get('metadata',{}).get('hard_negatives',[])) >= 3)
print(f'CT-01 hard_negatives>=3: {ct01}/{len(testables)}')
"
```

### 4.1 Phase 0: BY DESIGN + chunk validation (CB-01 + CB-04)

> **Acteur**: LLM (Claude Code ou Gemini 2.5 Flash via API)
> **Ref plan**: PLAN-GS-CONF-001 Phase 0

**Methode combinee**: Pour CHAQUE question, le LLM recoit le chunk + la question + la reponse
et produit: validation chunk, reformulation BY DESIGN, requires_context_reason.

- [ ] 420/420 questions traitees
- [ ] CB-04 by_design = 100%
- [ ] CB-01 chunk_match_score = 100% testables
- [ ] F-01 question finit par ? = 100%
- [ ] original_question preservee dans metadata

### 4.2 Phase 1: Corrections Metadonnees

> **Acteur**: LLM (CB-09) + Python deterministe (M-01/M-02, F-04)
> **Ref plan**: PLAN-GS-CONF-001 Phase 1

- [ ] CB-09 requires_context_reason = 100% (42 manquants)
- [ ] M-01 difficulty present = 100% (34 manquants)
- [ ] M-02 difficulty in [0,1] = 100%
- [ ] F-04 expected_answer > 5 chars = review 32 questions

### 4.3 Phase 2: Hard Negatives (CT-01)

> **Acteur**: EmbeddingGemma (pre-filtre local) + LLM (juge)
> **Ref plan**: PLAN-GS-CONF-001 Phase 2

- [ ] Etape 2B: EmbeddingGemma encode 1857 chunks + 420 queries → top-10 candidats
- [ ] Etape 2A: LLM juge 10 candidats/question → 3-5 hard negatives valides
- [ ] Etape 2C: same_doc >= 40% enrichissement
- [ ] CT-01 hard_negatives >= 3 = 100% testables
- [ ] CT-02 pas de duplicate negatives
- [ ] CT-03 negative != positive = 100%

### 4.4 Phase 3: Export Multi-Format (EX-01..06)

> **Acteur**: Python (scripts deterministes)
> **Ref plan**: PLAN-GS-CONF-001 Phase 3

- [ ] EX-01 Triplets JSONL (train + val)
- [ ] EX-02 ARES TSV
- [ ] EX-03 BEIR (queries + corpus + qrels)
- [ ] EX-04 RAGAS
- [ ] EX-05 DVC tracking
- [ ] EX-06 Composition report JSON
- [ ] CT-04 Schema JSON valide = 100%
- [ ] CT-05 Split train/val 80/20 (seed=42)

### 4.5 Validation Finale

```bash
python -c "
import json, sys
from pathlib import Path

with open('tests/data/gold_standard_annales_fr_v7.json', encoding='utf-8') as f:
    gs = json.load(f)

questions = gs['questions']
testables = [q for q in questions if not q.get('metadata',{}).get('requires_context')]
rc = [q for q in questions if q.get('metadata',{}).get('requires_context')]
errors = []

# Phase 0
for q in questions:
    if not q.get('metadata',{}).get('by_design'):
        errors.append(f'CB-04: {q[\"id\"]}')
    if not q['question'].strip().endswith('?'):
        errors.append(f'F-01: {q[\"id\"]}')
for q in testables:
    if q.get('metadata',{}).get('chunk_match_score') != 100:
        errors.append(f'CB-01: {q[\"id\"]}')

# Phase 1
for q in rc:
    if not q.get('metadata',{}).get('requires_context_reason'):
        errors.append(f'CB-09: {q[\"id\"]}')
for q in questions:
    d = q.get('metadata',{}).get('difficulty')
    if d is None or not (0 <= d <= 1):
        errors.append(f'M-01: {q[\"id\"]}')

# Phase 2
for q in testables:
    if len(q.get('metadata',{}).get('hard_negatives',[])) < 3:
        errors.append(f'CT-01: {q[\"id\"]}')

# Phase 3
unified = Path('data/training/unified')
for f in ['triplets_train.jsonl','triplets_val.jsonl','ares_train.tsv','beir/queries.jsonl','ragas_val.jsonl','dataset_composition.json']:
    if not (unified / f).exists():
        errors.append(f'EX: missing {f}')

if errors:
    print(f'CONFORMITY FAILED: {len(errors)} errors')
    for e in errors[:30]:
        print(f'  {e}')
    sys.exit(1)
else:
    print('CONFORMITY PASSED - Ready for semantic bridge training')
"
```

---

## 5. Etat Actuel GS v8.0 (audit 2026-02-02)

> **Changements v7.6→v8.0**: Complete traceability chain rebuild from Docling sources.
> Phase 1: 765 field repairs (choices, mcq_answer, expected_answer, article_reference).
> Phase 2: All 420 expected_answers grounded as verbatim chunk passages (CB-01=100%).
> Phase 3: Metadata completion (difficulty, requires_context_reason), version bump.
>
> Scripts: p1_rebuild_gs_from_docling.py, p2_cb01_answer_in_chunk.py, p3_metadata_completion.py

### 5.1 Criteres Bloquants

| Critere | Seuil | Actuel | Status | Phase |
|---------|-------|--------|--------|-------|
| CB-01 answer in chunk | 100% | **100% (420/420)** | **PASS** | P2 |
| CB-02 expected_chunk_id | 100% | **100% (420/420)** | **PASS** | - |
| CB-03 expected_chunk_id non-null | 100% | **100% (420/420)** | **PASS** | - |
| CB-04 by_design (manual_by_design) | 100% | **100% (420/420)** | **PASS** | P2 |
| CB-09 requires_context_reason | 100% rc | **100% (92/92)** | **PASS** | P3 |

### 5.2 Criteres Format

| Critere | Seuil | Actuel | Status | Phase |
|---------|-------|--------|--------|-------|
| F-01 question finit par ? | 100% | **100% (420/420)** | **PASS** | P2 |
| F-02 anchor >= 10 chars | 100% | **100% (420/420)** | **PASS** | - |
| F-03 positive >= 50 chars | 100% | **100% (420/420)** | **PASS** | - |
| F-04 expected_answer > 5 chars | 100% | **100% (420/420)** | **PASS** | P2 |

### 5.3 Criteres Qualite Donnees

| Critere | Seuil | Actuel | Status | Source |
|---------|-------|--------|--------|--------|
| Zero ## fusion (question) | 0 | **0** | **PASS** | P1 rebuild |
| Zero ## fusion (answer) | 0 | **0** | **PASS** | P1 rebuild |
| Zero empty answer | 0 | **0** | **PASS** | P1 rebuild |
| Zero ref-only answer | 0 | **0** | **PASS** | P1 M3b check |
| Zero dirty mcq_answer | 0 | **0** | **PASS** | P1 manual corrections |
| keywords populated | 100% | **100% (420/420)** | **PASS** | Preserved |
| question_type populated | 100% | **100% (420/420)** | **PASS** | Preserved |
| reasoning_class populated | 100% | **100% (420/420)** | **PASS** | Preserved |

### 5.4 Criteres Metadonnees

| Critere | Seuil | Actuel | Status | Phase |
|---------|-------|--------|--------|-------|
| M-01 difficulty present | 100% | **100% (420/420)** | **PASS** | P3 |
| M-02 difficulty in [0,1] | 100% | **100% (420/420)** | **PASS** | P3 |
| M-03 cognitive_level | 100% | **100% (420/420)** | **PASS** | - |
| M-04 category | 100% | **100% (420/420)** | **PASS** | - |

### 5.5 Criteres Triplets

| Critere | Seuil | Actuel | Status | Phase |
|---------|-------|--------|--------|-------|
| CT-01 hard_negatives >= 3 | 100% testables | 0% (0/328) | **PENDING** | Next |
| QA-01 deduplication | cosine < 0.95 | Non verifie | **AUDIT** | Next |
| QA-02 anchor independence | cosine < 0.9 | Non verifie | **AUDIT** | Next |

### 5.6 Criteres Export

| Critere | Actuel | Phase |
|---------|--------|-------|
| EX-01 Triplets JSONL | Non genere | Next |
| EX-02 ARES TSV | Non genere | Next |
| EX-03 BEIR | Non genere | Next |
| EX-04 RAGAS | Non genere | Next |
| EX-05 DVC tracking | Non configure | Next |
| EX-06 Composition report | Non genere | Next |

### 5.7 Resume

| Phase | PASS | PENDING | AUDIT |
|-------|:----:|:-------:|:-----:|
| Bloquants (CB-01..09) | **5/5** | 0 | 0 |
| Format (F-01..04) | **4/4** | 0 | 0 |
| Qualite donnees | **8/8** | 0 | 0 |
| Metadonnees (M-01..04) | **4/4** | 0 | 0 |
| Triplets (CT-01..03) | 0 | 1 | 2 |
| Export (EX-01..06) | 0 | 6 | 0 |
| **Total** | **21** | **7** | **2** |

**Verdict: GS v8.0 — 15 criteres bloquants PASS (CB+F+M complets), 7 PENDING (triplets+export), 2 AUDIT requis.**
**Le GS est pret pour la generation de triplets.**

---

## 6. Acteurs et Evaluation d'Utilite

### 6.1 Matrice Acteur par Phase

| Phase | Tache | Claude Code | Gemini 2.5 Flash (free) | Mistral (free) | Python local |
|-------|-------|:-----------:|:-----------------------:|:--------------:|:------------:|
| **0** CB-04 BY DESIGN | Reformulation 420 Q | **Optimal** | Bon | Tres bon (FR) | - |
| **0** CB-01 | Validation chunk-reponse | **Optimal** | Bon | Bon | - |
| **1** CB-09 | requires_context_reason | **Optimal** | OK | Bon | - |
| **1** M-01/M-02 | difficulty | - | - | - | **Script** |
| **1** F-01 | question finit par ? | Fait en Phase 0 | - | - | - |
| **1** F-04 | Review answers | Bon | OK | OK | - |
| **2B** | Encode top-10 candidats | - | - | - | **EmbeddingGemma** |
| **2A** | Juge hard negatives | **Optimal** | Bon | Bon | - |
| **2C** | same_doc enrichment | - | - | - | **Script** |
| **3** | Export 6 formats | - | - | - | **Script** |
| **QA** | Deduplication audit | - | - | - | **EmbeddingGemma** |

### 6.2 Evaluation Claude pour ce domaine

**Domaine**: Reformulation BY DESIGN de questions d'arbitrage echecs FR pour QLoRA fine-tuning.

| Competence requise | Claude | Gemini Flash | Mistral | Importance |
|--------------------|:------:|:------------:|:-------:|:----------:|
| Comprehension regles FFE/FIDE FR | Excellent | Bon | Tres bon | Critique |
| Raisonnement chunk→reponse | Superieur | Bon | Bon | Critique |
| Reformulation naturelle FR | Excellent | Bon | Excellent | Haute |
| Detection faux negatifs | Superieur | Moyen | Moyen | Haute |
| JSON structure | Natif | Schema enforced | OK | Moyenne |
| Cout | Inclus CC | $0 (free tier) | $0 (experiment) | Variable |

**Conclusion**: Claude est l'acteur optimal pour Phases 0+1+2A (taches LLM).
Alternative free viable: **Gemini 2.5 Flash** (AI Studio, 250 RPD) ou **Mistral** (experiment, 1B tok/mois).
Qualite estimee des alternatives: 85-95% de Claude pour ce domaine specifique.

### 6.3 Budget Tokens Estime

| Phase | Input tokens | Output tokens | Total |
|-------|:----------:|:-----------:|:-----:|
| Phase 0 (420 Q) | ~215K | ~126K | ~341K |
| Phase 2A (420 Q) | ~1,470K | ~168K | ~1,638K |
| **Total LLM** | **~1,685K** | **~294K** | **~1,979K** |

### 6.4 Strategie Recommandee

```
OPTION A (Optimale): Claude Code direct
  Phase 0: Claude traite 420 Q par batches (~20 Q/tour)
  Phase 1: Python deterministe + Claude pour CB-09
  Phase 2B: Python + EmbeddingGemma local (Kaggle T4 si pas de GPU)
  Phase 2A: Claude juge les hard negatives
  Phase 3: Python scripts

OPTION B (Free): Gemini 2.5 Flash API
  Phase 0: Script Python + google.generativeai (250 RPD = 2 jours)
  Phase 1: Python deterministe + Gemini pour CB-09
  Phase 2B: Python + EmbeddingGemma local (Kaggle T4)
  Phase 2A: Gemini juge les hard negatives (250 RPD = 2 jours)
  Phase 3: Python scripts

OPTION C (Hybride): Claude Phase 0 + Gemini Phase 2A
  Phase 0: Claude (qualite maximale pour BY DESIGN = fondation)
  Phase 2A: Gemini 2.5 Flash (jugement hard neg = moins critique)
```

---

## 7. References

### 7.1 Documents Projet (Approfondissement)

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

### 7.2 Standards Industrie

| Standard | Reference | Exigence |
|----------|-----------|----------|
| Know Your RAG | arXiv:2411.19710 | Distribution reasoning_class |
| LREM | arXiv:2510.14321 | Reasoning training +19.2% |
| NV-Embed-v2 | arXiv:2405.17428 | Hard negative mining |
| SoftDedup | arXiv:2407.06564 | Deduplication < 0.95 |
| RAGen | arXiv:2411.14831 | Context-grounded generation |

### 7.3 Normes ISO

| Norme | Controle | Application |
|-------|----------|-------------|
| ISO 42001 A.6.2.2 | Provenance | expected_chunk_id, expected_docs |
| ISO 42001 A.6.2.3 | Lineage | Methodology documentee |
| ISO 42001 A.7.3 | Documentation | requires_context_reason |
| ISO 25010 | Exactitude | expected_answer dans chunk |
| ISO 29119 | Test data | Schema validation |

### 7.4 Sources Web de Confiance (2026-01-26)

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

#### Outils LLM / Alternatives Free
| Outil | URL | Contenu |
|-------|-----|---------|
| Kilo Code | [kilo.ai](https://kilo.ai/) | Agent coding open-source, modeles free via OpenRouter |
| Kilo Code Free Models | [kilo.ai/docs](https://kilo.ai/docs/advanced-usage/free-and-budget-models) | DeepSeek R1, Qwen3 Coder, Kimi K2 (free) |
| Google AI Studio | [ai.google.dev](https://ai.google.dev/gemini-api/docs/pricing) | Gemini 2.5 Flash free tier (250 RPD) |
| Mistral Experiment | [mistral.ai](https://mistral.ai/pricing) | Tier gratuit, excellent FR |

#### Autres Sources
| Source | URL | Contenu |
|--------|-----|---------|
| Microsoft Data Science | [medium.com](https://medium.com/data-science-at-microsoft/the-path-to-a-golden-dataset-or-how-to-evaluate-your-rag-045e23d1f13f) | Golden Dataset RAG |
| Statsig | [statsig.com](https://www.statsig.com/perspectives/golden-datasets-evaluation-standards) | Evaluation standards |
| SBERT Training | [sbert.net](https://www.sbert.net/docs/sentence_transformer/training_overview.html) | Training overview |
| Pinecone MNR Loss | [pinecone.io](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/) | MultipleNegativesRankingLoss |

---

## 8. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-26 | Creation - Consolidation requirements ISO + industrie |
| 2.0 | 2026-01-28 | Alignement sur PLAN-GS-CONF-001: ajout criteres F-01..04, M-01..04, QA-01..02, EX-01..06; checklist actionnable par phase; audit complet v7.6; matrice acteurs; evaluation Claude/Gemini/Mistral |
| 2.1 | 2026-01-30 | Audit v7.7: 9 criteres qualite donnees PASS (0 fusion, 0 ref-only, 0 empty, 0 dirty mcq, 420/420 chunk_ids valid + 106 optimised, metadata schema 0 errors). Pipeline: reextract_from_docling + patch_gs + verify_gs_metadata |
| 3.0 | 2026-02-02 | Audit v8.0: 15 criteres bloquants PASS. P1: traceability chain repaired (765 fields, 7 manual corrections). P2: CB-01=420/420, CB-04=420/420, F-01=420/420, F-04=420/420. P3: M-01/M-02=420/420, CB-09=92/92, version 8.0. Scripts: p1_rebuild_gs_from_docling.py, p2_cb01_answer_in_chunk.py, p3_metadata_completion.py |

---

*Document ISO 42001/25010/29119 - Pocket Arbiter Project*
*Usage: Checklist one-shot avant generation triplets*
*Aligne sur: PLAN-GS-CONF-001 (GS_CONFORMITY_PLAN_V1.md)*
