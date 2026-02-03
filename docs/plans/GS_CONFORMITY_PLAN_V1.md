# Plan: Mise en Conformité GS + Fine-tuning Semantic Bridge

> Date: 2026-01-26 (maj 2026-02-03)
> Sources: GS_CONFORMITY_CHECKLIST.md, GS_ANNALES_V7_OPTIMIZATION_SPEC.md, TRIPLET_GENERATION_SPEC.md, UNIFIED_TRAINING_DATA_SPEC.md, FINETUNING_RESOURCES.md, FINETUNING_LESSONS.md, **GS_SCHEMA_V2.md**
> Objectif: GS conforme → Triplets valides → Fine-tune EmbeddingGemma
> Schema: **v2.0** - 8 groupes fonctionnels (voir [specs/GS_SCHEMA_V2.md](../specs/GS_SCHEMA_V2.md))

---

## 0. SEMANTIC BRIDGE - REQUIREMENTS CRITIQUES

### 0.1 Modèle Cible
- **Model**: `google/embeddinggemma-300m-qat-q4_0-unquantized` (**QAT obligatoire pour QLoRA**)
- **Pourquoi QAT**: Poids entraînés avec simulation de quantization → robustes au bruit 4-bit
  - QAT + QLoRA = pas de distribution shift (le modèle s'attend à être quantifié)
  - Full + QLoRA = dégradation (poids non préparés pour la quantization)
  - Cohérence end-to-end: QAT → QLoRA → TFLite int4 → LiteRT Android
- **Framework**: sentence-transformers >= 3.0 + peft
- **Loss**: `CachedMultipleNegativesRankingLoss`
- **Dimensions**: 768D (ou MRL: 512/256/128)
- **Context**: 2048 tokens
- **Ref**: FINETUNING_RESOURCES.md §3.4

### 0.2 Data Requirements pour Fine-tuning

| Critère | Seuil | Source |
|---------|-------|--------|
| Format triplets | anchor/positive/negative | TRIPLET_GENERATION_SPEC §2.1 |
| BY DESIGN reformulation | 100% | UNIFIED_TRAINING_DATA_SPEC §1.3 |
| Hard negatives same_doc | >= 40% | FINETUNING_RESOURCES §3.3 |
| Batch size | >= 32 (ideal 64-128) | FINETUNING_LESSONS §2 |
| NO_DUPLICATES sampler | Obligatoire | FINETUNING_LESSONS §2 |
| Val split | 20% (100% GS) | UNIFIED_TRAINING_DATA_SPEC §3.5.3 |
| Seed fixe | 42 | ISO 12207 reproducibility |

### 0.3 Prompts EmbeddingGemma (OBLIGATOIRE)

```python
# Queries (MUST USE)
"task: search result | query: {question}"

# Documents (MUST USE)
"title: {title} | text: {chunk_text}"
```
**Ref**: FINETUNING_RESOURCES.md §3.3

### 0.4 Hyperparametres Recommandés

| Param | Valeur | Ref |
|-------|--------|-----|
| learning_rate | 2e-5 | FINETUNING_RESOURCES §4.1 |
| epochs | 1-5 | FINETUNING_RESOURCES §4.1 |
| warmup_ratio | 0.1 | FINETUNING_RESOURCES §4.1 |
| LoRA rank | 8 | FINETUNING_RESOURCES §4.2 |
| LoRA alpha | 16 | FINETUNING_RESOURCES §4.2 |
| LoRA dropout | 0.1 | FINETUNING_RESOURCES §4.2 |
| Early stopping | patience=3 | FINETUNING_LESSONS §5.2 |

### 0.5 Sources Externes Officielles

| Source | URL | Contenu |
|--------|-----|---------|
| Google AI Dev | ai.google.dev/gemma/docs/embeddinggemma | Guide officiel |
| Kaggle (Nilay Chauhan) | kaggle.com/code/nilaychauhan/fine-tune-embeddinggemma | Notebook officiel |
| HuggingFace | huggingface.co/blog/embeddinggemma | Architecture, MRL |
| Google AI Edge | ai.google.dev/edge/mediapipe | Déploiement Android |
| GitHub ai-edge-torch | github.com/google-ai-edge/ai-edge-torch | Conversion TFLite |

### 0.6 Workflow Fine-tuning → Déploiement

```
PHASE 1: GS CONFORME (ce plan)
    ├── CB-04 BY DESIGN: reformulation avec chunk visible
    ├── CB-01: chunk_match_score = 100%
    ├── Hard negatives >= 40% same_doc
    └── Export triplets JSONL

PHASE 2: FINE-TUNING (Kaggle T4)
    ├── Model: embeddinggemma-300m-qat-q4_0-unquantized
    ├── Loss: CachedMultipleNegativesRankingLoss
    ├── Data: triplets_train.jsonl (~1000+ triplets)
    └── Output: embeddinggemma-chess-fr/ (adapters)

PHASE 3: CONVERSION TFLite
    ├── Tool: ai-edge-torch
    ├── Quantization: int4/int8
    └── Output: embeddinggemma-chess-fr.tflite (~180 MB)

PHASE 4: DEPLOIEMENT ANDROID
    └── LiteRT runtime → inference on-device <15ms
```

---

## 1. REQUIREMENTS (depuis les specs)

### 1.1 Critères Bloquants (GS_CONFORMITY_CHECKLIST.md)

| ID | Critère | Seuil | Actuel | Delta |
|----|---------|-------|--------|-------|
| CB-01 | chunk_match_score = 100 | 100% testables | 2.9% (12/420) | **408** |
| CB-02 | expected_chunk_id existe | 100% | 100% | ✓ |
| CB-03 | expected_chunk_id non-null | 100% testables | 100% | ✓ |
| CB-04 | BY DESIGN (reformulation chunk visible) | 100% | **0%** | **420** |
| CB-09 | requires_context_reason | 100% si rc=true | 54% (50/92) | **42** |
| CB-05 | Zero recursive generation | 0% | 0% | ✓ |
| CB-06 | Lineage documentée (original_question) | 100% | ~100% | Audit |
| CB-07 | expected_docs present | 100% | ~100% | Audit |
| CB-08 | expected_pages present | >= 80% | ~80% | Audit |

### 1.2 Critères Qualité (GS_CONFORMITY_CHECKLIST.md)

| ID | Critère | Seuil | Actuel | Delta |
|----|---------|-------|--------|-------|
| CQ-01 | reasoning_class | 100% | 100% | ✓ |
| CQ-08 | expected_answer non-vide | 100% | 100% | ✓ |

> **Note**: CQ-02..CQ-07 definis dans GS_CONFORMITY_CHECKLIST.md §2.
> Ce sont des criteres SHOULD (non-bloquants pour l'execution des phases).

### 1.3 Critères Triplets (TRIPLET_GENERATION_SPEC.md)

| ID | Critère | Seuil | Actuel | Delta |
|----|---------|-------|--------|-------|
| CT-01 | same_doc_diff_page negatives | >= 40% | 0% | **420** |
| CT-02 | Pas duplicate negatives | 0% | N/A | - |
| CT-03 | Negative != Positive | 100% | N/A | - |
| CT-04 | Schema JSON valide | 100% | N/A | - |
| CT-05 | Split train/val 80/20 | seed=42 | N/A | - |

### 1.4 Critères Format (TRIPLET_GENERATION_SPEC.md S2.2)

| ID | Critère | Seuil | Actuel | Delta |
|----|---------|-------|--------|-------|
| F-01 | anchor finit par ? | 100% | 301/420 (71.7%) | **119** |
| F-02 | anchor >= 10 chars | 100% | 420/420 | ✓ |
| F-03 | positive >= 50 chars | 100% | 420/420 | ✓ |
| F-04 | expected_answer > 5 chars | 100% | 388/420 | **32** |

### 1.5 Critères Métadonnées (GS_ANNALES_V7_OPTIMIZATION_SPEC.md)

| ID | Critère | Seuil | Actuel | Delta |
|----|---------|-------|--------|-------|
| M-01 | difficulty présent | 100% | 386/420 (91.9%) | **34** |
| M-02 | difficulty in [0,1] | 100% | 383/420 | **37** |
| M-03 | cognitive_level | 100% | 420/420 | ✓ |
| M-04 | category | 100% | 420/420 | ✓ |

### 1.6 Critères Qualité Avancés (UNIFIED_TRAINING_DATA_SPEC.md)

| ID | Critère | Seuil | Actuel | Delta |
|----|---------|-------|--------|-------|
| QA-01 | Deduplication inter-questions | cosine < 0.95 | Non vérifié | **Audit** |
| QA-02 | Anchor independence | cosine(anchor, positive) < 0.9 | Non vérifié | **Audit** |

### 1.7 Critères Export Multi-Format (UNIFIED_TRAINING_DATA_SPEC.md)

| ID | Critère | Format | Status |
|----|---------|--------|--------|
| EX-01 | Triplets JSONL | anchor/positive/negative | **À générer** |
| EX-02 | ARES TSV | Query/Document/Answer/Label | **À générer** |
| EX-03 | BEIR | queries.jsonl + corpus.jsonl + qrels.tsv | **À générer** |
| EX-04 | RAGAS | question/answer/contexts/ground_truth | **À générer** |
| EX-05 | DVC tracking | dvc add + push | **À faire** |
| EX-06 | Composition report JSON | dataset_composition.json | **À générer** |

---

## 2. DELTA TOTAL

| Catégorie | Questions à corriger | Priorité |
|-----------|---------------------|----------|
| **CB-04 BY DESIGN** | 420 (reformulation LLM) | **P0 CRITIQUE** |
| CB-01 chunk_match_score | 408 (validation LLM) | P0 |
| CB-09 requires_context_reason | 42 | P1 |
| F-01 question finit par ? | 119 | P1 |
| F-04 expected_answer court | 32 (review) | P2 |
| M-01/M-02 difficulty | 37 | P1 |
| CT-01 hard_negatives | 420 | P1 |
| QA-01/02 Deduplication | Audit requis | P2 |
| EX-01..06 Multi-format export | 6 formats | P1 |

---

## 3. FICHIERS

| Fichier | Rôle |
|---------|------|
| `tests/data/gold_standard_annales_fr_v7.json` | GS source (420 Q) |
| `corpus/processed/chunks_mode_b_fr.json` | Corpus chunks (1857) |
| `docs/schemas/triplet_schema.json` | Schema validation |

---

## 4. PHASES

### Phase 0: Validation LLM chunk_id + BY DESIGN (CB-01 + CB-04)

**Problème Critique:**
- CB-01: 97.1% des chunk_id pointent vers des chunks qui ne permettent PAS de dériver la réponse
- CB-04: 0% des questions ont été reformulées BY DESIGN (avec chunk visible)

**Méthode combinée:** Pour CHAQUE question:

```
CONTEXTE (chunk du réglement):
{chunk_text}

QUESTION OFFICIELLE (examen DNA):
{original_question}

RÉPONSE OFFICIELLE:
{expected_answer}

TÂCHES:
1. VALIDATION: Le chunk permet-il de DÉRIVER cette réponse? (oui/non/partiel)
   - Si non: quel chunk contiendrait la réponse? (chercher dans corpus)

2. REFORMULATION BY DESIGN: Reformule la question en langage courant
   CONTRAINTES:
   - La réponse DOIT être trouvable dans le CONTEXTE
   - Garde le même sens que la question originale
   - Style oral, naturel, finissant par ?
   - Vocabulaire courant (éviter jargon technique)

OUTPUT JSON:
{
  "chunk_validated": true/false,
  "chunk_match_score": 100/0,
  "suggested_chunk_id": "...", // si chunk_validated=false
  "reformulated_question": "Question reformulée BY DESIGN ?",
  "requires_context": true/false,
  "requires_context_reason": "..." // si requires_context=true
}
```

**Output dans GS:**
```python
q['question'] = result['reformulated_question']  # BY DESIGN
q['metadata']['original_question'] = original  # Garder trace
q['metadata']['chunk_validated_llm'] = result['chunk_validated']
q['metadata']['chunk_match_score'] = result['chunk_match_score']
q['metadata']['by_design'] = True
```

**Validation Phase 0:**
```python
for q in questions:
    assert q['metadata'].get('by_design'), f"CB-04 FAIL: {q['id']}"
    assert q['question'].strip().endswith('?'), f"F-01 FAIL: {q['id']}"

validated = sum(1 for q in testables if q['metadata'].get('chunk_validated_llm'))
assert validated >= 0.9 * len(testables), "CB-01 FAIL"
```

### Phase 1: Corrections GS Métadonnées

> Note: F-01 (questions finissant par ?) est géré en Phase 0 via BY DESIGN

#### 1.1 Ajouter requires_context_reason (CB-09)
```python
VALID_REASONS = [
    "answer_requires_calculation",
    "answer_requires_context_position",
    "answer_requires_external_data",
    "answer_is_reformulation",
    "chunk_not_in_corpus"
]

for q in questions:
    if q['metadata'].get('requires_context') and not q['metadata'].get('requires_context_reason'):
        # determine_reason(): classification LLM (Claude/Gemini)
        # Input: question + metadata → Output: one of VALID_REASONS
        # Implementation: LLM analyse question+chunk characteristics
        q['metadata']['requires_context_reason'] = determine_reason(q)  # LLM call
```

#### 1.2 Compléter difficulty (M-01/M-02)
```python
for q in questions:
    diff = q['metadata'].get('difficulty')
    if diff is None or not (0 <= diff <= 1):
        if q['metadata'].get('annales_source', {}).get('success_rate'):
            q['metadata']['difficulty'] = 1 - q['metadata']['annales_source']['success_rate']
        else:
            mapping = {'Remember': 0.2, 'Understand': 0.4, 'Apply': 0.6, 'Analyze': 0.8}
            q['metadata']['difficulty'] = mapping.get(q['metadata'].get('cognitive_level'), 0.5)
```

#### 1.3 Review expected_answer courts (F-04)
Les 32 questions avec answer < 5 chars doivent être vérifiées manuellement.

**Validation Phase 1:**
```python
for q in questions:
    assert q['question'].strip().endswith('?'), f"F-01 FAIL: {q['id']}"
    if q['metadata'].get('requires_context'):
        assert q['metadata'].get('requires_context_reason'), f"CB-09 FAIL: {q['id']}"
    assert 0 <= q['metadata'].get('difficulty', -1) <= 1, f"M-01 FAIL: {q['id']}"
```

### Phase 2: Génération Hard Negatives (CT-01)

**Méthode hybride en 2 étapes:**

#### Étape 2A: Claude LLM-as-Judge (teacher sémantique)

> **Rationale**: Un LLM raisonne sur le *contenu* (ce chunk parle du meme sujet mais
> ne contient PAS la reponse) là où un embedding ne voit qu'une distance cosine.
> Pour 420 questions FR techniques (arbitrage echecs), le raisonnement > la distance.
> Ref: Databricks embedding finetuning (LLM-as-judge), NVIDIA SDG RAG (4 critères qualité)

**Prompt Claude** — pour CHAQUE question testable, on presente les 10 chunks
les plus proches (pre-filtres par EmbeddingGemma) et Claude juge:

```
QUESTION: {question}
RÉPONSE ATTENDUE: {expected_answer}
CHUNK POSITIF (contient la réponse): {positive_chunk_text}

CANDIDATS NÉGATIFS (même document ou sémantiquement proches):
[1] {chunk_id_1}: {chunk_text_1[:200]}
[2] {chunk_id_2}: {chunk_text_2[:200]}
...
[10] {chunk_id_10}: {chunk_text_10[:200]}

TÂCHES:
1. Pour chaque candidat, évalue:
   - Ce chunk permet-il de répondre à la question? (oui/non)
   - Pourquoi est-ce un bon hard negative? (thématique proche mais réponse absente)
2. Classe les candidats du meilleur hard negative au pire
3. Rejette les faux négatifs (chunks qui contiennent AUSSI la réponse)

OUTPUT JSON:
{
  "hard_negatives": [
    {
      "chunk_id": "...",
      "rank": 1,
      "is_false_negative": false,
      "reason": "Parle des missions de l'arbitre mais pas de la mise en place des cavaliers"
    },
    ...
  ],
  "rejected_false_negatives": [
    {"chunk_id": "...", "reason": "Contient aussi la réponse (Article 2.1)"}
  ]
}
```

**Avantages Claude vs teacher embedding:**
- Détecte les **faux négatifs** (chunks qui contiennent la réponse → rejet)
- Justification textuelle traçable (ISO 42001 A.6.2.3)
- Comprend le jargon FFE/FIDE en français
- Coût: ~$0.01/question × 420 = ~$4.20 total

#### Étape 2B: EmbeddingGemma pre-filtrage (candidats pour Claude)

> EmbeddingGemma fournit les 10 candidats que Claude juge ensuite.
> Le modèle QAT reste unique dans le pipeline (coherence QLoRA → TFLite → LiteRT).
> NOTE: Utiliser le QAT (pas le full) car c'est ce modèle qui sera fine-tuné.
> Les hard negatives doivent être "hard" pour le QAT, pas pour le full.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# IMPORTANT: QAT model = celui qui sera fine-tuné via QLoRA
# Les candidats hard negatives doivent refléter CE que CE modèle "voit"
model = SentenceTransformer("google/embeddinggemma-300m-qat-q4_0-unquantized")

# Encoder tout le corpus une seule fois
corpus_texts = [c["text"] for c in all_chunks]
corpus_embs = model.encode(corpus_texts, batch_size=128, show_progress_bar=True)

# Pour chaque question: top-10 candidats négatifs
for q in testable_questions:
    q_emb = model.encode(q["question"])
    positive_id = q["expected_chunk_id"]

    # Cosine similarity avec tout le corpus
    scores = np.dot(corpus_embs, q_emb) / (
        np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(q_emb)
    )

    # Exclure le positif, trier par score décroissant
    candidates = [
        (all_chunks[i], float(scores[i]))
        for i in np.argsort(-scores)
        if all_chunks[i]["id"] != positive_id
    ][:10]

    # → Envoyer ces 10 candidats à Claude pour jugement (Étape 2A)
    q["metadata"]["hard_negative_candidates"] = candidates
```

#### Étape 2C: Enrichissement same_doc (CT-01 >= 40%)

```python
# Après jugement Claude, vérifier le ratio same_doc
for q in testable_questions:
    hns = q["metadata"]["hard_negatives"]  # jugés par Claude
    positive_source = chunks[q["expected_chunk_id"]]["source"]

    same_doc = [hn for hn in hns if chunks[hn["chunk_id"]]["source"] == positive_source]
    same_doc_ratio = len(same_doc) / len(hns) if hns else 0

    # Si < 40% same_doc: ajouter des candidats same_doc non-jugés
    if same_doc_ratio < 0.4:
        same_doc_candidates = [
            c for c in all_chunks
            if c["source"] == positive_source
            and c["id"] != q["expected_chunk_id"]
            and c["id"] not in {hn["chunk_id"] for hn in hns}
        ]
        # Ajouter les plus proches sémantiquement
        # (déjà encodés par EmbeddingGemma)
```

**Output dans GS:**
```python
q['metadata']['hard_negatives'] = [
    {
        'chunk_id': '...',
        'source': 'same_doc',           # ou 'cross_doc'
        'rank': 1,                       # classé par Claude
        'reason': '...',                 # justification Claude (ISO 42001)
        'is_false_negative': False,      # validé par Claude
        'embedding_score': 0.72          # score EmbeddingGemma
    },
    ...
]
q['metadata']['hard_negative_mining'] = {
    'method': 'hybrid_claude_embeddinggemma',
    'pre_filter': 'google/embeddinggemma-300m-qat-q4_0-unquantized',
    'judge': 'claude-opus-4-5-20251101',
    'num_candidates': 10,
    'num_selected': 5,
    'false_negatives_rejected': 2
}
```

**Validation Phase 2:**
```python
for q in testables:
    hns = q['metadata'].get('hard_negatives', [])
    assert len(hns) >= 3, f"CT-01 FAIL: {q['id']}"
    # Tous validés par Claude (pas de faux négatifs)
    assert all(not hn['is_false_negative'] for hn in hns), f"CT-03 FAIL: {q['id']}"

# Ratio same_doc >= 40%
total_hns = sum(len(q['metadata']['hard_negatives']) for q in testables)
same_doc_hns = sum(
    1 for q in testables
    for hn in q['metadata']['hard_negatives']
    if hn['source'] == 'same_doc'
)
assert same_doc_hns / total_hns >= 0.4, f"CT-01 same_doc < 40%: {same_doc_hns}/{total_hns}"
```

### Phase 3: Export Multi-Format (EX-01..06)

#### 3.1 Triplets JSONL (EX-01)
**Format (TRIPLET_GENERATION_SPEC.md S2.1):**
```json
{
  "anchor": "Question BY DESIGN finissant par ?",
  "positive": "Texte du chunk (>= 50 chars)",
  "negative": "Texte hard negative (>= 50 chars)",
  "metadata": {
    "source": "gold_standard",
    "question_id": "ffe:annales:...",
    "chunk_id": "...",
    "difficulty": 0.0-1.0,
    "reasoning_class": "summary|fact_single|reasoning|arithmetic",
    "negative_mining": {
      "method": "topk_percpos",
      "source": "same_doc|cross_doc",
      "score": 0.72
    },
    "validation": {
      "human_reviewed": true,
      "chunk_validated_llm": true,
      "by_design": true
    }
  }
}
```

**Output:**
- `data/training/unified/triplets_train.jsonl`
- `data/training/unified/triplets_val.jsonl`

#### 3.2 ARES TSV (EX-02)
```tsv
Query	Document	Answer	Context_Relevance_Label
Question BY DESIGN ?	Chunk text...	expected_answer	1
```
**Output:** `data/training/unified/ares_train.tsv`, `data/training/unified/ares_val.tsv`

#### 3.3 BEIR (EX-03)
```bash
data/training/unified/beir/
├── queries.jsonl   # {"_id": "FR-ANN-001", "text": "Question"}
├── corpus.jsonl    # {"_id": "chunk_id", "title": "Article X", "text": "..."}
└── qrels.tsv       # query_id\tcorpus_id\tscore
```

#### 3.4 RAGAS (EX-04)
```json
{"question": "...", "answer": "", "contexts": ["chunk_text"], "ground_truth": "expected_answer"}
```
**Output:** `data/training/unified/ragas_val.jsonl`

#### 3.5 DVC Tracking (EX-05)
```bash
dvc add data/training/unified/
git add data/training/unified.dvc data/training/.gitignore
```

#### 3.6 Composition Report (EX-06)
```json
{
  "version": "1.0",
  "seed": 42,
  "source": {
    "gold_standard": "gold_standard_annales_fr_v7.json",
    "gold_standard_version": "7.7",
    "chunks_file": "chunks_mode_b_fr.json"
  },
  "statistics": {
    "total_questions": 420,
    "testable": 328,
    "requires_context": 92,
    "by_design_reformulated": 420
  },
  "splits": {
    "train": {"count": 262, "percentage": 80},
    "val": {"count": 66, "percentage": 20}
  },
  "hard_negative_distribution": {
    "same_doc_diff_page": ">= 40%",
    "cross_doc_semantic": "~60%"
  },
  "quality_gates": {
    "CB-01_chunk_match_100": true,
    "CB-04_by_design": true,
    "val_100_percent_gs": true
  }
}
```
**Output:** `data/training/unified/dataset_composition.json`

**Split (CT-05):**
- Train: 80% (testables ~262)
- Val: 20% (testables ~66)
- Stratifié par reasoning_class
- Seed: 42

**Validation Phase 3:**
```python
import jsonschema, json

# Schema validation
with open('docs/schemas/triplet_schema.json') as f:
    schema = json.load(f)
for triplet in triplets:
    jsonschema.validate(triplet, schema)

# Val = 100% gold_standard
for t in val_triplets:
    assert t['metadata']['source'] == 'gold_standard'

# No data leakage
train_ids = {t['metadata']['question_id'] for t in train_triplets}
val_ids = {t['metadata']['question_id'] for t in val_triplets}
assert train_ids.isdisjoint(val_ids)

# All formats generated
from pathlib import Path
unified = Path('data/training/unified')
assert (unified / 'triplets_train.jsonl').exists()
assert (unified / 'triplets_val.jsonl').exists()
assert (unified / 'ares_train.tsv').exists()
assert (unified / 'beir/queries.jsonl').exists()
assert (unified / 'ragas_val.jsonl').exists()
assert (unified / 'dataset_composition.json').exists()
```

---

## 5. VALIDATION FINALE

> **Note**: Ce script est un controle rapide "one-shot" en une commande.
> Il couvre un SOUS-ENSEMBLE des criteres des quality gates §6.4:
> CB-04, CB-01, CB-09, F-01, M-01, CT-01, EX-01..06 (7 criteres sur ~30).
> Les gates §6.4 sont la reference normative complete (ISO 29119).

```bash
python -c "
import json
import sys
from pathlib import Path

with open('tests/data/gold_standard_annales_fr_v7.json', encoding='utf-8') as f:
    gs = json.load(f)

questions = gs['questions']
testables = [q for q in questions if not q.get('metadata',{}).get('requires_context')]
errors = []

# CB-04: BY DESIGN (CRITIQUE)
for q in questions:
    if not q.get('metadata',{}).get('by_design'):
        errors.append(f'CB-04: {q[\"id\"]} (not by_design)')

# CB-01: chunk_match_score = 100 pour testables
for q in testables:
    if q.get('metadata',{}).get('chunk_match_score') != 100:
        errors.append(f'CB-01: {q[\"id\"]}')

# CB-09: requires_context_reason
rc = [q for q in questions if q.get('metadata',{}).get('requires_context')]
for q in rc:
    if not q.get('metadata',{}).get('requires_context_reason'):
        errors.append(f'CB-09: {q[\"id\"]}')

# F-01: question finit par ?
for q in questions:
    if not q['question'].strip().endswith('?'):
        errors.append(f'F-01: {q[\"id\"]}')

# M-01: difficulty in [0,1]
for q in questions:
    d = q.get('metadata',{}).get('difficulty')
    if d is None or not (0 <= d <= 1):
        errors.append(f'M-01: {q[\"id\"]}')

# CT-01: hard_negatives >= 3
for q in testables:
    if len(q.get('metadata',{}).get('hard_negatives',[])) < 3:
        errors.append(f'CT-01: {q[\"id\"]}')

# EX-01..06: All formats generated
unified = Path('data/training/unified')
required_files = [
    'triplets_train.jsonl', 'triplets_val.jsonl',
    'ares_train.tsv', 'beir/queries.jsonl',
    'ragas_val.jsonl', 'dataset_composition.json'
]
for f in required_files:
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

## 6. AUTOCONTRÔLE QUALITÉ

> **ISO 29119**: Vérification et validation à chaque niveau
> **ISO 42001 A.7.3**: Documentation des décisions IA
> **QUALITY_REQUIREMENTS.md §4**:
>   - §4.1 Gold Standard: TD-01..05 (taille, couverture, adversariales, validation, dedup)
>   - §4.2 Triplets: TT-01..05 (hard neg, anchor indep, diversity, grounded, collapse)
>   - §4.3 Benchmarks: TB-01..05 (MTEB, MMTEB, BEIR, RAGAS, ARES) — post-fine-tuning
>   - §4.4 Synthetic: TS-01..05 (ratio, LLM-judge, human review, dedup, diversity) — si applicable
>   - §4.5 DVC: TV-01..04 (version, reproducibility, lineage, schema)

### 6.1 Niveaux de qualité

```
Q1 ─ Par question     Autocontrôle LLM (confidence + flags)
Q2 ─ Par batch         Spot-check humain 10%
Q3 ─ Par phase         Quality gate (critères + régression)
Q4 ─ Par export        Schema validation + composition report
Q5 ─ Global            Validation finale + benchmark recall
```

### 6.2 Q1 — Autocontrôle par question (LLM self-assessment)

Pour CHAQUE question traitée par le LLM, le JSON de sortie DOIT inclure:

```json
{
  "quality_check": {
    "confidence": 0.92,
    "flags": [],
    "reformulation_preserves_meaning": true,
    "chunk_derivability": "certain|probable|doubtful|impossible",
    "needs_human_review": false
  }
}
```

**Règles d'escalade:**
| Condition | Action |
|-----------|--------|
| `confidence < 0.7` | Flag pour review humain |
| `chunk_derivability == "doubtful"` | Flag + chercher chunk alternatif |
| `chunk_derivability == "impossible"` | Marquer `requires_context=true` avec reason |
| `reformulation_preserves_meaning == false` | Rejeter, garder question originale |
| `flags` non vide | Review humain obligatoire |

**Métriques Q1 attendues:**
| Métrique | Cible | Alerte |
|----------|-------|--------|
| Taux confidence >= 0.7 | >= 90% | < 85% → revoir prompt |
| Taux needs_human_review | <= 15% | > 25% → revoir prompt |
| Taux chunk_derivability certain | >= 70% testables | < 60% → problème corpus |

### 6.3 Q2 — Spot-check par batch

**Protocole:**
```
Pour chaque batch de ~20 questions:
  1. LLM traite les 20 questions
  2. Sauvegarder: gs_v7_batch_{n}.json (checkpoint)
  3. Humain review 2 questions au hasard (10%)
     - La reformulation garde le même sens?
     - La réponse est dérivable du chunk identifié?
     - Le style est naturel, finit par ?
     - Les metadata sont cohérentes?
  4. Si 1/2 échoue → review les 20, ajuster le prompt
  5. Si 2/2 échoue → STOP, revoir la méthode
  6. Si 0/2 échoue → batch validé, continuer
```

**Checkpoints:**
```
tests/data/checkpoints/
├── gs_v7_phase0_batch_01.json    # Questions 1-20
├── gs_v7_phase0_batch_02.json    # Questions 21-40
├── ...
├── gs_v7_phase0_batch_21.json    # Questions 401-420
├── gs_v7_phase0_VALIDATED.json   # Merge final après gate
└── spot_check_log.jsonl          # Log des reviews humains
```

**Format spot_check_log.jsonl:**
```json
{"batch": 1, "question_id": "ffe:annales:clubs:007:...", "reviewer": "human", "pass": true, "notes": ""}
{"batch": 1, "question_id": "ffe:annales:clubs:014:...", "reviewer": "human", "pass": true, "notes": ""}
```

> **Note d'assemblage**: Les gate functions ci-dessous sont presentees comme
> blocs independants. En script unifie, `regression_check()` (§6.6) doit etre
> defini AVANT les gates qui l'appellent. Imports communs: `from pathlib import Path`,
> `import json`, `import jsonschema`.

### 6.4 Q3 — Quality gates inter-phases

#### GATE 0→1: BY DESIGN + chunk validation

| # | Critère | Seuil | Bloquant |
|---|---------|-------|:--------:|
| G0-1 | CB-04 by_design | 100% | OUI |
| G0-2 | CB-01 chunk_match_score=100 testables | >= 90% | OUI |
| G0-3 | F-01 finit par ? | 100% | OUI |
| G0-4 | original_question préservée | 100% | OUI |
| G0-5 | Spot-check: 0 échec sur sample 10% | 0 échec | OUI |
| G0-6 | Q1 confidence >= 0.7 | >= 90% | NON (alerte) |
| G0-7 | REGRESSION CB-02, CB-03 | PASS | OUI |
| G0-8 | REGRESSION CQ-01, CQ-08 | PASS | OUI |
| G0-9 | REGRESSION F-02, F-03, M-03, M-04 | PASS | OUI |
| G0-10 | CB-05 zero recursive generation | 0% | OUI |
| G0-11 | CB-06 lineage documentée | 100% | OUI |
| G0-12 | CB-07 expected_docs present | 100% | OUI |
| G0-13 | CB-08 expected_pages present | >= 80% | OUI |

```python
from pathlib import Path
import json as _json

# Script gate Phase 0 → Phase 1
def gate_phase0(gs):
    errors, warnings = [], []

    questions = gs['questions']
    testables = [q for q in questions if not q.get('metadata',{}).get('requires_context')]

    # Critères bloquants
    for q in questions:
        if not q.get('metadata',{}).get('by_design'):
            errors.append(f"G0-1 FAIL: {q['id']} not by_design")
        if not q['question'].strip().endswith('?'):
            errors.append(f"G0-3 FAIL: {q['id']} not ending with ?")
        if not q.get('metadata',{}).get('original_question'):
            errors.append(f"G0-4 FAIL: {q['id']} no original_question")

    validated = sum(1 for q in testables if q.get('metadata',{}).get('chunk_match_score') == 100)
    if validated < 0.9 * len(testables):
        errors.append(f"G0-2 FAIL: {validated}/{len(testables)} chunk_match=100")

    # Régression
    for q in questions:
        if not q.get('expected_chunk_id'):
            errors.append(f"G0-7 REGRESSION CB-02: {q['id']}")
        if not q.get('metadata',{}).get('reasoning_class'):
            errors.append(f"G0-8 REGRESSION CQ-01: {q['id']}")
        if not q.get('expected_answer','').strip():
            errors.append(f"G0-8 REGRESSION CQ-08: {q['id']}")
        if len(q['question'].strip()) < 10:
            errors.append(f"G0-9 REGRESSION F-02: {q['id']}")
        # G0-9 REGRESSION F-03: requires corpus lookup, deferred to regression_check()
        if not q.get('metadata', {}).get('cognitive_level'):
            errors.append(f"G0-9 REGRESSION M-03: {q['id']}")
        if not q.get('metadata', {}).get('category'):
            errors.append(f"G0-9 REGRESSION M-04: {q['id']}")

    # G0-10: CB-05 zero recursive generation
    for q in questions:
        if q.get('metadata',{}).get('generation_depth', 0) > 0:
            errors.append(f"G0-10 FAIL: {q['id']} recursive generation detected")

    # G0-11: CB-06 lineage (meme check que G0-4, scope normatif different)
    # G0-4 = preservation donnees; G0-11 = ISO 42001 A.6.2.3 tracabilite lineage
    for q in questions:
        if not q.get('metadata',{}).get('original_question'):
            errors.append(f"G0-11 FAIL: {q['id']} no lineage (original_question)")

    # G0-12: CB-07 expected_docs
    for q in questions:
        if not q.get('metadata',{}).get('expected_docs'):
            errors.append(f"G0-12 FAIL: {q['id']} no expected_docs")

    # G0-13: CB-08 expected_pages >= 80%
    with_pages = sum(1 for q in questions if q.get('metadata',{}).get('expected_pages'))
    if with_pages < 0.8 * len(questions):
        errors.append(f"G0-13 FAIL: expected_pages {with_pages}/{len(questions)} < 80%")

    # G0-5: Spot-check log (tous pass)
    spot_log = Path('tests/data/checkpoints/spot_check_log.jsonl')
    if spot_log.exists():
        with open(spot_log, encoding='utf-8') as f:
            checks = [_json.loads(line) for line in f if line.strip()]
        fails = [c for c in checks if not c.get('pass')]
        if fails:
            errors.append(f"G0-5 FAIL: {len(fails)} spot-check failures")
    else:
        errors.append("G0-5 FAIL: spot_check_log.jsonl not found")

    # G0-6: Alerte non-bloquante
    qc = [q for q in questions if q.get('metadata',{}).get('quality_check',{}).get('confidence',1) < 0.7]
    if len(qc) > 0.1 * len(questions):
        warnings.append(f"G0-6 WARN: {len(qc)}/{len(questions)} low confidence")

    return errors, warnings
```

#### GATE 1→2: Métadonnées complètes

| # | Critère | Seuil | Bloquant |
|---|---------|-------|:--------:|
| G1-1 | CB-09 requires_context_reason | 100% rc | OUI |
| G1-2 | M-01 difficulty present | 100% | OUI |
| G1-3 | M-02 difficulty in [0,1] | 100% | OUI |
| G1-4 | F-04 expected_answer > 5 chars review | 100% traité | OUI |
| G1-5 | REGRESSION: tous critères Phase 0 | PASS | OUI |

```python
from pathlib import Path

def gate_phase1(gs):
    errors, warnings = [], []
    questions = gs['questions']
    rc = [q for q in questions if q.get('metadata',{}).get('requires_context')]

    # G1-1: CB-09
    for q in rc:
        if not q.get('metadata',{}).get('requires_context_reason'):
            errors.append(f"G1-1 FAIL: {q['id']} no requires_context_reason")

    # G1-2 + G1-3: M-01/M-02
    for q in questions:
        d = q.get('metadata',{}).get('difficulty')
        if d is None:
            errors.append(f"G1-2 FAIL: {q['id']} no difficulty")
        elif not (0 <= d <= 1):
            errors.append(f"G1-3 FAIL: {q['id']} difficulty={d} not in [0,1]")

    # G1-4: F-04 review status
    short_answers = [q for q in questions if len(q.get('expected_answer','').strip()) <= 5]
    for q in short_answers:
        if not q.get('metadata',{}).get('short_answer_reviewed'):
            errors.append(f"G1-4 FAIL: {q['id']} short answer not reviewed")

    # G1-5: Regression Phase 0
    reg_errors = regression_check(gs, phase_completed=0)
    errors.extend(f"G1-5 {e}" for e in reg_errors)

    return errors, warnings
```

#### GATE 2→3: Hard negatives validés

| # | Critère | Seuil | Bloquant |
|---|---------|-------|:--------:|
| G2-1 | CT-01 hard_negatives >= 3 | 100% testables | OUI |
| G2-2 | CT-02 pas de duplicate negatives | 0% | OUI |
| G2-3 | CT-03 negative != positive | 100% | OUI |
| G2-4 | same_doc ratio | >= 40% | OUI |
| G2-5 | Spot-check hard negatives 10% | 0 échec | OUI |
| G2-6 | Faux négatifs rejetés par LLM | 100% documentés | OUI |
| G2-7 | REGRESSION: tous critères Phases 0+1 | PASS | OUI |

```python
from pathlib import Path
import json as _json

def gate_phase2(gs, chunks_by_id):
    errors, warnings = [], []
    questions = gs['questions']
    testables = [q for q in questions if not q.get('metadata',{}).get('requires_context')]

    total_hns, same_doc_hns = 0, 0

    for q in testables:
        hns = q.get('metadata',{}).get('hard_negatives', [])
        positive_id = q.get('expected_chunk_id','')

        # G2-1: >= 3 hard negatives
        if len(hns) < 3:
            errors.append(f"G2-1 FAIL: {q['id']} has {len(hns)} hard negatives")

        neg_ids = set()
        for hn in hns:
            hn_id = hn.get('chunk_id','')
            # G2-3: negative != positive
            if hn_id == positive_id:
                errors.append(f"G2-3 FAIL: {q['id']} negative == positive ({hn_id})")
            # G2-2: duplicates
            if hn_id in neg_ids:
                errors.append(f"G2-2 FAIL: {q['id']} duplicate negative {hn_id}")
            neg_ids.add(hn_id)
            # G2-4: same_doc ratio
            total_hns += 1
            if hn.get('source') == 'same_doc':
                same_doc_hns += 1
            # G2-6: false negatives documented
            if hn.get('is_false_negative') and not hn.get('reason'):
                errors.append(f"G2-6 FAIL: {q['id']} false negative not documented")

    # G2-4: global same_doc ratio
    if total_hns > 0 and same_doc_hns / total_hns < 0.4:
        errors.append(f"G2-4 FAIL: same_doc ratio {same_doc_hns}/{total_hns} < 40%")

    # G2-5: Spot-check (same pattern as G0-5)
    spot_log = Path('tests/data/checkpoints/spot_check_phase2_log.jsonl')
    if spot_log.exists():
        import json as _json
        with open(spot_log, encoding='utf-8') as f:
            checks = [_json.loads(line) for line in f if line.strip()]
        fails = [c for c in checks if not c.get('pass')]
        if fails:
            errors.append(f"G2-5 FAIL: {len(fails)} spot-check failures")
    else:
        errors.append("G2-5 FAIL: spot_check_phase2_log.jsonl not found")

    # G2-7: Regression Phases 0+1
    reg_errors = regression_check(gs, phase_completed=1)
    errors.extend(f"G2-7 {e}" for e in reg_errors)

    return errors, warnings
```

#### GATE 3→Fine-tuning: Export validé

| # | Critère | Seuil | Bloquant |
|---|---------|-------|:--------:|
| G3-1 | EX-01..06 tous formats générés | 6/6 | OUI |
| G3-2 | CT-04 schema JSON valide | 100% | OUI |
| G3-3 | CT-05 split 80/20 seed=42 | Exact | OUI |
| G3-4 | No data leakage train/val | 0 overlap | OUI |
| G3-5 | Val = 100% gold_standard | TRUE | OUI |
| G3-6 | QA-01 deduplication < 5% + QA-02 anchor < 0.9 | < 5% / < 0.9 | OUI |
| G3-7 | REGRESSION complète Phases 0+1+2 | PASS | OUI |

```python
def gate_phase3(gs):
    import json as _json
    import jsonschema
    from pathlib import Path

    errors, warnings = [], []
    unified = Path('data/training/unified')

    # G3-1: All 6 formats exist
    required = {
        'EX-01': 'triplets_train.jsonl',
        'EX-01b': 'triplets_val.jsonl',
        'EX-02': 'ares_train.tsv',
        'EX-03': 'beir/queries.jsonl',
        'EX-04': 'ragas_val.jsonl',
        'EX-05': 'triplets_train.jsonl.dvc',  # DVC tracked
        'EX-06': 'dataset_composition.json',
    }
    for ex_id, path in required.items():
        if not (unified / path).exists():
            errors.append(f"G3-1 FAIL: {ex_id} missing {path}")

    # G3-2: Schema validation
    schema_path = Path('docs/schemas/triplet_schema.json')
    if schema_path.exists():
        with open(schema_path, encoding='utf-8') as f:
            schema = _json.load(f)
        for split in ['triplets_train.jsonl', 'triplets_val.jsonl']:
            fpath = unified / split
            if fpath.exists():
                with open(fpath, encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            jsonschema.validate(_json.loads(line), schema)
                        except jsonschema.ValidationError as e:
                            errors.append(f"G3-2 FAIL: {split}:{i} {e.message[:80]}")

    # G3-3: Split 80/20 seed=42
    comp_path = unified / 'dataset_composition.json'
    if comp_path.exists():
        with open(comp_path, encoding='utf-8') as f:
            comp = _json.load(f)
        splits = comp.get('splits', {})
        if splits.get('train', {}).get('percentage') != 80:
            errors.append(f"G3-3 FAIL: train split != 80%")
        if splits.get('val', {}).get('percentage') != 20:
            errors.append(f"G3-3 FAIL: val split != 20%")
        if comp.get('seed') != 42:
            errors.append(f"G3-3 FAIL: seed != 42 (got {comp.get('seed')})")

    # G3-4: No data leakage
    train_ids, val_ids = set(), set()
    for split, id_set in [('triplets_train.jsonl', train_ids), ('triplets_val.jsonl', val_ids)]:
        fpath = unified / split
        if fpath.exists():
            with open(fpath, encoding='utf-8') as f:
                for line in f:
                    t = _json.loads(line)
                    id_set.add(t.get('metadata',{}).get('question_id',''))
    overlap = train_ids & val_ids
    if overlap:
        errors.append(f"G3-4 FAIL: {len(overlap)} question_ids in both train/val")

    # G3-5: Val = 100% gold_standard
    val_path = unified / 'triplets_val.jsonl'
    if val_path.exists():
        with open(val_path, encoding='utf-8') as f:
            for line in f:
                t = _json.loads(line)
                if t.get('metadata',{}).get('source') != 'gold_standard':
                    errors.append(f"G3-5 FAIL: val contains non-GS source")
                    break

    # G3-6: QA-01 deduplication + QA-02 anchor independence (BLOQUANT)
    if comp_path.exists():
        dedup_rate = comp.get('quality_audits', {}).get('duplicate_rate')
        if dedup_rate is not None and dedup_rate >= 0.05:
            errors.append(f"G3-6 FAIL: duplicate_rate {dedup_rate:.1%} >= 5%")
        elif dedup_rate is None:
            errors.append("G3-6 FAIL: deduplication audit not performed")
        # QA-02/TT-02: cosine(anchor, positive) < 0.9
        anchor_sim = comp.get('quality_audits', {}).get('max_anchor_positive_cosine')
        if anchor_sim is not None and anchor_sim >= 0.9:
            errors.append(f"G3-6 FAIL: anchor independence {anchor_sim:.2f} >= 0.9")
        elif anchor_sim is None:
            errors.append("G3-6 FAIL: anchor independence audit not performed")

    # G3-7: Regression complète
    reg_errors = regression_check(gs, phase_completed=2)
    errors.extend(f"G3-7 {e}" for e in reg_errors)

    return errors, warnings
```

### 6.5 Checkpoint et rollback

**Stratégie par phase:**

| Phase | Checkpoint | Rollback |
|-------|-----------|----------|
| Phase 0 | `gs_v7_phase0_batch_{n}.json` toutes les 20 Q | Recharger batch n-1 |
| Phase 1 | `gs_v7_phase1_COMPLETE.json` | Recharger phase0_VALIDATED |
| Phase 2 | `gs_v7_phase2_batch_{n}.json` toutes les 20 Q | Recharger batch n-1 |
| Phase 3 | `data/training/unified/` versionné DVC | `dvc checkout` |

**Git commits = snapshots:**
```
Chaque phase validée (gate PASS) → git commit
Si gate FAIL → corriger et re-tester, PAS de commit
Si correction impossible → git revert au commit précédent
```

### 6.6 Script de régression unifié

```python
def regression_check(gs, phase_completed):
    """Vérifie que les critères des phases précédentes sont toujours PASS."""
    errors = []
    questions = gs['questions']
    testables = [q for q in questions if not q.get('metadata',{}).get('requires_context')]
    rc = [q for q in questions if q.get('metadata',{}).get('requires_context')]

    # Critères TOUJOURS vérifiés (invariants)
    for q in questions:
        if not q.get('expected_chunk_id'):
            errors.append(f"REGRESSION CB-02: {q['id']}")
        if not q.get('metadata',{}).get('reasoning_class'):
            errors.append(f"REGRESSION CQ-01: {q['id']}")
        if not q.get('expected_answer','').strip():
            errors.append(f"REGRESSION CQ-08: {q['id']}")
        if len(q['question'].strip()) < 10:
            errors.append(f"REGRESSION F-02: {q['id']}")
        if not q.get('metadata',{}).get('cognitive_level'):
            errors.append(f"REGRESSION M-03: {q['id']}")
        if not q.get('metadata',{}).get('category'):
            errors.append(f"REGRESSION M-04: {q['id']}")

    if phase_completed >= 0:
        for q in questions:
            if not q.get('metadata',{}).get('by_design'):
                errors.append(f"REGRESSION CB-04: {q['id']}")
            if not q['question'].strip().endswith('?'):
                errors.append(f"REGRESSION F-01: {q['id']}")

    if phase_completed >= 1:
        for q in rc:
            if not q.get('metadata',{}).get('requires_context_reason'):
                errors.append(f"REGRESSION CB-09: {q['id']}")
        for q in questions:
            d = q.get('metadata',{}).get('difficulty')
            if d is None or not (0 <= d <= 1):
                errors.append(f"REGRESSION M-01: {q['id']}")

    if phase_completed >= 2:
        for q in testables:
            hns = q.get('metadata',{}).get('hard_negatives',[])
            if len(hns) < 3:
                errors.append(f"REGRESSION CT-01: {q['id']}")
            # F-03: positive chunk text >= 50 chars (vérifiable via chunk)
            # Vérifié dans validate_quality_gates(), pas ici (besoin du corpus)

    return errors
```

---

## 7. COMMITS

```bash
# Phase 0: BY DESIGN + chunk validation
git commit -m "fix(gs): v7.7 - BY DESIGN reformulation + chunk validation

- CB-04: 420/420 questions reformulated with chunk visible
- CB-01: chunk_id validated by LLM (chunk_match_score=100)
- F-01: all questions end with ?
- Preserves original_annales for traceability

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Phase 1: Metadata fixes
git commit -m "fix(gs): v7.7.1 - metadata completion

- CB-09: requires_context_reason for all rc=true
- M-01: difficulty in [0,1] for all

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Phase 2: Hard negatives
git commit -m "feat(gs): v7.7.2 - hard negatives hybrid Claude+EmbeddingGemma

- CT-01: 3-5 hard negatives per question, validated by LLM-as-judge
- same_doc_diff_page >= 40%
- Method: EmbeddingGemma pre-filter (top-10) + Claude judge (rank + false negative rejection)
- ISO 42001: justification textuelle pour chaque negative

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Phase 3: Export triplets multi-format
git commit -m "feat(triplets): unified training data for semantic bridge

- 420 questions → ~1260 triplets
- Formats: JSONL, ARES TSV, BEIR, RAGAS
- Split: 80/20 train/val (seed=42)
- EmbeddingGemma prompts applied
- DVC tracked

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## 8. MÉTRIQUES CIBLES

### 8.1 Conformité GS

| Métrique | Cible |
|----------|-------|
| CB-04 BY DESIGN | 100% |
| CB-01 chunk_match_score=100 | 100% testables |
| CB-09 requires_context_reason | 100% |
| F-01 question finit par ? | 100% |
| M-01 difficulty in [0,1] | 100% |

### 8.2 Triplets pour Semantic Bridge

| Métrique | Cible | Justification |
|----------|-------|---------------|
| CT-01 hard_negatives >= 3 | 100% testables | Minimum pour contrastive learning |
| same_doc_diff_page ratio | >= 40% | FINETUNING_RESOURCES §3.3 |
| false_negatives_rejected | 100% filtrés | Claude LLM-as-judge (CT-03) |
| Total triplets | ~1260 | 420 Q × 3 negatives |
| Train/val split | 80/20 | UNIFIED_TRAINING_DATA_SPEC |
| Val = 100% GS | TRUE | Pas de synthétique en val |

### 8.3 Fine-tuning (Post-export)

| Métrique | Cible Baseline | Cible Fine-tuned |
|----------|----------------|------------------|
| Recall@5 FR | 91.56% | >= 95% |
| nDCG@10 | ~60% | >= 70% |
| Hard cases resolved | 46/150 fail | <= 25/150 |

---

## 9. RÉFÉRENCES ARXIV/INDUSTRIE

| Ref | Titre | Usage |
|-----|-------|-------|
| arXiv:2405.17428 | NV-Embed-v2 | Hard negative mining (relative_margin) |
| arXiv:2407.15831 | NV-Retriever | TopK-PercPos (pre-filtrage EmbeddingGemma) |
| arXiv:2411.14831 | RAGen | BY DESIGN validation |
| arXiv:2409.08239 | Source2Synth | Grounded generation |
| arXiv:2411.19710 | Know Your RAG | reasoning_class |
| arXiv:2407.06564 | SoftDedup | Deduplication |
| Databricks Blog | Embedding Finetuning | LLM-as-judge pour hard negatives |
| NVIDIA SDG RAG | Synthetic Data Generation | 4 critères qualité LLM |

### 9.1 Architecture Hard Negatives Hybride

```
Phase 2 Pipeline:

  EmbeddingGemma-300m                    Claude Opus 4.5
  (pre-filtrage)                         (LLM-as-judge)
  ┌──────────────┐                       ┌──────────────┐
  │ Encode 1857  │    top-10 candidats   │ Pour chaque  │
  │ chunks +     │ ──────────────────→   │ question:    │
  │ 420 queries  │                       │ - Juge 10    │
  │              │                       │ - Rejette FN │
  │ cosine sim   │                       │ - Classe 1→5 │
  │ → top 10     │                       │ - Justifie   │
  └──────────────┘                       └──────┬───────┘
                                                │
                                    5 hard negatives validés
                                    + reason (ISO 42001)
                                    + same_doc >= 40%
```

---

## 10. STATUS

- [ ] Phase 0: BY DESIGN + chunk validation
- [ ] Phase 1: Metadata fixes
- [ ] Phase 2: Hard negatives (hybrid Claude + EmbeddingGemma)
- [ ] Phase 3: Multi-format export
- [ ] Validation finale

---

## 11. LESSONS LEARNED

> **Note**: Zero dette technique. Tous les findings actionnables sont corriges.
> Cette section documente les lecons tirees des erreurs de processus git
> et les clarifications post-audit.

| Finding | Description | ISO | Lecon |
|---------|-------------|-----|-------|
| L-1 | Cycle add/delete/add/delete hard negatives (760 lignes) | 12207 | Feature branches pour code experimental |
| L-2 | Commit 0008c9f melange 4 changements non-lies | 12207 | Un concern = un commit atomique |
| L-3 | Type `data(gs):` non-conventionnel | 12207 | Types valides: feat/fix/test/docs/refactor/chore |
| M-7 | Finding fantome dans matrice de tracabilite audit | 29119 | Verifier 1:1 entre matrice et descriptions de commits. M-7 "invariants GS" n'avait pas de description ni d'action concrete — N/A |
| L-8 | Secret API Cerebras dans historique git | 27001 | git filter-repo applique. Cle a rotation obligatoire. Ne jamais committer de secrets, meme temporairement |
