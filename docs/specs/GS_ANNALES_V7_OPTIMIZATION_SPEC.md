# Optimisation Gold Standard Annales v7 pour Triplets

> **Document ID**: SPEC-GS-OPT-001
> **ISO Reference**: ISO 42001, ISO 25010, ISO 29119
> **Version**: 1.0
> **Date**: 2026-01-25
> **Statut**: OBLIGATOIRE
> **Classification**: Critique
> **Sources**: arXiv, HuggingFace, NVIDIA, Sentence Transformers, COLING 2025

---

## 0. Contexte

### 0.1 Fichiers Gold Standard

| Fichier | Status | Usage | Questions |
|---------|--------|-------|-----------|
| `gold_standard_annales_fr_v7.json` | **ACTIF** | Training triplets | 420 |
| `gold_standard_fr.json` | **DEPRECATED** | Legacy (remplace par adversarial_questions.json) | 318 |
| `adversarial_questions.json` | **ACTIF** | Evaluation adversariale | 105 |

### 0.2 Etat Actuel v7.3.0

| Metrique | Valeur | Cible Industrie | Gap |
|----------|--------|-----------------|-----|
| Questions totales | 420 | >= 400 | ✅ OK |
| Questions avec "?" | 287 (68%) | 100% | ❌ -32% |
| question_type complete | 386 (92%) | 100% | ⚠️ -8% |
| Diversity (4 classes) | 2 classes | 4 classes | ❌ Missing |
| Hard negatives | Non generes | 100% | ❌ TODO |

---

## 1. Standards Industrie (State of the Art 2024-2025)

### 1.1 Training Triplets - Sources Cles

| Source | Reference | Contribution Cle |
|--------|-----------|------------------|
| **NV-Embed-v2** | [arXiv:2405.17428](https://arxiv.org/abs/2405.17428) | Positive-aware hard negative mining, MTEB #1 |
| **NV-Retriever** | [arXiv:2407.15831](https://arxiv.org/abs/2407.15831) | TopK-PercPos false negative removal |
| **GTE** | [arXiv:2308.03281](https://arxiv.org/abs/2308.03281) | Multi-stage contrastive learning |
| **Know Your RAG** | [arXiv:2411.19710](https://arxiv.org/abs/2411.19710) | 4-class taxonomy, statement extraction |
| **Sentence Transformers v3** | [HuggingFace Blog](https://huggingface.co/blog/train-sentence-transformers) | Training best practices |

### 1.2 Taxonomie Question-Context (Know Your RAG - COLING 2025)

| Classe | Definition | % Cible | Actuel v7 |
|--------|------------|---------|-----------|
| **fact_single** | Reponse = 1 unite info dans contexte | 40-50% | ~51% (factual) |
| **summary** | Reponse = multiple unites info | 15-25% | ~37% (scenario) |
| **reasoning** | Reponse inferable mais non explicite | 10-20% | ~3% (comparative) |
| **unanswerable** | Info absente du contexte | 10-15% | 0% ❌ |

> **CRITICAL**: 95% des Q&A generes par prompts simples sont fact_single.
> Solution: Statement extraction strategy (answer-first).

### 1.3 Hard Negative Mining (NV-Embed-v2)

```
┌─────────────────────────────────────────────────────────────────────┐
│  POSITIVE-AWARE HARD NEGATIVE MINING (NV-Retriever)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Probleme: Naive mining → 70% false negatives (MS-MARCO)            │
│                                                                      │
│  Solution: TopK-PercPos                                             │
│  ├── Teacher model: E5-Mistral-7B pour scoring                      │
│  ├── Positive score: sim(query, positive)                           │
│  ├── Negative threshold: 95% du positive score                      │
│  └── Rejet: candidates avec score > threshold = false negatives     │
│                                                                      │
│  Resultat: Hard negatives sans contamination positive               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Two-Stage Training (NV-Embed, GTE)

| Stage | Focus | Data | Loss |
|-------|-------|------|------|
| **Stage 1** | Retrieval | Triplets + hard negatives | MultipleNegativesRankingLoss |
| **Stage 2** | Generalisation | Non-retrieval tasks blend | Contrastive sans in-batch |

---

## 2. Optimisations Requises GS Annales v7

### 2.1 P0 - CRITIQUE (Avant generation triplets)

#### 2.1.1 Normaliser format questions

```python
# PROBLEME: 133/420 questions sans "?"
# SOLUTION: Ajouter "?" si manquant

def normalize_question(q: str) -> str:
    q = q.strip()
    if not q.endswith('?'):
        # Verifier si c'est une vraie question
        if any(q.lower().startswith(w) for w in
               ['que ', 'quel', 'quand', 'comment', 'pourquoi', 'où', 'qui']):
            q += ' ?'
    return q
```

#### 2.1.2 Completer question_type

```python
# PROBLEME: 34 questions avec question_type=None
# SOLUTION: Inference depuis cognitive_level + answer_type

INFERENCE_MAP = {
    ('Remember', 'extractive'): 'factual',
    ('Remember', 'multiple_choice'): 'factual',
    ('Apply', 'multiple_choice'): 'scenario',
    ('Understand', 'abstractive'): 'procedural',
    ('Analyze', '*'): 'comparative',
}
```

#### 2.1.3 Ajouter reasoning_class (Know Your RAG)

> **Note**: Le champ `reasoning_type` existe deja (single-hop/multi-hop/temporal).
> `reasoning_class` est un champ **supplementaire** derive de `reasoning_type` + `question_type`.

```json
{
  "metadata": {
    "reasoning_type": "multi-hop",        // EXISTANT (RAGAS/BEIR)
    "reasoning_class": "summary",         // NOUVEAU (Know Your RAG)
    "reasoning_class_method": "inferred"  // NOUVEAU (tracabilite)
  }
}
```

**Mapping `reasoning_class` = f(reasoning_type, question_type)**:

| reasoning_type | question_type | reasoning_class |
|----------------|---------------|-----------------|
| single-hop | factual | fact_single |
| single-hop | procedural | fact_single |
| multi-hop | scenario | summary |
| multi-hop | factual | summary |
| multi-hop | comparative | reasoning |
| temporal | * | reasoning |

**Distribution Actuelle v7.3.0 (calculee)**:

| reasoning_class | Count | % | Cible Know Your RAG | Status |
|-----------------|-------|---|---------------------|--------|
| fact_single | 200 | 47.6% | 40-50% | ✅ OK |
| summary | 219 | 52.1% | 15-25% | ⚠️ Eleve |
| reasoning | 1 | 0.2% | 10-20% | ❌ Manquant |

> **Note**: Le desequilibre summary/reasoning est attendu pour des questions d'examen (scenarios).

### 2.2 P1 - MAJEUR (Generation triplets)

#### 2.2.1 Schema Triplet Cible

```json
{
  "anchor": "Question en francais, style oral naturel ?",
  "positive": "Texte exact du chunk source (expected_chunk_id)",
  "negative": "Hard negative mine avec TopK-PercPos",
  "metadata": {
    "source": "gold_standard",
    "question_id": "ffe:annales:rules:001:a3f2b8c1",
    "chunk_id": "LA-octobre2025.pdf-p027-parent120-child00",
    "reasoning_class": "fact_single",
    "difficulty": 0.16,
    "negative_mining": {
      "method": "topk_percpos",
      "teacher_model": "intfloat/multilingual-e5-large",
      "positive_score": 0.89,
      "negative_score": 0.72,
      "threshold_ratio": 0.95
    }
  }
}
```

#### 2.2.2 Pipeline Hard Negative Mining

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

# 1. Charger teacher model (multilingual pour FR)
teacher = SentenceTransformer("intfloat/multilingual-e5-large")

# 2. Charger corpus chunks
corpus = load_chunks("corpus/processed/chunks_for_embedding_fr.json")

# 3. Positive-aware mining
def mine_with_topk_percpos(
    queries: list[str],
    positives: list[str],
    corpus: list[str],
    threshold_ratio: float = 0.95
) -> list[str]:
    """
    TopK-PercPos: Reject negatives with score > 95% of positive score
    """
    q_emb = teacher.encode(queries)
    p_emb = teacher.encode(positives)
    c_emb = teacher.encode(corpus)

    hard_negatives = []
    for i, (q, p) in enumerate(zip(q_emb, p_emb)):
        pos_score = cosine_similarity(q, p)
        threshold = pos_score * threshold_ratio

        # Find hard negatives below threshold
        candidates = [(j, cosine_similarity(q, c))
                      for j, c in enumerate(c_emb)]
        candidates = [(j, s) for j, s in candidates
                      if s < threshold and s > 0.3]  # min similarity
        candidates.sort(key=lambda x: -x[1])  # hardest first

        if candidates:
            hard_negatives.append(corpus[candidates[0][0]])
        else:
            hard_negatives.append(None)  # fallback needed

    return hard_negatives
```

#### 2.2.3 Training Configuration (Sentence Transformers v3)

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Model: QLoRA fine-tuning
model = SentenceTransformer(
    "google/embeddinggemma-300m-qat-q4_0-unquantized"
)

# Loss: Cached MNR pour memory efficiency
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

# Training args (best practices)
args = SentenceTransformerTrainingArguments(
    output_dir="./embeddinggemma-chess-fr",
    num_train_epochs=3,
    per_device_train_batch_size=32,  # Larger = more in-batch negatives
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # CRITICAL for MNR
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
)
```

### 2.3 P2 - MINEUR (Quality improvements)

#### 2.3.1 Corriger encoding chunk_ids

```python
# 6 chunk_ids avec encoding issues (parité → parit�)
ENCODING_FIXES = {
    "C04_2025_26_Coupe_de_la_parit�": "C04_2025_26_Coupe_de_la_parite",
}
```

#### 2.3.2 Diversity metrics

```python
from collections import Counter
import numpy as np

def compute_diversity_metrics(questions: list[dict]) -> dict:
    """Compute diversity metrics per Know Your RAG taxonomy"""

    # 1. Class distribution
    classes = Counter(q['metadata']['reasoning_class'] for q in questions)
    total = len(questions)

    # 2. Entropy (higher = more diverse)
    probs = [c/total for c in classes.values()]
    entropy = -sum(p * np.log(p) for p in probs if p > 0)

    # 3. Coverage
    expected_classes = {'fact_single', 'summary', 'reasoning'}
    coverage = len(set(classes.keys()) & expected_classes) / len(expected_classes)

    return {
        'distribution': dict(classes),
        'entropy': entropy,
        'coverage': coverage,
        'is_balanced': entropy >= 0.8 and coverage >= 0.8
    }
```

---

## 3. Schema Cible GS Annales v7.4.0

### 3.1 Champs a Ajouter

```json
{
  "id": "ffe:annales:rules:001:a3f2b8c1",
  "question": "Quand dit-on qu'un joueur a le trait ?",  // DOIT finir par ?
  "metadata": {
    "question_type": "factual",           // REQUIS (non None)
    "reasoning_class": "fact_single",     // NOUVEAU (Know Your RAG)
    "reasoning_class_method": "inferred", // NOUVEAU
    "triplet_ready": true                 // NOUVEAU (validation flag)
  }
}
```

### 3.2 Validation Schema

```python
def validate_for_triplets(question: dict) -> tuple[bool, list[str]]:
    """Validate question is ready for triplet generation"""
    errors = []

    # P0 checks
    if not question['question'].strip().endswith('?'):
        errors.append("Question must end with ?")

    if question['metadata'].get('question_type') is None:
        errors.append("question_type is required")

    if 'reasoning_class' not in question['metadata']:
        errors.append("reasoning_class is required (Know Your RAG)")

    # P1 checks
    if not question.get('expected_chunk_id'):
        errors.append("expected_chunk_id required for positive extraction")

    return len(errors) == 0, errors
```

---

## 4. Metriques Cibles

### 4.1 Pre-Triplet Generation

| Metrique | Actuel | Cible | Standard |
|----------|--------|-------|----------|
| Questions avec "?" | 68% | 100% | Format standard |
| question_type complete | 92% | 100% | Taxonomie |
| reasoning_class | 0% | 100% | Know Your RAG |
| Diversity entropy | N/A | >= 0.8 | COLING 2025 |

### 4.2 Post-Triplet Generation

| Metrique | Cible | Standard |
|----------|-------|----------|
| Triplets generes | 420 | 1:1 avec GS |
| Hard negatives valides | 100% | TopK-PercPos |
| False negative rate | < 5% | NV-Retriever |
| Same-doc negatives | >= 40% | NV-Embed-v2 |

### 4.3 Training Evaluation

| Metrique | Cible | Standard |
|----------|-------|----------|
| Triplet accuracy (eval) | >= 85% | TripletEvaluator |
| MTEB Retrieval FR | >= 0.70 NDCG@10 | MTEB median |
| Recall@5 GS | >= 90% | ISO 25010 |

---

## 5. Checklist Implementation

### 5.1 Phase 1: Preparation GS v7.4.0

- [ ] Script: Ajouter "?" aux 133 questions manquantes
- [ ] Script: Inferer question_type pour 34 questions None
- [ ] Script: Ajouter reasoning_class (Know Your RAG mapping)
- [ ] Script: Corriger 6 encoding issues chunk_ids
- [ ] Validation: 100% triplet_ready=true
- [ ] Export: `gold_standard_annales_fr_v7.4.0.json`

### 5.2 Phase 2: Generation Triplets

- [ ] Extraire positives depuis corpus via expected_chunk_id
- [ ] Miner hard negatives avec TopK-PercPos
- [ ] Valider ratio false negatives < 5%
- [ ] Generer `triplets_annales_fr.jsonl`
- [ ] Rapport: `triplets_generation_report.json`

### 5.3 Phase 3: Training

- [ ] Split 80/20 train/val (seed=42)
- [ ] QLoRA fine-tuning EmbeddingGemma
- [ ] Evaluation TripletEvaluator
- [ ] Benchmark MTEB FR custom

---

## 6. References

### 6.1 Papers Academiques

| Reference | Application |
|-----------|-------------|
| [NV-Embed arXiv:2405.17428](https://arxiv.org/abs/2405.17428) | Hard negative mining, two-stage training |
| [NV-Retriever arXiv:2407.15831](https://arxiv.org/abs/2407.15831) | TopK-PercPos method |
| [GTE arXiv:2308.03281](https://arxiv.org/abs/2308.03281) | Multi-stage contrastive learning |
| [Know Your RAG arXiv:2411.19710](https://arxiv.org/abs/2411.19710) | 4-class taxonomy, diversity |
| [RAGBench arXiv:2407.11005](https://arxiv.org/abs/2407.11005) | TRACe evaluation framework |

### 6.2 Ressources Techniques

| Resource | URL |
|----------|-----|
| Sentence Transformers v3 | [HuggingFace Blog](https://huggingface.co/blog/train-sentence-transformers) |
| NV-Embed-v2 Model | [HuggingFace](https://huggingface.co/nvidia/NV-Embed-v2) |
| MTEB Leaderboard | [HuggingFace Spaces](https://huggingface.co/spaces/mteb/leaderboard) |
| Pinecone MNR Loss | [Pinecone Learn](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/) |

### 6.3 Standards ISO

| Standard | Application |
|----------|-------------|
| ISO 42001 A.6.2.2 | Provenance, lineage |
| ISO 25010 | Quality metrics |
| ISO 29119 | Test data requirements |

---

## 7. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-25 | Creation - Standards industrie pour optimisation GS v7 |

---

*Document ISO 42001/25010 - Pocket Arbiter Project*
*Conforme aux standards: NV-Embed-v2, GTE, Know Your RAG, Sentence Transformers v3*
