# Guide Fine-tuning EmbeddingGemma - MRL + LoRA

> **Document ID**: RES-LORA-001
> **ISO Reference**: ISO 42001, ISO 25010
> **Version**: 1.3
> **Date**: 2026-01-21
> **Statut**: Draft
> **Classification**: Interne
> **Auteur**: Claude Opus 4.5
> **Mots-cles**: fine-tuning, LoRA, MRL, EmbeddingGemma, embeddings, retrieval

---

## 1. Objectif

Ce document consigne toutes les informations necessaires AVANT la production du notebook de fine-tuning EmbeddingGemma pour Pocket Arbiter.

> **Note importante**: Le precedent fine-tuning a ete realise avec un notebook de mauvaise qualite. Ce guide documente l'approche recommandee basee sur les sources officielles Google AI et les best practices de la communaute.

---

## 2. Sources Officielles

### 2.1 Google AI Dev

| Document | URL | Contenu cle |
|----------|-----|-------------|
| **Fine-tune EmbeddingGemma** | [ai.google.dev](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers) | Guide officiel Sentence Transformers |
| **LoRA Tuning Gemma** | [ai.google.dev](https://ai.google.dev/gemma/docs/core/lora_tuning) | LoRA avec Keras/JAX |
| **EmbeddingGemma Overview** | [ai.google.dev](https://ai.google.dev/gemma/docs/embeddinggemma) | Architecture, MRL, specs |
| **EmbeddingGemma Model Card** | [ai.google.dev](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) | Specs techniques detaillees |
| **MediaPipe LLM Inference** | [ai.google.dev](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android) | Deploiement Android |

### 2.2 Kaggle Notebooks

| Notebook | URL | Auteur |
|----------|-----|--------|
| **Fine-tune EmbeddingGemma** | [kaggle.com](https://www.kaggle.com/code/nilaychauhan/fine-tune-embeddinggemma) | Nilay Chauhan (Google) |
| **Fine-tune Gemma LoRA Keras** | [kaggle.com](https://www.kaggle.com/code/nilaychauhan/fine-tune-gemma-models-in-keras-using-lora) | Nilay Chauhan (Google) |
| **EmbeddingGemma Model** | [kaggle.com](https://www.kaggle.com/models/google/embeddinggemma) | Google Official |
| **Gemma 3 LoRA Medical Q&A** | [kaggle.com](https://www.kaggle.com/code/gpreda/fine-tune-gemma-3-270m-using-lora-for-medical-q-a) | Gabriel Preda |
| **RAG Sentence Transformer + Gemma** | [kaggle.com](https://www.kaggle.com/code/jamesvasanth/rag-model-sentence-transformer-gemma-chatbot) | James Vasanth |

### 2.3 HuggingFace / Community

| Resource | URL | Contenu |
|----------|-----|---------|
| **HuggingFace EmbeddingGemma Blog** | [huggingface.co](https://huggingface.co/blog/embeddinggemma) | Architecture, fine-tuning complet |
| **Train Sentence Transformers v3** | [huggingface.co](https://huggingface.co/blog/train-sentence-transformers) | Guide officiel ST v3 |
| **Medium GDE - LoRA On-Device** | [medium.com](https://medium.com/google-developer-experts/fine-tuning-gemma-with-lora-for-on-device-inference-android-ios-web-with-separate-lora-weights-f05d1db30d86) | Sasha Denisov (GDE) |
| **Aurelio AI - ST Fine-tuning** | [aurelio.ai](https://www.aurelio.ai/learn/sentence-transformers-fine-tuning) | Best practices |
| **Pinecone - MNR Loss** | [pinecone.io](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/) | MultipleNegativesRankingLoss |

### 2.4 Best Practices / Overfitting

| Resource | URL | Focus |
|----------|-----|-------|
| **Zilliz - Overfitting Prevention** | [zilliz.com](https://zilliz.com/ai-faq/what-should-i-do-if-the-finetuning-process-for-a-sentence-transformer-model-overfits-quickly-for-example-training-loss-gets-much-lower-than-validation-loss-early-on) | Early stopping, dropout, regularization |
| **Zilliz - ST Parameters** | [zilliz.com](https://zilliz.com/ai-faq/what-parameters-can-be-adjusted-when-finetuning-a-sentence-transformer-eg-learning-rate-batch-size-number-of-epochs-and-how-do-they-impact-training) | Hyperparameters impact |
| **Unsloth - LoRA Hyperparameters** | [unsloth.ai](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) | LoRA rank, alpha, settings |
| **SBERT Training Overview** | [sbert.net](https://www.sbert.net/docs/sentence_transformer/training_overview.html) | Callbacks, logging |

### 2.5 Generation synthetique de triplets

| Resource | URL | Focus |
|----------|-----|-------|
| **LlamaIndex finetune-embedding** | [github.com/run-llama](https://github.com/run-llama/finetune-embedding) | Pipeline complet LLM → triplets |
| **Databricks Embedding Finetuning** | [databricks.com](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning) | Synthetic queries + LLM-as-judge |
| **HuggingFace ModernBERT Synthetic** | [huggingface.co](https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data) | Synthetic Data Generator |
| **Gemini Batch API** | [ai.google.dev](https://ai.google.dev/gemini-api/docs/batch-api) | 50% moins cher, batch JSONL |
| **Google GenAI SDK** | [github.com/googleapis](https://github.com/googleapis/python-genai) | SDK Python officiel 2025+ |
| **Pinecone MNR Loss** | [pinecone.io](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/) | MultipleNegativesRankingLoss |
| **SBERT mine_hard_negatives** | [sbert.net](https://sbert.net/docs/package_reference/util.html) | Hard negative mining natif |
| **Synthetic Data LLM Fine-tuning** | [labelyourdata.com](https://labelyourdata.com/articles/llm-fine-tuning/synthetic-data) | Best practices 2025 |
| **Glean Enterprise RAG** | [jxnl.co](https://jxnl.co/writing/2025/03/06/fine-tuning-embedding-models-for-enterprise-rag-lessons-from-glean/) | Quality > Quantity, user feedback |
| **NVIDIA SDG RAG** | [nvidia.dev](https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/) | 4 criteres qualite, answerability filter |
| **Phil Schmid RAG Fine-tune** | [philschmid.de](https://www.philschmid.de/fine-tune-embedding-model-for-rag) | 6.3k samples, +7% performance |

---

## 3. Specifications EmbeddingGemma

### 3.1 Architecture

| Spec | Valeur |
|------|--------|
| **Parametres** | 308M (300M dans certaines refs) |
| **Base** | Gemma 3 transformer, bi-directional attention |
| **Dimensions output** | 768 (truncatable: 512, 256, 128) |
| **Context window** | 2,048 tokens |
| **Langues** | 100+ (multilingual) |
| **RAM quantized** | < 200MB |
| **Latence** | < 22ms sur EdgeTPU |

### 3.2 Matryoshka Representation Learning (MRL)

**Principe**: Information importante concentree dans les premieres dimensions. Truncation sans retraining.

```python
from sentence_transformers import SentenceTransformer

# Option 1: Truncation a l'initialisation
model = SentenceTransformer("google/embeddinggemma-300m", truncate_dim=256)

# Option 2: Truncation par query
query_emb = model.encode_query(query, truncate_dim=256)  # (256,) au lieu de (768,)
```

**Benchmarks MRL**:
| Dimensions | Performance relative | Storage | Speedup |
|------------|---------------------|---------|---------|
| 768 (full) | 100% | 100% | 1x |
| 512 | 99.5% | 67% | 1.5x |
| 256 | 99% | 33% | 2x |
| 128 | 97% | 17% | 3x |

### 3.3 Prompts requis

EmbeddingGemma necessite des prompts specifiques:

```python
# Queries
"task: [task] | query: [content]"

# Documents
"title: [title | none] | text: [content]"

# Tasks disponibles
PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "clustering": "task: clustering | query: ",
    "classification": "task: classification | query: ",
    "STS": "task: sentence similarity | query: ",
    "reranking": "task: search result | query: ",
}
```

---

## 4. Hyperparametres Recommandes

### 4.1 Sweet Spots (Sentence Transformers)

| Parametre | Valeur recommandee | Notes |
|-----------|-------------------|-------|
| **learning_rate** | `2e-5` | Starting point, reduire si overfitting |
| **num_train_epochs** | `1-5` | 1 suffisant avec gros dataset, 3-5 pour petits |
| **per_device_train_batch_size** | `32-128` | Plus grand = meilleur signal MNR Loss |
| **warmup_ratio** | `0.1` | 10% des steps |
| **fp16** | `True` | Mixed precision |
| **weight_decay** | `0.01` | Regularisation L2 |

### 4.2 Sweet Spots (LoRA specifique)

| Parametre | Valeur recommandee | Notes |
|-----------|-------------------|-------|
| **rank** | `4-16` | 4-8 simple, 16-32 complexe |
| **alpha** | `= rank` ou `rank/2` | Stabilite vs agressivite |
| **dropout** | `0.1-0.3` | Augmenter si overfitting |
| **target_modules** | `["q_proj", "v_proj"]` | Attention layers |

### 4.3 Configuration Complete

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Model
model = SentenceTransformer("google/embeddinggemma-300m")

# Loss (recommandee pour gros batch virtuels)
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

# Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="./embeddinggemma-chess-fr",

    # Hyperparametres core
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,

    # Precision
    fp16=True,
    bf16=False,

    # Batch sampling (crucial pour MNR Loss)
    batch_sampler=BatchSamplers.NO_DUPLICATES,

    # Prompts mapping
    prompts={
        "anchor": model.prompts["query"],
        "positive": model.prompts["document"],
    },

    # Evaluation & Logging
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    logging_steps=10,

    # W&B integration
    run_name="embeddinggemma-chess-fr-v1",
    report_to="wandb",  # ou "tensorboard"

    # Early stopping (via callback)
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

---

## 5. Prevention Overfitting

### 5.1 Signaux d'alerte

| Signal | Description | Action |
|--------|-------------|--------|
| Train loss << Val loss | Gap croissant | Early stopping |
| Val loss stagnant | Plateau apres quelques epochs | Arreter |
| Val loss remonte | Overfitting clair | Rollback best checkpoint |

### 5.2 Techniques de prevention

#### Early Stopping
```python
from transformers import EarlyStoppingCallback

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # 3 evals sans amelioration
            early_stopping_threshold=0.001,  # Seuil minimum
        )
    ],
)
```

#### Dropout augmente
```python
# Pour architecture BERT-like
model[0].auto_model.config.hidden_dropout_prob = 0.3  # default 0.1
model[0].auto_model.config.attention_probs_dropout_prob = 0.3
```

#### Layer freezing (petit dataset)
```python
# Geler les couches basses, entrainer seulement top 2-4
for name, param in model.named_parameters():
    if "encoder.layer" in name:
        layer_num = int(name.split(".")[2])
        if layer_num < 8:  # Geler couches 0-7
            param.requires_grad = False
```

#### Learning rate reduit
```python
# Si overfitting, reduire LR
args.learning_rate = 5e-6  # au lieu de 2e-5
```

### 5.3 Monitoring checkpoints

```python
# Logging detaille toutes les N steps
args = SentenceTransformerTrainingArguments(
    logging_steps=10,           # Log frequemment
    eval_steps=50,              # Eval reguliere
    save_steps=50,              # Sauvegarder souvent
    save_total_limit=5,         # Garder 5 checkpoints
    logging_first_step=True,    # Log step 0
)
```

---

## 6. Dataset Pocket Arbiter

### 6.1 Strategie deux tiers

> **Principe**: Gold Standard = source primaire (qualite), Synthetique = augmentation (volume)

```
┌─────────────────────────────────────────────────────────────┐
│                    HIERARCHIE TRIPLETS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TIER 1: GOLD STANDARD (prioritaire, humain-valide)        │
│  ├── 193 questions (150 FR + 43 INTL)                      │
│  ├── expected_chunk_id connu → positive                    │
│  ├── mine_hard_negatives → negatives                       │
│  └── Split: 80% train / 20% val (JAMAIS touche)            │
│                                                             │
│  TIER 2: SYNTHETIC (augmentation, LLM-genere)              │
│  ├── Gemini API genere ~3 Q/chunk (~5500 Q)                │
│  ├── Filtre qualite (LLM-as-judge)                         │
│  └── Ajoute au TRAIN seulement (pas val/test)              │
│                                                             │
│  RESULTAT FINAL:                                            │
│  ├── Train: 154 gold + ~4000 synth = ~4150 triplets        │
│  └── Val:   39 gold (intouche, evaluation fiable)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Sources Gold Standard

| Fichier | Questions | Hard Cases | Documents |
|---------|-----------|------------|-----------|
| `tests/data/gold_standard_fr.json` | 150 | 46 (30.6%) | 28 PDF FFE |
| `tests/data/gold_standard_intl.json` | 43 | 12 (27.9%) | 1 PDF FIDE |
| **Total** | **193** | **58** | **29** |

### 6.3 Format requis (triplets)

```python
{
    "anchor": "Question utilisateur",
    "positive": "Texte du chunk correct",
    "negative": "Texte d'un chunk similaire mais incorrect (hard negative)"
}
```

### 6.4 TIER 1: Triplets depuis Gold Standard (PRIORITAIRE)

> **Source primaire**: Les 193 questions du gold standard sont deja validees par des humains.
> On connait le `expected_chunk_id` → on a le positive.
> On mine les hard negatives via retrieval.

#### 6.4.1 Extraction positive depuis Gold Standard

```python
"""
Extraire triplets depuis gold_standard_fr.json.
Le positive est deja connu (expected_chunk_id).
"""
import json
from pathlib import Path

def load_gold_standard_positives(
    gold_path: str,
    chunks_path: str
) -> list[dict]:
    """
    Extraire paires (question, positive_chunk) depuis gold standard.

    Returns:
        Liste de {"anchor": question, "positive": chunk_text, "chunk_id": id}
    """
    # Charger gold standard
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    # Charger chunks (index par ID)
    with open(chunks_path, encoding="utf-8") as f:
        chunks_list = json.load(f)
    chunks_by_id = {c["id"]: c["text"] for c in chunks_list}

    pairs = []
    for q in gold["questions"]:
        chunk_id = q.get("expected_chunk_id") or q.get("expected_chunks", [None])[0]

        if chunk_id and chunk_id in chunks_by_id:
            pairs.append({
                "anchor": q["question"],
                "positive": chunks_by_id[chunk_id],
                "chunk_id": chunk_id,
                "difficulty": q.get("difficulty", "medium"),
            })
        else:
            print(f"Warning: chunk {chunk_id} not found for Q: {q['question'][:50]}")

    return pairs


# Usage
pairs_fr = load_gold_standard_positives(
    "tests/data/gold_standard_fr.json",
    "corpus/processed/chunks_for_embedding_fr.json"
)
print(f"Gold Standard FR: {len(pairs_fr)} paires (anchor, positive)")
# Expected: ~150 paires
```

#### 6.4.2 Generation hard negatives

```python
def generate_hard_negatives(
    question: str,
    positive_chunk_id: str,
    db: Database,
    model: SentenceTransformer,
    top_k: int = 10
) -> str:
    """
    Generer un hard negative pour une question.

    Hard negative = chunk semantiquement proche mais incorrect.
    Typiquement rang 2-5 dans les resultats retrieval.
    """
    # 1. Embed question
    q_emb = model.encode_query(question)

    # 2. Retrieve top-k
    results = db.search(q_emb, top_k=top_k)

    # 3. Filtrer le positif
    negatives = [r for r in results if r.id != positive_chunk_id]

    # 4. Prendre rang 2-3 (pas le plus distant)
    if len(negatives) >= 2:
        return negatives[1].text  # Rang 2
    elif negatives:
        return negatives[0].text
    else:
        return None
```

#### 6.4.3 Script complet Gold Standard → Triplets

```python
import json
from datasets import Dataset
from sentence_transformers import SentenceTransformer

def prepare_training_dataset(
    gold_standard_path: str,
    chunks_db_path: str,
    output_path: str
) -> Dataset:
    """Preparer dataset triplets pour fine-tuning."""

    # Charger gold standard
    with open(gold_standard_path) as f:
        gold = json.load(f)

    # Charger modele pour hard negatives
    model = SentenceTransformer("google/embeddinggemma-300m")

    # Charger chunks
    db = load_chunks_db(chunks_db_path)

    triplets = []
    for item in gold["questions"]:
        question = item["question"]
        positive_text = get_chunk_text(db, item["expected_chunk_id"])
        negative_text = generate_hard_negatives(
            question,
            item["expected_chunk_id"],
            db,
            model
        )

        if positive_text and negative_text:
            triplets.append({
                "anchor": question,
                "positive": positive_text,
                "negative": negative_text,
            })

    # Split train/val 80/20
    dataset = Dataset.from_list(triplets)
    splits = dataset.train_test_split(test_size=0.2, seed=42)

    # Sauvegarder
    splits.save_to_disk(output_path)

    return splits

# Usage
dataset = prepare_training_dataset(
    "tests/data/gold_standard_fr.json",
    "corpus/corpus_fr.db",
    "data/training_triplets"
)
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")
```

### 6.5 TIER 2: Generation synthetique (Claude Code + Gemini API)

> **Augmentation**: Utiliser Claude Code comme orchestrateur et Gemini API pour generer des questions supplementaires.
> Ces triplets synthetiques s'ajoutent au **train set seulement** (jamais au val/test).
> Le val/test reste 100% Gold Standard pour evaluation fiable.

#### 6.5.1 Sources et references

| Resource | URL | Description |
|----------|-----|-------------|
| **LlamaIndex finetune-embedding** | [github.com/run-llama](https://github.com/run-llama/finetune-embedding) | Pipeline complet sans labeled data |
| **Databricks Embedding Finetuning** | [databricks.com](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning) | Synthetic queries + LLM-as-judge |
| **HuggingFace Synthetic Generator** | [huggingface.co](https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data) | ModernBERT + synthetic data |
| **Gemini Batch API** | [ai.google.dev](https://ai.google.dev/gemini-api/docs/batch-api) | 50% moins cher, batch JSONL |
| **Sentence Transformers mine_hard_negatives** | [sbert.net](https://sbert.net/docs/package_reference/util.html) | Hard negative mining natif |
| **Pinecone MNR Loss** | [pinecone.io](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/) | Best practices MNR |

#### 6.5.2 Workflow recommande

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE GENERATION TRIPLETS                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. CHUNKS (1827 FR)                                                │
│       │                                                              │
│       ▼                                                              │
│  2. GEMINI API (gemini-2.0-flash)                                   │
│       │  ├── Generate 3-5 questions/chunk                           │
│       │  ├── JSON structured output                                  │
│       │  └── Batch API (50% moins cher)                             │
│       ▼                                                              │
│  3. VALIDATION HUMAINE (Human-in-the-loop)                          │
│       │  ├── Review sample 10%                                       │
│       │  ├── Ajuster prompts si necessaire                          │
│       │  └── Filtrer questions hors-sujet                           │
│       ▼                                                              │
│  4. HARD NEGATIVES (mine_hard_negatives)                            │
│       │  ├── EmbeddingGemma baseline                                │
│       │  ├── relative_margin=0.05 (NV-Retriever)                    │
│       │  └── CrossEncoder rescore (optionnel)                       │
│       ▼                                                              │
│  5. DATASET FINAL                                                   │
│       ├── ~5000-9000 triplets (3-5 Q x 1827 chunks)                 │
│       ├── Format: {"anchor", "positive", "negative"}                │
│       └── Split: 80% train / 20% val                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 6.5.3 Configuration Gemini API

```bash
# Configuration environnement (VS Code / Claude Code)
# Linux/macOS
export GEMINI_API_KEY='votre_cle_ici'

# Windows PowerShell
$env:GEMINI_API_KEY='votre_cle_ici'
```

> **Obtenir la cle**: [Google AI Studio](https://aistudio.google.com/) → "Get API key" → Nouveau projet

#### 6.5.4 Script generation questions synthetiques

```python
"""
Script generation triplets synthetiques pour Pocket Arbiter.
Utilise Gemini API (gemini-2.0-flash) pour generer des questions.

Usage:
    python scripts/pipeline/generate_triplets.py
"""
import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Google GenAI SDK (2025+)
from google import genai
from google.genai import types

# Configuration
GEMINI_MODEL = "gemini-2.0-flash"  # Ou gemini-2.5-flash
QUESTIONS_PER_CHUNK = 3
OUTPUT_DIR = Path("data/synthetic_triplets")


@dataclass
class GeneratedQuestion:
    """Question generee par LLM."""
    question: str
    question_type: str  # "factual", "procedural", "definition"
    difficulty: str     # "easy", "medium", "hard"
    source_chunk_id: str


def setup_gemini_client() -> genai.Client:
    """Configurer client Gemini avec cle API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY non definie. Voir README.")
    return genai.Client(api_key=api_key)


def generate_questions_for_chunk(
    client: genai.Client,
    chunk_text: str,
    chunk_id: str,
    document_title: str,
    num_questions: int = 3
) -> list[GeneratedQuestion]:
    """
    Generer des questions synthetiques pour un chunk.

    Args:
        client: Client Gemini configure
        chunk_text: Texte du chunk source
        chunk_id: ID unique du chunk
        document_title: Titre du document source
        num_questions: Nombre de questions a generer

    Returns:
        Liste de GeneratedQuestion
    """
    # Prompt systeme optimise pour reglement echecs
    system_prompt = """Tu es un expert en reglements d'echecs (FFE/FIDE).
Tu generes des questions que des arbitres poseraient en situation reelle.

REGLES STRICTES:
1. Questions en francais, style oral/naturel
2. La reponse DOIT etre dans le texte fourni
3. Varier les types: factuel, procedural, definition
4. Utiliser le jargon arbitre (cadence, forfait, appariement, etc.)
5. Inclure des questions difficiles (formulation indirecte)

EXEMPLES DE STYLE:
- "C'est quoi le delai pour le forfait en rapide?"
- "Un joueur peut contester une decision de l'arbitre?"
- "Comment on gere un telephone qui sonne?"
"""

    user_prompt = f"""Document: {document_title}

Texte reglementaire:
\"\"\"
{chunk_text}
\"\"\"

Genere exactement {num_questions} questions variees.

Reponds en JSON valide:
{{
  "questions": [
    {{
      "question": "...",
      "question_type": "factual|procedural|definition",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.7,  # Creativite moderee
                max_output_tokens=1024,
            )
        )

        # Parser JSON
        result = json.loads(response.text)

        return [
            GeneratedQuestion(
                question=q["question"],
                question_type=q["question_type"],
                difficulty=q["difficulty"],
                source_chunk_id=chunk_id
            )
            for q in result["questions"]
        ]

    except Exception as e:
        print(f"Erreur generation chunk {chunk_id}: {e}")
        return []


def generate_triplets_batch(
    client: genai.Client,
    chunks: list[dict],
    output_path: Path,
    questions_per_chunk: int = 3,
    validation_callback: Optional[callable] = None
) -> list[dict]:
    """
    Generer triplets pour tous les chunks avec validation optionnelle.

    Args:
        client: Client Gemini
        chunks: Liste de chunks {"id", "text", "metadata"}
        output_path: Chemin sauvegarde JSON
        questions_per_chunk: Questions par chunk
        validation_callback: Fonction validation humaine (optionnel)

    Returns:
        Liste de triplets {"anchor", "positive", "negative"}
    """
    all_pairs = []  # (question, positive_chunk)

    print(f"Generation questions pour {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(chunks)}")

        questions = generate_questions_for_chunk(
            client=client,
            chunk_text=chunk["text"],
            chunk_id=chunk["id"],
            document_title=chunk.get("metadata", {}).get("source", "Unknown"),
            num_questions=questions_per_chunk
        )

        for q in questions:
            pair = {
                "anchor": q.question,
                "positive": chunk["text"],
                "chunk_id": chunk["id"],
                "question_type": q.question_type,
                "difficulty": q.difficulty,
            }

            # Validation humaine optionnelle
            if validation_callback:
                if validation_callback(pair):
                    all_pairs.append(pair)
            else:
                all_pairs.append(pair)

    # Sauvegarder pairs intermediaires
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path / "synthetic_pairs.json", "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"Genere {len(all_pairs)} paires (anchor, positive)")
    return all_pairs


def interactive_validation(pair: dict) -> bool:
    """
    Validation humaine interactive (Human-in-the-loop).

    Returns:
        True si valide, False si rejete
    """
    print(f"\n{'='*60}")
    print(f"Question: {pair['anchor']}")
    print(f"Type: {pair['question_type']} | Difficulty: {pair['difficulty']}")
    print(f"Chunk preview: {pair['positive'][:200]}...")
    print(f"{'='*60}")

    while True:
        choice = input("Valider (y), Modifier (m), Ignorer (i), Quitter (q)? ").lower()

        if choice == 'y':
            return True
        elif choice == 'i':
            return False
        elif choice == 'm':
            pair['anchor'] = input("Nouvelle question: ")
            return True
        elif choice == 'q':
            raise KeyboardInterrupt("Validation interrompue")
        else:
            print("Choix invalide. Reessayer.")


# ============================================================
# USAGE EXEMPLE
# ============================================================
if __name__ == "__main__":
    # 1. Setup client
    client = setup_gemini_client()

    # 2. Charger chunks
    with open("corpus/processed/chunks_for_embedding_fr.json") as f:
        chunks = json.load(f)

    # 3. Generer paires (sans validation pour batch)
    pairs = generate_triplets_batch(
        client=client,
        chunks=chunks[:100],  # Test sur 100 chunks d'abord
        output_path=OUTPUT_DIR,
        questions_per_chunk=QUESTIONS_PER_CHUNK,
        validation_callback=None  # Ou interactive_validation pour review
    )

    print(f"\nGenere {len(pairs)} paires synthetiques")
```

#### 6.5.5 Hard negative mining avec Sentence Transformers

```python
"""
Ajouter hard negatives aux paires synthetiques.
Utilise mine_hard_negatives de Sentence Transformers.
"""
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

def add_hard_negatives(
    pairs_path: str,
    output_path: str,
    model_name: str = "google/embeddinggemma-300m",
    num_negatives: int = 1,
    relative_margin: float = 0.05  # NV-Retriever optimal
) -> Dataset:
    """
    Transformer paires (anchor, positive) en triplets avec hard negatives.

    Args:
        pairs_path: JSON avec paires synthetiques
        output_path: Chemin sauvegarde dataset
        model_name: Modele embedding pour mining
        num_negatives: Nombre de negatives par paire
        relative_margin: Marge relative (0.05 = 95% similarity max)

    Returns:
        Dataset avec triplets
    """
    import json

    # Charger paires
    with open(pairs_path) as f:
        pairs = json.load(f)

    # Convertir en Dataset
    dataset = Dataset.from_list([
        {"anchor": p["anchor"], "positive": p["positive"]}
        for p in pairs
    ])

    # Charger modele
    model = SentenceTransformer(model_name)

    # Collecter corpus (tous les positives)
    corpus = list(set(p["positive"] for p in pairs))

    # Mining hard negatives (methode NV-Retriever)
    dataset_with_negatives = mine_hard_negatives(
        dataset=dataset,
        model=model,
        corpus=corpus,
        num_negatives=num_negatives,
        relative_margin=relative_margin,  # Negative <= 95% similar as positive
        sampling_strategy="top",           # Prendre les plus difficiles
        use_faiss=True,                    # Accelerer avec FAISS
        batch_size=64,
        output_format="triplet",           # (anchor, positive, negative)
    )

    # Sauvegarder
    dataset_with_negatives.save_to_disk(output_path)

    print(f"Dataset final: {len(dataset_with_negatives)} triplets")
    return dataset_with_negatives


# Usage
if __name__ == "__main__":
    dataset = add_hard_negatives(
        pairs_path="data/synthetic_triplets/synthetic_pairs.json",
        output_path="data/training_triplets",
        num_negatives=1,
        relative_margin=0.05
    )
```

#### 6.5.6 Gemini Batch API (production, 50% moins cher)

> **Avantages**: 50% du cout standard, pas de rate limit/minute, turnaround 24h (souvent plus rapide)
> **Doc officielle**: [ai.google.dev/gemini-api/docs/batch-api](https://ai.google.dev/gemini-api/docs/batch-api)

Pour generation massive (1827 chunks x 3 questions = ~5500 requests):

```python
"""
Script complet Gemini Batch API pour generation triplets.
50% moins cher, pas de rate limit, turnaround ~1-6h.

Usage:
    python scripts/pipeline/batch_generate_triplets.py
"""
import json
import time
import os
from pathlib import Path
from google import genai
from google.genai import types

# Configuration
GEMINI_MODEL = "gemini-2.0-flash"  # ou "gemini-3-flash-preview"
OUTPUT_DIR = Path("data/batch_triplets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_client() -> genai.Client:
    """Configurer client Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY non definie")
    return genai.Client(api_key=api_key)


def create_batch_jsonl(chunks: list[dict], output_file: str) -> str:
    """
    Creer fichier JSONL pour batch job.
    Format: {"key": "chunk-id", "request": {...}}
    """
    system_prompt = """Tu es un expert en reglements d'echecs (FFE/FIDE).
Genere exactement 3 questions que des arbitres poseraient en situation reelle.
Questions en francais, style oral/naturel.
La reponse DOIT etre dans le texte fourni.

Reponds en JSON valide:
{"questions": [{"question": "...", "type": "factual|procedural|definition", "difficulty": "easy|medium|hard"}]}"""

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            request = {
                "key": chunk["id"],
                "request": {
                    "contents": [{
                        "parts": [{"text": f"Texte reglementaire:\n{chunk['text'][:3000]}"}],
                        "role": "user"
                    }],
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.7
                    }
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"Cree {output_file} avec {len(chunks)} requetes")
    return output_file


def submit_batch_job(client: genai.Client, jsonl_file: str) -> str:
    """Upload fichier et creer batch job."""
    # 1. Upload fichier via File API
    print("Uploading fichier JSONL...")
    uploaded_file = client.files.upload(
        file=jsonl_file,
        config=types.UploadFileConfig(
            display_name="pocket-arbiter-triplets",
            mime_type="jsonl"
        )
    )
    print(f"Uploaded: {uploaded_file.name}")

    # 2. Creer batch job
    print("Creating batch job...")
    batch_job = client.batches.create(
        model=GEMINI_MODEL,
        src=uploaded_file.name,
        config={"display_name": "pocket-arbiter-triplets-job"}
    )

    print(f"Job cree: {batch_job.name}")
    print(f"Status initial: {batch_job.state.name}")
    return batch_job.name


def wait_for_completion(client: genai.Client, job_name: str, poll_interval: int = 60):
    """Attendre completion du batch job."""
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }

    print(f"Polling status pour job: {job_name}")

    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name

        print(f"[{time.strftime('%H:%M:%S')}] Status: {state}")

        if state in completed_states:
            print(f"\n{'='*50}")
            print(f"Job termine: {state}")
            return batch_job

        print(f"Attente {poll_interval}s...")
        time.sleep(poll_interval)


def download_results(client: genai.Client, batch_job) -> list[dict]:
    """Telecharger et parser resultats."""
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job echoue: {batch_job.error}")
        return []

    # Resultats dans un fichier
    if batch_job.dest and batch_job.dest.file_name:
        result_file = batch_job.dest.file_name
        print(f"Telechargement resultats: {result_file}")

        file_content = client.files.download(file=result_file)
        content = file_content.decode('utf-8')

        results = []
        for line in content.strip().split("\n"):
            if line:
                parsed = json.loads(line)
                results.append(parsed)

        print(f"Telecharge {len(results)} reponses")
        return results

    # Resultats inline (petit batch)
    elif batch_job.dest and batch_job.dest.inlined_responses:
        print("Resultats inline")
        return [
            {"key": f"inline-{i}", "response": r.response}
            for i, r in enumerate(batch_job.dest.inlined_responses)
            if r.response
        ]

    return []


def parse_questions_from_results(results: list[dict]) -> list[dict]:
    """Extraire questions des reponses Gemini."""
    all_questions = []

    for result in results:
        chunk_id = result.get("key", "unknown")

        # Extraire texte de la reponse
        try:
            if "response" in result and result["response"]:
                response = result["response"]
                if "candidates" in response:
                    text = response["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    text = response.get("text", "{}")

                # Parser JSON
                data = json.loads(text)
                questions = data.get("questions", [])

                for q in questions:
                    all_questions.append({
                        "chunk_id": chunk_id,
                        "question": q.get("question", ""),
                        "type": q.get("type", "unknown"),
                        "difficulty": q.get("difficulty", "medium"),
                    })
            elif "error" in result:
                print(f"Erreur chunk {chunk_id}: {result['error']}")

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Parse error chunk {chunk_id}: {e}")

    return all_questions


# ============================================================
# PIPELINE COMPLET
# ============================================================
def run_batch_pipeline(chunks_path: str, max_chunks: int = None):
    """
    Pipeline complet: chunks → batch API → questions.

    Args:
        chunks_path: Chemin vers chunks JSON
        max_chunks: Limite (None = tous)
    """
    # 1. Setup
    client = setup_client()

    # 2. Charger chunks
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if max_chunks:
        chunks = chunks[:max_chunks]
    print(f"Chunks a traiter: {len(chunks)}")

    # 3. Creer JSONL
    jsonl_file = str(OUTPUT_DIR / "batch_requests.jsonl")
    create_batch_jsonl(chunks, jsonl_file)

    # 4. Soumettre batch job
    job_name = submit_batch_job(client, jsonl_file)

    # 5. Attendre completion (poll toutes les 2 min)
    batch_job = wait_for_completion(client, job_name, poll_interval=120)

    # 6. Telecharger resultats
    results = download_results(client, batch_job)

    # 7. Sauvegarder resultats bruts
    with open(OUTPUT_DIR / "batch_results_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 8. Parser questions
    questions = parse_questions_from_results(results)
    print(f"\nQuestions generees: {len(questions)}")

    # 9. Sauvegarder questions
    with open(OUTPUT_DIR / "synthetic_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    return questions


# Usage
if __name__ == "__main__":
    # Test avec 10 chunks d'abord
    questions = run_batch_pipeline(
        "corpus/processed/chunks_for_embedding_fr.json",
        max_chunks=10  # Retirer pour traiter tous les chunks
    )
    print(f"\nTermine! {len(questions)} questions generees")
```

#### 6.5.7 Lister et gerer les batch jobs

```python
# Lister tous les jobs
batch_jobs = client.batches.list()
for job in batch_jobs:
    print(f"{job.name}: {job.state.name}")

# Annuler un job en cours
client.batches.cancel(name="batches/123456")

# Supprimer un job
client.batches.delete(name="batches/123456")
```

#### 6.5.7 Estimation tokens Pocket Arbiter

| Metrique | Valeur |
|----------|--------|
| **Chunks FR** | 1,827 |
| **Chunks INTL** | 974 |
| **Total chunks** | 2,801 |
| **Tokens moy/chunk** | ~215 |
| **Input/requete** | 364 tokens (chunk + system prompt) |
| **Output/requete** | 200 tokens (3 questions JSON) |
| **Total input** | 1.02M tokens |
| **Total output** | 0.56M tokens |
| **Total** | **1.58M tokens** |
| **Cout Batch API (50%)** | **~$0.12** |

#### 6.5.8 Alternatives: Generation sur Kaggle/HuggingFace

##### Option A: Kaggle Notebook (GPU gratuit)

> **Avantages**: 30h GPU/semaine gratuit, Gemini API integre, notebooks officiels Google
> **Limitations**: Quota API Gemini identique (free tier)

| Resource | URL | Description |
|----------|-----|-------------|
| **Gemini Synthetic Data** | [GoogleCloudPlatform/synthetic_data_generation](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/data-generation/synthetic_data_generation_using_gemini.ipynb) | Notebook officiel Google |
| **Gemini API Starter** | [kaggle.com/prathameshbang](https://www.kaggle.com/code/prathameshbang/gemini-api-starter-notebook) | Template Kaggle + Gemini |
| **Fine-tuning RAG LlamaIndex** | [kaggle.com/hiarsl](https://www.kaggle.com/code/hiarsl/fine-tuning-embeddings-for-rag-with-llamaindex) | Embedding fine-tuning complet |

##### Option B: HuggingFace Synthetic Data Generator (gratuit)

> **Avantages**: 100% gratuit via HF Inference API, Llama 3.1 8B/70B, ~50 samples/min
> **Limitations**: Taux limite, moins configurable que Gemini

| Resource | URL | Description |
|----------|-----|-------------|
| **Synthetic Data Generator** | [huggingface.co/blog/synthetic-data-generator](https://huggingface.co/blog/synthetic-data-generator) | Outil officiel HF |
| **GitHub argilla-io** | [github.com/argilla-io/synthetic-data-generator](https://github.com/argilla-io/synthetic-data-generator) | Code source Apache 2.0 |
| **DataDreamer** | [huggingface.co/blog/asoria/datadreamer-datasets](https://huggingface.co/blog/asoria/datadreamer-datasets) | Alternative Python |
| **HF Inference API** | [huggingface.co/inference-api](https://huggingface.co/inference-api) | API gratuite (rate limited) |

```bash
# Installation HF Synthetic Generator
pip install synthetic-dataset-generator

# Necessite HF_TOKEN pour push datasets
export HF_TOKEN="hf_xxx"
```

##### Option C: Notebooks embedding fine-tuning (reference)

| Resource | URL | Description |
|----------|-----|-------------|
| **Phil Schmid RAG Fine-tuning** | [github.com/philschmid](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-embedding-model-for-rag.ipynb) | **Notebook complet** triplets + synthetic |
| **Sentence Transformers v3** | [huggingface.co/blog/train-sentence-transformers](https://huggingface.co/blog/train-sentence-transformers) | Doc officielle training |
| **ModernBERT RAG** | [huggingface.co/blog/sdiazlor](https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data) | Distilabel synthetic |
| **Triplet Margin Loss PyTorch** | [medium.com/@rjnclarke](https://medium.com/@rjnclarke/fine-tune-an-embedding-model-with-triplet-margin-loss-in-pytorch-62bf00865a6c) | Implementation manuelle |
| **AWS BGE Fine-tuning** | [aws.amazon.com/blogs](https://aws.amazon.com/blogs/machine-learning/fine-tune-a-bge-embedding-model-using-synthetic-data-from-amazon-bedrock/) | Bedrock synthetic |
| **LlamaIndex Fine-tuning** | [llamaindex.ai/blog](https://www.llamaindex.ai/blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971) | Framework complet |
| **Synthetic Generation Gist** | [gist.github.com/Krilecy](https://gist.github.com/Krilecy/999e74da1cd412b36abdc502c7fe8ede) | Script generation |

#### 6.5.9 Standards Industrie - Redaction Questions Synthetiques

> **Sources**: [Glean Enterprise RAG](https://jxnl.co/writing/2025/03/06/fine-tuning-embedding-models-for-enterprise-rag-lessons-from-glean/), [NVIDIA SDG](https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/), [Aurelio AI](https://www.aurelio.ai/learn/sentence-transformers-fine-tuning), [HuggingFace ModernBERT](https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data)

##### A. Principes fondamentaux (consensus industrie)

| Principe | Description | Source |
|----------|-------------|--------|
| **Quality > Quantity** | Prioritiser la qualite du signal sur le volume | Glean |
| **Blend Real + Synthetic** | Combiner donnees reelles (Gold Standard) et synthetiques | Label Your Data |
| **Diversity obligatoire** | Varier formulations, types, difficultes | NVIDIA, LlamaIndex |
| **Independence** | Questions independantes du texte source (pas de reformulation) | NVIDIA |
| **Realism** | Questions comme les vrais utilisateurs les poseraient | Glean, NVIDIA |

##### B. Quatre criteres NVIDIA pour questions synthetiques

Les prompts de generation doivent viser:

1. **Query Independence**: La question ne doit pas reprendre les mots exacts du chunk
2. **Query Realisticness**: Formulation naturelle, comme un vrai utilisateur
3. **Query Diversity**: Varier types (factuel, procedural, definition)
4. **Relevance to Context**: La reponse DOIT etre dans le chunk

##### C. Dataset sizes recommandes (benchmarks)

| Source | Taille | Performance |
|--------|--------|-------------|
| **Phil Schmid (HuggingFace)** | 6,300 paires | +7% performance |
| **Aurelio AI** | 4,719 paires | Amelioration significative |
| **HuggingFace ModernBERT** | 500 rows/dataset | Suffisant pour fine-tuning |
| **Glean Enterprise** | Signal quality prioritaire | User feedback > synthetic |

##### D. Validation dual-layer (recommande)

```
┌─────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  LAYER 1: STATISTIQUE (automatique)                         │
│  ├── Embedding similarity filter (cosine > 0.7)             │
│  ├── Filtrer empty/NaN                                       │
│  ├── Detecter duplicates                                     │
│  └── Answerability check (NVIDIA: 94% precision)            │
│                                                              │
│  LAYER 2: HUMAN-IN-THE-LOOP (echantillon)                   │
│  ├── Review 10% des questions generees                       │
│  ├── Valider coherence domaine                               │
│  └── Ajuster prompts si necessaire                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 6.5.10 Categories de questions Pocket Arbiter

> **Implementation**: `scripts/pipeline/generate_triplets_hf.py`

##### Corpus FFE (francais uniquement)

| Categorie | Description | Exemples |
|-----------|-------------|----------|
| **arbitre_terrain** | Cas particuliers en competition | "Quel sera l'Elo d'un joueur apres 9 rondes?", "Un portable sonne, quelle sanction?" |
| **arbitre_organisateur** | Organisation tournoi, formation arbitrale | "Conditions pour devenir AF2?", "Delai d'homologation?" |
| **question_joueur** | Questions orales de joueurs (langage familier) | "J'ai le droit de proposer nulle quand?", "C'est quoi le departage?" |

##### Corpus FIDE (anglais)

| Categorie | Description | Exemples |
|-----------|-------------|----------|
| **arbiter_field** | Specific cases during tournament | "What is the penalty for a mobile phone ringing?" |
| **arbiter_organizer** | Tournament organization, certification | "Requirements to become FIDE Arbiter?" |
| **player_question** | Oral questions from players | "Can I offer a draw before making my move?" |

##### Prompt FFE (extrait)

```python
"""Tu es un arbitre d'echecs FFE experimente (AF3 minimum).
Tu generes des questions REALISTES que l'on te pose sur le terrain ou en formation.

TROIS CATEGORIES DE QUESTIONS (varier obligatoirement):

1. ARBITRE TERRAIN - Cas particuliers concrets en competition
2. ARBITRE ORGANISATEUR - Organisation tournoi ou formation arbitrale
3. JOUEUR - Questions orales d'un joueur (langage familier, verbal)

REGLES STRICTES:
- Langue: FRANCAIS uniquement
- La reponse DOIT etre trouvable dans le texte fourni
- Style: questions naturelles, pas academiques
- Jargon FFE: Elo, cadence, forfait, appariement, homologation, departage
"""
```

#### 6.5.11 Estimation volume et qualite

| Metrique | Valeur | Notes |
|----------|--------|-------|
| **Chunks total** | 2801 | FR + INTL |
| **Questions/chunk** | 3 | Recommande |
| **Paires brutes** | ~8400 | Avant filtrage |
| **Taux filtrage** | ~10-20% | Questions hors-sujet |
| **Triplets finaux** | 6700-7500 | Apres hard negatives |
| **Split train/val** | 80/20 | Standard |

#### 6.5.12 Qualite: LLM-as-Judge (optionnel)

```python
def filter_with_llm_judge(
    client: genai.Client,
    pairs: list[dict],
    threshold: float = 0.7
) -> list[dict]:
    """
    Filtrer questions avec LLM-as-judge.
    Garde seulement questions de qualite >= threshold.
    """
    filtered = []

    judge_prompt = """Evalue cette question pour entrainement RAG echecs (0-1):

Question: {question}
Chunk source: {chunk_preview}

Criteres:
- Pertinence: La reponse est dans le chunk?
- Naturalite: Un arbitre poserait cette question?
- Clarte: Question non ambigue?

Reponds JSON: {{"score": 0.X, "reason": "..."}}"""

    for pair in pairs:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=judge_prompt.format(
                question=pair["anchor"],
                chunk_preview=pair["positive"][:500]
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        result = json.loads(response.text)
        if result["score"] >= threshold:
            filtered.append(pair)

    return filtered
```

### 6.6 Resume: Composition finale dataset

| Split | Source | Volume | Usage |
|-------|--------|--------|-------|
| **Train** | Gold Standard (80%) | ~154 triplets | Base qualite |
| **Train** | Synthetique (Gemini) | ~4000 triplets | Augmentation |
| **Train TOTAL** | - | **~4150 triplets** | Fine-tuning |
| **Val** | Gold Standard (20%) | ~39 triplets | Early stopping |
| **Test** | Gold Standard reserve | - | Evaluation finale |

> **Important**: Le val/test reste 100% Gold Standard.
> Cela garantit une evaluation fiable sur des questions humain-validees.

#### Script merge final

```python
from datasets import Dataset, concatenate_datasets

def create_final_dataset(
    gold_triplets_path: str,
    synthetic_triplets_path: str,
    val_ratio: float = 0.2
) -> dict:
    """
    Creer dataset final avec Gold Standard + Synthetique.

    Returns:
        {"train": Dataset, "val": Dataset}
    """
    # Charger Gold Standard
    gold = Dataset.load_from_disk(gold_triplets_path)

    # Split Gold Standard 80/20
    gold_splits = gold.train_test_split(test_size=val_ratio, seed=42)
    gold_train = gold_splits["train"]
    gold_val = gold_splits["test"]

    # Charger Synthetique (train only)
    synthetic = Dataset.load_from_disk(synthetic_triplets_path)

    # Merge train
    train_final = concatenate_datasets([gold_train, synthetic])
    train_final = train_final.shuffle(seed=42)

    print(f"Train: {len(train_final)} ({len(gold_train)} gold + {len(synthetic)} synth)")
    print(f"Val: {len(gold_val)} (100% gold)")

    return {
        "train": train_final,
        "val": gold_val,
    }


# Usage
dataset = create_final_dataset(
    "data/gold_triplets",
    "data/synthetic_triplets",
    val_ratio=0.2
)
```

---

## 7. Evaluation

### 7.1 Metrics cibles

| Metrique | Baseline | Objectif | Mesure |
|----------|----------|----------|--------|
| **Recall@5 FR** | 91.56% | 95%+ | `InformationRetrievalEvaluator` |
| **Recall@5 INTL** | 93.22% | 95%+ | `InformationRetrievalEvaluator` |
| **Hard cases FR** | 46/150 echecs | <20/150 | Count manual |
| **nDCG@10** | ~60% | 75%+ | `InformationRetrievalEvaluator` |

### 7.2 Evaluator configuration

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Preparer donnees eval
queries = {i: q["question"] for i, q in enumerate(gold_standard)}
corpus = {chunk.id: chunk.text for chunk in all_chunks}
relevant_docs = {i: [q["expected_chunk_id"]] for i, q in enumerate(gold_standard)}

# Creer evaluator
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="chess-fr-retrieval",
    show_progress_bar=True,

    # Metrics a calculer
    mrr_at_k=[1, 5, 10],
    ndcg_at_k=[5, 10],
    accuracy_at_k=[1, 5, 10],
    precision_recall_at_k=[5, 10],
    map_at_k=[10],
)
```

### 7.3 Logging W&B

```python
import wandb

# Init W&B
wandb.init(
    project="pocket-arbiter-embeddings",
    name="embeddinggemma-chess-fr-v1",
    config={
        "model": "google/embeddinggemma-300m",
        "dataset": "gold_standard_fr.json",
        "epochs": 3,
        "batch_size": 64,
        "learning_rate": 2e-5,
    }
)

# Log custom metrics apres training
wandb.log({
    "recall@5_baseline": 0.9156,
    "recall@5_finetuned": 0.95,
    "hard_cases_before": 46,
    "hard_cases_after": 18,
})
```

---

## 8. Deploiement

### 8.1 Export modele fine-tune

```python
# Sauvegarder localement
model.save_pretrained("./models/embeddinggemma-chess-fr-v1")

# Push to HuggingFace Hub
model.push_to_hub("pocket-arbiter/embeddinggemma-chess-fr")
```

### 8.2 Re-generation embeddings corpus

```python
from sentence_transformers import SentenceTransformer

# Charger modele fine-tune
model = SentenceTransformer("./models/embeddinggemma-chess-fr-v1")

# Re-generer tous les embeddings
for chunk in all_chunks:
    new_embedding = model.encode_document(chunk.text)
    db.update_embedding(chunk.id, new_embedding)
```

### 8.3 Tests regression

```bash
# Verifier que recall n'a pas baisse
python -m pytest scripts/pipeline/test_retrieval.py -v -k "recall"

# Benchmark complet
python scripts/pipeline/benchmark_retrieval.py --model ./models/embeddinggemma-chess-fr-v1
```

---

## 9. Checklist Pre-Notebook

### 9.1 Prerequis

- [ ] GPU disponible (T4 minimum, A100 recommande)
- [ ] `sentence-transformers>=3.0` installe
- [ ] `transformers>=4.56.0` installe
- [ ] `wandb` configure (optionnel mais recommande)
- [ ] Acces Kaggle/HuggingFace pour EmbeddingGemma

### 9.2 Donnees

- [ ] `gold_standard_fr.json` valide (150 questions)
- [ ] `corpus_fr.db` accessible avec chunks
- [ ] Script generation hard negatives teste
- [ ] Split train/val prepare (80/20)

### 9.3 Configuration

- [ ] Hyperparametres choisis (voir section 4)
- [ ] Evaluator configure
- [ ] Early stopping callback pret
- [ ] Logging W&B/TensorBoard active

### 9.4 Post-training

- [ ] Script export modele pret
- [ ] Script re-generation embeddings pret
- [ ] Tests regression prepares

---

## 10. Erreurs a eviter

| Erreur | Consequence | Solution |
|--------|-------------|----------|
| Batch size trop petit | MNR Loss inefficace | Minimum 32, ideal 64-128 |
| Pas de prompts | Embeddings sous-optimaux | Toujours utiliser prompts EmbeddingGemma |
| LR trop eleve | Overfitting rapide | Commencer a 2e-5, reduire si necessaire |
| Pas d'early stopping | Overfitting | Callback + patience 3 |
| Hard negatives random | Training signal faible | Utiliser retrieval-based negatives |
| Oublier NO_DUPLICATES | False negatives dans batch | `batch_sampler=BatchSamplers.NO_DUPLICATES` |
| Pas de val split | Impossible detecter overfitting | 80/20 split obligatoire |

---

## 11. Variantes EmbeddingGemma et Deploiement Mobile

> **Sources**: [Google Developers Blog](https://developers.googleblog.com/en/introducing-embeddinggemma/), [AI Edge RAG Guide](https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android), [HuggingFace EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)

### 11.1 Variantes disponibles

| Variante | Source | Taille | Format | Usage |
|----------|--------|--------|--------|-------|
| `google/embeddinggemma-300m` | [HuggingFace](https://huggingface.co/google/embeddinggemma-300m) | ~1.2 GB | PyTorch | **Fine-tuning LoRA** |
| `google/embeddinggemma-300m-qat-q4_0-unquantized` | HuggingFace | ~600 MB | PyTorch Q4 | **Fine-tuning QLoRA** |
| `litert-community/embeddinggemma-300m` | [HuggingFace](https://huggingface.co/litert-community/embeddinggemma-300m) | 179-196 MB | .tflite | **Deploiement mobile** |

### 11.2 Comparaison QLoRA vs LoRA sur TFLite

| Critere | QLoRA (HuggingFace) | LoRA sur TFLite |
|---------|---------------------|-----------------|
| **Fichier** | `.safetensors` PyTorch | `.tflite` LiteRT |
| **Fine-tuning** | ✅ OUI | ❌ NON (frozen) |
| **Librairie** | sentence-transformers | LiteRT runtime |
| **GPU requis** | Oui (training) | Non (inference) |
| **Taille modele** | ~600 MB | ~180 MB |
| **Adapters** | Trainable | Non supporte |

**Conclusion**: Le modele TFLite de 110-180 MB est le **resultat final** apres conversion, pas le point de depart du fine-tuning.

### 11.3 Performances EmbeddingGemma (benchmarks officiels)

| Metrique | Valeur | Notes |
|----------|--------|-------|
| **MTEB Rank** | #1 (<500M params) | Meilleur modele multilingual open <500M |
| **Latence inference** | <15 ms | 256 tokens sur EdgeTPU |
| **RAM quantifie** | <200 MB | Avec QAT int4/int8 |
| **Dimensions** | 768D (ou 128/256/512 MRL) | Truncation sans perte majeure |
| **Contexte** | 2048 tokens | vs 512 pour la plupart des concurrents |
| **Langues** | 100+ | Francais inclus |

### 11.4 Workflow Pocket Arbiter : Fine-tuning → Deploiement

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: FINE-TUNING (PC/Cloud - Kaggle/Colab)                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ google/embeddinggemma-300m (HuggingFace)                  │  │
│  │         ↓                                                  │  │
│  │ QLoRA fine-tuning avec triplets FFE (~5000 triplets)      │  │
│  │         ↓                                                  │  │
│  │ embeddinggemma-chess-fr/ (adapters fusionnes)             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  PHASE 2: EXPORT MOBILE                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Conversion TFLite + quantization int4/int8                │  │
│  │         ↓                                                  │  │
│  │ embeddinggemma-chess-fr.tflite (~110-180 MB)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  PHASE 3: DEPLOIEMENT ANDROID (AI Edge RAG SDK)                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ LiteRT runtime → inference on-device <15ms                │  │
│  │ Devices cibles: Pixel 8/9, Samsung S23/S24                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.5 Integration AI Edge RAG SDK

> **Reference**: [AI Edge RAG Guide Android](https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android)

```kotlin
// RagPipeline.kt - Integration EmbeddingGemma fine-tune
private val embedder: Embedder<String> = GeckoEmbeddingModel(
    context = context,
    modelPath = "embeddinggemma-chess-fr.tflite",  // Modele fine-tune
    sequenceLength = 1024,
    outputDimension = 768  // Ou 256 pour MRL truncate
)

// Retrieval avec embeddings domaine echecs
val retrievalRequest = RetrievalRequest.create(
    prompt = "Quelle est la sanction si un portable sonne?",
    config = RetrievalConfig.create(
        topK = 5,
        minScore = 0.7f,
        taskType = TaskType.QUESTION_ANSWERING
    )
)
retrievalAndInferenceChain.invoke(retrievalRequest, callback)
```

### 11.6 Comparaison approches fine-tuning

| Approche | Avantages | Inconvenients | Pour Pocket Arbiter |
|----------|-----------|---------------|---------------------|
| **QLoRA sur HuggingFace** | Adapte au domaine echecs, meilleur recall | Necessite GPU, conversion TFLite | ✅ Recommande |
| **Export TFLite direct** (sans fine-tuning) | Simple, rapide | Performance baseline seulement | ⚠️ Fallback |
| **LoRA sur TFLite** | N/A | Non supporte (format inference-only) | ❌ Impossible |

### 11.7 Projection gains fine-tuning domaine echecs

| Metrique | Baseline (EmbeddingGemma) | Apres QLoRA Fine-tuning | Gain estime |
|----------|---------------------------|-------------------------|-------------|
| Recall@5 FR | 91.56% | 94-97% | +3-6% |
| Latence mobile | <15 ms | <15 ms | = |
| Taille APK | +180 MB | +180 MB | = |
| Hard cases resolus | 104/150 | 125-140/150 | +20-35% |

---

## 12. Benchmarks de reference

### 12.1 Gains observes (domain-specific fine-tuning)

| Domaine | Avant | Apres | Gain | Source |
|---------|-------|-------|------|--------|
| Legal | 59.5% nDCG | 82.5% nDCG | +38% | [Voyage AI](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/) |
| Mortgage | 59.9% nDCG | 62.1% nDCG | +4% | [SugiV](https://blog.sugiv.fyi/mortgage-embeddinggemma) |
| Medical (MIRIAD) | 83.4% nDCG | 88.6% nDCG | +5% | [HuggingFace](https://huggingface.co/blog/embeddinggemma) |

### 12.2 Projection Pocket Arbiter

| Metrique | Baseline | Projection conservative | Projection optimiste |
|----------|----------|------------------------|---------------------|
| Recall@5 FR | 91.56% | 94% | 97% |
| Hard cases | 46/150 | 25/150 | 15/150 |
| nDCG@10 | ~60% | 70% | 80% |

---

## Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-21 | Claude Opus 4.5 | Creation initiale - consolidation sources |
| 1.1 | 2026-01-21 | Claude Opus 4.5 | Strategie 2 tiers (Gold Standard + Synthetique), section 6.5 Gemini API |
| 1.2 | 2026-01-21 | Claude Opus 4.5 | Ajout standards industrie (NVIDIA, Glean, Aurelio), 3 categories metier FFE/FIDE |
| 1.3 | 2026-01-21 | Claude Opus 4.5 | Section 11: Variantes EmbeddingGemma, workflow deploiement mobile, AI Edge RAG SDK |

---

*Document ISO 42001/25010 - Pocket Arbiter Project*
