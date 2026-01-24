# Ressources Fine-tuning EmbeddingGemma - MRL + LoRA

> **Document ID**: RES-FT-RESOURCES-001
> **ISO Reference**: ISO 42001, ISO 25010
> **Version**: 1.2
> **Date**: 2026-01-24
> **Statut**: Reference
> **Classification**: Interne
> **Origine**: Extrait de LORA_FINETUNING_GUIDE.md (sections 1-5)
> **Mots-cles**: fine-tuning, LoRA, MRL, EmbeddingGemma, references, hyperparameters

---

## 1. Objectif

Ce document consolide les ressources et references techniques pour le fine-tuning EmbeddingGemma.

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

### 2.3 HuggingFace / Community

| Resource | URL | Contenu |
|----------|-----|---------|
| **HuggingFace EmbeddingGemma Blog** | [huggingface.co](https://huggingface.co/blog/embeddinggemma) | Architecture, fine-tuning complet |
| **Train Sentence Transformers v3** | [huggingface.co](https://huggingface.co/blog/train-sentence-transformers) | Guide officiel ST v3 |
| **Aurelio AI - ST Fine-tuning** | [aurelio.ai](https://www.aurelio.ai/learn/sentence-transformers-fine-tuning) | Best practices |
| **Pinecone - MNR Loss** | [pinecone.io](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/) | MultipleNegativesRankingLoss |

### 2.4 Best Practices / Overfitting

| Resource | URL | Focus |
|----------|-----|-------|
| **Zilliz - Overfitting Prevention** | [zilliz.com](https://zilliz.com/ai-faq/what-should-i-do-if-the-finetuning-process-for-a-sentence-transformer-model-overfits-quickly-for-example-training-loss-gets-much-lower-than-validation-loss-early-on) | Early stopping, dropout |
| **Zilliz - ST Parameters** | [zilliz.com](https://zilliz.com/ai-faq/what-parameters-can-be-adjusted-when-finetuning-a-sentence-transformer-eg-learning-rate-batch-size-number-of-epochs-and-how-do-they-impact-training) | Hyperparameters |
| **Unsloth - LoRA Hyperparameters** | [unsloth.ai](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) | LoRA rank, alpha |
| **SBERT Training Overview** | [sbert.net](https://www.sbert.net/docs/sentence_transformer/training_overview.html) | Callbacks, logging |

### 2.5 Generation synthetique de triplets

| Resource | URL | Focus |
|----------|-----|-------|
| **LlamaIndex finetune-embedding** | [github.com/run-llama](https://github.com/run-llama/finetune-embedding) | Pipeline LLM → triplets |
| **Databricks Embedding Finetuning** | [databricks.com](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning) | Synthetic + LLM-as-judge |
| **NVIDIA SDG RAG** | [nvidia.dev](https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/) | 4 criteres qualite |
| **Phil Schmid RAG Fine-tune** | [philschmid.de](https://www.philschmid.de/fine-tune-embedding-model-for-rag) | 6.3k samples, +7% |
| **Glean Enterprise RAG** | [jxnl.co](https://jxnl.co/writing/2025/03/06/fine-tuning-embedding-models-for-enterprise-rag-lessons-from-glean/) | Quality > Quantity |

---

## 3. Specifications EmbeddingGemma

### 3.1 Architecture

| Spec | Valeur |
|------|--------|
| **Parametres** | 308M |
| **Base** | Gemma 3 transformer, bi-directional attention |
| **Dimensions output** | 768 (truncatable: 512, 256, 128 via MRL) |
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
    "STS": "task: sentence similarity | query: ",
}
```

### 3.4 Modeles EmbeddingGemma - Fine-tuning vs Deploiement

> **RECOMMANDATION**: Utiliser **QLoRA** avec le modele QAT pour le fine-tuning Pocket Arbiter.

| Usage | Model ID | Source | Taille | Recommandation |
|-------|----------|--------|--------|----------------|
| **Fine-tuning QLoRA** | `google/embeddinggemma-300m-qat-q4_0-unquantized` | [HuggingFace](https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized) | ~600 MB | **RECOMMANDE** |
| **Fine-tuning full** | `google/embeddinggemma-300m` | [HuggingFace](https://huggingface.co/google/embeddinggemma-300m) | ~1.2 GB | Fallback si QLoRA echoue |
| **Deploiement Android** | `litert-community/embeddinggemma-300m` | [HuggingFace](https://huggingface.co/litert-community/embeddinggemma-300m) | 179-196 MB | Production |

#### 3.4.1 Modele RECOMMANDE: `google/embeddinggemma-300m-qat-q4_0-unquantized`

> **Pourquoi QLoRA > Full Fine-tuning:**
> - **VRAM**: 8-12 GB vs 24+ GB (T4 suffisant vs A100 requis)
> - **Vitesse**: 2-3x plus rapide (gradient checkpointing)
> - **Stabilite**: Pas de catastrophic forgetting
> - **Qualite post-quant**: QAT preserve la qualite apres quantization TFLite

- **Framework**: sentence-transformers >= 3.0 + peft
- **Loss**: `CachedMultipleNegativesRankingLoss` (memory-efficient)
- **LoRA config**: rank=8, alpha=16, dropout=0.1
- **Target modules**: `["q_proj", "v_proj", "k_proj", "o_proj"]`
- **Reference**: [Google QAT Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)

#### 3.4.2 Modele Fallback: `google/embeddinggemma-300m`

- **Usage**: Full fine-tuning si QLoRA incompatible
- **Prerequis**: GPU A100 40GB+ ou multi-GPU
- **Note**: Ne supporte PAS float16, utiliser float32 ou bfloat16
- **Quand utiliser**: Seulement si QLoRA donne des resultats inferieurs

#### 3.4.3 Modele Deploiement: `litert-community/embeddinggemma-300m`

Variantes TFLite disponibles:

| Sequence Length | Taille | Inference CPU | Inference GPU | RAM (CPU) |
|-----------------|--------|---------------|---------------|-----------|
| 256 tokens | 179 MB | 66 ms | 64 ms | 110 MB |
| 512 tokens | 179 MB | 169 ms | 119 ms | 123 MB |
| 1024 tokens | 183 MB | 549 ms | 241 ms | 169 MB |
| 2048 tokens | 196 MB | 2455 ms | 683 ms | 333 MB |

**Recommandation Pocket Arbiter**: 256 tokens (chunks < 450 tokens tronques)

### 3.5 Workflow QLoRA Fine-tuning → Deploiement Android

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: FINE-TUNING QLoRA (Kaggle T4 / Colab)                     │
│  ├── Model: google/embeddinggemma-300m-qat-q4_0-unquantized         │
│  ├── Framework: sentence-transformers + peft (QLoRA)                │
│  ├── Loss: CachedMultipleNegativesRankingLoss                       │
│  ├── Data: triplets GS v6.7.0 (UNIFIED_TRAINING_DATA_SPEC.md)       │
│  ├── VRAM: ~10 GB (T4 16GB OK)                                      │
│  └── Output: embeddinggemma-chess-fr/ (adapters LoRA)               │
│                         ↓                                           │
│  PHASE 1.5: MERGE ADAPTERS                                          │
│  ├── Merge LoRA adapters into base model                            │
│  └── Output: embeddinggemma-chess-fr-merged/ (full weights)         │
│                         ↓                                           │
│  PHASE 2: CONVERSION TFLite                                         │
│  ├── Tool: ai-edge-torch (https://github.com/google-ai-edge/ai-edge-torch)
│  ├── Quantization: Mixed precision e4_a8_f4_p4 (int4 + int8)        │
│  └── Output: embeddinggemma-chess-fr.tflite (~180 MB)               │
│                         ↓                                           │
│  PHASE 3: DEPLOIEMENT ANDROID                                       │
│  ├── Runtime: LiteRT (XNNPACK CPU ou GPU)                           │
│  ├── RAG SDK: com.google.ai.edge.localagents:localagents-rag:0.1.0  │
│  ├── Embedder: Custom via Embedder<String> interface                │
│  └── Tokenizer: SentencePiece (separe du modele)                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.6 Compatibilite Google AI Edge RAG SDK

| Aspect | Officiel (Gecko) | Custom (EmbeddingGemma) |
|--------|------------------|-------------------------|
| Embedder | `Gecko_256_f32.tflite` | `embeddinggemma.tflite` |
| Interface | Built-in GeckoEmbedder | Custom `Embedder<String>` |
| Dimensions | 768 | 768 (ou MRL: 512/256/128) |
| Tokenizer | Integre | SentencePiece separe |
| Integration | Directe | Via LiteRT API |

**Reference**: [AI Edge RAG Android Guide](https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android)

---

## 4. Hyperparametres Recommandes

### 4.1 Sweet Spots (Sentence Transformers)

| Parametre | Valeur recommandee | Notes |
|-----------|-------------------|-------|
| **learning_rate** | `2e-5` | Reduire si overfitting |
| **num_train_epochs** | `1-5` | 1 suffisant avec gros dataset |
| **per_device_train_batch_size** | `32-128` | Plus grand = meilleur signal MNR |
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

model = SentenceTransformer("google/embeddinggemma-300m")
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

args = SentenceTransformerTrainingArguments(
    output_dir="./embeddinggemma-chess-fr",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
)
```

---

## 5. Prevention Overfitting

### 5.1 Signaux d'alerte

| Signal | Description | Action |
|--------|-------------|--------|
| Train loss << Val loss | Gap croissant | Early stopping |
| Val loss stagnant | Plateau | Arreter |
| Val loss remonte | Overfitting clair | Rollback best checkpoint |

### 5.2 Techniques de prevention

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
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )
    ],
)
```

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-24 | Extraction depuis LORA_FINETUNING_GUIDE.md (sections 1-5) |
| 1.1 | 2026-01-24 | Ajout sections 3.4-3.6: Modeles, workflow TFLite, compatibilite RAG SDK |
| 1.2 | 2026-01-24 | **CORRECTION CRITIQUE**: QLoRA = methode RECOMMANDEE (pas optionnelle), modele QAT prioritaire, workflow mis a jour |

---

*Document ISO 42001/25010 - Pocket Arbiter Project*
