# Solutions Vector-Based - Optimisation RAG

> **Pocket Arbiter** - Recherche HuggingFace/Kaggle/Google AI Dev
> Date: 2026-01-22 | Version: 1.2 | ISO 25010, ISO 42001

---

## 1. Solutions Compression (Zéro RAM Runtime)

### 1.1 Tableau Comparatif

| Solution | Gain | RAM Runtime | Complexité | Recall Impact |
|----------|------|-------------|------------|---------------|
| **Matryoshka 768→256D** | 3x storage | Aucune | Trivial | -1% |
| **Binary Quantization** | 32x compression | Aucune | Moyen | -4% |
| **Model2Vec Static** | 500x faster | Aucune | Moyen | -13% |
| **E5-small-v2** | 14x faster | ~200MB | Trivial | EN only |

### 1.2 Matryoshka Representation Learning (MRL)

**Sources**:
- [HuggingFace Blog - Matryoshka](https://huggingface.co/blog/matryoshka)
- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)

**Principe**: Embeddings entraînés pour stocker l'information importante dans les premières dimensions. Truncation sans retraining.

**Implémentation**:
```python
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

model = SentenceTransformer(
    "google/embeddinggemma-300m-qat-q4_0-unquantized",
    truncate_dim=256  # au lieu de 768
)

# Ou manuellement:
embeddings = model.encode(texts)
truncated = F.normalize(embeddings[:, :256], dim=-1)
```

**Benchmarks**:
| Dimensions | Performance | Storage | Speedup |
|------------|-------------|---------|---------|
| 768 (full) | 100% | 100% | 1x |
| 512 | 99.5% | 67% | 1.5x |
| 256 | 99% | 33% | 2x |
| 128 | 97% | 17% | 3x |

**Applicabilité Pocket Arbiter**: ✅ Immédiate (EmbeddingGemma supporte MRL nativement)

---

### 1.3 Binary Quantization

**Sources**:
- [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization)
- [Qdrant - Binary Quantization 40x Faster](https://qdrant.tech/articles/binary-quantization/)

**Principe**: Convertir float32 → 1 bit (signe uniquement). 32x compression.

**Implémentation**:
```python
import numpy as np

def binary_quantize(embeddings: np.ndarray) -> np.ndarray:
    """Quantize embeddings to binary (32x compression)."""
    return np.packbits((embeddings > 0).astype(np.uint8), axis=-1)

def hamming_distance(a: bytes, b: bytes) -> int:
    """Fast comparison via XOR + popcount."""
    return bin(int.from_bytes(a, 'big') ^ int.from_bytes(b, 'big')).count('1')

# Pipeline recommandé (3-step):
# 1. Binary search (RAM) → top-100 candidats
# 2. Float32 rescoring (disk) → top-10
# 3. (Optionnel) Cross-encoder API si connecté
```

**Benchmarks**:
| Méthode | Recall@100 | Speedup | Compression |
|---------|------------|---------|-------------|
| Float32 baseline | 100% | 1x | 1x |
| Binary + rescore | 96% | 40x | 32x |
| Binary seul | 85% | 40x | 32x |

**Applicabilité Pocket Arbiter**: ✅ Moyen terme (nécessite dual-index)

---

### 1.4 Model2Vec Static Embeddings

**Sources**:
- [Model2Vec GitHub](https://github.com/MinishLab/model2vec)
- [HuggingFace Blog - Static Embeddings](https://huggingface.co/blog/static-embeddings)

**Principe**: Distiller un modèle transformer en lookup table statique. Pas d'inférence.

**Implémentation**:
```python
from model2vec.distill import distill

# Distillation one-shot (30 sec CPU, pas de dataset)
static_model = distill(
    model_name="google/embeddinggemma-300m-qat-q4_0-unquantized",
    pca_dims=256
)
static_model.save_pretrained("./embeddinggemma-m2v-256")

# Usage (500x faster)
from model2vec import StaticModel
model = StaticModel.from_pretrained("./embeddinggemma-m2v-256")
embeddings = model.encode(["Texte à encoder"])
```

**Modèle multilingue existant**: [M2V_multilingual_output](https://huggingface.co/minishlab/M2V_multilingual_output)

**Benchmarks**:
| Modèle | Params | Perf vs mpnet | Speedup CPU | Speedup GPU |
|--------|--------|---------------|-------------|-------------|
| all-mpnet-base-v2 | 110M | 100% | 1x | 1x |
| static-retrieval-mrl-en-v1 | ~5M | 87.4% | 400x | 25x |
| M2V_multilingual | ~5M | 86.5% | 400x | 25x |

**Applicabilité Pocket Arbiter**: ⚠️ À évaluer (perte potentielle sur FR technique)

---

### 1.5 E5-small-v2 (Alternative légère)

**Source**: [E5 Benchmark](https://research.aimultiple.com/open-source-embedding-models/)

**Specs**: 118M params, 384D, 512 tokens max

**Benchmarks**:
- Latency: 16ms (14x faster que 7B models)
- Top-5 Accuracy: 100% (meilleur que modèles 7B)
- **Limitation**: English only

**Applicabilité Pocket Arbiter**: ❌ Non (corpus FR)

---

## 2. Solutions Amélioration Recall

### 2.1 Tableau Comparatif

> **CONTRAINTE VISION.md §5.1**: 100% offline, RAM < 500MB, latence < 5s
> Seules les techniques **INDEX-TIME** sont applicables

| Solution | Gain Recall | RAM Runtime | Complexité | Phase |
|----------|-------------|-------------|------------|-------|
| **Contextual Retrieval** | -49% à -67% failures | 0 (index) | Moyen | Index |
| **Semantic Chunking** | +9% | Aucune | Moyen | Index |
| **Fine-tuning MRL+LoRA** | +16-38% | Aucune | Élevé | Index |
| **Proposition-level** | +15-25% | Aucune | Moyen | Index |

### 2.2 Contextual Retrieval (Anthropic 2024)

**Sources**:
- [Anthropic - Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [ArXiv - Contextual Embeddings](https://arxiv.org/abs/2401.05831)

**Principe**: Ajouter contexte documentaire à chaque chunk AVANT embedding (index-time, pas query-time).

**Prompt Claude pour contexte**:
```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document
for the purposes of improving search retrieval of the chunk. Answer only with the
succinct context and nothing else.
```

**Implémentation**:
```python
def contextual_chunk(chunk: str, document: str, llm) -> str:
    """Add document context to chunk (index-time)."""
    prompt = f"""<document>
{document[:10000]}  # Truncate if too long
</document>
<chunk>
{chunk}
</chunk>
Provide 1-2 sentences of context situating this chunk."""

    context = llm.generate(prompt)  # ~50 tokens output
    return f"{context}\n\n{chunk}"

# À l'indexation:
for chunk in chunks:
    contextualized = contextual_chunk(chunk.text, full_document, claude)
    embedding = model.encode(contextualized)
    db.insert(embedding, chunk_id=chunk.id)
```

**Benchmarks Anthropic**:
| Configuration | Failure Rate | Réduction |
|---------------|--------------|-----------|
| Baseline (embeddings seuls) | 100% (ref) | - |
| Contextual Embeddings | 65% | **-35%** |
| Contextual Embeddings + BM25 | 51% | **-49%** |
| + Reranker (Cohere) | 33% | **-67%** |

**Coût estimation Pocket Arbiter**:
- ~2000 chunks × ~100 tokens input + ~50 tokens output
- Claude 3 Haiku: ~$0.002/chunk → **~$4 total**
- One-time index-time cost

**Applicabilité Pocket Arbiter**:
- Mode A (HybridChunker): ✅ **APPLICABLE** - post-processing chunks
- Mode B (LangChain): ✅ **APPLICABLE** - post-processing chunks
- **Priorité**: P1 (amélioration significative, coût index-time unique)

---

### 2.3 Semantic Chunking

**Sources**:
- [LangChain - Semantic Chunking](https://python.langchain.com/docs/how_to/semantic-chunker/)
- [Greg Kamradt - Chunking Strategies](https://www.youtube.com/watch?v=8OJC21T2SL4)

**Principe**: Diviser par similarité sémantique (cosine threshold) plutôt que tokens fixes.

**Algorithme**:
1. Split document en phrases/sentences
2. Embed chaque phrase
3. Calculer cosine similarity entre phrases adjacentes
4. Couper quand similarity < threshold (changement de topic)

**Implémentation**:
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")

# Breakpoint basé sur percentile de similarité
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,  # Top 5% dissimilarity = chunk boundary
)

chunks = chunker.split_text(document)
```

**Comparaison avec Mode A/B**:

| Aspect | Mode A (Hybrid) | Mode B (LangChain) | Semantic |
|--------|-----------------|--------------------|-----------|
| Méthode | Headings + tokens | Tokens fixes | Similarité |
| Boundaries | Structure doc | Arbitraires | Sémantiques |
| Avantage | Respecte structure | Simple | Topics cohérents |
| Inconvénient | Dépend parsing | Coupe mid-topic | Plus lent |

**Benchmarks**:
| Chunking | Recall@5 | Coherence |
|----------|----------|-----------|
| Fixed-size (512 tokens) | 78% | Low |
| Semantic | 87% | **High** |
| Gain | **+9%** | - |

**Applicabilité Pocket Arbiter**:
- Mode A (HybridChunker): ⚠️ **PARTIEL** - déjà headings-aware, semantic serait redondant
- Mode B (LangChain): ✅ **APPLICABLE** - remplacer RecursiveCharacterTextSplitter
- **Effort**: Moyen (re-chunking corpus)
- **Priorité**: P2 (après Contextual Retrieval)

---

### 2.4 Fine-tuning Domain-Specific (MRL + LoRA)

> ⚠️ **Note**: Le précédent fine-tuning a été réalisé avec un notebook de mauvaise qualité.
> Cette section documente l'approche recommandée basée sur les sources Google AI officielles.

**Sources principales**:
- [Google AI - Fine-tune EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers)
- [Google AI - LoRA Tuning Gemma](https://ai.google.dev/gemma/docs/core/lora_tuning)
- [Medium GDE - Fine-Tuning Gemma LoRA On-Device](https://medium.com/google-developer-experts/fine-tuning-gemma-with-lora-for-on-device-inference-android-ios-web-with-separate-lora-weights-f05d1db30d86)
- [HuggingFace - EmbeddingGemma](https://huggingface.co/blog/embeddinggemma)
- [Voyage AI - Legal Embeddings](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/)
- [SugiV - Fine-tuning EmbeddingGemma Mortgage](https://blog.sugiv.fyi/mortgage-embeddinggemma)

**Principe**: Combiner MRL (Matryoshka) + LoRA pour fine-tuner EmbeddingGemma sur le domaine arbitrage échecs.

#### 2.4.1 Pourquoi MRL + LoRA?

| Technique | Avantage | Application Pocket Arbiter |
|-----------|----------|----------------------------|
| **MRL** | Truncation sans perte (768→256D) | 3x compression storage |
| **LoRA** | Poids séparés (~10MB), base model inchangé | Updates OTA légères |
| **Combiné** | Fine-tuning efficace + déploiement flexible | Optimal mobile |

#### 2.4.2 Architecture recommandée

```
EmbeddingGemma-300M (base, 307M params)
         │
         ├── LoRA adapters (rank=4-16, ~10MB)
         │   └── Entraînés sur gold_standard_fr.json
         │
         └── MRL truncation (768→256D)
             └── Appliqué après LoRA inference
```

#### 2.4.3 Implémentation (Sentence Transformers v3+)

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from datasets import Dataset
import torch

# 1. Charger le modèle avec prompts
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("google/embeddinggemma-300M").to(device)

# 2. Préparer dataset triplets depuis gold_standard_fr.json
# Format: {"anchor": query, "positive": chunk_text, "negative": hard_negative}
train_data = []
for q in gold_standard:
    train_data.append({
        "anchor": q["question"],
        "positive": q["expected_chunk_text"],  # Texte du chunk attendu
        "negative": q["hard_negative_text"]    # Chunk similaire mais incorrect
    })

train_dataset = Dataset.from_list(train_data)

# 3. Configuration entraînement (paramètres Google AI recommandés)
args = SentenceTransformerTrainingArguments(
    output_dir="embeddinggemma-chess-fr",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Mixed precision
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="best",
    load_best_model_at_end=True,
)

# 4. Loss function optimale pour retrieval
loss = losses.MultipleNegativesRankingLoss(model)

# 5. Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

# 6. Fine-tuning
trainer.train()

# 7. Sauvegarder (poids complets, pas LoRA séparé avec ST)
model.save_pretrained("./embeddinggemma-chess-fr")
```

#### 2.4.4 Alternative: LoRA séparé avec Keras (On-Device)

Pour déploiement Android avec poids LoRA séparés (MediaPipe):

```python
import keras_hub
import os

os.environ["KERAS_BACKEND"] = "jax"

# 1. Charger modèle base
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b")

# 2. Activer LoRA (rank 4-16 recommandé)
model.backbone.enable_lora(rank=8)

# 3. Entraîner sur dataset
model.fit(train_data, epochs=3, batch_size=1)

# 4. Sauvegarder poids LoRA séparément
# → Convertir en FlatBuffers pour MediaPipe
# → Base model: ~2GB, LoRA weights: ~10MB
```

#### 2.5.5 Dataset Pocket Arbiter

**Source**: `tests/data/gold_standard_fr.json` (150 questions, 46 hard cases)

| Champ requis | Source | Description |
|--------------|--------|-------------|
| `question` | gold_standard | Query utilisateur |
| `expected_chunk_text` | corpus chunks | Texte du chunk correct |
| `hard_negative_text` | À générer | Chunk similaire mais faux |

**Génération hard negatives**:
```python
def generate_hard_negatives(question: str, positive_chunk: str, all_chunks: list):
    """Trouver chunks sémantiquement proches mais incorrects."""
    # 1. Embed question
    q_emb = model.encode(question)

    # 2. Trouver top-10 chunks (hors positif)
    candidates = retrieve_similar(q_emb, all_chunks, top_k=10)

    # 3. Filtrer le positif
    negatives = [c for c in candidates if c.id != positive_chunk.id]

    # 4. Retourner le plus difficile (rank 2-3)
    return negatives[0] if negatives else None
```

#### 2.5.6 Benchmarks attendus

| Métrique | Baseline | Post Fine-tuning | Source |
|----------|----------|------------------|--------|
| Recall@5 FR | 91.56% | 95-98% | Projection |
| nDCG@10 | ~60% | ~75% | [Mortgage domain](https://blog.sugiv.fyi/mortgage-embeddinggemma) |
| Hard cases | 46/150 | <20/150 | Objectif |

**Benchmark référence** (domain-specific fine-tuning):
- Legal domain: +38% nDCG@10 ([Voyage AI](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/))
- Mortgage domain: +4% nDCG@10 ([SugiV](https://blog.sugiv.fyi/mortgage-embeddinggemma))

**Applicabilité Pocket Arbiter**: ✅ **Priorité haute** (gold standard disponible, EmbeddingGemma MRL natif)

---

### 2.9 Proposition-Level Indexing

**Source**: [EMNLP 2024 - Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845.pdf)

**Principe**: Indexer au niveau proposition (phrase atomique) plutôt que chunk.

**Implémentation**:
```python
def extract_propositions(text: str) -> list[str]:
    """Extract atomic propositions from text."""
    # Utiliser LLM ou NLP pour extraire propositions
    # Ex: "Le pat est nul. Il survient quand..."
    #   → ["Le pat est une position nulle", "Le pat survient quand..."]
    pass

# Indexer chaque proposition séparément
for chunk in chunks:
    propositions = extract_propositions(chunk.text)
    for prop in propositions:
        emb = model.encode(prop)
        db.insert(emb, metadata={"parent_chunk": chunk.id, "text": prop})
```

**Benchmarks**: +15-25% recall vs passage-level sur NQ/TriviaQA

**Applicabilité Pocket Arbiter**: ⚠️ Nécessite re-chunking du corpus

---

## 3. Recommandations Pocket Arbiter

### 3.1 Court Terme (Indexing-time, zéro RAM runtime)

| Priorité | Solution | Effort | Gain Attendu | Mode A | Mode B |
|----------|----------|--------|--------------|--------|--------|
| **P0** | ⭐ **MRL + Fine-tuning EmbeddingGemma** | 1-2j | +5-15% recall hard cases | ✅ | ✅ |
| **P1** | ⭐ **Contextual Retrieval** (Anthropic) | 4h | -35% à -49% failures | ✅ | ✅ |
| **P1** | Matryoshka 768→256D | 1h | 3x storage, -1% recall | ✅ | ✅ |
| **P2** | Semantic Chunking | 4h | +9% recall | ⚠️ | ✅ |
| **P2** | Binary quantization | 2h | 32x compression | ✅ | ✅ |
| **P3** | Late Chunking | 1j | +10-12% | ❌ | ❌ |

### 3.1.1 Plan Fine-tuning MRL+LoRA (Priorité P0)

> ⚠️ **Remplace le notebook précédent de mauvaise qualité**

**Étapes**:
1. **Préparer dataset** (2h)
   - Extraire triplets depuis `gold_standard_fr.json`
   - Générer hard negatives via retrieval actuel
   - Split train/val 80/20

2. **Fine-tuning** (GPU T4/A100, 4h)
   - `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss`
   - 5 epochs, lr=2e-5, batch=8
   - Early stopping sur val loss

3. **Évaluation** (1h)
   - Recall@5 sur gold standard complet
   - Comparaison baseline vs fine-tuned

4. **Déploiement** (2h)
   - Export modèle fine-tuné
   - Re-génération embeddings corpus
   - Tests régression

**Ressources requises**:
- Colab Pro (T4 GPU) ou Kaggle
- `sentence-transformers>=3.0`
- `transformers>=4.56.0`

### 3.2 ~~Moyen Terme (Mode connecté, API LLM)~~ - HORS-SCOPE

> ❌ **HORS-SCOPE v1.0**: VISION.md §5.1 impose 100% offline
> Ces techniques nécessitent API LLM query-time, incompatibles avec la contrainte offline.

| ~~Priorité~~ | ~~Solution~~ | ~~Effort~~ | ~~Gain Attendu~~ | **Statut** |
|----------|----------|--------|--------------|------------|
| ~~P1~~ | ~~HyDE via Claude API~~ | ~~4h~~ | ~~+10-15%~~ | ❌ HORS-SCOPE |
| ~~P2~~ | ~~RAG-Fusion~~ | ~~4h~~ | ~~+5-10%~~ | ❌ HORS-SCOPE |

**Note**: Ces techniques pourraient être envisagées pour une version v2.0 "mode connecté" (VISION.md §8).

### 3.3 Long Terme (Refonte)

| Priorité | Solution | Effort | Gain Attendu |
|----------|----------|--------|--------------|
| **P2** | LoRA séparé + MediaPipe | 2j | OTA updates légères |
| **P3** | Proposition-level indexing | 2j | +15-25% recall |
| **P3** | Model2Vec distillation | 1j | 500x speedup, -13% recall |

---

## 4. Sources

### Google AI Dev (Officielles)
- [Google AI - Fine-tune EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers)
- [Google AI - LoRA Tuning Gemma](https://ai.google.dev/gemma/docs/core/lora_tuning)
- [Google AI - EmbeddingGemma Inference](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)
- [Google AI - MediaPipe LLM Inference](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android)
- [Google Developers Blog - EmbeddingGemma](https://developers.googleblog.com/en/introducing-embeddinggemma/)
- [Google Developers Blog - Gemma 3 270M On-Device](https://developers.googleblog.com/own-your-ai-fine-tune-gemma-3-270m-for-on-device/)

### Fine-tuning LoRA + MRL
- [Medium GDE - Fine-Tuning Gemma LoRA On-Device](https://medium.com/google-developer-experts/fine-tuning-gemma-with-lora-for-on-device-inference-android-ios-web-with-separate-lora-weights-f05d1db30d86) ⭐
- [ArXiv - Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [ArXiv - Temporal-aware MRL + LoRA](https://arxiv.org/html/2601.05549)
- [Aniket Rege - MRL from Ground Up](https://aniketrege.github.io/blog/2024/mrl/)
- [Aurelio AI - Sentence Transformers Fine-Tuning](https://www.aurelio.ai/learn/sentence-transformers-fine-tuning)

### Domain-Specific Fine-tuning
- [Voyage AI - Legal Embeddings](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/) (+38% nDCG)
- [SugiV - EmbeddingGemma Mortgage](https://blog.sugiv.fyi/mortgage-embeddinggemma) (+4% nDCG)
- [Redis - Fine-tune Embeddings](https://redis.io/blog/get-better-rag-by-fine-tuning-embedding-models/)

### Compression
- [HuggingFace - Matryoshka](https://huggingface.co/blog/matryoshka)
- [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization)
- [Qdrant - Binary Quantization](https://qdrant.tech/articles/binary-quantization/)
- [Model2Vec GitHub](https://github.com/MinishLab/model2vec)
- [HuggingFace - Static Embeddings](https://huggingface.co/blog/static-embeddings)

### Amélioration Recall
- [Zilliz - HyDE](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)
- [ArXiv - RAG-Fusion](https://arxiv.org/abs/2402.03367)
- [ArXiv - RQ-RAG](https://arxiv.org/html/2404.00610v1)
- [EMNLP 2024 - Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845.pdf)

### Chunking Avancé (v1.2)
- [Anthropic - Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) ⭐ **-49% failures**
- [Jina AI - Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) (+10-12%)
- [ArXiv - Late Chunking](https://arxiv.org/abs/2409.04701)
- [LangChain - Semantic Chunking](https://python.langchain.com/docs/how_to/semantic-chunker/) (+9%)
- [Greg Kamradt - Chunking Strategies](https://www.youtube.com/watch?v=8OJC21T2SL4)

### Benchmarks
- [AIMultiple - Embedding Models](https://research.aimultiple.com/open-source-embedding-models/)
- [HuggingFace - EmbeddingGemma](https://huggingface.co/blog/embeddinggemma)
- [RAG with EmbeddingGemma - Gemma Cookbook](https://deepwiki.com/google-gemini/gemma-cookbook/4.4.1-rag-with-embeddinggemma)

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-20 | Document initial - recherche HuggingFace/Kaggle |
| 1.1 | 2026-01-21 | Section MRL+LoRA fine-tuning, sources Google AI Dev, plan implémentation |
| 1.2 | 2026-01-22 | **+3 techniques**: Contextual Retrieval (Anthropic -49%), Late Chunking (Jina +10%), Semantic Chunking (+9%). Évaluation applicabilité Mode A/B |
| 1.3 | 2026-01-23 | **Benchmark Chunking**: Dual-size 81.72% (-5.22%), Semantic 82.89% (-4.05%) vs Baseline 86.94%. RÉGRESSION. Recherche web 2025-2026: Query Expansion, SPLADE, Context Engineering (RAGFlow). |

---

## 5. Benchmark Résultats (2026-01-23)

### 5.1 Chunking Optimizations Testées

| Mode | Configuration | Chunks FR | Recall@5 | Delta vs Baseline |
|------|---------------|-----------|----------|-------------------|
| **Baseline** | 450t single-size | ~2500 | **86.94%** | - |
| Mode A | Dual-size 256+450t | 6161 | 81.72% | **-5.22%** ❌ |
| Mode B | SemanticChunker (percentile 90) | 2558 | 82.89% | **-4.05%** ❌ |

### 5.2 Analyse

**CONCLUSION**: Les optimisations chunking ont **RÉGRESSÉ** le recall.

| Cause | Mode A | Mode B |
|-------|--------|--------|
| Dilution | 2.5x chunks → bruit dans top-5 | - |
| Boundaries | - | Frontières sémantiques ≠ formulation queries |
| Recommandation | ❌ Revert | ❌ Revert |

### 5.3 Pistes Alternatives (Web Search 2026-01-23)

| Technique | Source | Gain Attendu | Effort | Applicabilité |
|-----------|--------|--------------|--------|---------------|
| **Query Expansion** | [arXiv 2501.07391](https://arxiv.org/abs/2501.07391) | +5-10% | 4h | ✅ Offline |
| **SPLADE term expansion** | [Neo4j RAG](https://neo4j.com/blog/genai/advanced-rag-techniques/) | +3-8% | 8h | ✅ Offline |
| **Contextual Retrieval** | [Anthropic](https://www.anthropic.com/news/contextual-retrieval) | -35% failures | 4h + $4 | ✅ Index-time |
| **Context Engineering** | [RAGFlow 2025](https://ragflow.io/blog/rag-review-2025-from-rag-to-context) | Architecture | 2j | ⚠️ Complexe |
| **Fine-tuning MRL+LoRA** | Voir §2.4 | +5-15% | 1j GPU | ✅ Priorité P0 |

### 5.4 Recommandation Mise à Jour

1. **Conserver baseline** (450t single-size, 86.94% recall)
2. **Priorité P0**: Fine-tuning MRL+LoRA (gold standard disponible)
3. **Priorité P1**: Contextual Retrieval ($4 one-time)
4. **Priorité P2**: Query expansion (synonymes + reformulation)

---

*Document ISO 25010/42001 - Pocket Arbiter Project*
