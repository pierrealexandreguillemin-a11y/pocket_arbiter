# Solutions Vector-Based - Optimisation RAG

> **Pocket Arbiter** - Recherche HuggingFace/Kaggle
> Date: 2026-01-20 | ISO 25010, ISO 42001

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

| Solution | Gain Recall | RAM Runtime | Complexité | Latence |
|----------|-------------|-------------|------------|---------|
| **HyDE** | +10-45% | API LLM | Moyen | +500ms |
| **RAG-Fusion** | +5-15% | API LLM | Moyen | +300ms |
| **RQ-RAG** | +1.9% SOTA | 7B model | Élevé | +1s |
| **Fine-tuning** | +16-38% | Aucune | Élevé | 0 |
| **Proposition-level** | +15-25% | Aucune | Moyen | 0 |

### 2.2 HyDE (Hypothetical Document Embeddings)

**Sources**:
- [Zilliz - HyDE](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)
- [Haystack Documentation](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)

**Principe**: LLM génère un "faux document" répondant à la query, puis embed ce document pour retrieval.

**Implémentation**:
```python
def hyde_retrieve(query: str, llm, embed_model, db, top_k=5):
    """HyDE: Generate hypothetical doc, embed it, retrieve."""
    # 1. Generate hypothetical answer
    prompt = f"Écris un paragraphe répondant à: {query}"
    hypothetical_doc = llm.generate(prompt)

    # 2. Embed the hypothetical document (not the query!)
    hyde_embedding = embed_model.encode(hypothetical_doc)

    # 3. Retrieve using hypothetical embedding
    return db.search(hyde_embedding, top_k=top_k)
```

**Benchmarks**:
| Dataset | BM25 | Contriever | HyDE | Gain |
|---------|------|------------|------|------|
| NQ | 32.0 | 41.3 | 49.8 | +20% |
| TriviaQA | 67.7 | 68.2 | 73.4 | +8% |
| MSMARCO | 22.8 | 31.6 | 36.2 | +15% |

**Applicabilité Pocket Arbiter**: ⚠️ Nécessite API LLM (mode connecté)

---

### 2.3 RAG-Fusion

**Sources**:
- [ArXiv - RAG-Fusion](https://arxiv.org/abs/2402.03367)
- [GitHub - rag-fusion](https://github.com/Raudaschl/rag-fusion)

**Principe**: LLM génère N variations de la query, retrieve pour chaque, fusionne via RRF.

**Implémentation**:
```python
def rag_fusion(query: str, llm, embed_model, db, n_variations=3, top_k=5):
    """RAG-Fusion: Multiple query variations + RRF."""
    # 1. Generate query variations
    prompt = f"Génère {n_variations} reformulations de: {query}"
    variations = [query] + llm.generate(prompt).split("\n")

    # 2. Retrieve for each variation
    all_results = []
    for var in variations:
        emb = embed_model.encode(var)
        results = db.search(emb, top_k=top_k * 2)
        all_results.append(results)

    # 3. Reciprocal Rank Fusion
    return reciprocal_rank_fusion(all_results, k=60)[:top_k]
```

**Applicabilité Pocket Arbiter**: ⚠️ Nécessite API LLM (mode connecté)

---

### 2.4 RQ-RAG (Refine Query)

**Source**: [ArXiv - RQ-RAG](https://arxiv.org/html/2404.00610v1)

**Principe**: Modèle 7B fine-tuné pour dynamiquement:
- Rewriter: reformuler queries ambiguës
- Decomposer: diviser queries multi-hop
- Disambiguator: clarifier termes ambigus

**Benchmarks**: +1.9% vs SOTA sur single-hop QA

**Applicabilité Pocket Arbiter**: ❌ Trop lourd (7B model runtime)

---

### 2.5 Fine-tuning Domain-Specific

**Sources**:
- [Voyage AI - Legal Embeddings](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/)
- [Redis - Fine-tune Embeddings](https://redis.io/blog/get-better-rag-by-fine-tuning-embedding-models/)

**Principe**: Fine-tuner le modèle d'embedding sur des paires (query, document) du domaine.

**Implémentation**:
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Préparer données d'entraînement
train_examples = [
    InputExample(texts=["Règle du pat?", "Le pat est une position..."]),
    InputExample(texts=["Forfait retard?", "Un joueur est déclaré forfait..."]),
]

# 2. Fine-tuner
model = SentenceTransformer("google/embeddinggemma-300m-qat-q4_0-unquantized")
train_dataloader = DataLoader(train_examples, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)
```

**Benchmarks** (legal domain):
| Métrique | Before | After | Gain |
|----------|--------|-------|------|
| nDCG@10 | 0.5949 | 0.8245 | +38% |
| Recall@10 | 0.72 | 0.88 | +22% |

**Applicabilité Pocket Arbiter**: ✅ Possible (nécessite dataset QA annoté)

---

### 2.6 Proposition-Level Indexing

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

| Priorité | Solution | Effort | Gain Attendu |
|----------|----------|--------|--------------|
| **P0** | Matryoshka 768→256D | 1h | 3x storage, -1% recall |
| **P1** | Fine-tuning sur gold standard | 1j | +10-20% recall |
| **P2** | Binary quantization | 2h | 32x compression |

### 3.2 Moyen Terme (Mode connecté, API LLM)

| Priorité | Solution | Effort | Gain Attendu |
|----------|----------|--------|--------------|
| **P1** | HyDE via Claude API | 4h | +10-15% recall hard cases |
| **P2** | RAG-Fusion | 4h | +5-10% recall |

### 3.3 Long Terme (Refonte)

| Priorité | Solution | Effort | Gain Attendu |
|----------|----------|--------|--------------|
| **P3** | Proposition-level indexing | 2j | +15-25% recall |
| **P3** | Model2Vec distillation | 1j | 500x speedup, -13% recall |

---

## 4. Sources

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
- [Voyage AI - Legal Embeddings](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/)

### Benchmarks
- [AIMultiple - Embedding Models](https://research.aimultiple.com/open-source-embedding-models/)
- [EmbeddingGemma](https://huggingface.co/blog/embeddinggemma)

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-20 | Document initial - recherche HuggingFace/Kaggle |

---

*Document ISO 25010/42001 - Pocket Arbiter Project*
