# Pipeline de Retrieval - Pocket Arbiter

> Documentation technique du pipeline RAG et mesure du recall
> ISO 25010 - Performance efficiency

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INDEXATION (offline)                                                       │
│  ─────────────────────                                                      │
│                                                                             │
│  PDFs FFE ──→ Extraction ──→ Chunking ──→ Embedding ──→ Storage            │
│    29 docs    PyMuPDF       400 tokens   EmbeddingGemma  SQLite            │
│                             2794 chunks   768D vectors   + FTS5            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL (runtime) ← RECALL MESURE ICI                                    │
│  ────────────────────                                                       │
│                                                                             │
│  User Query ──→ Embedding ──→ Hybrid Search ──→ Reranker ──→ Top-5         │
│                 768D query    70% BM25 +       BGE-v2-m3    chunks          │
│                               30% vector                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  GENERATION (Phase 3 - A IMPLEMENTER)                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  Top-5 chunks + Query ──→ LLM (Gemma 3) ──→ Reponse avec citations         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Composants du pipeline

### 1. Extraction PDF (PyMuPDF)

**Fichier:** `scripts/pipeline/pdf_extractor.py`

```python
# Extraction texte brut par page
for page in doc:
    text = page.get_text()
```

**Limites connues:**
- Tableaux mal extraits (perte structure)
- En-tetes/pieds de page inclus
- Colonnes fusionnees incorrectement
- OCR non supporte (PDFs images)

### 2. Chunking (400 tokens)

**Fichier:** `scripts/pipeline/sentence_chunker.py`

- Strategie: SentenceSplitter (LlamaIndex)
- Taille: 400 tokens (optimise recall +1.67%)
- Overlap: 50 tokens

### 3. Embedding (EmbeddingGemma)

**Fichier:** `scripts/pipeline/embeddings.py`

- Modele: `google/embeddinggemma-300m-qat-q4_0-unquantized`
- Dimension: 768
- Prompts: "Retrieval-query" / "Retrieval-document"

### 4. Recherche Hybride (BM25 + Vector)

**Fichier:** `scripts/pipeline/export_search.py`

```python
# Poids RRF
DEFAULT_VECTOR_WEIGHT = 0.3  # Embedding generique
DEFAULT_BM25_WEIGHT = 0.7    # Plus efficace pour FR normatif
RRF_K = 60
```

**FTS5 Configuration:**
```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    tokenize='unicode61 remove_diacritics 2'  -- Accents FR
);
```

### 5. Query Expansion

**Fichier:** `scripts/pipeline/query_expansion.py`

- Dictionnaire synonymes chess FR
- Snowball French stemmer (optionnel)
- Stopwords FR

### 6. Reranking (Cross-Encoder)

**Fichier:** `scripts/pipeline/reranker.py`

- Modele: `BAAI/bge-reranker-v2-m3` (multilingual)
- Fallback: `BAAI/bge-reranker-base`

## Mesure du Recall

### Definition

```
Recall@k = Pages attendues trouvees dans top-k / Total pages attendues
```

### Gold Standard

**Fichier:** `tests/data/questions_fr.json`

- 30 questions manuelles
- Pages attendues identifiees par lecture humaine
- Independant du systeme de retrieval (ISO 42001)

### Benchmark

**Fichier:** `scripts/pipeline/tests/test_recall.py`

```python
result = benchmark_recall(
    db_path,
    questions_file,
    model,
    top_k=5,
    use_hybrid=True,
    tolerance=2,      # Pages adjacentes acceptees
    reranker=reranker,
    top_k_retrieve=30,
)
```

### Resultats actuels

| Version | Recall@5 | Questions echouees |
|---------|----------|-------------------|
| Vector-only | 48.89% | - |
| + tolerance=2 | 70.00% | - |
| + hybrid | 73.33% | - |
| + 400-token chunks | 75.00% | - |
| + gold standard v4.2 | 78.33% | FR-Q04, Q18, Q22, Q25 |

**Objectif ISO 25010:** Recall >= 80%

## Problemes connus et solutions

### Extraction PDF

| Probleme | Impact | Solution |
|----------|--------|----------|
| Tableaux mal extraits | Perte info structuree | Camelot/Tabula |
| En-tetes repetitifs | Bruit dans chunks | Post-processing |
| OCR absent | PDFs images ignores | pytesseract |

### Retrieval

| Probleme | Impact | Solution |
|----------|--------|----------|
| Embedding generique | -15-20% recall | Fine-tuning domaine |
| Pas de stemming FR | -10-15% recall | Snowball (ajoute) |
| Synonymes manquants | -5% recall | Dictionnaire (ajoute) |

## Fine-tuning (Phase 2)

Pipeline de fine-tuning EmbeddingGemma pour le domaine echecs FR:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Generation donnees synthetiques                                  │
│  Chunks corpus ──→ LLM (Claude/GPT) ──→ Questions synthetiques             │
│  2794 chunks       2-3 questions/chunk   ~6000 paires (query, chunk)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Hard Negative Mining                                              │
│  Pour chaque (query, positive): rechercher chunks similaires non-pertinents │
│  → Triplets (anchor, positive, negative)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: Fine-tuning                                                       │
│  Model: EmbeddingGemma-300M   Loss: MultipleNegativesRankingLoss           │
│  Data: ~6000 triplets         Time: ~30 min (CPU)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: Evaluation                                                        │
│  Benchmark recall sur gold standard → Objectif: >= 80%                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Fichiers:**
- `scripts/training/generate_synthetic_data.py`
- `scripts/training/hard_negative_mining.py`
- `scripts/training/finetune_embeddinggemma.py`
- `scripts/training/evaluate_finetuned.py`

## Fichiers cles

```
scripts/
├── pipeline/
│   ├── pdf_extractor.py      # Extraction PDF
│   ├── sentence_chunker.py   # Chunking
│   ├── embeddings.py         # EmbeddingGemma
│   ├── export_sdk.py         # Creation DB
│   ├── export_search.py      # Recherche hybride
│   ├── query_expansion.py    # Synonymes + stemmer
│   ├── reranker.py           # Cross-encoder
│   └── tests/
│       └── test_recall.py    # Benchmark recall
└── training/
    ├── generate_synthetic_data.py  # Generation questions LLM
    ├── hard_negative_mining.py     # Mining negatifs difficiles
    ├── finetune_embeddinggemma.py  # Fine-tuning
    ├── evaluate_finetuned.py       # Evaluation recall
    └── tests/                      # Tests unitaires
```

## References

- [ISO 25010](https://www.iso.org/standard/35733.html) - Performance efficiency
- [ISO 42001](https://www.iso.org/standard/81230.html) - AI traceability
- [SQLite FTS5](https://sqlite.org/fts5.html) - Full-text search
- [Snowball Stemmer](https://snowballstem.org/algorithms/french/stemmer.html) - French stemmer
