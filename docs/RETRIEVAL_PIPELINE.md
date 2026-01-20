# Pipeline de Retrieval - Pocket Arbiter

> **Document ID**: DOC-RETR-001
> **ISO Reference**: ISO/IEC 25010 - Performance efficiency
> **Version**: 2.2
> **Date**: 2026-01-20
> **Statut**: Approuve

---

## Vue d'ensemble (v4.0)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INDEXATION (offline) - Pipeline v4.0                                       │
│  ─────────────────────                                                      │
│                                                                             │
│  PDFs FFE ──→ Extraction ──→ Chunking ──→ Embedding ──→ Storage            │
│    29 docs    Docling ML    Parent-Child  768D vectors  SQLite + FTS5      │
│                             1454 FR chunks                                  │
│                             764 INTL chunks                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL (runtime) - Recall 97.06%                                        │
│  ────────────────────                                                       │
│                                                                             │
│  User Query ──→ Embedding ──→ Vector Search ──→ source_filter ──→ Top-5    │
│                 768D query    Cosine sim      (optionnel)        chunks     │
│                               ↓                                             │
│                         glossary_boost                                      │
│                         (x3.5 definitions)                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  GENERATION (Phase 3 - A IMPLEMENTER)                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  Top-5 chunks + Query ──→ LLM (Gemma 3) ──→ Reponse avec citations         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Composants du pipeline

### 1. Extraction PDF (Docling ML)

**Fichier:** `scripts/pipeline/extract_docling.py`

```python
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert(pdf_path)
```

**Avantages vs PyMuPDF:**
- Extraction ML (meilleure qualite)
- Structure preservee (tableaux, listes)
- Multi-format (PDF, DOCX, images)

### 2. Chunking Parent-Child (NVIDIA 2025)

**Fichier:** `scripts/pipeline/parent_child_chunker.py`

| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Parent size | 1024 tokens | Contexte large pour retrieval |
| Child size | 450 tokens | Unite semantique coherente |
| Overlap | 15% (~68 tokens) | NVIDIA research optimal |

**Strategie:**
- RecursiveCharacterTextSplitter (LangChain)
- Preserve structure: `\n\n` > `\n` > `. ` > ` `
- Metadata heritage parent-child

### 3. Table Summaries (ISO 42001)

**Fichier:** `scripts/pipeline/table_multivector.py`

- Claude Code genere summary par table (~40-50 tokens)
- Embedding du summary (plus semantique que table brute)
- 111 tables FR traitees (source: table_summaries_claude.json)

### 4. Embedding

**Fichier:** `scripts/pipeline/embeddings.py`

| Parametre | Valeur |
|-----------|--------|
| Modele | `google/embeddinggemma-300m-qat` |
| Dimension | 768 |
| Normalisation | L2 (cosine ready) |

### 5. Recherche Vectorielle (Optimal)

**Fichier:** `scripts/pipeline/export_search.py`

```python
def retrieve_similar(
    db_path: Path,
    query_embedding: np.ndarray,
    top_k: int = 5,
    source_filter: str | None = None,  # NEW: filtrage document
    glossary_boost: float | None = None,  # NEW: boost definitions
) -> list[dict]:
```

**Benchmark (2026-01-19):**

| Mode | Recall FR | Status |
|------|-----------|--------|
| Vector-only | **97.06%** | OPTIMAL |
| + source_filter | **100%** | Edge cases |
| Hybrid (BM25+Vector) | 89.46% | Regression |

> **Note:** Hybrid search (BM25 60% + Vector 40%) teste mais **moins performant** que vector-only sur ce gold standard. Garde comme option pour autres use cases.

### 6. Source Filter (Cross-document fix)

```python
# Filtrage par document source
results = retrieve_similar(db, emb, source_filter="Statuts")
results = retrieve_similar(db, emb, source_filter="LA-octobre")
```

**Patterns auto-detection (`smart_retrieve`):**

| Keyword | Source Filter |
|---------|---------------|
| "objectifs", "fédération" | Statuts |
| "cadence", "rapide", "blitz" | LA-octobre |
| "éthique", "déontologie" | Code_ethique |

### 7. Glossary Boost (DNA 2025)

```python
# Boost x3.5 pour questions definition (avec fallback intelligent)
results = retrieve_with_glossary_boost(db, emb, "Qu'est-ce que le roque?")
```

**Detection automatique patterns:**
- "qu'est-ce que", "c'est quoi", "définition de"
- "que signifie", "que veut dire"
- Glossaire pages 67-70 LA-octobre + table summaries

**Features avancees (v2.1):**
- **Fallback intelligent**: Si boost actif mais 0 chunk glossaire → retry sans boost
- **Logging JSONL**: `logs/retrieval.jsonl` pour analytics (1 JSON par ligne)
- **Module dedie**: `scripts/pipeline/retrieval_logger.py`

### 8. Hybrid Search (Disponible, non optimal)

**Fichier:** `scripts/pipeline/export_search.py`

```python
# Poids RRF (Reciprocal Rank Fusion)
DEFAULT_VECTOR_WEIGHT = 0.4
DEFAULT_BM25_WEIGHT = 0.6
RRF_K = 60

def retrieve_hybrid(db_path, query_embedding, query_text, top_k=5):
    # Combine vector + BM25 avec RRF
```

**FTS5 Configuration:**
```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    tokenize='unicode61 remove_diacritics 2'
);
```

> **Status:** Implemente mais vector-only plus performant (97.06% vs 89.46%). Hybrid utile pour queries avec numeros d'article specifiques.

### 9. Query Expansion

**Fichier:** `scripts/pipeline/query_expansion.py`

- Dictionnaire synonymes chess FR (~50 termes)
- Snowball French stemmer (optionnel)
- Stopwords FR

### 10. Reranking (Optionnel)

**Fichier:** `scripts/pipeline/reranker.py`

- Modele: `BAAI/bge-reranker-v2-m3`
- Usage: Pool large (100) -> rerank -> top-k final
- Status: Implemente, CPU lent (~2h pour benchmark)

---

## Mesure du Recall

### Definition

```
Recall@k = Pages attendues trouvees dans top-k / Total pages attendues
Tolerance: ±2 pages adjacentes acceptees
```

### Gold Standard v5.22

**Fichiers:**
- `tests/data/gold_standard_fr.json` (134 questions, 45 hard cases)
- `tests/data/gold_standard_intl.json` (25 questions)

| Corpus | Questions | Hard Cases | Recall@5 | Audit |
|--------|-----------|------------|----------|-------|
| FR | 134 | 45 | 91.17% | v5.22 (2026-01-20) |
| INTL | 25 | - | - | - |
| **Total** | **159** | **45** | ISO 25010 PASS |

**Analyse echecs (14 questions)**: `docs/research/RECALL_FAILURE_ANALYSIS_2026-01-20.md`
**Optimisations zero-runtime**: `docs/research/OFFLINE_OPTIMIZATIONS_2026-01-20.md`

### Resultats (2026-01-19)

| Mode | Recall FR | Recall INTL | Status |
|------|-----------|-------------|--------|
| Vector-only | **97.06%** | 80.00% | PASS (>= 90%) |
| + source_filter | **100%** | - | Edge cases resolus |
| Hybrid | 89.46% | - | Regression |

**Objectif ISO 25010:** Recall >= 90% - **ATTEINT**

---

## Fichiers cles (v4.0)

```
scripts/pipeline/
├── extract_docling.py        # Extraction PDF (Docling ML)
├── parent_child_chunker.py   # Chunking Parent 1024/Child 450
├── table_multivector.py      # Tables + LLM summaries
├── embeddings.py             # EmbeddingGemma 768D
├── export_sdk.py             # Creation SQLite DB
├── export_search.py          # Vector search + source_filter + glossary_boost
├── retrieval_logger.py       # Logging JSONL structuré (analytics)
├── query_expansion.py        # Synonymes + stemmer FR
├── reranker.py               # Cross-encoder (optionnel)
└── tests/
    ├── test_recall.py        # Benchmark recall
    └── test_export_search.py # Tests recherche
```

---

## Statistiques corpus

| Corpus | Chunks | Child | Tables | DB Size |
|--------|--------|-------|--------|---------|
| FR | 1454 | 1343 | 111 | 7.58 MB |
| INTL | 764 | 764 | 0 | 4.21 MB |
| **Total** | **2218** | 2107 | 111 | **11.79 MB** |

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-15 | Creation initiale (PyMuPDF, 400 tokens) |
| 2.0 | 2026-01-19 | **Rewrite complet**: Docling, Parent-Child, vector-only optimal, source_filter, glossary_boost |
| 2.1 | 2026-01-19 | **Fallback + logging**: Fallback intelligent, logging structure `logs/retrieval_log.txt` |
| 2.2 | 2026-01-20 | **Research docs**: Lien analyse echecs + optimisations zero-runtime, gold standard v5.22 |

---

## References

- [ISO 25010](https://www.iso.org/standard/35733.html) - Performance efficiency
- [ISO 42001](https://www.iso.org/standard/81230.html) - AI traceability
- [Docling](https://github.com/DS4SD/docling) - Document extraction ML
- [NVIDIA RAG 2025](https://developer.nvidia.com/blog/rag-101/) - Parent-Child chunking
- [SQLite FTS5](https://sqlite.org/fts5.html) - Full-text search
