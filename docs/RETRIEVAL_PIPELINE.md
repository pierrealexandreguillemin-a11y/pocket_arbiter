# Pipeline de Retrieval - Pocket Arbiter

> **Document ID**: DOC-RETR-001
> **ISO Reference**: ISO/IEC 25010 - Performance efficiency
> **Version**: 5.0
> **Date**: 2026-01-24
> **Statut**: DUAL-RAG Architecture (FR actif, INTL obsolete)
> **Scope**: **RAG FRANCE UNIQUEMENT** (voir VISION.md v2.0)

---

## 0. Avertissement Dual-RAG (VISION v2.0)

> **ARCHITECTURE DUAL-RAG: DEUX PIPELINES SEPARES**
> Cause: Pollution mutuelle des corpus due a specificite metier et scopes differents.

| Pipeline | Corpus | Database | Status |
|----------|--------|----------|--------|
| **RAG FR** | 29 docs FFE | corpus_mode_b_**fr**.db | **ACTIF** |
| RAG INTL | FIDE (incomplet) | corpus_mode_b_intl.db | **OBSOLETE** |

**IMPORTANT**: Ne jamais mixer les databases FR et INTL dans une meme requete.

---

## Vue d'ensemble (v6.0 - RAG FR)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INDEXATION (offline) - Pipeline v6.0 (ISO 42001 + EmbeddingGemma 308M)     │
│  ─────────────────────                                                      │
│                                                                             │
│  PDFs FFE ──→ Extraction ──→ Chunking (DUAL) ──→ Embedding ──→ Storage     │
│    29 docs    Docling ML    MODE A: HybridChunker    768D vectors  SQLite   │
│               DoclingDocument  2540 FR / 1412 INTL   EmbeddingGemma + FTS5  │
│               (provenance)  MODE B: LangChain        308M params            │
│                               1331 FR / 866 INTL                            │
│                             100% page provenance (both modes)               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL (runtime) - Recall FR 91.56%, INTL 93.22%                        │
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
doc = result.document
doc_dict = doc.export_to_dict()  # Preserve provenance (ISO 42001)
```

**Avantages vs PyMuPDF:**
- Extraction ML (meilleure qualite)
- Structure preservee (tableaux, listes)
- Multi-format (PDF, DOCX, images)
- **DoclingDocument JSON** preserve page provenance (Discussion #1012)

**Page Provenance (ISO 42001 A.6.2.2):**
- `export_to_markdown()` = LOSSY (perd page_no)
- `export_to_dict()` = LOSSLESS (preserve prov[].page_no)
- Source: [Docling Discussion #1012](https://github.com/docling-project/docling/discussions/1012)

### 2. Chunking Dual-Mode (Parent-Child, ISO 42001)

**Tokenizer unifié:** `google/embeddinggemma-300m` (308M params)
- Cohérence chunking/embedding (ISO 42001 A.6.2.2)
- Remplace tiktoken cl100k_base

| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Parent size | 1024 tokens | arXiv 2025: contexte large |
| Child size | 450 tokens | Chroma 2025: sweet spot |
| Overlap | 15% (154/68) | NVIDIA 2025: optimal |
| Page provenance | 100% | prov[].page_no obligatoire |

**Mode A (HybridChunker):** `scripts/pipeline/chunker_hybrid.py`
- HybridChunker (Docling native) + RecursiveCharacterTextSplitter
- Préserve frontières document (paragraphes, sections)
- 2540 FR / 1412 INTL chunks

**Mode B (LangChain):** `scripts/pipeline/chunker_langchain.py`
- MarkdownHeaderTextSplitter + section fusion + RecursiveCharacterTextSplitter
- Fusionne petites sections (<1024 tokens) avant parent split
- 1331 FR / 866 INTL chunks (plus denses)

**Page Provenance (v6.0 - ISO 42001 A.6.2.2):**
- Mode A: `chunk.meta.doc_items[].prov[].page_no` (natif Docling)
- Mode B: Mapping depuis `docling_document.texts[].prov[].page_no`
- 100% coverage pour les deux modes (pas de fallback dégradé)

### 3. Table Summaries (ISO 42001)

**Fichier:** `scripts/pipeline/table_multivector.py`

- Claude Code genere summary par table (~40-50 tokens)
- Embedding du summary (plus semantique que table brute)
- 111 tables FR traitees (source: table_summaries_claude.json)

### 4. Embedding (ISO 42001 A.6.2.2 Conforme)

**Fichier:** `scripts/pipeline/embeddings.py`

| Parametre | Valeur | Source |
|-----------|--------|--------|
| Modele | `google/embeddinggemma-300m-qat-q4_0` | HuggingFace |
| Dimension | 768 (MRL: 128/256/512) | Google |
| Batch size | 128 | Google recommendation |
| Normalisation | L2 (cosine ready) | Standard |
| Encodage | Asymétrique (query/document) | Google official |
| Prompts | `task: search result \| query:` | Google official |
| Titles | Section injectée dans prompt | +4% relevance |

**Conformité Google/HuggingFace:**
- `encode_query()` pour requêtes utilisateur
- `encode_document()` pour chunks avec titles
- Prompts officiels: `title: {section} | text: {chunk}`

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
├── chunker.py   # Chunking Parent 1024/Child 450
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
| 2.3 | 2026-01-22 | **COMPLETE**: Recall 91.17% atteint, Phase 1B triplets synthetiques 5434 questions |
| 4.1 | 2026-01-23 | **Benchmark Chunking Optimizations**: Dual-size 81.72%, Semantic 82.89% vs Baseline 86.94% - RÉGRESSION. Recommandation: conserver baseline 450t single-size. |

---

## References

- [ISO 25010](https://www.iso.org/standard/35733.html) - Performance efficiency
- [ISO 42001](https://www.iso.org/standard/81230.html) - AI traceability
- [Docling](https://github.com/DS4SD/docling) - Document extraction ML
- [NVIDIA RAG 2025](https://developer.nvidia.com/blog/rag-101/) - Parent-Child chunking
- [SQLite FTS5](https://sqlite.org/fts5.html) - Full-text search
