# Strategie de Chunking - Pocket Arbiter

> **Document ID**: DOC-CHUNK-001
> **ISO Reference**: ISO/IEC 25010:2023, ISO/IEC 82045-1
> **Version**: 1.1
> **Date**: 2026-01-16
> **Statut**: Approuve
> **Classification**: Technique

---

## 1. Contexte et problematique

### 1.1 Probleme identifie

Le recall initial du systeme RAG etait de **34.67%** (cible ISO: 80%).

**Root cause analyse** :
- Chunking fixed-size (256 tokens) fragmente les Articles reglementaires
- Exemple: Article 4 "toucher-jouer" decoupe en 7 chunks incoherents
- Le retrieval trouve le mauvais chunk de la bonne page

### 1.2 Impact metier

| Symptome | Cause | Impact |
|----------|-------|--------|
| Chunk "j'adoube" retourne | Fragmentation Article 4.2 | Reponse incomplete |
| Mauvaise page retournee | Contexte semantique perdu | Hallucination potentielle |
| Recall 34% vs 80% cible | Chunking inadequat | Non-conformite ISO 25010 |

---

## 2. Recherche - State of the Art 2025

### 2.1 Sources consultees

| Source | URL | Date |
|--------|-----|------|
| Firecrawl | https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025 | 2025 |
| RAG About It | https://ragaboutit.com/the-chunking-strategy-shift-why-semantic-boundaries-cut-your-rag-errors-by-60/ | 2025 |
| Databricks | https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089 | 2025 |
| Weaviate | https://weaviate.io/blog/chunking-strategies-for-rag | 2025 |
| LangCopilot | https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide | 2025 |
| Stack Overflow | https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/ | 2024 |

### 2.2 Findings cles

#### Finding 1: Chunking = 60% de l'accuracy RAG

> "Chunking strategy determines roughly **60% of your RAG system's accuracy**.
> Not the embedding model. Not the reranker. Not even the language model."
> — Firecrawl, 2025

**Implication**: Priorite absolue sur la strategie de chunking avant toute autre optimisation.

#### Finding 2: Fixed-size chunking INADEQUAT pour documents reglementaires

> "Here's what happens when you use fixed-size chunking on a regulatory document:
> A financial services firm found their system confidently citing compliance rules
> that were actually **three chunks stitched together incorrectly**."
> — RAG About It, 2025

**Implication**: Le chunking actuel (256 tokens fixed) est inapproprie pour les reglements FFE/FIDE.

#### Finding 3: Semantic chunking = +35-60% amelioration

> "Teams switching from fixed-size to semantic chunking experience:
> - Irrelevant context drops by **35%**
> - Retrieval precision climbs by **20-40%**
> - Hallucination rates decline proportionally"
> — RAG About It, 2025

> "LongRAG architectures report **35% reduction in context loss** on legal and structured documents."
> — RAG About It, 2025

**Implication**: Migration vers semantic chunking obligatoire.

#### Finding 4: Approche recommandee pour documents structures

> "For structure-rich documents (legal, financial, technical):
> 1. Extract document structure: headers, sections, subsections, tables
> 2. Identify semantic units at the lowest structure level (subsections, not pages)
> 3. Set minimum chunk size (200 tokens) and maximum (1000 tokens)
> 4. If a semantic unit exceeds max size, recursively split on sub-boundaries"
> — Databricks, 2025

**Implication**: Chunker par Article/Section, pas par nombre de tokens fixe.

#### Finding 5: Parametres optimaux

> "Optimal configuration: **256-512 tokens** with **10-20% overlap**"
> — LangCopilot, 2025

> "A good baseline is **512 tokens** chunk size with **50-100 tokens overlap**"
> — Weaviate, 2025

**Implication**: Augmenter chunk_size de 256 a 512 tokens.

#### Finding 6: Hybrid search ameliore recall

> "BM25 + vector search combination improves recall by **15-25%** on keyword-heavy queries"
> — Multiple sources, 2025

**Implication**: Implementer recherche hybride BM25 + cosine similarity.

#### Finding 7: Reranking cross-encoder

> "Cross-encoder reranking improves precision@5 by **10-15%** after initial retrieval"
> — Multiple sources, 2025

**Implication**: Ajouter etape de reranking post-retrieval.

---

## 3. Strategie adoptee

### 3.1 Architecture chunking

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE CHUNKING v2.0                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PDF ──► Extract ──► Detect Structure ──► Semantic Chunk ──► DB │
│                            │                    │                │
│                            ▼                    ▼                │
│                    ┌───────────────┐    ┌─────────────┐         │
│                    │ Article 4.1   │    │ Chunk 512t  │         │
│                    │ Article 4.2   │    │ + overlap   │         │
│                    │ Article 4.3   │    │ + metadata  │         │
│                    └───────────────┘    └─────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Parametres chunking v2.0 (defaut code)

| Parametre | v1.0 (ancien) | v2.0 (defaut) | Justification |
|-----------|---------------|---------------|---------------|
| Strategie | Fixed-size | **Semantic (Article)** | Finding 2, 3 |
| chunk_size | 256 tokens | **512 tokens** | Finding 5 |
| overlap | 50 tokens (19%) | **128 tokens (25%)** | Finding 5, contexte preservé |
| min_chunk | 50 chars | **100 tokens** | Eviter fragments |
| max_chunk | 256 tokens | **1024 tokens** | Articles longs |
| split_on | Sentence | **Article > Section > Sentence** | Finding 4 |

### 3.2.1 Parametres corpus actuel (chunks_fr_v3.json)

> **Note**: Version optimisee suite a recherche (RECALL_OPTIMIZATION_PLAN.md).

| Parametre | v2.2 (ancien) | **v3 (actuel)** | Justification |
|-----------|---------------|-----------------|---------------|
| strategy | `semantic_article` | `semantic_article` | Inchange |
| max_tokens | 600 | **400** | Research optimal pour factoid queries |
| overlap_tokens | 120 (20%) | **80** (20%) | Proportionnel |
| min_chunk_tokens | 200 | **100** | Garder petits articles pertinents |
| total_chunks | 2710 | **2794** | +84 chunks (granularite accrue) |

**Resultats v3**:
- Recall@5: 73.33% → **75.00%** (+1.67%)
- Pipeline: hybrid search + BGE reranking + tolerance=2

### 3.3 Detection structure reglementaire

Patterns detectes pour chunking semantique :

```python
ARTICLE_PATTERNS = [
    r"^Article\s+(\d+(?:\.\d+)*)",      # Article 4.1, Article 4.2.3
    r"^(\d+\.\d+(?:\.\d+)*)\s",         # 4.1 Le toucher-jouer
    r"^Chapitre\s+(\d+)",               # Chapitre 2
    r"^Section\s+(\d+)",                # Section 1
    r"^Annexe\s+([A-Z])",               # Annexe A
]
```

### 3.4 Retrieval hybride

```
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE v2.0                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query ──► Embed ──┬──► Vector Search (cosine) ──┐              │
│                    │                              │              │
│                    └──► BM25 Search (keywords) ──┤              │
│                                                   ▼              │
│                                          ┌───────────────┐      │
│                                          │ Fusion RRF    │      │
│                                          │ (Reciprocal   │      │
│                                          │  Rank Fusion) │      │
│                                          └───────┬───────┘      │
│                                                  ▼              │
│                                          ┌───────────────┐      │
│                                          │ Rerank        │      │
│                                          │ Cross-Encoder │      │
│                                          └───────┬───────┘      │
│                                                  ▼              │
│                                            Top-K Results        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Conformite ISO

### 4.1 ISO 25010 - Qualite produit

| Caracteristique | Exigence | Implementation |
|-----------------|----------|----------------|
| Functional Suitability | Recall >= 80% | Semantic chunking + hybrid search |
| Functional Suitability | Precision >= 70% | Reranking cross-encoder |
| Reliability | 0% hallucination | Chunks coherents, contexte preserve |

### 4.2 ISO 82045 - Document metadata

| Exigence | Implementation |
|----------|----------------|
| Tracabilite | Chunk ID = {corpus}-{doc}-{article}-{seq} |
| Metadata | source, page, article, section preserved |
| Integrite | Chunks ne coupent pas mid-article |

### 4.3 ISO 42001 - IA responsable

| Controle | Implementation |
|----------|----------------|
| A.6.2.3 Tracabilite donnees | Structure article preservee dans metadata |
| A.7.2 Verification IA | Tests recall automatises CI/CD |
| A.9.3 Gestion incidents | Metriques recall/precision monitorees |

---

## 5. Plan d'implementation

### 5.1 Phase 1: Semantic chunking (Priorite HAUTE)

1. Modifier `chunker.py` pour detecter structure Article
2. Chunker par Article/Section au lieu de fixed-size
3. Preserver hierarchie dans metadata

### 5.2 Phase 2: Parametres optimises

1. chunk_size: 256 → 512 tokens
2. max_chunk: 256 → 1024 tokens
3. overlap: 50 → 128 tokens (25%)

### 5.3 Phase 3: Hybrid search

1. Ajouter index BM25 (SQLite FTS5)
2. Implementer fusion RRF
3. Ponderation: 0.7 vector + 0.3 BM25

### 5.4 Phase 4: Reranking

1. Integrer cross-encoder (ms-marco-MiniLM)
2. Rerank top-20 → top-5
3. Seuil confiance minimum

---

## 6. Strategies de chunking disponibles

### 6.1 Comparaison des 4 strategies implementees

| Script | Dependance | Methode | Usage corpus |
|--------|------------|---------|--------------|
| `chunker.py` | tiktoken (natif) | Detection Article + fallback sentence | **chunks_fr_v2.x.json** (actif) |
| `sentence_chunker.py` | llama-index-core | LlamaIndex SentenceSplitter | chunks_sentence_fr.json |
| `semantic_chunker.py` | langchain-experimental | LangChain SemanticChunker | chunks_semantic_fr.json |
| `similarity_chunker.py` | sentence-transformers | Cosine similarity breaks | chunks_similarity_fr.json |

### 6.2 Recommandations Google vs Implementation

| Aspect | Google AI Edge | Implementation actuelle |
|--------|----------------|------------------------|
| Chunking | Non specifie (SDK = retrieval only) | Custom Article-based |
| Embeddings | EmbeddingGemma TFLite | sentence-transformers (Python) |
| Vector Store | SqliteVectorStore | Compatible (export_sdk.py) |

> **Note**: Google AI Edge RAG SDK ne fournit pas de chunking - uniquement retrieval/inference.
> Le chunking est donc 100% custom, inspire des best practices (Firecrawl, LangCopilot 2025).

---

## 7. Metriques cibles

| Metrique | Avant | Actuel | Cible | Methode mesure |
|----------|-------|--------|-------|----------------|
| Recall@5 FR | 34.67% | **75.00%** (XFAIL) | **>= 80%** | Gold standard 30 questions |
| Recall@5 INTL | TBD | TBD | **>= 70%** | Gold standard questions |
| Precision@5 | TBD | TBD | **>= 70%** | Evaluation manuelle |
| Hallucination | TBD | TBD | **0%** | Tests adversaires |
| Latence | TBD | TBD | **< 500ms** | Benchmark |

> **Statut Recall**: 75% avec pipeline complet (hybrid + reranking + 400 tokens).
> Gap -5% vers cible 80%. Prochaine etape: query expansion ou multilingual embedding.

---

## 8. Risques et mitigation

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| Chunks trop gros depassent context LLM | Moyenne | Eleve | max_chunk = 1024 tokens |
| BM25 moins performant sur FR | Faible | Moyen | Stemmer francais + stopwords |
| Reranking trop lent mobile | Moyenne | Moyen | Modele quantifie, top-20 only |
| Structure Article non detectee | Faible | Eleve | Fallback fixed-size |

---

## 9. References

### 9.1 Documentation projet
- `docs/AI_POLICY.md` - Politique IA (ISO 42001)
- `docs/TEST_PLAN.md` - Plan de tests (ISO 29119)
- `docs/QUALITY_REQUIREMENTS.md` - Exigences qualite (ISO 25010)

### 9.2 Sources externes
- [Best Chunking Strategies for RAG in 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Why Semantic Boundaries Cut RAG Errors by 60%](https://ragaboutit.com/the-chunking-strategy-shift-why-semantic-boundaries-cut-your-rag-errors-by-60/)
- [Ultimate Guide to Chunking Strategies - Databricks](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [Chunking Strategies - Weaviate](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Document Chunking for RAG - LangCopilot](https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide)

---

## 9. Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-15 | Claude Code | Creation initiale |
| 1.1 | 2026-01-16 | Claude Opus 4.5 | Mise a jour v3 (400 tokens, recall 75%) |
