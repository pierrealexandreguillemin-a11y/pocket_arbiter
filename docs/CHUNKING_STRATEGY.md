# Chunking Strategy - Pocket Arbiter

> **Document ID**: SPEC-CHUNK-001
> **ISO Reference**: ISO/IEC 25010 S4.2, ISO/IEC 42001, ISO/IEC 12207 S7.3.3
> **Version**: 3.1
> **Date**: 2026-01-18
> **Statut**: En cours
> **Classification**: Technique
> **Auteur**: Claude Opus 4.5
> **Mots-cles**: chunking, RAG, embeddings, retrieval, performance

---

## 1. Objectif

Definir la strategie de chunking optimale pour le systeme RAG Pocket Arbiter,
avec pour cible un **Recall@5 >= 90%** sur le gold standard (68 questions).

---

## 2. Baseline et Etat Actuel

### 2.1 Configuration Initiale (v1-v2)
| Parametre | Valeur | Resultat |
|-----------|--------|----------|
| Chunker | LlamaIndex SentenceSplitter | - |
| Chunk size | 512 tokens | - |
| Overlap | 128 tokens (25%) | - |
| Recall@5 | 78.33% | XFAIL |

### 2.2 Configuration Actuelle (v3 - 2026-01-18)
| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Chunker | LangChain RecursiveCharacterTextSplitter | Hierarchie semantique |
| Chunk size | 450 tokens | Sweet-spot RAG 2025-2026 |
| Overlap | 100 tokens (22%) | Best practice 20-25% |
| Min chunk | 50 tokens | Evite fragments inutiles |
| Tokenizer | cl100k_base (tiktoken) | Standard OpenAI |
| Total chunks | 1244 | Reduction depuis 2794 |

**Separateurs hierarchiques** (ordre de priorite):
```python
REGULATORY_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
```

---

## 3. Roadmap d'Optimisation

### 3.1 Etape 1: RecursiveCharacterTextSplitter + Overlap (DONE)
**Gain attendu**: +4-8% recall

**Implementation**: `scripts/pipeline/sentence_chunker.py`
- Switch SentenceSplitter -> RecursiveCharacterTextSplitter
- Chunk size: 512 -> 450 tokens
- Overlap: 128 -> 100 tokens (22%)
- Separateurs hierarchiques pour reglements

**Statut**: COMPLETE (2026-01-18)

### 3.2 Etape 2: Parent-Document Retrieval (DONE)
**Gain attendu**: +8-15% recall

**Implementation**: `scripts/pipeline/parent_child_chunker.py`
- Chunks enfants (300 tokens, 20% overlap) pour embeddings/recherche
- Chunks parents (800 tokens, 12.5% overlap) pour contexte retourne
- Mapping child -> parent dans metadata
- **Resultats**: 718 parents, 1997 children, 2.78 children/parent

```python
PARENT_CHUNK_SIZE = 800   # Rich context for LLM
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 300    # Precise semantic units
CHILD_CHUNK_OVERLAP = 60
```

**Statut**: COMPLETE (2026-01-18)

### 3.3 Etape 3: Metadata Forte (DONE)
**Gain attendu**: +3-7% recall

**Implementation**: Integre dans `parent_child_chunker.py`
| Champ | Source | Exemple |
|-------|--------|---------|
| article_num | Regex `Art\.\s*(\d+[\.\d]*)` | "Art. 9.1.2" |
| section | Headers parsing | "Chapitre 3" |
| page | Extraction JSON | 42 |
| parent_id | Mapping child->parent | "source-p1-parent0" |

**Statut**: COMPLETE (2026-01-18)

### 3.4 Etape 4: Extraction Tables (IN PROGRESS)
**Gain attendu**: +5-12% recall

**Implementation**: `scripts/pipeline/table_extractor.py`
- Camelot (lattice + stream methods)
- Detection automatique type table (cadence, penalty, elo, tiebreak)
- Conversion table -> text pour embeddings

**Types de tables detectes**:
- cadence: Tables de temps de reflexion
- penalty: Grilles de penalites
- elo: Tableaux de classement Elo
- tiebreak: Systemes de departage

**Statut**: EN COURS (2026-01-18)

---

## 4. Metriques de Validation

### 4.1 Gold Standard
- **Source**: `tests/data/questions_fr.json`
- **Questions**: 68 (ISO 29119 >= 50)
- **Documents**: 28

### 4.2 Resultats Benchmark
| Etape | Recall@5 Cible | Recall@5 Reel | Statut |
|-------|----------------|---------------|--------|
| Baseline (v2) | - | 78.33% | Reference |
| Etape 1-3 (v3) | 82-86% | **85.29%** | **ATTEINT** |
| + Tables | 88-92% | TBD | En cours |
| Cible finale | >=90% | - | Objectif |

### 4.3 Benchmark Command
```bash
python -m scripts.pipeline.benchmark \
  --model google/embeddinggemma-300m-qat-q4_0-unquantized \
  --db corpus/processed/corpus_fr_v3.db \
  --questions tests/data/questions_fr.json \
  --top-k 5
```

---

## 5. Conformite ISO

### 5.1 ISO/IEC 25010 - Performance Efficiency
- **Metric**: Recall@5 >= 80% (cible 90%)
- **Validation**: Benchmark automatise sur gold standard
- **Tracabilite**: Resultats loggues avec configuration

### 5.2 ISO/IEC 42001 - AI Traceability
- **Chunking auditable**: Configuration versionnee
- **Metadata complete**: Source, page, article tracables
- **Reproductibilite**: Meme input -> meme output

### 5.3 ISO/IEC 12207 S7.3.3 - Implementation
- **Documentation**: Ce document + docstrings
- **Tests**: `pytest scripts/pipeline/test_sentence_chunker.py`
- **CI/CD**: Validation pre-commit

---

## 6. References Industrie

### 6.1 Sources consultees (2025-2026)
- [NVIDIA: Finding the Best Chunking Strategy](https://developer.nvidia.com/blog/finding-best-chunking-strategy) - 15% overlap optimal
- [LangChain: Parent Document Retriever](https://python.langchain.com/docs/how_to/parent_document_retriever/) - Pattern reference
- [Weaviate: Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag) - Best practices
- [Databricks: Ultimate Guide to Chunking](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [Firecrawl: Best Chunking Strategies 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)

### 6.2 Best Practices Appliquees
| Recommandation | Valeur industrie | Notre config |
|----------------|------------------|--------------|
| Chunk size | 256-512 tokens | 450 (recursive), 300 (child) |
| Overlap | 10-20% | 22% (recursive), 20% (child) |
| Parent size | 500-2000 tokens | 800 tokens |
| Child size | 100-500 tokens | 300 tokens |
| Children/parent | 2-4 | 2.78 (mesure) |

---

## 7. Fichiers Associes

| Fichier | Role |
|---------|------|
| `scripts/pipeline/sentence_chunker.py` | Chunker recursif (Step 1) |
| `scripts/pipeline/parent_child_chunker.py` | Parent-child retrieval (Step 2-3) |
| `scripts/pipeline/table_extractor.py` | Extraction tables (Step 4) |
| `scripts/pipeline/token_utils.py` | Utilitaires tokenization |
| `scripts/pipeline/embeddings.py` | Generation embeddings |
| `scripts/pipeline/export_sdk.py` | Export DB SQLite |
| `corpus/processed/chunks_recursive_fr.json` | Chunks recursifs (1244) |
| `corpus/processed/chunks_parent_child_fr.json` | Parents + children (718+1997) |

---

## 8. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-16 | SentenceSplitter 512/128 tokens (baseline 78.33%) |
| 2.0 | 2026-01-17 | Chunks 400 tokens (recall 75%) |
| 3.0 | 2026-01-18 | RecursiveCharacterTextSplitter 450/100, Etape 1-4 |
| 3.1 | 2026-01-18 | Parent-child chunker, metadata, tables, **recall 85.29%** |

---

*Ce document est maintenu dans le cadre du systeme de conformite ISO du projet Pocket Arbiter.*
