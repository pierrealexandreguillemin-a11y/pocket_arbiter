# Dependencies Documentation - Pocket Arbiter

> **Document ID**: SPEC-DEP-001
> **ISO Reference**: ISO/IEC 12207 S7.3.3 - Configuration Management
> **Version**: 1.1
> **Date**: 2026-01-29
> **Statut**: Approuve
> **Classification**: Technique

---

## 1. Objectif

Documenter les dependances du projet Pocket Arbiter avec justification technique
basee sur la recherche industrie et les benchmarks 2025-2026.

---

## 2. Dependencies Pipeline

### 2.1 Extraction PDF

| Package | Version | Source | Justification |
|---------|---------|--------|---------------|
| `docling` | >=2.68.0 | IBM / LF AI & Data | ML-based extraction (vs rules-based). Tables + sections. MIT license. |

**Recherche**:
- Docling vs PyMuPDF: +15% accuracy sur tables complexes (IBM benchmark 2025)
- Docling vs Camelot: ML adaptatif vs rules rigides
- Source: [Docling GitHub](https://github.com/DS4SD/docling)

### 2.2 Tokenization

| Package | Version | Source | Justification |
|---------|---------|--------|---------------|
| `tiktoken` | >=0.8.0 | OpenAI | cl100k_base encoder. Implementation C rapide. Standard industrie. |

**Configuration**:
```python
TOKENIZER_NAME = "cl100k_base"  # Compatible GPT-4, embeddings OpenAI
```

**Recherche**:
- tiktoken vs SentencePiece: 10x plus rapide (benchmark OpenAI)
- cl100k_base: vocabulaire 100k tokens, optimal pour texte francais

### 2.3 Text Splitting

| Package | Version | Source | Justification |
|---------|---------|--------|---------------|
| `langchain-text-splitters` | >=0.3.0 | LangChain | RecursiveCharacterTextSplitter. Hierarchie semantique. |

**Configuration** (chunker.py):
```python
# NVIDIA 2025: 15% overlap optimal (FinanceBench)
# arXiv 2025: 512-1024 tokens pour contexte large
# Chroma 2025: 400-512 tokens sweet spot (85-90% recall)

PARENT_CHUNK_SIZE = 1024
PARENT_CHUNK_OVERLAP = 154   # 15% de 1024

CHILD_CHUNK_SIZE = 450
CHILD_CHUNK_OVERLAP = 68     # 15% de 450

REGULATORY_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
```

**Recherche**:
- NVIDIA: [Finding Best Chunking Strategy](https://developer.nvidia.com/blog/finding-best-chunking-strategy)
- Chroma: 400-512 tokens = 85-90% recall@5
- arXiv 2025: Larger context improves LLM synthesis

---

## 3. Dependencies Embeddings

### 3.1 Embedding Model

| Package | Version | Source | Justification |
|---------|---------|--------|---------------|
| `sentence-transformers` | >=5.2.0 | UKP Lab / Hugging Face | EmbeddingGemma support. MRL (Matryoshka). |

**Configuration**:
```python
MODEL = "google/embeddinggemma-300m-qat-q4_0-unquantized"
MAX_TOKENS = 2048  # EmbeddingGemma context window
EMBEDDING_DIM = 384  # MRL: truncatable to 256, 128
```

**Recherche**:
- EmbeddingGemma: Google 2025, 308M params, MTEB rank top-10
- MRL: Matryoshka Representation Learning (variable dimensions)
- Source: [Google AI Blog](https://ai.googleblog.com/)

---

## 4. Dependencies On-Device

### 4.1 Runtime Mobile

| Package | Version | Source | Justification |
|---------|---------|--------|---------------|
| `ai-edge-litert` | >=2.1.0 | Google AI Edge | LiteRT (successor TFLite). Android/iOS. |

**Configuration**:
- Export: `.task` format (MediaPipe compatible)
- Quantization: INT8 dynamic range
- Target: ARM64 (mobile), x86_64 (dev)

**Recherche**:
- LiteRT: Google rebrand TFLite 2025
- Source: [Google AI Edge SDK](https://ai.google.dev/edge)

---

## 5. Dependencies Testing (ISO 29119)

| Package | Version | Source | Justification |
|---------|---------|--------|---------------|
| `pytest` | >=8.0.0 | pytest-dev | Framework testing standard Python |
| `pytest-cov` | >=5.0.0 | pytest-dev | Coverage reporting. Gate 80%. |

**Configuration**:
```bash
pytest --cov=scripts --cov-fail-under=80
```

---

## 6. Dependencies Quality (ISO 25010)

| Package | Version | Source | Justification |
|---------|---------|--------|---------------|
| `ruff` | >=0.8.0 | Astral | Linter + formatter. Rust-based (rapide). |
| `mypy` | >=1.13.0 | Python | Type checking strict. |

**Configuration** (pyproject.toml):
```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.mypy]
strict = true
```

---

## 7. Versions Verification

Toutes les versions specifiees sont les **latest stable** au 2026-01-19:

| Package | Specified | Latest Stable | Status |
|---------|-----------|---------------|--------|
| docling | >=2.68.0 | 2.68.0 | OK |
| tiktoken | >=0.8.0 | 0.8.0 | OK |
| langchain-text-splitters | >=0.3.0 | 0.3.6 | OK |
| sentence-transformers | >=5.2.0 | 5.2.0 | OK |
| ai-edge-litert | >=2.1.0 | 2.1.0 | OK |
| pytest | >=8.0.0 | 8.3.4 | OK |
| pytest-cov | >=5.0.0 | 6.0.0 | OK |
| ruff | >=0.8.0 | 0.9.3 | OK |
| mypy | >=1.13.0 | 1.14.1 | OK |

---

## 8. References

### 8.1 Recherche Industrie
- NVIDIA 2025: "Finding the Best Chunking Strategy for RAG"
- Chroma 2025: "Chunking Strategies for Vector Databases"
- arXiv 2025: "Optimal Context Windows for RAG Systems"

### 8.2 Documentation Officielle
- [Docling](https://github.com/DS4SD/docling)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Sentence Transformers](https://www.sbert.net/)
- [Google AI Edge](https://ai.google.dev/edge)

---

## 9. Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-19 | Creation initiale avec justifications recherche |
| 1.1 | 2026-01-29 | Suppression FlagEmbedding (reranker abandonne), toolchain pyproject.toml |

---

*Ce document est maintenu dans le cadre du systeme de conformite ISO du projet Pocket Arbiter.*
