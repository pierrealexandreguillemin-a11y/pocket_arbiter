# Roadmap Projet - Pocket Arbiter

> **Document ID**: PLAN-RDM-001
> **ISO Reference**: ISO/IEC 12207:2017
> **Version**: 1.3
> **Date**: 2026-01-19
> **Effort total estime**: 205h (~13-15 semaines)

---

## Vue d'ensemble

```
PHASE 0 ✅ COMPLETE
    │
    ▼
PHASE 1A ✅ COMPLETE + ENHANCED ──► Pipeline Extract + Chunk (1454 FR + 764 INTL)
    │
    ▼
PHASE 1B ✅ 95% COMPLETE ──► Pipeline Embed + Index (Recall FR 86.76%)
    │
    ▼
PHASE 2A (S5-7) ──► Android Setup + Retrieval
    │
    ▼
PHASE 2B (S8-9) ──► Android UI + Integration
    │
    ▼
PHASE 3 (S10-12) ─► LLM Synthesis + Anti-hallucination
    │
    ▼
PHASE 4A (S13) ───► Evaluation Vector Store RAM
    │
    ▼
PHASE 4B (S14) ───► Optimisation Generale
    │
    ▼
PHASE 5 (S15-16) ─► Beta + Release
    │
    ▼
PRODUCTION
```

---

## Phase 0 - Fondations (COMPLETE)

| Deliverable | Statut |
|-------------|--------|
| Structure projet | ✅ |
| Documentation ISO | ✅ |
| Hooks pre-commit | ✅ |
| Configuration gates | ✅ |

---

## Phase 1A - Pipeline Extract + Chunk (COMPLETE + ENHANCED)

**Duree**: Semaine 1-2 | **Effort**: 20h | **Statut**: ✅ COMPLETE + ENHANCED

### Objectif
Extraire le contenu textuel des PDF et segmenter en chunks parent-child.

### Deliverables

| Fichier | Description | Statut |
|---------|-------------|--------|
| `scripts/pipeline/extract_docling.py` | Extraction PDF via Docling ML | ✅ UPGRADE |
| `scripts/pipeline/parent_child_chunker.py` | Segmentation Parent 1024/Child 450 | ✅ NEW |
| `scripts/pipeline/table_multivector.py` | Tables + LLM summaries | ✅ NEW |
| `corpus/processed/chunks_for_embedding_fr.json` | 1454 chunks FR | ✅ |
| `corpus/processed/chunks_for_embedding_intl.json` | 764 chunks INTL | ✅ |

### Evolutions vs Roadmap Original

| Aspect | Roadmap Original | Etat Actuel | Deviation |
|--------|-----------------|-------------|-----------|
| Extraction | `extract_pdf.py` (PyMuPDF) | `extract_docling.py` (Docling ML) | UPGRADE |
| Chunking | `chunker.py` 256 tokens | `parent_child_chunker.py` Parent 1024/Child 450 | UPGRADE |
| Overlap | 50 tokens (20%) | 15% (NVIDIA 2025) | OPTIMISE |
| Chunks FR | 2047 | 1454 (1343 child + 111 tables) | -29% (moins redondant) |
| Chunks INTL | 1105 | 764 | -31% |

### Resultats

| Critere | Cible | Reel | Statut |
|---------|-------|------|--------|
| PDFs FR extraits | 28 | 28 | ✅ |
| PDFs INTL extraits | 1 | 1 | ✅ |
| Chunks FR | ~500 | 1454 | ✅ |
| Chunks INTL | ~100 | 764 | ✅ |
| Coverage tests | 80% | 87% | ✅ |
| Complexite cyclomatique | B max | B | ✅ |

> **Note**: Chunks reduits grace au Parent-Child chunking (NVIDIA 2025 research).
> Moins de redondance, meilleure precision semantique.

### Specifications
- Voir `docs/specs/PHASE1A_SPECS.md`
- Schema: `docs/CHUNKING_STRATEGY.md`

---

## Phase 1B - Pipeline Embed + Export SDK (95% COMPLETE)

**Duree**: Semaine 3-4 | **Effort**: 30h | **Statut**: ✅ 95% COMPLETE

### Objectif
Generer embeddings et exporter au format Google AI Edge RAG SDK.

### Stack validee (2026-01-19)

| Composant | Choix | Specs | Source |
|-----------|-------|-------|--------|
| Embedding | **EmbeddingGemma-300m-qat** | 768D, quantized | [HuggingFace](https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized) |
| Reranker | **BGE-reranker-v2-m3** | Cross-encoder multilingual | [BAAI](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Search | **Hybrid** | BM25=0.7 + Vector=0.3 + RRF | Custom |
| Vector Store | **SqliteVectorStore** | SDK natif, persistant, FTS5 | [Google AI Edge](https://github.com/google-ai-edge/ai-edge-apis) |

### Deliverables

| Fichier | Description | Statut |
|---------|-------------|--------|
| `scripts/pipeline/embeddings.py` | Generation vecteurs 768D | ✅ |
| `scripts/pipeline/export_sdk.py` | Export format SqliteVectorStore | ✅ |
| `scripts/pipeline/export_search.py` | Hybrid BM25+Vector+RRF | ✅ NEW |
| `scripts/pipeline/reranker.py` | Cross-encoder reranking | ✅ NEW |
| `corpus/processed/corpus_fr.db` | Base SQLite FR (7.58 MB) | ✅ |
| `corpus/processed/corpus_intl.db` | Base SQLite INTL (4.21 MB) | ✅ |

### Resultats (2026-01-19)

| Critere | Cible | Reel | Statut |
|---------|-------|------|--------|
| Recall FR | ≥ 80% | **86.76%** (hybrid+rerank) | ✅ PASS |
| Recall FR (vector-only) | ≥ 80% | **84.31%** | ✅ PASS |
| Recall INTL | ≥ 70% | **80.00%** (vector, tol=2) | ✅ PASS |
| DB size total | < 100 MB | **11.79 MB** | ✅ PASS |
| Coverage tests | ≥ 80% | **87%** | ✅ PASS |

### Definition of Done

| Critere | Cible | Bloquant | Statut |
|---------|-------|----------|--------|
| Recall FR | ≥ 80% | OUI | ✅ 86.76% |
| Recall INTL | ≥ 70% | NON | ✅ 80.00% |
| 0% hallucination adversarial | 30/30 pass | OUI | PENDING |
| DB size | < 100 MB | NON | ✅ 11.79 MB |
| Coverage tests | ≥ 80% | OUI | ✅ 87% |

---

## Phase 2A - Android Setup + Retrieval

**Duree**: Semaine 5-7 | **Effort**: 40h

### Objectif
Creer app Android avec retrieval fonctionnel (sans LLM).

### Stack Android

| Composant | Version |
|-----------|---------|
| Min SDK | 29 (Android 10) |
| Target SDK | 34 |
| Compose | 1.5+ |
| RAG SDK | `localagents-rag:0.1.0` |
| MediaPipe | `tasks-genai:0.10.22` |

### Deliverables

| Fichier | Description |
|---------|-------------|
| `android/` | Projet Android Studio |
| `RagPipeline.kt` | Configuration SDK |
| `EmbeddingEngine.kt` | Wrapper EmbeddingGemma |
| Tests unitaires Kotlin | JUnit5 |

### Definition of Done

| Critere | Validation |
|---------|------------|
| Build APK reussi | `./gradlew assembleDebug` |
| Retrieval top-5 | Test manuel 5 questions |
| Latence < 500ms | Benchmark |
| RAM < 300MB | Android Profiler |

---

## Phase 2B - Android UI + Integration

**Duree**: Semaine 8-9 | **Effort**: 30h

### Objectif
Interface utilisateur complete avec navigation.

### Ecrans

| Ecran | Description |
|-------|-------------|
| HomeScreen | Selection corpus FR/INTL |
| QueryScreen | Saisie question |
| ResultScreen | Reponse + citations |

### Definition of Done

| Critere | Validation |
|---------|------------|
| UX complete | Review VISION.md |
| Disclaimer visible | Screenshot |
| Tests UI passent | Espresso/Compose |
| Accessibilite | Test TalkBack |

---

## Phase 3 - LLM Synthesis

**Duree**: Semaine 10-12 | **Effort**: 40h

### Objectif
Integrer LLM pour synthese avec grounding strict anti-hallucination.

### Stack LLM

| Composant | Choix | Specs |
|-----------|-------|-------|
| LLM | Gemma 3 270M | ~200MB RAM, 2-3s |
| Backup | Gemma 3 1B | Si qualite < 70% |

### Prompt template

```
Tu es un assistant pour arbitres d'echecs.
REGLES STRICTES:
1. Reponds UNIQUEMENT a partir des passages ci-dessous
2. Si l'info n'est pas dans les passages: "Information non trouvee"
3. Cite TOUJOURS la source: [Nom document, Page X]
4. Ne jamais inventer ou extrapoler

PASSAGES:
{retrieved_chunks}

QUESTION: {user_query}
```

### Definition of Done

| Critere | Cible | Bloquant |
|---------|-------|----------|
| Hallucination | 0% | OUI |
| Fidelite | ≥ 85% | OUI |
| Latence totale | < 5s | NON |
| Citations | 100% presentes | OUI |

---

## Phase 4A - Evaluation Vector Store RAM

**Duree**: Semaine 13 | **Effort**: 10h

### Objectif
Valider performance RAM du SqliteVectorStore en conditions reelles.

### Implementations a tester

| Implementation | Description | Effort |
|----------------|-------------|--------|
| SqliteVectorStore | SDK default (choix principal) | 0h |
| sqlite-vec custom | Alternative si RAM > 300MB | ~10h |

### Protocole

```
1. RAM baseline (app sans vector store)
2. RAM apres chargement index (~3000 chunks)
3. RAM pic pendant query (10 queries)
4. Latence moyenne query (ms)
5. Repeter 3x sur device mid-range (4GB RAM)
```

### Criteres decision

- RAM embedder + vector store < 300 MB → Valide
- RAM > 300 MB → Tester sqlite-vec custom
- Latence degradee > 2x → Optimiser requetes

---

## Phase 4B - Optimisation Generale

**Duree**: Semaine 14 | **Effort**: 15h

### Objectif
Performance et qualite production.

### Definition of Done

| Critere | Cible |
|---------|-------|
| RAM pic | < 500 MB (4GB device) |
| Latence P95 | < 5s |
| Crash-free | ≥ 99% |
| Taille APK | < 50MB (hors modeles) |
| Coverage | ≥ 60% |

---

## Phase 5 - Beta + Release

**Duree**: Semaine 15-16 | **Effort**: 20h

### Objectif
Validation utilisateur et release production.

### Deliverables

| Fichier | Description |
|---------|-------------|
| `docs/USER_GUIDE.md` | Guide utilisateur |
| `docs/RELEASE_NOTES.md` | Notes de version |
| APK release signe | Distribution |

### Definition of Done

| Critere | Cible |
|---------|-------|
| NPS | ≥ 7/10 (5 arbitres) |
| Bugs critiques | 0 |
| Checklist ISO 42001 | 100% |

---

## Metriques globales

| Phase | Metrique | Cible | Bloquant |
|-------|----------|-------|----------|
| 1B | Recall FR | ≥ 80% | OUI |
| 1B | Recall INTL | ≥ 70% | NON |
| 3 | Hallucination | 0% | OUI |
| 3 | Fidelite | ≥ 85% | OUI |
| 4 | RAM | < 500MB | OUI |
| 4 | Latence totale | < 5s | NON |
| 5 | NPS | ≥ 7/10 | NON |

---

## Risques residuels

| Risque | Probabilite | Plan B |
|--------|-------------|--------|
| Recall < 80% avec EmbeddingGemma | FAIBLE | Tester Gecko-110m-en |
| EmbeddingGemma non dispo sentence-transformers | FAIBLE | Utiliser ONNX ou TFLite directement |
| Gemma 270M qualite < 70% | MOYEN | Passer a Gemma 3 1B |
| RAM > 300MB (embedder + store) | FAIBLE | Reduire dimensions MRL (512 ou 256) |
| SqliteVectorStore lent sur 3000 chunks | FAIBLE | Optimiser requetes, pagination |

---

## Findings & Ameliorations (2026-01-19)

### Ameliorations Non Planifiees
| Module | Role | ISO Ref |
|--------|------|---------|
| `parent_child_chunker.py` | Parent-Child chunking (NVIDIA 2025 research) | ISO 25010 |
| `table_multivector.py` | Tables + LLM summaries | ISO 42001 |
| `reranker.py` | Cross-encoder bge-reranker-v2-m3 | ISO 25010 |
| `export_search.py` | Hybrid BM25+Vector+RRF | ISO 25010 |
| `query_expansion.py` | Snowball FR + synonymes | ISO 25010 |

### Metriques Finales Phase 1B

| Corpus | Chunks | Recall@5 (vector) | Recall@5 (hybrid+rerank) | ISO Status |
|--------|--------|-------------------|--------------------------|------------|
| FR | 1454 | 84.31% | **86.76%** | ✅ PASS (≥80%) |
| INTL | 764 | **80.00%** | 66.00% | ✅ PASS vector (≥70%) |

> **Note**: Le reranking ameliore FR (+2.45%) mais degrade INTL (-14%).
> Recommandation: utiliser vector-only pour INTL, hybrid+rerank pour FR.

### Questions Gold Standard

| Corpus | Questions | Documents | ISO 29119 (>=50) |
|--------|-----------|-----------|------------------|
| FR | 68 | 28 | ✅ PASS |
| INTL | 25 | 1 | ✅ PASS (total 93) |

---

## Effort par phase

| Phase | Effort | Cumule |
|-------|--------|--------|
| 1A | 20h | 20h |
| 1B | 30h | 50h |
| 2A | 40h | 90h |
| 2B | 30h | 120h |
| 3 | 40h | 160h |
| 4A | 10h | 170h |
| 4B | 15h | 185h |
| 5 | 20h | 205h |
| **TOTAL** | **205h** | |

---

## Documents associes

| Document | Description |
|----------|-------------|
| `docs/specs/PHASE1A_SPECS.md` | Specifications Phase 1A |
| `docs/CHUNK_SCHEMA.md` | Schema JSON chunks |
| `docs/AI_POLICY.md` | Politique IA ISO 42001 |
| `docs/QUALITY_REQUIREMENTS.md` | Exigences qualite ISO 25010 |
| `docs/TEST_PLAN.md` | Plan de tests ISO 29119 |

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-14 | Creation initiale |
| 1.1 | 2026-01-14 | Phase 1A complete (2047 FR + 1105 INTL chunks) |
| 1.2 | 2026-01-14 | Stack Phase 1B: FAISS -> SqliteVectorStore, EmbeddingGemma-300m specs |
| 1.3 | 2026-01-19 | Phase 1A enhanced (Docling, Parent-Child), Phase 1B 95% (Recall FR 86.76%), Findings section |
