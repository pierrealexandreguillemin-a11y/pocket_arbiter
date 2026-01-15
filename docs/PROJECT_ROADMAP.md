# Roadmap Projet - Pocket Arbiter

> **Document ID**: PLAN-RDM-001
> **ISO Reference**: ISO/IEC 12207:2017
> **Version**: 1.2
> **Date**: 2026-01-14
> **Effort total estime**: 205h (~13-15 semaines)

---

## Vue d'ensemble

```
PHASE 0 ✅ COMPLETE
    │
    ▼
PHASE 1A ✅ COMPLETE ──► Pipeline Extract + Chunk (2047 FR + 1105 INTL)
    │
    ▼
PHASE 1B (S3-4) ──► Pipeline Embed + Index
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

## Phase 1A - Pipeline Extract + Chunk (COMPLETE)

**Duree**: Semaine 1-2 | **Effort**: 20h | **Statut**: ✅ COMPLETE

### Objectif
Extraire le contenu textuel des PDF et segmenter en chunks de 256 tokens.

### Deliverables

| Fichier | Description | Statut |
|---------|-------------|--------|
| `scripts/pipeline/extract_pdf.py` | Extraction PDF via PyMuPDF | ✅ |
| `scripts/pipeline/chunker.py` | Segmentation 256 tokens, overlap 50 | ✅ |
| `corpus/processed/chunks_fr.json` | 2047 chunks FR | ✅ |
| `corpus/processed/chunks_intl.json` | 1105 chunks INTL | ✅ |

### Resultats

| Critere | Cible | Reel | Statut |
|---------|-------|------|--------|
| PDFs FR extraits | 28 | 28 | ✅ |
| PDFs INTL extraits | 1 | 1 | ✅ |
| Chunks FR | ~500 | 2047 | ✅ |
| Chunks INTL | ~100 | 1105 | ✅ |
| Coverage tests | 80% | 93% | ✅ |
| Complexite cyclomatique | B max | B | ✅ |

> **Note**: Le nombre de chunks est superieur aux previsions car le corpus
> contient 404 pages (vs estimation initiale). Le chunking 256 tokens avec
> overlap 20% produit ~5 chunks/page en moyenne.

### Specifications
- Voir `docs/specs/PHASE1A_SPECS.md`
- Schema: `docs/CHUNK_SCHEMA.md`

---

## Phase 1B - Pipeline Embed + Export SDK

**Duree**: Semaine 3-4 | **Effort**: 30h

### Objectif
Generer embeddings et exporter au format Google AI Edge RAG SDK.

### Stack validee (2026-01-14)

| Composant | Choix | Specs | Source |
|-----------|-------|-------|--------|
| Embedding | **EmbeddingGemma-300m** | 768D, 179MB TFLite | [HuggingFace](https://huggingface.co/litert-community/embeddinggemma-300m) |
| Quantization | Mixed Precision | int4 embed/ff, int8 attn | e4_a8_f4_p4 |
| RAM CPU | **110 MB** (256 seq) | Compatible mid-range | Benchmark S25 Ultra |
| Inference | **66 ms** (256 seq) | XNNPACK 4 threads | Benchmark S25 Ultra |
| Vector Store | **SqliteVectorStore** | SDK natif, persistant | [Google AI Edge](https://github.com/google-ai-edge/ai-edge-apis) |
| Fallback | Gecko-110m-en | 114MB, 126MB RAM, 147ms | Si EmbeddingGemma indisponible |

### Deliverables

| Fichier | Description |
|---------|-------------|
| `scripts/pipeline/embeddings.py` | Generation vecteurs 768D (sentence-transformers) |
| `scripts/pipeline/export_sdk.py` | Export format SqliteVectorStore |
| `corpus/processed/corpus_fr.db` | Base SQLite avec vecteurs FR |
| `corpus/processed/corpus_intl.db` | Base SQLite avec vecteurs INTL |
| `corpus/processed/embeddings_fr.npy` | Vecteurs numpy FR (backup) |

### Definition of Done

| Critere | Cible | Bloquant |
|---------|-------|----------|
| Recall FR | ≥ 80% | OUI |
| Recall INTL | ≥ 70% | NON |
| 0% hallucination adversarial | 30/30 pass | OUI |
| DB size | < 100 MB | NON |
| Coverage tests | ≥ 80% | OUI |

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
