# Roadmap Projet - Pocket Arbiter

> **Document ID**: PLAN-RDM-001
> **ISO Reference**: ISO/IEC 12207:2017
> **Version**: 1.0
> **Date**: 2026-01-14
> **Effort total estime**: 210h (~13-16 semaines)

---

## Vue d'ensemble

```
PHASE 0 ✅ COMPLETE
    │
    ▼
PHASE 1A (S1-2) ──► Pipeline Extract + Chunk
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

## Phase 1A - Pipeline Extract + Chunk

**Duree**: Semaine 1-2 | **Effort**: 20h

### Objectif
Extraire le contenu textuel des PDF et segmenter en chunks de 256 tokens.

### Deliverables

| Fichier | Description |
|---------|-------------|
| `scripts/pipeline/extract_pdf.py` | Extraction PDF via PyMuPDF |
| `scripts/pipeline/chunker.py` | Segmentation 256 tokens, overlap 50 |
| `corpus/processed/chunks_fr.json` | ~500 chunks FR |
| `corpus/processed/chunks_intl.json` | ~100 chunks INTL |

### Definition of Done

| Critere | Validation |
|---------|------------|
| 100% PDF extraits | Script execute sans erreur |
| ~500 chunks FR | Comptage output |
| Tests ≥80% coverage | `pytest --cov-fail-under=80` |
| Schema documente | `docs/CHUNK_SCHEMA.md` existe |

### Specifications
- Voir `docs/specs/PHASE1A_SPECS.md`

---

## Phase 1B - Pipeline Embed + Index

**Duree**: Semaine 3-4 | **Effort**: 30h

### Objectif
Generer embeddings et creer index vectoriel pour retrieval.

### Stack validee

| Composant | Choix | Specs |
|-----------|-------|-------|
| Embedding | EmbeddingGemma-300M | 768D, 179MB, 66ms |
| Index | FAISS (IVF_FLAT) | cosine, < 50MB |

### Deliverables

| Fichier | Description |
|---------|-------------|
| `scripts/pipeline/embeddings.py` | Generation vecteurs 768D |
| `scripts/pipeline/indexer.py` | Creation index FAISS |
| `scripts/pipeline/export_android.py` | Export format SDK |
| `corpus/processed/index_fr.faiss` | Index vectoriel FR |

### Definition of Done

| Critere | Cible | Bloquant |
|---------|-------|----------|
| Recall FR | ≥ 80% | OUI |
| Recall INTL | ≥ 70% | NON |
| 0% hallucination adversarial | 30/30 pass | OUI |
| Index < 50 MB | Mesure | NON |

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

**Duree**: Semaine 13 | **Effort**: 15h

### Objectif
Evaluer et comparer implementations vector store pour RAM.

### Implementations a tester

| Implementation | Description | Effort |
|----------------|-------------|--------|
| SqliteVectorStore | SDK default | 0h |
| sqlite-vec custom | Alternative legere | ~10h |
| FAISS JNI | In-memory | ~20h |

### Protocole

```
1. RAM baseline (app sans vector store)
2. RAM apres chargement index (~500 chunks)
3. RAM pic pendant query (10 queries)
4. Latence moyenne query (ms)
5. Repeter 3x par implementation
```

### Criteres decision

- Gain RAM > 50 MB → Implementer alternative
- Gain RAM < 50 MB → Garder SDK default
- Latence degradee > 2x → Garder SDK default

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
| Recall < 80% avec EmbeddingGemma | FAIBLE | Tester all-MiniLM-L12-v2 |
| Gemma 270M qualite < 70% | MOYEN | Passer a Gemma 1B |
| RAM > 500MB | FAIBLE | Reduire seq_length, dimensions |
| SDK 0.1.0 instable | FAIBLE | Figer version, workarounds |

---

## Effort par phase

| Phase | Effort | Cumule |
|-------|--------|--------|
| 1A | 20h | 20h |
| 1B | 30h | 50h |
| 2A | 40h | 90h |
| 2B | 30h | 120h |
| 3 | 40h | 160h |
| 4A | 15h | 175h |
| 4B | 15h | 190h |
| 5 | 20h | 210h |
| **TOTAL** | **210h** | |

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
