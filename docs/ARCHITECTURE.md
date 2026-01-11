# Architecture Technique - Pocket Arbiter

> **Document ID**: SPEC-ARCH-001
> **ISO Reference**: ISO/IEC 12207:2017 - Architecture logicielle
> **Version**: 1.0
> **Date**: 2026-01-11
> **Statut**: Draft
> **Classification**: Interne

---

## 1. Vue d'ensemble

Application mobile Android 100% offline pour l'aide aux arbitres d'echecs via RAG (Retrieval-Augmented Generation).

### 1.1 Contraintes architecturales

| Contrainte | Valeur | Justification |
|------------|--------|---------------|
| Execution | 100% offline | Salles de tournoi mal connectees |
| RAM max | 500 MB | Compatibilite mid-range |
| Latence max | 5 secondes | Decisions rapides en tournoi |
| Taille APK | < 100 MB | Telechargement raisonnable |
| Taille assets | < 500 MB | Stockage mobile |

### 1.2 Standards appliques

- **ISO/IEC 12207:2017** - Processus cycle de vie
- **ISO/IEC 25010:2011** - Qualite logicielle
- **ISO/IEC 29119:2013** - Tests logiciels
- **ISO/IEC 42001:2023** - Gouvernance IA

---

## 2. Architecture globale

```
+------------------------------------------------------------------+
|                        POCKET ARBITER                             |
+------------------------------------------------------------------+
|                                                                   |
|  +---------------------+    +---------------------+               |
|  |   PRESENTATION      |    |   DATA LAYER        |               |
|  |   (Android/Kotlin)  |    |   (Python Pipeline) |               |
|  +---------------------+    +---------------------+               |
|  | - MainActivity      |    | - PDF Extraction    |               |
|  | - QueryFragment     |    | - Text Chunking     |               |
|  | - ResultFragment    |    | - Embedding Gen     |               |
|  | - SettingsFragment  |    | - Index Building    |               |
|  +----------+----------+    +----------+----------+               |
|             |                          |                          |
|             v                          v                          |
|  +---------------------+    +---------------------+               |
|  |   DOMAIN LAYER      |    |   ASSETS            |               |
|  |   (Android/Kotlin)  |    |   (Generated)       |               |
|  +---------------------+    +---------------------+               |
|  | - QueryUseCase      |    | - corpus_fr.faiss   |               |
|  | - RetrievalUseCase  |    | - corpus_intl.faiss |               |
|  | - SynthesisUseCase  |    | - chunks_fr.json    |               |
|  +----------+----------+    | - chunks_intl.json  |               |
|             |               | - embedding.tflite  |               |
|             v               | - llm.tflite        |               |
|  +---------------------+    +---------------------+               |
|  |   INFERENCE LAYER   |                                          |
|  |   (MediaPipe/TFLite)|                                          |
|  +---------------------+                                          |
|  | - EmbeddingEngine   |                                          |
|  | - FAISSIndex        |                                          |
|  | - LLMEngine         |                                          |
|  +---------------------+                                          |
|                                                                   |
+------------------------------------------------------------------+
```

---

## 3. Structure des repertoires

```
pocket_arbiter/
|
+-- android/                    # Application Android (Phase 2+)
|   +-- app/
|   |   +-- src/
|   |   |   +-- main/
|   |   |   |   +-- kotlin/com/arbiter/
|   |   |   |   |   +-- ui/             # Fragments, ViewModels
|   |   |   |   |   +-- domain/         # Use cases
|   |   |   |   |   +-- data/           # Repositories
|   |   |   |   |   +-- inference/      # ML engines
|   |   |   |   +-- res/                # Resources
|   |   |   |   +-- assets/             # ML models, indices
|   |   |   +-- test/                   # Unit tests
|   |   |   +-- androidTest/            # UI tests
|   |   +-- build.gradle.kts
|   +-- build.gradle.kts
|   +-- settings.gradle.kts
|
+-- corpus/                     # Documents sources (DVC tracked)
|   +-- fr/                     # Reglements FFE
|   +-- intl/                   # Reglements FIDE
|   +-- INVENTORY.md            # Liste des documents
|
+-- scripts/                    # Pipeline Python (Phase 1)
|   +-- pipeline/               # Scripts extraction/indexation
|   |   +-- __init__.py
|   |   +-- extract_pdf.py      # Extraction texte PDF
|   |   +-- chunker.py          # Decoupage en chunks
|   |   +-- embeddings.py       # Generation embeddings
|   |   +-- indexer.py          # Construction index FAISS
|   |   +-- tests/              # Tests unitaires pipeline
|   +-- iso/                    # Validation ISO (existant)
|   +-- requirements.txt
|
+-- models/                     # Modeles ML (DVC tracked)
|   +-- embeddings/             # Modele embeddings
|   +-- llm/                    # Modele LLM quantifie
|   +-- model_card.json         # Metadonnees modeles
|
+-- prompts/                    # Templates prompts LLM
|   +-- interpretation_v1.txt   # Prompt synthese
|   +-- CHANGELOG.md
|
+-- tests/                      # Donnees de test
|   +-- data/
|   |   +-- questions_fr.json   # Questions test FR
|   |   +-- questions_intl.json # Questions test INTL
|   |   +-- adversarial.json    # Tests anti-hallucination
|   +-- reports/                # Rapports de test
|
+-- docs/                       # Documentation
|   +-- VISION.md
|   +-- ARCHITECTURE.md         # Ce document
|   +-- AI_POLICY.md
|   +-- QUALITY_REQUIREMENTS.md
|   +-- TEST_PLAN.md
|   +-- DOC_CONTROL.md
|   +-- INDEX.md
|
+-- .iso/                       # Configuration ISO
|   +-- config.json
|   +-- checklists/
|   +-- templates/
|
+-- .githooks/                  # Git hooks
+-- .github/workflows/          # CI/CD
```

---

## 4. Planning des fichiers (limite 300 lignes)

### 4.1 Fichiers existants

| Fichier | Lignes | Statut | Action si > 300 |
|---------|--------|--------|-----------------|
| `.githooks/pre-commit.py` | 308 | WARN | Acceptable (hooks) |
| `scripts/iso/tests/test_checks.py` | 268 | OK | - |
| `scripts/iso/tests/test_gates.py` | 266 | OK | - |
| `scripts/iso/tests/test_setup_dvc.py` | 208 | OK | - |
| `scripts/iso/tests/test_validator.py` | 196 | OK | - |
| `scripts/iso/gates.py` | 171 | OK | - |
| `scripts/iso/setup_dvc.py` | 168 | OK | - |
| `scripts/iso/validate_project.py` | 143 | OK | - |

### 4.2 Fichiers futurs - Phase 1 (Pipeline)

| Fichier | Lignes estimees | Decomposition si > 300 |
|---------|-----------------|------------------------|
| `extract_pdf.py` | ~150 | - |
| `chunker.py` | ~200 | - |
| `embeddings.py` | ~100 | - |
| `indexer.py` | ~150 | - |
| `test_pipeline.py` | ~250 | - |

### 4.3 Fichiers futurs - Phase 2 (Android)

| Fichier | Lignes estimees | Decomposition si > 300 |
|---------|-----------------|------------------------|
| `MainActivity.kt` | ~150 | - |
| `QueryFragment.kt` | ~200 | - |
| `ResultFragment.kt` | ~250 | - |
| `QueryViewModel.kt` | ~150 | - |
| `EmbeddingEngine.kt` | ~200 | - |
| `FAISSRepository.kt` | ~250 | - |
| `LLMEngine.kt` | ~300 | Surveiller, split si depasse |

### 4.4 Fichiers futurs - Phase 3 (LLM)

| Fichier | Lignes estimees | Decomposition si > 300 |
|---------|-----------------|------------------------|
| `SynthesisUseCase.kt` | ~200 | - |
| `PromptBuilder.kt` | ~150 | - |
| `CitationExtractor.kt` | ~200 | - |
| `HallucinationGuard.kt` | ~150 | - |

### 4.5 Regles de decomposition

Si un fichier depasse 300 lignes :

1. **Extraction de classes** : Une responsabilite = une classe
2. **Extraction de modules** : Fonctions utilitaires dans `_utils.py`
3. **Separation tests** : `test_X.py` split en `test_X_unit.py` + `test_X_integration.py`
4. **Documentation externe** : Docstrings longues -> fichiers `.md` separes

---

## 5. Modules et responsabilites

### 5.1 Module Pipeline (Python)

```
scripts/pipeline/
|
+-- extract_pdf.py       # Responsabilite: Extraction texte brut
|   |-- extract_text(pdf_path) -> str
|   |-- clean_text(text) -> str
|   |-- extract_metadata(pdf_path) -> dict
|
+-- chunker.py           # Responsabilite: Decoupage semantique
|   |-- chunk_text(text, max_tokens) -> List[Chunk]
|   |-- overlap_chunks(chunks, overlap) -> List[Chunk]
|   |-- add_metadata(chunks, source) -> List[Chunk]
|
+-- embeddings.py        # Responsabilite: Generation vecteurs
|   |-- load_model(model_name) -> Model
|   |-- embed_text(text) -> np.array
|   |-- embed_batch(texts) -> np.array
|
+-- indexer.py           # Responsabilite: Index FAISS
|   |-- create_index(embeddings) -> faiss.Index
|   |-- save_index(index, path) -> None
|   |-- search(query_embedding, k) -> List[int]
```

### 5.2 Module Android (Kotlin)

```
android/app/src/main/kotlin/com/arbiter/
|
+-- ui/                  # Couche presentation
|   +-- MainActivity.kt          # Navigation, DI
|   +-- QueryFragment.kt         # Saisie question
|   +-- ResultFragment.kt        # Affichage reponse
|   +-- SettingsFragment.kt      # Choix corpus
|   +-- viewmodels/
|       +-- QueryViewModel.kt    # Etat UI query
|       +-- ResultViewModel.kt   # Etat UI result
|
+-- domain/              # Couche metier
|   +-- usecases/
|       +-- QueryUseCase.kt      # Orchestration requete
|       +-- RetrievalUseCase.kt  # Recherche chunks
|       +-- SynthesisUseCase.kt  # Generation reponse
|
+-- data/                # Couche donnees
|   +-- repositories/
|       +-- ChunkRepository.kt   # Acces chunks
|       +-- IndexRepository.kt   # Acces index FAISS
|   +-- models/
|       +-- Chunk.kt             # Modele chunk
|       +-- Query.kt             # Modele requete
|       +-- Response.kt          # Modele reponse
|
+-- inference/           # Couche ML
|   +-- EmbeddingEngine.kt       # Inference embeddings
|   +-- LLMEngine.kt             # Inference LLM
|   +-- FAISSWrapper.kt          # Wrapper FAISS JNI
```

---

## 6. Flux de donnees

### 6.1 Phase 1 - Indexation (offline, une fois)

```
PDF Reglement
    |
    v
[extract_pdf.py] --> Texte brut
    |
    v
[chunker.py] --> Chunks (500 tokens, 50 overlap)
    |
    v
[embeddings.py] --> Vecteurs 384-dim
    |
    v
[indexer.py] --> corpus_XX.faiss + chunks_XX.json
```

### 6.2 Phase 2+ - Requete (runtime, mobile)

```
Question utilisateur
    |
    v
[EmbeddingEngine] --> Vecteur query 384-dim
    |
    v
[FAISSWrapper] --> Top-K chunks pertinents
    |
    v
[PromptBuilder] --> Prompt avec contexte
    |
    v
[LLMEngine] --> Reponse synthetisee
    |
    v
[CitationExtractor] --> Citations formatees
    |
    v
Affichage ResultFragment
```

---

## 7. Dependances techniques

### 7.1 Pipeline Python (Phase 1)

| Dependance | Version | Usage |
|------------|---------|-------|
| PyMuPDF | >=1.24.0 | Extraction PDF |
| sentence-transformers | >=3.0.0 | Embeddings |
| faiss-cpu | >=1.8.0 | Index vecteur |
| tiktoken | >=0.7.0 | Comptage tokens |
| torch | >=2.2.0 | Backend ML |

### 7.2 Android (Phase 2+)

| Dependance | Version | Usage |
|------------|---------|-------|
| Kotlin | 1.9+ | Langage |
| Jetpack Compose | 1.5+ | UI |
| MediaPipe | 0.10+ | Inference ML |
| TensorFlow Lite | 2.14+ | Runtime modeles |

### 7.3 Modeles ML

| Modele | Taille | Usage |
|--------|--------|-------|
| all-MiniLM-L6-v2 | ~90 MB | Embeddings (fallback) |
| EmbeddingGemma | ~100 MB | Embeddings (cible) |
| Phi-3.5-mini-q4 | ~2 GB | LLM synthese |
| Gemma-2B-q4 | ~1.5 GB | LLM alternative |

---

## 8. Securite et confidentialite

### 8.1 Donnees

- **Aucune donnee personnelle** collectee
- **Aucune connexion reseau** en fonctionnement
- **Stockage local uniquement** (sandbox app)

### 8.2 IA (ISO 42001)

- **Grounding obligatoire** : toute reponse cite sa source
- **Disclaimer visible** : "Aide a la decision, pas decision officielle"
- **Traabilite** : logs locaux des requetes (opt-in)

---

## 9. Metriques qualite

| Metrique | Cible | Mesure |
|----------|-------|--------|
| Couverture tests | >= 60% | pytest-cov |
| Recall retrieval | >= 80% | Jeu de test |
| Hallucination | 0% | Tests adversaires |
| Latence | < 5s | Benchmark device |
| Crash-free | >= 99% | Firebase Crashlytics |

---

## 10. Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-11 | Claude Code | Creation initiale |

---

## 11. Approbations

| Role | Nom | Date | Signature |
|------|-----|------|-----------|
| Tech Lead | | | |
| Product Owner | | | |
