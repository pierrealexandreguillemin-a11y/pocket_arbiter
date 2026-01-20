# Architecture Technique - Pocket Arbiter

> **Document ID**: SPEC-ARCH-001
> **ISO Reference**: ISO/IEC 12207:2017 - Architecture logicielle
> **Version**: 2.0
> **Date**: 2026-01-19
> **Statut**: Approuve
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
|  | - QueryUseCase      |    | - corpus_fr.db      |               |
|  | - RetrievalUseCase  |    | - corpus_intl.db    |               |
|  | - SynthesisUseCase  |    | - chunks_fr.json    |               |
|  +----------+----------+    | - chunks_intl.json  |               |
|             |               | - embeddinggemma.   |               |
|             v               |   tflite            |               |
|  +---------------------+    | - llm.tflite        |               |
|  |   INFERENCE LAYER   |    +---------------------+               |
|  |   (LiteRT/RAG SDK)  |                                          |
|  +---------------------+                                          |
|  | - EmbeddingEngine   |                                          |
|  | - SqliteVectorStore |                                          |
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
|   +-- pipeline/               # Scripts extraction/indexation v4.0
|   |   +-- __init__.py
|   |   +-- extract_docling.py  # Extraction PDF (Docling ML)
|   |   +-- parent_child_chunker.py # Chunking Parent 1024/Child 450
|   |   +-- table_multivector.py # Tables + LLM summaries
|   |   +-- embeddings.py       # Generation embeddings 768D
|   |   +-- embeddings_config.py # Configuration modeles
|   |   +-- export_sdk.py       # Export SQLite (orchestration)
|   |   +-- export_search.py    # Vector search + source_filter + glossary_boost
|   |   +-- export_serialization.py # Serialisation embeddings
|   |   +-- export_validation.py # Validation exports
|   |   +-- query_expansion.py  # Synonymes chess FR + stemmer
|   |   +-- reranker.py         # Cross-encoder (optionnel)
|   |   +-- token_utils.py      # Utilitaires tokens (tiktoken)
|   |   +-- chunk_normalizer.py # Normalisation taille chunks
|   |   +-- utils.py            # Utilitaires communs
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
|   |   +-- gold_standard_fr.json   # Questions test FR (134 Q)
|   |   +-- gold_standard_intl.json # Questions test INTL (25 Q)
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

### 4.2 Fichiers Phase 1 (Pipeline) - Etat actuel

| Fichier | Lignes | Status | Notes |
|---------|--------|--------|-------|
| `extract_pdf.py` | 323 | OK | Extraction PDF |
| `chunker.py` | 400 | OK | Chunking Articles |
| `chunker_article.py` | 93 | OK | Detection boundaries |
| `sentence_chunker.py` | 346 | OK | Chunking phrases |
| `similarity_chunker.py` | 396 | OK | Chunking similarite |
| `semantic_chunker.py` | 432 | OK | Chunking semantique |
| `embeddings.py` | 434 | OK | Generation embeddings |
| `embeddings_config.py` | 90 | OK | Configuration modeles |
| `export_sdk.py` | 382 | OK | Orchestration export |
| `export_search.py` | 284 | OK | Recherche hybride |
| `export_serialization.py` | 66 | OK | Serialisation blobs |
| `export_validation.py` | 128 | OK | Validation exports |
| `token_utils.py` | 44 | OK | Utilitaires tokens |
| `chunk_normalizer.py` | 169 | OK | Normalisation chunks |
| `utils.py` | 198 | OK | Utilitaires communs |

### 4.3 Fichiers futurs - Phase 2 (Android)

| Fichier | Lignes estimees | Decomposition si > 300 |
|---------|-----------------|------------------------|
| `MainActivity.kt` | ~150 | - |
| `QueryFragment.kt` | ~200 | - |
| `ResultFragment.kt` | ~250 | - |
| `QueryViewModel.kt` | ~150 | - |
| `EmbeddingEngine.kt` | ~200 | - |
| `VectorStoreRepository.kt` | ~200 | - |
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
+-- extract_pdf.py       # Extraction texte brut PDF
|   |-- extract_text(pdf_path) -> str
|   |-- extract_metadata(pdf_path) -> dict
|
+-- chunker.py           # Chunking semantique par Articles
|   |-- chunk_by_article(text, max_tokens) -> List[Chunk]
|   |-- chunk_text_legacy(text, max_tokens, overlap) -> List[Chunk]
|   |-- chunk_document(extracted_data, corpus) -> List[Chunk]
|
+-- chunker_article.py   # Detection frontieres Articles
|   |-- detect_article_boundaries(text) -> List[dict]
|   |-- detect_article_match(line) -> str | None
|
+-- similarity_chunker.py # Chunking par similarite cosinus
|   |-- compute_similarity_breaks(sentences, model) -> List[int]
|   |-- chunk_document_similarity(text, model, ...) -> List[Chunk]
|
+-- token_utils.py       # Utilitaires tokens (tiktoken)
|   |-- get_tokenizer() -> tiktoken.Encoding
|   |-- count_tokens(text, tokenizer) -> int
|
+-- chunk_normalizer.py  # Normalisation taille chunks
|   |-- merge_by_max_tokens(chunks, max_tokens) -> List[str]
|   |-- filter_by_min_tokens(chunks, min_tokens) -> List[str]
|   |-- normalize_chunks(chunks, min_tokens, max_tokens) -> List[str]
|
+-- embeddings.py        # Generation vecteurs 768D
|   |-- load_embedding_model(model_id) -> SentenceTransformer
|   |-- embed_query(query, model) -> np.array[768]
|   |-- embed_documents(documents, model) -> np.array[N, 768]
|
+-- embeddings_config.py # Configuration modeles embeddings
|   |-- MODEL_ID, MODEL_ID_FULL, FALLBACK_MODEL_ID
|   |-- PROMPT_QUERY, PROMPT_DOCUMENT (prompts Google officiels)
|
+-- export_sdk.py        # Orchestration export SQLite
|   |-- create_vector_db(chunks, embeddings) -> Report
|   |-- export_corpus(chunks_file, embeddings_file, output_db) -> Report
|   |-- rebuild_fts_index(db_path) -> int
|
+-- export_search.py     # Recherche vectorielle/hybride
|   |-- retrieve_similar(db_path, query_embedding) -> List[Result]
|   |-- search_bm25(db_path, query_text) -> List[Result]
|   |-- retrieve_hybrid(db_path, embedding, text) -> List[Result]
|
+-- export_serialization.py # Serialisation embeddings
|   |-- embedding_to_blob(embedding) -> bytes
|   |-- blob_to_embedding(blob, dim) -> np.array
|
+-- export_validation.py # Validation exports
|   |-- validate_export(db_path, expected_chunks) -> List[str]
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
|       +-- ChunkRepository.kt       # Acces chunks
|       +-- VectorStoreRepository.kt # Acces SqliteVectorStore
|   +-- models/
|       +-- Chunk.kt                 # Modele chunk
|       +-- Query.kt                 # Modele requete
|       +-- Response.kt              # Modele reponse
|
+-- inference/           # Couche ML (Google AI Edge RAG SDK)
|   +-- EmbeddingEngine.kt       # EmbeddingGemma via LiteRT
|   +-- LLMEngine.kt             # Gemma 3 via MediaPipe
|   +-- RagPipeline.kt           # RetrievalAndInferenceChain
```

---

## 6. Flux de donnees

### 6.1 Phase 1 - Indexation (offline, une fois) - v4.0

```
PDF Reglement (29 docs)
    |
    v
[extract_docling.py] --> Texte + Tables (Docling ML)
    |
    v
[parent_child_chunker.py] --> Chunks Parent 1024/Child 450 (15% overlap)
    |                          1454 FR + 764 INTL chunks
    |
    v
[table_multivector.py] --> Table summaries (111 FR)
    |
    v
[embeddings.py] --> Vecteurs 768-dim (EmbeddingGemma-300m)
    |
    v
[export_sdk.py] --> corpus_XX.db (SqliteVectorStore + FTS5)
```

### 6.2 Phase 2+ - Requete (runtime, mobile)

```
Question utilisateur
    |
    v
[EmbeddingEngine] --> Vecteur query 768-dim (EmbeddingGemma TFLite)
    |
    v
[SqliteVectorStore] --> Top-K chunks pertinents (cosine similarity)
    |
    v
[PromptBuilder] --> Prompt avec contexte
    |
    v
[LLMEngine] --> Reponse synthetisee (Gemma 3)
    |
    v
[CitationExtractor] --> Citations formatees
    |
    v
Affichage ResultFragment
```

---

## 7. Dependances techniques

### 7.1 Pipeline Python (Phase 1) - v4.0

| Dependance | Version | Usage |
|------------|---------|-------|
| docling | >=2.31.0 | Extraction PDF ML |
| sentence-transformers | >=3.0.0 | Embeddings 768D |
| tiktoken | >=0.7.0 | Comptage tokens |
| torch | >=2.2.0 | Backend ML |
| numpy | >=1.24.0 | Vecteurs |
| langchain-text-splitters | >=0.4.0 | RecursiveCharacterTextSplitter |
| nltk | >=3.9.0 | Snowball stemmer FR |

### 7.2 Android (Phase 2+)

| Dependance | Version | Usage |
|------------|---------|-------|
| Kotlin | 1.9+ | Langage |
| Jetpack Compose | 1.5+ | UI |
| LiteRT | 1.4.0+ | Runtime TFLite |
| AI Edge RAG SDK | 0.1.0+ | SqliteVectorStore, Embedder |
| MediaPipe GenAI | 0.10.22+ | LLM Inference |

### 7.3 Modeles ML

| Modele | Taille | RAM CPU | Usage |
|--------|--------|---------|-------|
| EmbeddingGemma-300m | 179 MB | 110 MB | Embeddings (principal) |
| Gecko-110m-en | 114 MB | 126 MB | Embeddings (fallback) |
| Gemma 3 1B | ~600 MB | ~400 MB | LLM synthese |
| Gemma 3 270M | ~200 MB | ~150 MB | LLM alternative (leger) |

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

| Metrique | Cible | Actuel | Mesure |
|----------|-------|--------|--------|
| Couverture tests | >= 80% | **87%** | pytest-cov |
| Recall retrieval FR | >= 90% | **97.06%** | Gold standard v5.7 |
| Recall retrieval INTL | >= 70% | **80.00%** | Gold standard |
| Hallucination | 0% | TBD | Tests adversaires |
| Latence | < 5s | TBD | Benchmark device |
| Crash-free | >= 99% | TBD | Firebase Crashlytics |

---

## 10. Historique

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-11 | Claude Code | Creation initiale |
| 1.1 | 2026-01-14 | Claude Code | Migration FAISS -> SqliteVectorStore, EmbeddingGemma-300m |
| 1.2 | 2026-01-15 | Claude Code | Ajout modules Phase 1A (chunkers, search, token_utils, chunk_normalizer) |
| 2.0 | 2026-01-19 | Claude Opus 4.5 | **v4.0**: Docling ML, Parent-Child chunking, Recall 97.06%, source_filter, glossary_boost |

---

## 11. Approbations

| Role | Nom | Date | Signature |
|------|-----|------|-----------|
| Tech Lead | | | |
| Product Owner | | | |
