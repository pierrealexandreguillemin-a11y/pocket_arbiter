# Roadmap Projet - Pocket Arbiter

> **Document ID**: PLAN-RDM-001
> **ISO Reference**: ISO/IEC 12207:2017
> **Version**: 2.0
> **Date**: 2026-01-29
> **Effort total estime**: 215h (~14-16 semaines)

---

## Vue d'ensemble

```
PHASE 0 ✅ COMPLETE
    │
    ▼
PHASE 1A ✅ COMPLETE + ENHANCED ──► Pipeline v6.0 Dual-Mode (Mode A: 2540 FR, Mode B: 1331 FR)
    │
    ▼
PHASE 1B ✅ COMPLETE ──► Pipeline Embed + Index (Recall FR 91.17%, INTL 93.22%)
    │
    ▼
PHASE 1C (OPTIONAL) ──► Fine-tuning MRL+LoRA EmbeddingGemma (+5-15% recall)
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
| `scripts/pipeline/chunker_hybrid.py` | Mode A: HybridChunker + Parent 1024/Child 450 | ✅ NEW v6.0 |
| `scripts/pipeline/chunker_langchain.py` | Mode B: LangChain + section fusion + Parent 1024/Child 450 | ✅ NEW v6.0 |
| `scripts/pipeline/table_multivector.py` | Tables + LLM summaries | ✅ NEW |
| `corpus/processed/chunks_mode_a_fr.json` | Mode A: 2540 chunks FR (HybridChunker) | ✅ |
| `corpus/processed/chunks_mode_b_fr.json` | Mode B: 1331 chunks FR (LangChain+fusion) | ✅ |
| `corpus/processed/chunks_mode_a_intl.json` | Mode A: 1412 chunks INTL | ✅ |
| `corpus/processed/chunks_mode_b_intl.json` | Mode B: 866 chunks INTL | ✅ |

### Evolutions vs Roadmap Original

| Aspect | Roadmap Original | Etat Actuel | Deviation |
|--------|-----------------|-------------|-----------|
| Extraction | `extract_pdf.py` (PyMuPDF) | `extract_docling.py` (Docling ML) | UPGRADE |
| Chunking | `chunker.py` 256 tokens | **Dual-mode v6.0**: HybridChunker (A) + LangChain (B) | UPGRADE |
| Parent/Child | N/A | Parent 1024t / Child 450t / 15% overlap | NEW (NVIDIA 2025) |
| Tokenizer | tiktoken | EmbeddingGemma (unifié chunking+embedding) | UPGRADE |
| Mode A FR | N/A | 2540 chunks (HybridChunker preserves structure) | NEW |
| Mode B FR | N/A | 1331 chunks (LangChain + section fusion) | NEW |
| Mode A INTL | N/A | 1412 chunks | NEW |
| Mode B INTL | N/A | 866 chunks | NEW |
| Page provenance | N/A | 100% coverage (ISO 42001 A.6.2.2) | NEW |

### Resultats

| Critere | Cible | Reel | Statut |
|---------|-------|------|--------|
| PDFs FR extraits | 28 | 28 | ✅ |
| PDFs INTL extraits | 1 | 1 | ✅ |
| Chunks FR | ~500 | 1827 | ✅ |
| Chunks INTL | ~100 | 974 | ✅ |
| Sections coverage | 95%+ | 99.9%+ | ✅ |
| Coverage tests | 80% | 87% | ✅ |
| Complexite cyclomatique | B max | B | ✅ |

> **Note**: Pipeline v5.0 avec MarkdownHeaderTextSplitter pour couverture sections 99.9%+.
> Total: 2801 chunks (1827 FR + 974 INTL).

### Specifications
- Voir `docs/specs/PHASE1A_SPECS.md`
- Schema: `docs/CHUNKING_STRATEGY.md`

---

## Phase 1B - Pipeline Embed + Export SDK (COMPLETE)

**Duree**: Semaine 3-4 | **Effort**: 30h | **Statut**: ✅ COMPLETE

### Objectif
Generer embeddings et exporter au format Google AI Edge RAG SDK.

### Stack validee (2026-01-22)

| Composant | Choix | Specs | Source |
|-----------|-------|-------|--------|
| Embedding | **EmbeddingGemma-300M (full)** | 768D, 308M params, full precision | [HuggingFace](https://huggingface.co/google/embeddinggemma-300m) |
| Tokenizer | **EmbeddingGemma tokenizer** | Unifié chunking + embedding (ISO 42001) | Google |
| Search | **Vector-only** | Cosine similarity (optimal apres audit) | Custom |
| Vector Store | **SqliteVectorStore** | SDK natif, persistant, FTS5 | [Google AI Edge](https://github.com/google-ai-edge/ai-edge-apis) |

> **Note**: Hybrid search (BM25+Vector) teste mais moins performant (89.46% vs 97.06%).

### Deliverables

| Fichier | Description | Statut |
|---------|-------------|--------|
| `scripts/pipeline/embeddings.py` | Generation vecteurs 768D | ✅ |
| `scripts/pipeline/export_sdk.py` | Export format SqliteVectorStore | ✅ |
| `scripts/pipeline/export_search.py` | Hybrid BM25+Vector+RRF + glossary boost | ✅ ENHANCED |
| `corpus/processed/corpus_mode_b_fr.db` | Base SQLite FR (9.8 MB) - ACTIF | ✅ |
| `corpus/processed/corpus_mode_a_intl.db` | Base SQLite INTL (OBSOLETE) | ⏸️ A reconstruire |

### Resultats (2026-01-21)

| Critere | Cible | Reel | Statut |
|---------|-------|------|--------|
| Recall FR | ≥ 90% | **91.17%** (420+ Q v7, smart_retrieve, tol=2) | ✅ PASS |
| Recall INTL | ≥ 70% | **93.22%** (43 Q, vector, tol=2) | ✅ PASS |
| Gold Standard FR | ≥ 50 Q | **420+ questions v7** (46 hard cases) | ✅ ISO 29119 |
| Gold Standard INTL | ≥ 30 Q | **43 questions** (12 hard cases) | ✅ ISO 29119 |
| DB size total | < 100 MB | **~15 MB** | ✅ PASS |
| Coverage tests | ≥ 80% | **87%** | ✅ PASS |

### Definition of Done

| Critere | Cible | Bloquant | Statut |
|---------|-------|----------|--------|
| Recall FR | ≥ 90% | OUI | ✅ **91.17%** (420+ Q v7) |
| Recall INTL | ≥ 70% | NON | ✅ **93.22%** (43 Q) |
| Gold Standard | ≥ 50 Q FR + 30 Q INTL | OUI | ✅ **463+ questions** |
| 0% hallucination adversarial | 30/30 pass | OUI | PENDING (Phase 3) |
| DB size | < 100 MB | NON | ✅ ~15 MB |
| Coverage tests | ≥ 80% | OUI | ✅ 87% |

---

## Phase 1C - Fine-tuning EmbeddingGemma (OPTIONAL)

**Duree**: 1-2 jours | **Effort**: 10h | **Statut**: PLANNED

### Objectif
Ameliorer recall sur hard cases via fine-tuning domain-specific MRL + LoRA.

### Justification
- 46 hard cases FR (30.6% des questions)
- 12 hard cases INTL (27.9% des questions)
- Gain attendu: +5-15% recall sur hard cases

### Stack

| Composant | Choix | Source |
|-----------|-------|--------|
| Model | EmbeddingGemma-300M | [Google AI](https://ai.google.dev/gemma/docs/embeddinggemma) |
| Framework | Sentence Transformers v3 | [SBERT](https://sbert.net/) |
| Loss | CachedMultipleNegativesRankingLoss | Best for retrieval |
| MRL | Truncation 768→256D | 3x compression |

### Deliverables

| Fichier | Description |
|---------|-------------|
| `docs/research/LORA_FINETUNING_GUIDE.md` | Guide complet pre-notebook | ✅ DONE |
| `notebooks/finetune_embeddinggemma.ipynb` | Notebook Kaggle/Colab | TODO |
| `models/embeddinggemma-chess-fr/` | Modele fine-tune | TODO |

### Definition of Done

| Critere | Cible | Bloquant |
|---------|-------|----------|
| Recall FR | ≥ 95% | NON |
| Hard cases FR | < 25/150 | NON |
| Overfitting | Val loss stable | OUI |

### Documentation
- Guide complet: `docs/research/LORA_FINETUNING_GUIDE.md`
- Sources: `docs/ISO_VECTOR_SOLUTIONS.md`

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

### Metriques RAGAs (a implementer)

| Metrique | Description | Cible |
|----------|-------------|-------|
| **Faithfulness** | Reponse fidele aux chunks recuperes | >= 90% |
| **Answer Relevancy** | Reponse pertinente a la question | >= 85% |
| **Context Precision** | Chunks recuperes pertinents | >= 80% |

> **Note**: RAGAs necessite `expected_answer` dans le gold standard.
> A ajouter lors de l'implementation Phase 3.
> Ref: [RAGAs Framework](https://github.com/explodinggradients/ragas)

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

| Phase | Metrique | Cible | Reel | Bloquant |
|-------|----------|-------|------|----------|
| 1B | Recall FR | ≥ 90% | **91.17%** (420+ Q v7) | OUI ✅ |
| 1B | Recall INTL | ≥ 70% | **93.22%** (43 Q) | NON ✅ |
| 1B | Gold Standard | ≥ 80 Q | **463+ questions** (420+ FR + 43 INTL) | OUI ✅ |
| 1C | Recall FR (post fine-tune) | ≥ 95% | TBD | NON |
| 3 | Hallucination | 0% | TBD | OUI |
| 3 | Fidelite | ≥ 85% | TBD | OUI |
| 4 | RAM | < 500MB | TBD | OUI |
| 4 | Latence totale | < 5s | TBD | NON |
| 5 | NPS | ≥ 7/10 | TBD | NON |

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

## Findings & Ameliorations (2026-01-21)

### Pipeline v5.0

**Chunking ameliore** avec `MarkdownHeaderTextSplitter` pour couverture sections 99.9%+:
- Voir: `docs/CHUNKING_STRATEGY.md`
- Total: 2801 chunks (1827 FR + 974 INTL)

### Gold Standard v7

| Corpus | Questions | Hard Cases | Documents | Version |
|--------|-----------|------------|-----------|---------|
| FR | 420+ | 46 (30.6%) | 28 | v7 |
| INTL | 43 | 12 (27.9%) | 1 | v2.0 |
| **Total** | **463+** | **58** | **29** | - |

> Voir: `docs/research/AUDIT_GS_v5.25_2026-01-20.md`

### Gold Standard v6.0 - Annales-Based (✅ COMPLETE)

Pipeline d'extraction des annales DNA (examens officiels arbitres FFE):

| Statut | Description | Delivrable |
|--------|-------------|------------|
| ✅ | Extraction Docling (13 sessions 2018-2025) | `corpus/processed/annales_all/` |
| ✅ | Parser avec taxonomie standard | `scripts/evaluation/annales/parse_annales.py` |
| ✅ | Mapping articles → corpus | `scripts/evaluation/annales/map_articles_to_corpus.py` |
| ✅ | Génération Gold Standard v6 | `tests/data/gold_standard_annales_fr.json` |
| ✅ | Reformulation langage courant | `scripts/evaluation/annales/reformulate_questions.py` |
| ✅ | Validation réponses + pages | `scripts/evaluation/annales/validate_answers.py` |

**Statistiques actuelles** (13 sessions):

| Métrique | Valeur | Notes |
|----------|--------|-------|
| Questions extraites | 692 | Sessions 2018-2025 |
| **Questions Gold Standard v6** | **518** | Avec corrections + mapping |
| Mapping rate | 74.9% | 518/692 |
| Sessions | 11 | dec2018 à jun2025 |
| UV couvertes | UVR, UVC, UVO, UVT | 100% couverture |
| Documents corpus liés | 9 | LA, R01-R03, A02, C01, C03, J01 |
| **Page references** | **81%** | 420/518 questions avec pages |
| **Reformulation** | **24.5%** | Réduction longueur moyenne |
| Query variants | 1-3/question | Pour tests retrieval robustesse |

**Distribution par difficulté** (450 questions avec taux réussite):
- Easy (<30%): 273 questions
- Medium (30-60%): 129 questions
- Hard (>60%): 48 questions

**Taxonomie standard** (industry best practices):

| Field | Distribution |
|-------|--------------|
| `question_type` | factual (55.7%), scenario (40.9%), procedural (2.2%), comparative (1.2%) |
| `cognitive_level` | RECALL (55.7%), APPLY (40.9%), UNDERSTAND (2.2%), ANALYZE (1.2%) |
| `reasoning_type` | multi-hop (50.6%), single-hop (46.7%), temporal (2.6%) |
| `answer_type` | multiple_choice (66.7%), extractive (21.5%), abstractive (11.8%) |

**Reformulation par type** (question detection):

| Type | Count | % |
|------|-------|---|
| scenario | 283 | 54.6% |
| unknown | 151 | 29.2% |
| direct | 42 | 8.1% |
| instruction | 23 | 4.4% |
| selection | 10 | 1.9% |
| statement | 5 | 1.0% |
| conditional | 3 | 0.6% |
| reference | 1 | 0.2% |

**Couverture corpus FR**:

| Catégorie | Documents | Questions | % |
|-----------|-----------|-----------|---|
| Core Rules (LA) | 1/1 | 403 | 78.3% |
| Règlements FFE (R01-R03) | 3/3 | 53 | 10.3% |
| Championnats (A02) | 1/3 | 32 | 6.2% |
| Coupes (C01, C03) | 2/3 | 19 | 3.7% |
| Jeunes (J01) | 1/3 | 2 | 0.4% |
| **Total** | **8/28** | **515** | **100%** |

> Note: Documents non couverts (Admin/Legal, Regional, Féminin, Handicap) = hors scope examens nationaux

> Voir: `docs/specs/GOLD_STANDARD_V6_ANNALES.md`

### Benchmark Search Modes (150 Q FR)

| Mode | Recall@5 FR | Failed | Conclusion |
|------|-------------|--------|------------|
| **smart_retrieve** | **91.17%** | 14/420+ | ✅ OPTIMAL (prod) |
| **+ glossary_boost** | 92%+ | ~12/150 | ✅ Definitions |
| Vector-only | 89% | 16/150 | Baseline |

### Analyse 14 echecs FR

| Cause Racine | Questions | % |
|--------------|-----------|---|
| Langage oral/informel | Q95, Q98, Q103 | 21% |
| Cross-chapter content | Q85, Q86, Q132 | 21% |
| Mismatch terminologique | Q77, Q94 | 14% |
| Abreviations | Q98, Q119 | 14% |

> Voir: `docs/research/RECALL_FAILURE_ANALYSIS_2026-01-20.md`

### Ameliorations Non Planifiees

| Module | Role | ISO Ref |
|--------|------|---------|
| `chunker.py` | MarkdownHeader + Parent-Child (NVIDIA 2025) | ISO 25010 |
| `table_multivector.py` | Tables + LLM summaries (185 tables) | ISO 42001 |
| `export_search.py` | smart_retrieve + glossary_boost | ISO 25010 |
| `LORA_FINETUNING_GUIDE.md` | Guide fine-tuning MRL+LoRA | ISO 42001 |

### Metriques Finales Phase 1B

| Corpus | Chunks | Recall@5 (tol=2) | Hard Cases | ISO Status |
|--------|--------|------------------|------------|------------|
| FR | 1827 | **91.17%** | 46/420+ | ✅ PASS (≥90%) |
| INTL | 974 | **93.22%** | 12/43 | ✅ PASS (≥70%) |
| **Total** | **2801** | - | **58/463+** | ✅ |

### Plan Amelioration (Phase 1C)

| Action | Gain Attendu | Effort | Priorite |
|--------|--------------|--------|----------|
| Fine-tuning MRL+LoRA | +5-15% hard cases | 10h | P0 |
| Synonymes index-time | +3% | 1h | P1 |
| Abreviations expandues | +1% | 1h | P1 |

> Voir: `docs/research/LORA_FINETUNING_GUIDE.md`, `docs/ISO_VECTOR_SOLUTIONS.md`

---

## Effort par phase

| Phase | Effort | Cumule |
|-------|--------|--------|
| 1A | 20h | 20h |
| 1B | 30h | 50h |
| 1C (optional) | 10h | 60h |
| 2A | 40h | 100h |
| 2B | 30h | 130h |
| 3 | 40h | 170h |
| 4A | 10h | 180h |
| 4B | 15h | 195h |
| 5 | 20h | 215h |
| **TOTAL** | **215h** | |

---

## Documents associes

| Document | Description |
|----------|-------------|
| `docs/specs/PHASE1A_SPECS.md` | Specifications Phase 1A |
| `docs/CHUNK_SCHEMA.md` | Schema JSON chunks |
| `docs/CHUNKING_STRATEGY.md` | Strategie chunking v5.0 |
| `docs/AI_POLICY.md` | Politique IA ISO 42001 |
| `docs/QUALITY_REQUIREMENTS.md` | Exigences qualite ISO 25010 |
| `docs/TEST_PLAN.md` | Plan de tests ISO 29119 |
| `docs/ISO_VECTOR_SOLUTIONS.md` | Solutions vector-based RAG |
| `docs/research/LORA_FINETUNING_GUIDE.md` | Guide fine-tuning MRL+LoRA |
| `docs/research/RECALL_FAILURE_ANALYSIS_2026-01-20.md` | Analyse 14 echecs recall |

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-14 | Creation initiale |
| 1.1 | 2026-01-14 | Phase 1A complete (2047 FR + 1105 INTL chunks) |
| 1.2 | 2026-01-14 | Stack Phase 1B: FAISS -> SqliteVectorStore, EmbeddingGemma-300m specs |
| 1.3 | 2026-01-19 | Phase 1A enhanced (Docling, Parent-Child), Phase 1B 95% (Recall FR 86.76%), Findings section |
| 1.4 | 2026-01-19 | **Phase 1B COMPLETE** - Recall FR 97.06% (gold standard v5.7 audit, 23 corrections), vector-only optimal |
| 1.5 | 2026-01-19 | **source_filter** - Recall FR 100% avec filtrage document (FR-Q18, FR-Q50 resolus) |
| 1.6 | 2026-01-19 | **glossary_boost** - Boost x3.5 glossaire DNA 2025 pour questions definition |
| 1.7 | 2026-01-21 | **Pipeline v5.0** - 2801 chunks (1827 FR + 974 INTL), gold standard 193 Q, Phase 1C fine-tuning MRL+LoRA |
| 1.8 | 2026-01-22 | **Pipeline v6.0 Dual-Mode** - Mode A HybridChunker (2540 FR), Mode B LangChain+fusion (1331 FR), EmbeddingGemma 308M full, 100% page provenance |
| 1.9 | 2026-01-23 | **Benchmark Chunking Optimizations** - Dual-size 81.72% (-5.22%), Semantic 82.89% (-4.05%) vs Baseline 86.94%. Recommandation: conserver baseline 450t single-size. |
| 2.0 | 2026-01-29 | GS v7 (420+ FR), reranker supprime, pyproject.toml, metriques a jour |
