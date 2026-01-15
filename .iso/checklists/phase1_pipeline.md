# Checklist Phase 1 - Pipeline de donnees

> **Document ID**: CHK-PIPE-001
> **ISO Reference**: ISO/IEC 12207:2017 - Processus de developpement
> **Version**: 1.4
> **Date**: 2026-01-15
> **Statut**: Phase 1B INCOMPLETE - Recall 27% < 80% requis

---

## Phase 1A - Extract + Chunk (COMPLETE)

### Pre-requis
- [x] Specs extraction PDF documentees (`docs/specs/PHASE1A_SPECS.md`)
- [x] Specs chunking documentees (`docs/CHUNK_SCHEMA.md`)
- [x] Tests planifies dans TEST_PLAN.md
- [x] Corpus PDF present et inventorie (28 FR + 1 INTL)

### extract_pdf.py
- [x] Script executable sans erreur
- [x] Arguments CLI documentes (--help)
- [x] Gere tous les PDF du corpus (28 FR + 1 INTL)
- [x] Output : fichiers JSON structures
- [x] Logs informatifs (progression)
- [x] Tests unitaires (4 tests)
- [x] Gestion erreurs (PDF corrompu, fichier inexistant)

### chunker.py
- [x] Script executable sans erreur
- [x] Arguments CLI documentes
- [x] Chunking 256 tokens max
- [x] Overlap 50 tokens (20%)
- [x] Metadonnees preservees (source, page, section)
- [x] Output : JSON conforme CHUNK_SCHEMA.md
- [x] Tests unitaires (13 tests)
- [x] Limite ISO 82045 : 512 tokens max, 50 chars min

### Metriques Phase 1A

| Metrique | Cible | Reel | Statut |
|----------|-------|------|--------|
| PDFs FR extraits | 28 | 28 | OK |
| PDFs INTL extraits | 1 | 1 | OK |
| Chunks FR | ~500 | 2047 | OK |
| Chunks INTL | ~100 | 1105 | OK |
| Coverage tests | 80% | 93% | OK |
| Complexite | B max | B | OK |

### Approbation Phase 1A

| Role | Date | Validation |
|------|------|------------|
| Dev | 2026-01-14 | Commit 0e8b52d |
| Review | 2026-01-14 | Pre-push hooks OK |

---

## Phase 1B - Embed + Export SDK (INCOMPLETE - Recall Blocking)

### Pre-requis
- [x] Phase 1A complete (Gate: corpus_processed = True)
- [x] Specs embeddings documentees (PROJECT_ROADMAP.md v1.2)
- [x] Modele embedding valide localement (multilingual-e5-small fallback)
- [x] `tests/data/adversarial.json` cree (30 questions)

### Stack technique

| Composant | Production | Fallback (actuel) | Specs |
|-----------|------------|-------------------|-------|
| Embedding | EmbeddingGemma-300m | multilingual-e5-small | 768D / 384D |
| Vector Store | SqliteVectorStore | SqliteVectorStore | SQLite + BLOB |
| Recall cible | >= 80% | >= 20% | ISO 25010 |

### embeddings.py
- [x] Script executable sans erreur
- [x] Modele fallback : `intfloat/multilingual-e5-small` via sentence-transformers
- [x] Output : fichier numpy (.npy) 384D
- [x] Performance mesuree (103ms/chunk FR, 123ms/chunk INTL)
- [x] Tests unitaires (30 tests)

### export_sdk.py
- [x] Script executable sans erreur
- [x] Export SqliteVectorStore compatible SDK
- [x] Output : corpus_fr.db (8.0 MB), corpus_intl.db (4.25 MB)
- [x] Test de retrieval basique (top-5)
- [x] Metriques de recall documentees

### test_recall.py
- [x] Benchmark recall implementee
- [x] Validation adversariale (30/30 pass)
- [x] Seuil adaptatif selon modele (80% prod / 20% fallback)

### Metriques Phase 1B

| Metrique | Cible | Reel | Statut |
|----------|-------|------|--------|
| Embeddings FR | 2047 | 2047 | OK |
| Embeddings INTL | 1105 | 1105 | OK |
| Embedding dim | 768D (prod) / 384D (fallback) | 384D | OK (fallback) |
| DB size FR | < 50 MB | 8.0 MB | OK |
| DB size INTL | < 30 MB | 4.25 MB | OK |
| Recall FR (fallback) | >= 20% | 25.33% | OK |
| Adversarial | 30/30 pass | 30/30 | OK |
| Coverage tests | >= 80% | 84% | OK |
| Lint errors | 0 | 0 | OK |
| Type hints | 100% | 100% | OK |

### Validation Phase 1B

| Critere | Cible | Reel | Bloquant | Statut |
|---------|-------|------|----------|--------|
| Recall FR | >= 80% | 27% | OUI | **FAIL** |
| 0% hallucination | 30/30 | A retester | OUI | EN ATTENTE |
| DB size total | < 100 MB | 12.25 MB | NON | PASS |
| Coverage tests | >= 80% | 84% | OUI | PASS |

> **BLOQUANT**: Le recall de 27% (meme avec modele 768D) est insuffisant.
> Causes identifiees: encodage UTF-8 corrompu, modele non-adapte FR, mismatch semantique.
> Voir `docs/PHASE1B_REMEDIATION_PLAN.md` pour le plan de correction.

---

## Validation finale Phase 1

- [x] Phase 1A complete
- [ ] Phase 1B complete - **BLOQUANT: Recall 27% < 80%**
- [x] Pipeline complet : PDF -> SqliteVectorStore
- [x] Tous les scripts testes (>= 80% coverage)
- [ ] requirements.txt a jour
- [x] Documentation dans docs/ (ARCHITECTURE.md, PROJECT_ROADMAP.md)
- [ ] Recall >= 80% sur test set FR - **BLOQUANT**
- [ ] Test adversarial conforme ISO 42001 - **EN ATTENTE**

---

## Fichiers generes

| Fichier | Taille | Description |
|---------|--------|-------------|
| `corpus/processed/embeddings_fr.npy` | 3.0 MB | 2047 x 384 embeddings FR |
| `corpus/processed/embeddings_intl.npy` | 1.7 MB | 1105 x 384 embeddings INTL |
| `corpus/processed/corpus_fr.db` | 8.0 MB | Vector DB FR |
| `corpus/processed/corpus_intl.db` | 4.25 MB | Vector DB INTL |

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-11 | Creation initiale |
| 1.1 | 2026-01-14 | Phase 1A complete, metriques ajoutees |
| 1.2 | 2026-01-14 | Phase 1B: FAISS -> SqliteVectorStore, EmbeddingGemma-300m |
| 1.3 | 2026-01-15 | Phase 1B fallback model (multilingual-e5-small) - ERREUR |
| 1.4 | 2026-01-15 | CORRECTION: Phase 1B marquee INCOMPLETE, recall 27% < 80% bloquant |
