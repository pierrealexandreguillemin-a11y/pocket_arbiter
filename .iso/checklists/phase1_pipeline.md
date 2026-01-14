# Checklist Phase 1 - Pipeline de donnees

> **Document ID**: CHK-PIPE-001
> **ISO Reference**: ISO/IEC 12207:2017 - Processus de developpement
> **Version**: 1.1
> **Date**: 2026-01-14
> **Statut**: En cours

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

## Phase 1B - Embed + Index (EN ATTENTE)

### Pre-requis
- [ ] Phase 1A complete (Gate: corpus_processed = True)
- [ ] Specs embeddings documentees
- [ ] Modele embedding valide localement
- [ ] `tests/data/adversarial.json` cree (30 questions)

### embeddings.py
- [ ] Script executable sans erreur
- [ ] Modele embeddings choisi et documente (EmbeddingGemma-300M)
- [ ] Output : fichier numpy
- [ ] Performance mesuree (temps/chunk)
- [ ] Test de qualite embeddings

### indexer.py
- [ ] Script executable sans erreur
- [ ] Index FAISS cree
- [ ] Test de retrieval basique
- [ ] Metriques de recall documentees

### export_android.py
- [ ] Script executable sans erreur
- [ ] Format `<chunk_splitter>` conforme SDK
- [ ] Export compatible Google AI Edge SDK

### Validation Phase 1B

| Critere | Cible | Bloquant |
|---------|-------|----------|
| Recall FR | >= 80% | OUI |
| Recall INTL | >= 70% | NON |
| 0% hallucination adversarial | 30/30 pass | OUI |
| Index < 50 MB | Mesure | NON |

---

## Validation finale Phase 1

- [x] Phase 1A complete
- [ ] Phase 1B complete
- [ ] Pipeline complet : PDF -> Index
- [ ] Tous les scripts testes
- [ ] requirements.txt a jour
- [ ] Documentation dans docs/
- [ ] Recall >= 80% sur test set

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-11 | Creation initiale |
| 1.1 | 2026-01-14 | Phase 1A complete, metriques ajoutees |
