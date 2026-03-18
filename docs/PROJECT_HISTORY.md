# Pocket Arbiter — Project History

Chronologie factuelle des decisions et errements du projet.

## Ere 1 : Fondations (11-25 janvier 2026)

- Phase 0 : structure projet, ISO compliance, VISION, ARCHITECTURE, AI_POLICY
- Phase 1A : extraction PDF (docling), chunking (LangChain parent-child), embeddings (EmbeddingGemma-300M)
- Pipeline produit : 1857 chunks children (median 109 tokens), 1394 parents, 111 table summaries
- Recall mesure a 82.84% page-level tolerance=2 (sur 68 questions, ancien GS)
- Specs pipeline creees et modifiees 13-19 fois sans relancer le pipeline
- Probleme : parents et table summaries jamais integres dans l'index de recherche

## Ere 2 : Errements GS (28 janvier — 27 fevrier 2026)

- gs_scratch : tentative GS from scratch (584Q) → abandonne (71.5% garbage, templates mecaniques)
- 6 passes automatiques de correction chunk_ids sur le GS annales (152+101+25+290+238+7 fixes)
- Phase A : corrections metadata, regeneration 80 questions (P2), GO/NO-GO
- Phase B : tentative coverage → scripts supprimes (broken)
- Fine-tuning EmbeddingGemma tente → echec (recall degrade 82.84% → 65.69%, mauvais triplets)
- Pivot : gs_scratch abandonne, annales adopte comme GS primaire

## Ere 3 : Baseline et nettoyage (27-28 fevrier 2026)

- Baseline recall v8 mesure : 36.6% recall@5 chunk (vector-only), 37.2% (hybrid)
- Nettoyage Q/choices mismatch : 122 questions supprimees (bug extraction docling cross-UV)
- GS v9 : 403 questions (264 annales + 40 human + 99 adversarial)
- Baseline jamais re-mesure sur les 304 testables post-cleanup

## Ere 4 : Menage et redemarrage (mars 2026)

- Diagnostic : chunks trop petits, parents absents de l'index, table summaries non embedees
- Verification GS : chunk mappings corrects (30/30 sample), classifications fausses (answer_type 100% "multiple_choice")
- Archivage : 168 scripts, 46 docs, 19 artefacts data
- Cap : fix pipeline → re-mesure recall → decision (fine-tuning ou prompt engineering)

## Ere 5 : Pipeline v2 — chunker custom (mars 2026)

- Chantier 1 (menage) : DONE
- Chantier 2 (pipeline v2) : extract, chunker custom, indexer, search hybrid
  - Chunker custom : 350 lignes, regex maison, 3 bugs integrite decouverts au recall
  - Bug 1 : 7 parents avec texte mais 0 children (8020 tokens invisibles)
  - Bug 2 : 28 parents root vides (p000) avec 46 children orphelins
  - Bug 3 : parents geants (39K tokens, 55 children) — heading mal detecte
  - Recall baseline custom chunker : 28.2% → 62.1% apres 3 fixes (FTS5, adaptive_k, empty-parent)
- Search : hybrid cosine+BM25, RRF k=60, adaptive-k largest-gap (EMNLP 2025)
- Synonymes : 70 entries corpus-verified (A: intra-corpus + B: langage courant→corpus)

## Ere 6 : Pipeline v2 — chunker LangChain (mars 2026)

- Chunker custom remplace par LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
- Raison : chunker custom ignorait langchain-text-splitters (installe dans requirements.txt, prevu par ARCHITECTURE.md)
- Standards appliques : FloTorch 2026 (512 tok), Azure 2026 (20% overlap), NVIDIA (header-first), KX (table CCH)
- Table extraction AVANT header split (verifie : 98 tables LA, 22 seulement apres header split)
- 9 integrity gates (I1-I9) dans integrity.py
- Findings durant le re-chunking :
  - MarkdownHeaderTextSplitter fragmente les tables → extraction avant split
  - Page interpolation par CCH echoue (headings h4+ pas dans metadata) → line-level tracking
  - build_parents dict section ecrase sub-parents → index mapping
  - Heading-only Documents creent micro-chunks → pre-filter
  - CCH table summaries : indexer doit utiliser heading du chunker, pas premiers mots summary

## Findings techniques (mars 2026)

### Modele embedding
- EmbeddingGemma-300M QAT (sentence-transformers) ≠ .tflite (litert-community)
- Checkpoints differents : QAT entraine avec quantization awareness, .tflite converti depuis base
- Solution build-time : utiliser meme .tflite via tflite_runtime (pas de wheel Windows/Py3.13)
- Solution pragmatique : base model pour build, .tflite pour Android (meme source, delta quantization)
- EmbeddingGemma reste le seul modele on-device <500MB avec .tflite natif

### Modele generation
- Gemma 3n E2B : 2GB RAM (depasse spec 500MB VISION.md)
- Gemma 3 270M : 150MB RAM (dans spec) mais qualite inferieure
- Decision a prendre : relever contrainte RAM ou garder 270M

### Android / LiteRT
- AI Edge RAG SDK : deprecie, migration LiteRT-LM recommandee
- LiteRT-LM v0.1.0 : early preview, API peut changer
- Snowball FR : pas de port Kotlin officiel, utiliser org.tartarus.snowball (Java)
- FTS5 : disponible Android API 24+ (minSdk 26 = OK)

## Gemma model status (mars 2026)

- **Gemma 3n** (juin 2025) : nouveau modele mobile-first, MatFormer architecture
  - E2B : ~2 GB RAM, effective 2B params — candidat remplacement Gemma 3 270M
  - E4B : ~3 GB RAM, effective 4B params — plus capable
- **EmbeddingGemma-300M** : toujours le seul modele d'embeddings on-device Google, pas de successeur
- **LiteRT-LM** : successeur recommande pour inference LLM on-device

## Decisions cles

| Date | Decision | Raison |
|------|----------|--------|
| 18 jan | EmbeddingGemma-300M pour embeddings | Recommande Google pour RAG, 82.84% recall base |
| 18 jan | Gemma 3 270M IT pour generation | Plus petit Gemma 3, compatible MediaPipe, 200MB |
| 27 fev | gs_scratch abandonne | 71.5% garbage, qualite inacceptable |
| 27 fev | Annales = GS primaire | 264 vraies questions d'examen FFE, chunk_match verifie |
| 16 mar | Archivage scripts pipeline | Ont produit l'etat casse, plus simple de reecrire |
| 16 mar | AI Edge RAG SDK rejete | Deprecie, pas de pre-build desktop, pas de parent-child |
| 16 mar | Pipeline custom confirme | Python desktop (pre-build DB) + Kotlin Android (query) |
| 18 mar | Chunker custom → LangChain | Custom avait 3 bugs integrite, LangChain etait deja installe |
| 18 mar | Table extraction avant header split | Verifie empiriquement : 98 vs 22 tables |
| 18 mar | Page interpolation line-level | CCH match echouait pour h4+ headings |
