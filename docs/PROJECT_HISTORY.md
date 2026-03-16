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

## Gemma model status (mars 2026)

Recherche web effectuee le 16 mars 2026 :

- **Gemma 3n** (juin 2025) : nouveau modele mobile-first, MatFormer architecture
  - E2B : ~2 GB RAM, effective 2B params — candidat remplacement Gemma 3 270M
  - E4B : ~3 GB RAM, effective 4B params — plus capable
  - Multimodal (texte, image, video, audio), 32K context, 140+ langues
  - ~1.5x plus rapide que Gemma 3 4B avec meilleure qualite
  - LiteRT checkpoints disponibles sur HuggingFace
- **AI Edge RAG SDK** : pipeline officiel Google pour RAG on-device Android — **DEPRECIE**
  - Pas de pre-build desktop (JNI Android-only), pas de parent-child natif
  - Pattern `customEmbeddingData` interessant : embed child, return parent — a implementer nous-memes
  - Google recommande migration vers LiteRT-LM (qui n'a PAS de composants RAG)
  - Decision : ne pas utiliser le SDK, garder notre pipeline custom (SQLite + embeddings)
- **EmbeddingGemma-300M** : toujours le seul modele d'embeddings on-device Google, pas de successeur
- **Gemma 3 QAT** : modeles QAT disponibles pour 1B, 4B, 12B, 27B (avril 2025)
- **LiteRT-LM** : successeur recommande pour inference LLM on-device, supporte function calling
- **Brute-force cosine** : 1857 x 768D = sub-10ms, pas besoin d'ANN (HNSW/IVF)
- **Contraintes Android** : minSdk 26 (Android 8.0), arm64-v8a uniquement

**Impact** : Gemma 3n E2B candidat remplacement Gemma 3 270M pour la generation (meilleure qualite, 2GB RAM). Pipeline = Python desktop (pre-build DB) + Kotlin Android (query + cosine + generation). EmbeddingGemma-300M reste le choix embeddings.

Sources : deepmind.google, developers.googleblog.com, ai.google.dev, huggingface.co/google, github.com/google-ai-edge

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
