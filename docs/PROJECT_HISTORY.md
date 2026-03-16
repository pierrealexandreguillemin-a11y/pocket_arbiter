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

## Decisions cles

| Date | Decision | Raison |
|------|----------|--------|
| 18 jan | EmbeddingGemma-300M pour embeddings | Recommande Google pour RAG, 82.84% recall base |
| 18 jan | Gemma 3 270M IT pour generation | Plus petit Gemma 3, compatible MediaPipe, 200MB |
| 27 fev | gs_scratch abandonne | 71.5% garbage, qualite inacceptable |
| 27 fev | Annales = GS primaire | 264 vraies questions d'examen FFE, chunk_match verifie |
| 16 mar | Archivage scripts pipeline | Ont produit l'etat casse, plus simple de reecrire |
