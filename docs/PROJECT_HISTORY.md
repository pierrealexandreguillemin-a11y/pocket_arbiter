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
  - QAT-q4_0-unquantized abandonne pour build : TFLite = Mixed Precision (pas Q4_0), base aligne mieux
  - Base MTEB 69.67 vs QAT 69.31, seq_length 2048 (pas 256), LoRA standard si fine-tuning
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
| 19 mar | QAT → base pour build embeddings | TFLite Mixed Precision converti depuis base (pas QAT), +0.36 MTEB, elimine mismatch build/runtime, standard LoRA path |
| 19 mar | Row-as-chunk REVERTED | -6pp R@5 regression, Ragie warning confirme (header repetition degrade BM25) |
| 19 mar | Structured cells (level 3) | 4308 cells, keyword triggers strong/weak, three-way RRF, neutre recall mais fonctionnalite RAG |
| 19 mar | Recall final 60.1% | Gate R1 (70%) FAIL — LoRA fine-tuning necessaire |

## Ere 7 : Recall optimization — resultats (mars 2026)

- Chantier 3 termine : 8 optimisations testees, 4 appliquees, 2 skippees, 2 revertees
- Gain total : 56.7% → 60.1% recall@5 (+3.4pp)
  - Model switch QAT→base : +2.4pp
  - Enrichment (OPT 1-2-4) : +1.3pp (principalement R@1 : +3.7pp)
  - Level 3 structured cells : neutre (±0.3pp)
- Row-as-chunk (level 2) : -6pp regression, reverted
- Score calibration (OPT-7) : -5pp R@1, skipped
- Query decomposition (OPT-8) : 3/110 matches, skipped
- **Question ouverte** : pourquoi les techniques standard industrie (contextual retrieval -35% failures Anthropic, row-as-chunk Ragie) ont un impact marginal ou negatif sur CE corpus/modele ?
  - Hypotheses a explorer : seq_length 2048 vs modeles cloud 8K+, corpus reglementaire FR vs benchmarks EN, EmbeddingGemma 300M vs Voyage/OpenAI, taille corpus 1116 chunks vs benchmarks 10K+

## Ere 8 : Fine-tuning retrieval — abandonne (mars 2026)

- Chantier 4a planifie : SimCSE + ICT LoRA sur EmbeddingGemma-300M, Kaggle T4
- Spec ecrite (2026-03-20), kernel code, dataset prepare (1116 SimCSE + 1067 ICT pairs)
- Audit code review (2026-03-21) : **3 bugs critiques** trouves dans le kernel
  - Stage 2 sans LoRA re-attache (merge_and_unload detruit les adapters, Stage 2 casse)
  - SimCSE avec prompt asymetrique (query prompt sur document = pas du SimCSE)
  - Pas d'evaluation/early stopping malgre la spec qui le demande
- **Decision : ABANDON chantier 4a** — raisons cumulees :
  1. Aucune litterature ne valide SimCSE/ICT a l'echelle 1116 exemples (900x-73000x sous papers)
  2. Precedent fine-tune supervise a DEGRADE recall (82.84% → 65.69%)
  3. 8 optimisations retrieval testees (chantier 3) : +3.4pp total — rendements decroissants
  4. EmbeddingGemma = seul modele on-device <500MB, pas d'alternative
  5. GS = faux-ami (264/298 QCM annales ≠ queries terrain arbitres)
  6. Le levier generation est inexplore et plus prometteur
- **Pivot** : chantier 4 = GRPO fine-tuning modele generation (Gemma 3n)
  - Le retrieval a 60.1% R@5 reste en l'etat (EmbeddingGemma-300M base, non fine-tune)
  - La generation peut compenser : citations fideles, aveu d'ignorance, reformulation
  - GRPO = reward rule-based, pas de dependance au GS

## Decisions cles (suite)

| Date | Decision | Raison |
|------|----------|--------|
| 21 mar | Chantier 4a (LoRA retrieval) ABANDONNE | 3 bugs critiques, 0 precedent litterature a cette echelle, rendements decroissants, precedent echec |
| 21 mar | Pivot vers generation (GRPO Gemma 3n) | Levier inexplore, retrieval 60.1% = suffisant si generation compense |
