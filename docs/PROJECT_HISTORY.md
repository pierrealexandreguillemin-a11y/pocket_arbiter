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
| 21 mar | Pivot vers generation (CPT+SFT Gemma 270M) | Levier inexplore, retrieval 60.1% = suffisant si generation compense |
| 21 mar | ADR-001 : Gemma 3 270M IT (Option A) | Respecte spec RAM 500MB, gate rollback vers 1B si qualite <70% |
| 22 mar | TAPT DONE : ppl 37.74 → 7.98 | FFT fp32+AMP, 5 epochs, T4, Gate G1 PASS |
| 22 mar | Architecture 2 kernels Kaggle | SFT eval OOM sur vocab 262K, split TAPT + SFT-only |
| 23 mar | SFT v1 DONE : loss 3.64 → 1.29, overfit 1.33 | 3 epochs, 1622 train, overfit ratio sain |
| 23 mar | Eval 3 modeles DONE | SFT 0 empty (vs 71 base), 33% citations (vs 22%) |
| 23 mar | DVC tracking modeles | models/kaggle-output + models/kaggle-sft-output |
| 23 mar | SFT v2 step 60 selectionne | Loss descend a step 60 (1.97), remonte apres; overfit ratio 1.04 vs 1.33 v1 |
| 23 mar | Domain SFT peut nuire RAG faithfulness | Revue 17 papers : post-rationalisation, Dunning-Kruger, under-learning potentiellement meilleur |
| 23 mar | Benchmark base planifie | Pipeline RAG jamais teste avec modele base — erreur de brainstorming |
| 23 mar | SFT v3 : 2 epochs LR 1e-5 | Budget 0.00202, entre v2 (0.001) et v1 (0.006), 10 checkpoints |

## Ere 9 : Generation fine-tuning (mars 2026)

- Chantier 4 : TAPT + AdaptLLM SFT sur Gemma 3 270M IT (ADR-001)
- AdaptLLM regex mining : 1802 exercices, 6 types, FR connectors
- TAPT : 5 epochs FFT fp32+AMP, perplexity 37.74 → 7.98, Gate G1 PASS
- SFT : 3 epochs FFT fp16+grad_checkpoint, 1622 train / 180 eval
  - Loss curve : 3.64 → 1.05 (epoch 2) → 1.29 (epoch 3)
  - Token accuracy : 38.5% → 72.3% (epoch 1) → 67.7% (epoch 3)
  - Overfit ratio : 1.33 (eval/train CLM), flag = false
  - Epoch 2 (checkpoint-204) potentiellement meilleur que final (epoch 3)
- 13+ tentatives Kaggle : P100 defaut, OOM×3, 401 auth×2, fp16 error,
  eval OOM×3. Chaque finding documente dans skill kaggle-deployment.
- Architecture finale : 3 kernels (TAPT seul + SFT-only + eval 3 modeles)
- Eval comparative (2026-03-23) : base vs TAPT vs SFT v1 sur 298 questions
  - Base : 71 empty responses, 21.6% auto-citations
  - TAPT : 9 empty responses, 34.1% auto-citations
  - SFT v1 : **0 empty responses**, 33.0% auto-citations
  - Qualite : les 3 modeles faibles (270M) — hallucinations, repetitions, hors-sujet
  - Eval humaine PENDING sur 34 questions manuelles
- SFT v2 (2026-03-23) : 1 epoch LR 1e-5, base TAPT epoch 4 (checkpoint-88)
  - 6 checkpoints : 20, 40, 60, 80, 100, 102
  - Meilleur checkpoint : step 60 (loss 1.9736, token_accuracy 0.578)
  - Courbe loss : descend jusqu'a step 60, remonte ensuite (step 80: 2.19, step 100: 2.27)
  - Overfit ratio 1.04 (vs 1.33 v1) — amelioration significative
  - Training loss 2.3049, eval loss 1.9863 (vs 1.689 / 1.484 v1)
  - save_only_model=True : checkpoints 1.1 GB (vs 3.1 GB v1), budget disque OK
  - Runtime 6.9 min T4 (vs 20.5 min v1)
  - Eval v2 resultats (2026-03-23) :
    - Base : inchange (21.6% citations, 71 empty)
    - TAPT ep4 : quasi-identique ep5 (34.8% citations, 27/34 repetitif vs 26/34)
    - SFT v2 : **sous-apprend** — 70.6% reponses < 10 mots, median 5 mots (vs 18 v1)
    - SFT v2 cite des numeros de section (73.5%) mais ne developpe pas la reponse
    - 4 vraies reponses sur 34 sont factuellement correctes
    - Auto-citation 24.6% (vs 33.0% v1) — regression
  - **Diagnostic** : loss v2 descendait encore (moyenne mobile 2.13→2.08 steps 51-80)
    Mon erreur : j'ai lu la loss step-by-step (bruitee) au lieu de la moyenne mobile.
    Checkpoint-60 selectionne trop tot. 1 epoch LR 1e-5 = budget d'apprentissage insuffisant.
  - **SFT v3 planifie** : 2 epochs LR 1e-5, save_steps=20 (10 checkpoints)
  - **Finding critique (2026-03-23)** : revue litterature 17 papers (2020-2026) —
    domain SFT peut NUIRE a la faithfulness RAG (post-rationalisation, effet Dunning-Kruger).
    Under-learning potentiellement meilleur que sweet spot pour RAG grounding.
    Papers cles : Knowledge Conflicts Survey (EMNLP 2024), Tug-of-War (LREC-COLING 2024),
    Correctness != Faithfulness (ICTIR 2025), RAFT (Berkeley 2024), FACTS (Google 2025).
  - **Erreur de brainstorming** : pipeline RAG jamais teste avec modele base (sans FFT).
    Benchmark base planifie apres v3 pour valider si FFT est necessaire.
  - **Roadmap revisee** : v3 → benchmark base → comparer faithfulness → decision FFT/RAFT

### Artefacts generation (inventaire complet)

| Artefact | Emplacement local | Kaggle dataset |
|----------|-------------------|----------------|
| Base model | kaggle/model-gemma-270m/ | pguillemin/gemma-3-270m-it |
| TAPT checkpoint | models/kaggle-output/gemma-270m-cpt/ | pguillemin/gemma-270m-tapt-checkpoint |
| SFT checkpoint (final) | models/kaggle-sft-output/gemma-270m-cpt-sft/ | pguillemin/gemma-270m-sft-checkpoint |
| SFT epoch 1 | models/kaggle-sft-output/.../checkpoint-102/ | — |
| SFT epoch 2 | models/kaggle-sft-output/.../checkpoint-204/ | — |
| SFT epoch 3 | models/kaggle-sft-output/.../checkpoint-306/ | — |
| Training data | kaggle/dataset-generation/ | pguillemin/pocket-arbiter-gen-data |
| Eval data | kaggle/eval-data/ | pguillemin/pocket-arbiter-eval-data |
| SFT metrics | models/kaggle-sft-output/sft_metrics.json | — |
| TAPT metrics | models/kaggle-output/tapt_perplexity.json | — |
| Eval base | data/benchmarks/generation_eval_base.json | — |
| Eval TAPT | data/benchmarks/generation_eval_tapt.json | — |
| Eval SFT | data/benchmarks/generation_eval.json | — |
| Kernel TAPT+SFT | kaggle/kernel-generation/ | pocket-arbiter-cpt-sft-generation |
| Kernel SFT-only | kaggle/kernel-sft/ | pocket-arbiter-sft-generation |
| Kernel eval | kaggle/kernel-eval/ | pocket-arbiter-eval-generation-3-models |
