# Pocket Arbiter — Claude Code Memory

> Application RAG mobile Android pour arbitres d'echecs. Recherche semantique offline sur reglements FFE/FIDE.

## Etat du projet (avril 2026)

### Ce qui fonctionne
- **Corpus** : 28 PDFs FFE extraits avec heading levels (docling + docling-hierarchical-pdf)
- **Chunker** : LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter (450/50 overlap)
- **Chunks** : 1117 children (config 450/50 + 1 injected), 273 parents (cap 2048), 117 tables detectees, 117 table summaries
- **Contextes** : 1116 contextual retrieval entries (Anthropic 2024), generes par LLM, median 54 tokens
- **Enrichment** : abbreviation expansion (12 termes), chapter overrides (5 ranges LA), context prepend
- **Structured cells** : 4436 cells (level 3 TableRAG), keyword triggers strong/weak
- **Narrative rows** : 1420 rows (Table-to-Text), 4th RRF channel (w=0.5), +12.3pp tab recall
- **Synthetic queries** : 2232 Doc2Query (Gemma 4B), DISABLED (degrades recall, data in DB)
- **Pages** : line-level interpolation + pdfplumber page fix (120 children corrected), 99.3% GS pages couvertes (296/298)
- **GS** : 403 questions (298 testables), page-level matching, 297/298 chunk_ids v2-aligned
- **Modeles** : EmbeddingGemma-300M base (embeddings), Gemma 3n E2B candidat generation
- **ISO** : validation qualite (`scripts/iso/`), pre-commit hooks, 334 tests, 68% coverage
- **Indexer** : corpus_v2_fr.db (DVC tracked), 9/9 integrity gates (I1-I9) PASS
- **CCH** : heading hierarchy (h1 > h2 > h3) pour children ET table summaries (KX 2026)
- **Search** : hybrid cosine + BM25 FTS5 + structured cells + narrative rows + targeted rows, 5-way RRF k=60, gradient intent B.4, adaptive-k largest-gap (EMNLP 2025)
- **Targeted rows** : 45 row-chunks (6 tables prioritaires), forward fill, [col: val] format, unit suffixes, cosine-only canal 5
- **Gradient intent** : scoring continu 0-3.0 remplace trigger binaire, clipping max 1.5/1.0 (anti-shadowing)
- **Col normalization** : B.5, mapping abbreviations colonnes (cat.→Categorie, tps→Temps)
- **Synonymes** : Snowball FR stemmer + 70 synonymes chess

### Recall (chantier 3 — termine, chantier 5 — termine)
- **QAT baseline** : recall@5 = 56.7% (ancien modele)
- **Base-only** : recall@5 = 59.1% (+2.4pp model switch)
- **Enrichi (chantier 3)** : recall@5 = 60.1%, recall@1 = 38.9%, recall@10 = 63.8%, MRR = 0.479
- **Chantier 5 items 1-4** : recall@5 = **63.4%** (189/298), recall@10 = **67.4%** (201/298)
  - CORRECTION : 67.4% etait recall@**10**, PAS recall@5 (mislabel sessions precedentes)
  - LLM recall@5 = **79.3%** (2232 Q Gemma 4B), LLM recall@10 = 84.4%
- **Chantier 5 Phase 4 (targeted rows)** : recall@5 = **63.4%** (inchange — page-level prose matching)
  - Tables remontent dans le RRF (LA-table73 rank >10 → rank 6), gain **generation** pas recall
  - Human recall : 32/34 (94.1%) → 30/34 (88.2%) — 2 regressions (1 shadowing clipped, 1 trigger equipe)
  - Finding : GS mesure la page PROSE, pas la page TABLE. Targeted rows ameliorent le contexte LLM, pas le recall metric
  - Gradient intent clipped max 1.5/1.0 (anti-shadowing, isolation test documente)
- **Gate R1 (70%) : FAIL** — mais tranche haute industrie (55-65% corpus reglementaire offline)
- Row-as-chunk (level 2) : REVERTED, remplace par narrative rows (level 2b, +12.3pp tab)
- Doc2Query (canal 5) : DISABLED (degrades recall, data/benchmarks/doc2query_experiment.md)
- **Page interpolation fix** : 120 children page corrected via pdfplumber, recall ceiling 95%→100%
- **GS chunk_id realignment** : 298/298 chunk_ids updated v1→v2 format (297 children + 1 table_summary)
- **6 missing table_summaries** : summaries generees, embeddings, narrative rows (66), structured cells (154) injectes
- **1 child injected** : Interclubs p.7 (extraction gap docling, child + embedding + FTS)
- **FTS5 structured_cells** : fuzzy matching col_name + cell_value, prefix queries (term*)
- **Priority boost** : 10 tables data-driven (top GS frequency), +1.5 score additif
- **Intro filter (OPT-5)** : pages cover/section exclues du RRF
- **Canal 4 weight sweep** : neutre 0.0-0.7, maintenu w=0.5
- Root cause fix : `_build_text_to_page()` + `_extract_text_pages()` pour dense page tracking (text_pages)
- Ref : @data/benchmarks/page_interpolation_fix.md, @data/benchmarks/canal4_vs_phase3_audit.md

### Optimisations appliquees
- **OPT-1 DONE** : contextual retrieval (Anthropic 2024) — +3.7pp R@1 (principal gain)
- **OPT-2 DONE** : abbreviation expansion (12 termes, 1205 expansions)
- **OPT-4 DONE** : chapter overrides (85 chunks LA)
- **OPT-6 DONE** : config tuning 450/50 (teste 9 configs)
- **OPT-7 SKIPPED** : score calibration (4 formules, toutes degradent R@1)
- **OPT-8 SKIPPED** : query decomposition (3/110 matches = marginal)
- **Level 2 REVERTED** : row-as-chunk (-6pp, Ragie warning confirme)
- **Level 3 DONE** : structured cells (neutre recall, fonctionnalite RAG Android)
- **Model switch** : QAT → base (+2.4pp, aligne build/runtime)

### Retrieval table improvement (chantier 5 — termine)
- **Phase 1 DONE** : trigger tuning — recall-neutre, allow-list + col_name search
- **Phase 2 DONE** : narrative rows + 4-way RRF + page fix = **+7.4pp global** (57.4% → 64.8%)
  - narrate_table_rows() : Table-to-Text, +12.3pp tab recall
  - 4-way RRF canal separe (w=0.5) : elimine pollution prose (-7.3pp → -0.8pp)
  - fix page=None : _resolve_page() depuis best-scoring child, +5.4pp global
- **Doc2Query DISABLED** : 2232 synthetic queries, 70% complementaire, MAIS degrades recall
  - Root cause : questions generees pour SFT (thematique), pas retrieval (discriminative)
- **Phase 3 DONE** : injection contextuelle tables page±1 dans build_context()
  - _inject_neighbor_tables() : tables page X-1/X/X+1 injectees dans contexte LLM
  - Dedup, budget 500 mots, score inferieur aux resultats retrieves
  - 5 tests unitaires, recall-neutre (+0.3pp), gain qualitatif pour generation
- **Items 1-4 DONE** : FTS5 cells, priority boost (10 tables), intro filter, weight sweep
  - Recall@5 : 63.4% (189/298), recall@10 : 67.4% (201/298) — CORRECTION mislabel
  - Canal 4 vs Phase 3 audit : 82% overlap, Canal 4 maintenu (21 tables isolees)
- **Phase 4 DONE (2026-04-04)** : targeted row-chunks + gradient intent + col normalization
  - C.10 : 45 row-chunks (6 tables greedy set cover), forward fill, [col: val], unit suffixes
  - B.4 : gradient_intent_score() 0-3.0, clipped max 1.5/1.0 (anti-shadowing verifie)
  - B.5 : normalize_column_name() 17 mappings (cat.→Categorie, tps→Temps)
  - Recall@5 : 63.4% (inchange — finding : GS mesure page PROSE, pas page TABLE)
  - Tables remontent dans le RRF (gain contexte LLM, pas recall metric)
  - Human recall : 32/34 → 30/34 (2 regressions : 1 trigger equipe shadowing, 1 rebuild)
  - Shadowing isole : uncapped 76.5%, clipped 88.2%, channels OFF 91.2%
  - INTENT_WEIGHTS sanctuarises, ne plus toucher
  - Ref : @docs/superpowers/specs/2026-04-04-targeted-table-retrieval-design.md

### Fine-tuning retrieval (chantier 4a — ABANDONNE)
- SimCSE + ICT LoRA planifie, spec ecrite, kernel code, dataset prepare
- Audit code review (2026-03-21) : 3 bugs critiques (Stage 2 sans LoRA, prompt asymetrique, pas d'eval)
- **ABANDONNE** : 0 precedent litterature a 1116 exemples, precedent fine-tune a degrade recall, rendements decroissants
- EmbeddingGemma-300M base reste en l'etat (pas de fine-tuning)

### Generation fine-tuning (chantier 4)
- **TAPT v1 DONE** : Gemma 270M IT, FFT fp32+AMP, 5 epochs, ppl 37.74→7.98 (bugs: dropout 0.1, cosine, LR 5e-6)
- **TAPT v2 DONE** : params corriges (LR 5e-5, dropout 0.0, constant) → ppl 4.70, mais faithfulness catastrophique (ep1 = 4.2%)
- **TAPT v3 sweep DONE** : v1 params exacts, 5 epochs evaluated → **ep1 = sweet spot (46.2% citations, +2.3pp vs base)**
- **SFT v1-v4 INVALIDES** : toutes entrainees sur donnees regex garbage (AdaptLLM pattern matching)
  - Les 1802 "reading tasks" = connecteurs FR detectes par regex, pas de generation LLM
  - Reponses = bouts de texte copies (mediane 16 tokens, pas de citations, pas de format)
  - NEFTune alpha=5 present dans v1-v3 (pas valide sub-1B, mesure chat vibes pas faithfulness)
  - La litterature (17 papers) disait deja : SFT domain standard NUIT au RAG. Ignore.
  - **SFT v5** : generer VRAIES reponses avec LLM teacher (RAFT-style) avant tout SFT
- **Prompt v2** : 7 regles numerotees, reformulation, injection defense, contrainte longueur
- **Gen params** : temp=0.2, repetition_penalty=1.2, no_repeat_ngram_size=4, Google defaults

### TAPT v3 sweep — FINDING MAJEUR (2026-03-25)
- **TAPT ep1 46.2%** > base 43.9% > ep3 42.4% > ep2/ep5 40.2% > ep4 36.4%
- **1 epoch mild TAPT (LR 5e-6) = seul epoch qui bat base** sur faithfulness
- Apres ep1, degradation monotone : le modele internalise le corpus et ignore le contexte
- v2 params (LR 5e-5) = overcorrection confirmee (detruit instruction-following en 1 epoch)
- **SFT v4 doit utiliser TAPT ep1 (checkpoint-22) comme base**

### Eval v4 — RESULTATS CRITIQUES (2026-03-24)
- **Base 43.9%** > TAPT v1 ep5 36.4% > SFT v3 28.8% citations — **plus de FFT = moins de faithfulness**
- 0 empty pour les 3 modeles (min_new_tokens=10 corrige le bug)
- SFT v3 echo les questions au lieu de repondre (full-seq loss → apprend a predire le prompt)
- **Note** : eval v4 ne testait que ep5 (pire epoch). TAPT v3 sweep revele que ep1 bat base.

### 4 bugs training identifies (2026-03-24)
1. **attention_dropout=0.1** injecte → Google livre 0.0 (arXiv:2505.24788)
2. **cosine scheduler** → Google FFT guide utilise **constant** (WSO arXiv:2603.16127)
3. **Full-sequence loss** → TRL supporte **assistant-only** (echo = consequence directe)
4. **TAPT LR=5e-6** au lieu de 5e-5 → en fait 5e-6 est MEILLEUR pour faithfulness (v3 sweep confirme)

### Pipeline next steps — SFT v5 on 1B (LoRA + Unsloth, RAFT data)
- **RAFT data DONE** : Phase A (2232 questions) + Phase B (2142 answers filtrées) = sft_train_v5.jsonl
  - Teacher: Gemma 3 4B IT (NF4), RAFT hybrid 80% oracle / 15% abstain / 5% memorize
  - Format: "D'apres [source] (p.XX) : '[quote]'. [Answer]." + ##begin_quote## validation
  - 95.1% citations valides (quote_valid=true)
- **SFT v5 on 270M** : DONE mais 270M MORT (overfit 1.01, assistant-only loss). Jamais eval.
- **SFT v5 on 1B TRAINING** : DONE (2026-04-02) — Unsloth LoRA NF4, train_on_responses_only(), Kaggle T4
  - Base: Gemma 3 1B IT, LoRA R=16 alpha=32, 13.0M/664.1M trainable (2.0%)
  - train_loss=0.9091, eval_loss=3.3174, overfit=1.012, 55.1 min
  - 6 checkpoints (20,40,60,80,100,114), merged model 16bit sauve
  - OOM Phase 6 (compute_clm_loss apres merge) — metrics JSON perdu, modele OK
  - Kaggle install: --no-deps unsloth + trl==0.24.0 pinne (Kaggle a trl 1.0.0, transformers 5.0.0)
- **SFT v5 on 1B EVAL** : DONE (2026-04-03) — **GATE PASS : 60.1% > 56.7%**
  - Hang diagnostique : Unsloth save_pretrained_merged drop generation_config.json + use_cache=false
  - Token 106 (end_of_turn) manquait de la liste EOS → generation infinie → Tamil garbage → 128s/Q
  - Fix : eos_token_id=[1,106] + use_cache=True + smoke test timing guard (<30s)
  - SFT v5 : **60.1%** pipeline citations (hit=51.9%, miss=72.6%, abstain=18.5%, 0 empty, median 291 mots)
  - Base : 56.4% (hit=48.1%, miss=69.2%)
  - Delta : **+3.7pp** homogene hits+misses, 103 min runtime
  - Ref : data/benchmarks/eval_1b_sft_v5/
- **SFT v5 1B = MODELE GENERATION FINAL** (gate PASS, meilleur modele du projet)
- **Infra** : LiteRT-LM replaces MediaPipe (deprecated) for Android inference
- Ref : Pleias-RAG (2025) prouve qu'un 350M apprend les citations si donnees de qualite
- Ref : RAFT (Berkeley 2024) = format cible (oracle + distracteurs + citations verbatim)

### Eval methodology (generation)
- **Metrique primaire** : cited_pct (regex doc/page sur 264 annales) — PROXY, pas faithfulness (ICTIR 2025)
- **HHEM-2.1-Open** (Vectara) : T5-base hallucination classifier, supporte FR, faisable offline T4 — recommande pour eval v5.1
- **57% citations post-rationalisees** : Wallat et al. ICTIR 2025 Best Paper HM — cited_pct absolu ≠ faithfulness
- **FACTS Grounding** (Google 2025) : benchmark industrie = 3 LLM judges — non faisable budget 0EUR
- **FaithBench** (Vectara, EMNLP 2025) : annotations hallucination span-level, 4 severites
- **Ref complete** : @docs/GENERATION_EVAL_METHODOLOGY.md

### Postmortem 270M + Eval 1B ancien (2026-03-29)
- **270M = INUTILISABLE** : 34Q humaines, les 3 modeles (base/TAPT/SFT) hallucinent. 0% verbatim. ADR-001 gate declenchee.
- **1B base (ancien retrieval 63.1%)** : oracle 43.9%, pipeline 47.0%. Reponses qualitativement meilleures que 270M.
- **Post-rationalisation 1B** : MISS cite 60.6% > HIT cite 36.8% (ratio 1.65x). 0.8% abstention MISS.
- **Ref** : @data/benchmarks/retrieval_table_gap_analysis.md

### Eval 1B v4 — RESULTATS (2026-04-02, retrieval 67.4%)
- **Pipeline cited: 56.7%** (+9.7pp vs ancien run 47.0%) — gain direct du retrieval ameliore + Phase 3 injection
- **Oracle cited: 46.2%** (identique au 270M TAPT ep1 — plafond du prompt v2)
- **0 empty**, median 192-214 mots, 11.4% abstention
- **Post-rationalisation confirmee** : MISS cite 68.4% > HIT cite 49.2% (ratio 1.39x)
- **Abstention inversee** : HIT abstain 13.8% > MISS abstain 7.7% (le modele ne sait pas quand il ne sait pas)
- **1B base > tous les 270M** : pipeline 56.7% vs 270M sft80 48.7% vs 270M tapt_ep1 40.3% vs 270M base 24.8%
- **Ref** : @data/benchmarks/eval_1b_v2/

### Candidats generation post-270M
- **Gemma 3 1B IT** : IFEval 80.2%, ~400 MB, LiteRT natif. **CANDIDAT PRINCIPAL** (56.7% pipeline).
- **Gemma 3n E2B** : 2B eff, ~2 GB RAM, LiteRT .litertlm, mobile-first. Depasse spec 500MB.
- **Ministral 3B** : Apache 2.0, FR natif, 256K ctx. Pas de LiteRT (LLaMA.cpp).
- **Qwen3 1.7B** : MMLU 75.7, Apache 2.0. LiteRT non confirme.

### Embedding pipeline — question ouverte Keras vs sentence-transformers
- Le chemin officiel Google pour EmbeddingGemma est **Keras** (notebook Nilay Chauhan, Google)
- Le projet utilise **sentence-transformers** (PyTorch) depuis janvier 2026
- Les poids du modele sont identiques — le delta vient du backend (PyTorch vs TensorFlow)
- Le chemin Keras → TFLite est natif (conversion directe). sentence-transformers → TFLite passe par une conversion intermediaire
- **Delta non mesure** : impact reel inconnu. A quantifier (100 queries, cosine distance) avant toute decision
- Refs: kaggle.com/code/nilaychauhan/rag-with-embeddinggemma, docs/ISO_VECTOR_SOLUTIONS.md

### Kaggle deployment findings (session 2026-04-02)
- **Kaggle image v168** : Python 3.12, torch 2.10.0, transformers 5.0.0, trl 1.0.0, CUDA 12.8
- **Unsloth + Kaggle** : --no-deps OBLIGATOIRE (transformers 5.0.0 exclue par Unsloth pyproject.toml)
- **trl 1.0.0** : incompatible Unsloth (veut <=0.24.0). Pin trl==0.24.0 explicitement
- **bitsandbytes** : absent de l'image Kaggle. Hard dep d'Unsloth (crash sans). Installer explicitement
- **SFT v5 1B eval hang RESOLU** : Unsloth save_pretrained_merged drop generation_config.json + use_cache=false
  - Token 106 (end_of_turn) absent EOS list → generation infinie (128s vs 1.8s base)
  - Fix : generation_config.json + config.json fixes + eos_token_id=[1,106] dans kernel + smoke test guard
  - Ref : memory/feedback_unsloth_merge_eos.md

### References
- ADR-001 : Gemma 3 270M IT (Option A) — **GATE DECLENCHEE, 270M ABANDONNE**
- Specs : 2026-03-21-cpt-adaptllm, 2026-03-23-sft-v3, 2026-03-24-training-params-correction
- Artefacts : models/model_card.json
- **Question ouverte** : pourquoi les optimisations retrieval standard ont un impact marginal ?
- **Question ouverte** : retrieval de tables — pourquoi le search rate les tableaux associes aux questions prose ?
- **Question ouverte** : Keras vs sentence-transformers — delta a mesurer avant decision

## Commandes

- `python -m pytest scripts/iso/ scripts/pipeline/tests/ -m "not slow" -v` : Tests (ISO + pipeline, sans extraction PDF)
- `python -m pytest scripts/iso/ scripts/pipeline/tests/ -v` : Tests complets (inclut extraction PDF ~1h)
- `python -m pytest scripts/iso/ scripts/pipeline/tests/ --cov --cov-fail-under=80` : Tests avec coverage
- `python -m pre_commit run --all-files` : Quality hooks
- `python -m ruff check scripts/` : Lint
- `python scripts/pipeline/extract.py` : Re-extraire corpus (~1h, inclut LA 222 pages)
- `python -m dvc push` : Push DB versionnee

## Code style

- Python 3.10+ avec type hints obligatoires
- Docstrings Google style pour fonctions publiques
- Imports: stdlib, third-party, local (separes par ligne vide)
- Max 88 caracteres par ligne (ruff default)

## Structure

```
scripts/
  iso/              # Validation ISO (actif, 125 tests)
  pipeline/         # Pipeline actif (chunker, indexer, search, enrichment, recall)
  archive/          # Scripts archives (pipeline v1, evaluation, training)
corpus/
  fr/               # 29 PDF FFE
  intl/             # 1 PDF FIDE
  processed/        # Chunks, parents, table summaries, DB (DVC tracked)
tests/data/         # GS (gold_standard_annales_fr_v8_adversarial.json)
data/benchmarks/    # Recall baselines, experiments
docs/               # Specs, research, fondations
  superpowers/      # Specs et plans chantiers
models/             # model_card.json
```

## ISO Compliance (OBLIGATOIRE)

- ISO 27001 : Jamais lire .env, secrets/, *.pem, *.key
- ISO 29119 : Coverage >= 80% sur code actif (334 tests, 68.07% — recall.py/recall_report.py tirent la moyenne)
- ISO 25010 : Complexite cyclomatique <= B (xenon)
- ISO 12207 : Commits conventionnels (feat/fix/test/docs)
- ISO 42001 : Citations obligatoires, 0% hallucination

## Principes de developpement

- **DONNEES D'ABORD** : auditer le CONTENU des donnees (lire 10 samples) AVANT les hyperparametres
- **Pas de regex comme donnees SFT** : les reponses d'entrainement doivent etre generees par un modele, pas du pattern matching
- **Pas de LLM externe pour generer les donnees** : seuls nos modeles (Gemma 270M) generent les donnees d'entrainement (postmortem gs_scratch : LLM externe → 71.5% garbage)
- **Pas de SFT domain standard pour RAG** : la litterature (17 papers) montre que ca nuit a la faithfulness. Utiliser RAFT-style (citations + distracteurs)
- **KISS** : Solutions simples et directes, eviter over-engineering
- **DRY** : Factoriser le code commun
- **Production saine > existant rotte** : reecrire plutot que debugger du code empile
- Lire le fichier AVANT de le modifier
- Executer tests apres chaque modification
- Verifier les donnees contre les PDF sources (pas de texte invente)
- **DVC** : versionner la DB avant chaque experience (eviter les rebuilds inutiles)

## Environment

- **Virtualenv** : `.venv/`
- **Setup** :
  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate
  # Unix:    source .venv/bin/activate
  pip install -r requirements.txt
  pip install pre-commit xenon pip-audit  # dev tools
  python -m pre_commit install
  python -m pre_commit install --hook-type commit-msg --hook-type pre-push
  ```
- **CVE exceptions** : docs/CVE_EXCEPTIONS.md

## References

- @docs/PROJECT_HISTORY.md : Historique des errements et decisions
- @docs/VISION.md : Vision produit et architecture Dual-RAG
- @docs/ARCHITECTURE.md : Architecture technique (Android Kotlin layer)
- @docs/AI_POLICY.md : Politique anti-hallucination
- @docs/QUALITY_REQUIREMENTS.md : Exigences qualite
- @docs/specs/TRIPLET_GENERATION_SPEC.md : Spec triplets (si fine-tuning decide)
- @models/model_card.json : Specs modeles (EmbeddingGemma + Gemma 3)
- @docs/superpowers/specs/2026-03-19-recall-optimization-design.md : Spec recall optimization
- @docs/superpowers/specs/2026-03-19-structured-tables-design.md : Spec structured tables (level 3)
- @data/benchmarks/row_as_chunk_experiment.md : Experiment row-as-chunk (REVERTED, remplace par narrative rows)
- @data/benchmarks/doc2query_experiment.md : Doc2Query experiment (DISABLED, degrades recall)
- @data/benchmarks/retrieval_table_gap_analysis.md : Gap retrieval tables — search rate les tableaux associes
- @docs/GENERATION_EVAL_METHODOLOGY.md : Methodologie eval generation (ISO, ICTIR 2025, HHEM, FACTS, FaithBench)
