# Pocket Arbiter — Claude Code Memory

> Application RAG mobile Android pour arbitres d'echecs. Recherche semantique offline sur reglements FFE/FIDE.

## Etat du projet (mars 2026)

### Ce qui fonctionne
- **Corpus** : 28 PDFs FFE extraits avec heading levels (docling + docling-hierarchical-pdf)
- **Chunker** : LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter (450/50 overlap)
- **Chunks** : 1116 children (config 450/50), 282 parents (cap 2048), 117 tables detectees, 111 table summaries
- **Contextes** : 1116 contextual retrieval entries (Anthropic 2024), generes par LLM, median 54 tokens
- **Enrichment** : abbreviation expansion (12 termes), chapter overrides (5 ranges LA), context prepend
- **Structured cells** : 4308 cells (level 3 TableRAG), keyword triggers strong/weak, three-way RRF
- **Pages** : line-level interpolation, 95% GS pages couvertes (105/111)
- **GS** : 403 questions (298 testables), page-level matching
- **Modeles** : EmbeddingGemma-300M base (embeddings), Gemma 3n E2B candidat generation
- **ISO** : validation qualite (`scripts/iso/`), pre-commit hooks, 312 tests, 80% coverage
- **Indexer** : corpus_v2_fr.db (DVC tracked), 9/9 integrity gates (I1-I9) PASS
- **CCH** : heading hierarchy (h1 > h2 > h3) pour children ET table summaries (KX 2026)
- **Search** : hybrid cosine + BM25 FTS5 + structured cells, RRF k=60, adaptive-k largest-gap (EMNLP 2025)
- **Synonymes** : Snowball FR stemmer + 70 synonymes chess

### Recall (chantier 3 — termine)
- **QAT baseline** : recall@5 = 56.7% (ancien modele)
- **Base-only** : recall@5 = 59.1% (+2.4pp model switch)
- **Enrichi final** : recall@5 = 60.1%, recall@1 = 38.9%, recall@10 = 63.8%, MRR = 0.479
- **Gate R1 (70%) : FAIL** → LoRA fine-tuning necessaire
- Row-as-chunk (level 2) : REVERTED (-6pp, documente data/benchmarks/row_as_chunk_experiment.md)

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

### Fine-tuning retrieval (chantier 4a — ABANDONNE)
- SimCSE + ICT LoRA planifie, spec ecrite, kernel code, dataset prepare
- Audit code review (2026-03-21) : 3 bugs critiques (Stage 2 sans LoRA, prompt asymetrique, pas d'eval)
- **ABANDONNE** : 0 precedent litterature a 1116 exemples, precedent fine-tune a degrade recall, rendements decroissants
- EmbeddingGemma-300M base reste en l'etat (pas de fine-tuning)

### Generation fine-tuning (chantier 4)
- **TAPT v1 DONE** : Gemma 270M IT, FFT fp32+AMP, 5 epochs, ppl 37.74→7.98 (bugs: dropout 0.1, cosine, LR 5e-6 au lieu de 5e-5)
- **SFT v1** : 3ep LR 2e-5, TAPT ep5 → sur-apprend (echo 17.6%, overfit 1.33)
- **SFT v2** : 1ep LR 1e-5, TAPT ep4 → sous-apprend (median 5 mots, coupe trop tot)
- **SFT v3 DONE** : 2ep LR 1e-5, checkpoint-140 (MA loss 1.716, acc 0.620, overfit 1.08)
- **Prompt v2** : 7 regles numerotees, reformulation, injection defense, contrainte longueur
- **Gen params** : temp=0.2, repetition_penalty=1.2, no_repeat_ngram_size=4, Google defaults

### Eval v4 — RESULTATS CRITIQUES (2026-03-24)
- **Base 43.9%** > TAPT 36.4% > SFT v3 28.8% citations — **plus de FFT = moins de faithfulness**
- 0 empty pour les 3 modeles (min_new_tokens=10 corrige le bug)
- SFT v3 echo les questions au lieu de repondre (full-seq loss → apprend a predire le prompt)
- Base = meilleur lecteur fidele avec prompt v2 + gen params state-of-the-art
- **Paradoxe faithfulness confirme par les donnees** (17 papers + eval v4)

### 4 bugs training identifies (2026-03-24)
1. **attention_dropout=0.1** injecte → Google livre 0.0 (arXiv:2505.24788)
2. **cosine scheduler** → Google FFT guide utilise **constant** (WSO arXiv:2603.16127)
3. **Full-sequence loss** → TRL supporte **assistant-only** (echo = consequence directe)
4. **TAPT LR=5e-6** au lieu de 5e-5 (10x trop bas, erreur doc model_card)

### Pipeline correction planifie
- **TAPT v2** : params corriges (dropout=0.0, constant scheduler) → benchmark ppl
- **SFT v4** : sur TAPT v2 + assistant-only loss + prompt v2 dans training data → benchmark
- **Eval v5** : comparer avec eval v4 → decision FFT ou base suffit
- Spec : docs/superpowers/specs/2026-03-24-training-params-correction-design.md

### References
- ADR-001 : Gemma 3 270M IT (Option A)
- Specs : 2026-03-21-cpt-adaptllm, 2026-03-23-sft-v3, 2026-03-24-training-params-correction
- Artefacts : models/model_card.json
- **Question ouverte** : pourquoi les optimisations retrieval standard ont un impact marginal ?

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
- ISO 29119 : Coverage >= 80% sur code actif (312 tests, 80.06%)
- ISO 25010 : Complexite cyclomatique <= B (xenon)
- ISO 12207 : Commits conventionnels (feat/fix/test/docs)
- ISO 42001 : Citations obligatoires, 0% hallucination

## Principes de developpement

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
- @data/benchmarks/row_as_chunk_experiment.md : Experiment row-as-chunk (REVERTED)
