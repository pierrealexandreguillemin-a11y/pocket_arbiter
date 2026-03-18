# Pocket Arbiter — Claude Code Memory

> Application RAG mobile Android pour arbitres d'echecs. Recherche semantique offline sur reglements FFE/FIDE.

## Etat du projet (mars 2026)

### Ce qui fonctionne
- **Corpus** : 28 PDFs FFE extraits avec heading levels (docling + docling-hierarchical-pdf)
- **Chunker** : LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter (512/100 overlap)
- **Chunks** : 1073 children (median 323 tok), 282 parents (cap 2048), 117 tables detectees, 111 table summaries
- **Pages** : line-level interpolation, 95% GS pages couvertes (105/111)
- **GS** : 403 questions (298 testables), page-level matching
- **Modeles** : EmbeddingGemma-300M QAT (embeddings), Gemma 3n E2B candidat generation
- **ISO** : validation qualite (`scripts/iso/`), pre-commit hooks
- **Indexer** : corpus_v2_fr.db, 9/9 integrity gates (I1-I9) PASS
- **CCH** : heading hierarchy (h1 > h2 > h3) pour children ET table summaries (KX 2026)
- **Search** : hybrid cosine + BM25 FTS5, RRF k=60, adaptive-k largest-gap (EMNLP 2025), min_k=3
- **Synonymes** : Snowball FR stemmer + 70 synonymes chess (A: intra-corpus + B: langage courant→corpus)
- **Pipeline tests** : 146 fast PASS, 125 ISO PASS

### Recall baseline (chantier 3)
- **recall@5 = 56.7%** page-level (298 questions, reglages de base)
- recall@1 = 35.6%, recall@10 = 63.1%, MRR = 0.441
- Decision : < 60% → optimisations retrieval ou fine-tuning necessaires

### A faire
- **Recall improvement** : analyser les 20 pires echecs, calibrer adaptive_k
- **Classifications GS** : answer_type 100% faux ("multiple_choice"), reasoning_class ~55% faux
- Decider : fine-tuning embeddings ou prompt engineering selon resultats optimisation

## Commandes

- `python -m pytest scripts/iso/ scripts/pipeline/tests/ -m "not slow" -v` : Tests (ISO + pipeline, sans extraction PDF)
- `python -m pytest scripts/iso/ scripts/pipeline/tests/ -v` : Tests complets (inclut extraction PDF ~1h)
- `python -m pytest scripts/iso/ --cov --cov-config=.coveragerc --cov-fail-under=80` : Tests avec coverage
- `python -m pre_commit run --all-files` : Quality hooks
- `python -m ruff check scripts/` : Lint
- `python scripts/pipeline/extract.py` : Re-extraire corpus (~1h, inclut LA 222 pages)

## Code style

- Python 3.10+ avec type hints obligatoires
- Docstrings Google style pour fonctions publiques
- Imports: stdlib, third-party, local (separes par ligne vide)
- Max 88 caracteres par ligne (ruff default)

## Structure

```
scripts/
  iso/              # Validation ISO (actif, 125 tests)
  archive/          # Scripts archives (pipeline, evaluation, training)
corpus/
  fr/               # 29 PDF FFE
  intl/             # 1 PDF FIDE
  processed/        # Chunks, parents, table summaries, DB
tests/data/         # GS (gold_standard_annales_fr_v8_adversarial.json)
data/benchmarks/    # Baseline recall, audits
docs/               # Specs, research, fondations
  archive/          # Docs perimes
  superpowers/      # Specs et plans chantiers
models/             # model_card.json
```

## ISO Compliance (OBLIGATOIRE)

- ISO 27001 : Jamais lire .env, secrets/, *.pem, *.key
- ISO 29119 : Coverage >= 80% sur code actif
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
- @docs/superpowers/specs/2026-03-16-menage-design.md : Spec menage
- @docs/superpowers/plans/2026-03-16-menage.md : Plan menage
