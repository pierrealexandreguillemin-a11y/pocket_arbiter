# Pocket Arbiter — Claude Code Memory

> Application RAG mobile Android pour arbitres d'echecs. Recherche semantique offline sur reglements FFE/FIDE.

## Etat du projet (mars 2026)

### Ce qui fonctionne
- **Corpus** : 28 PDFs FFE extraits avec heading levels (docling + docling-hierarchical-pdf)
- **Chunks v2** : 1154 children (median 311 tok), 304 parents (median 863 tok), 111 table summaries
- **Pages** : 1154/1154 children ont une page source
- **GS** : 403 questions, chunk text retrouvable dans v2 (297/298 = 99.7%)
- **Modeles** : EmbeddingGemma-300M (embeddings), Gemma 3n E2B candidat generation
- **ISO** : validation qualite (`scripts/iso/`), pre-commit hooks
- **Pipeline tests** : 33 tests extract+chunker PASS, 125 tests ISO PASS

### En cours (chantier 2)
- **Task 3** : Indexer — CCH (Contextual Chunk Headers) + embed + SQLite DB
- **Task 4** : Search — cosine brute-force + adaptive k + parent lookup
- **Task 5** : Integration — build corpus_v2_fr.db

### A faire
- **Chantier 3** : Re-mesurer recall sur 304 testables propres
- **Classifications GS** : answer_type 100% faux ("multiple_choice"), reasoning_class ~55% faux
- Decider : fine-tuning embeddings ou prompt engineering selon resultats recall

## Commandes

- `python -m pytest scripts/iso/ scripts/pipeline/tests/ -v` : Tests (ISO + pipeline)
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
