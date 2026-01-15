# Pocket Arbiter - Claude Code Memory

> Application RAG mobile pour arbitres d'echecs
> Normes: ISO 25010, 42001, 12207, 29119, 27001

## Bash commands

- `python -m pytest scripts/ -v`: Run all tests
- `python -m pytest scripts/ --cov=scripts --cov-fail-under=80`: Tests with 80% coverage gate
- `python -m pre_commit run --all-files`: Run all quality hooks
- `python -m ruff check scripts/`: Lint Python code
- `python -m mypy scripts/`: Type check

## Code style

- Python 3.10+ avec type hints obligatoires
- Docstrings Google style pour fonctions publiques
- Imports: stdlib, third-party, local (separes par ligne vide)
- Max 88 caracteres par ligne (ruff default)

## Project structure

```
scripts/
  iso/          # Validation ISO (100% coverage)
  pipeline/     # Extraction PDF, chunking (Phase 1A)
corpus/
  fr/           # 29 PDF FFE
  intl/         # 1 PDF FIDE
tests/data/     # Questions gold standard
docs/           # Specs ISO
```

## ISO Compliance (OBLIGATOIRE)

- ISO 27001: Jamais lire .env, secrets/, *.pem, *.key
- ISO 29119: Coverage >= 80%, tests pour code executable uniquement
- ISO 25010: Complexite cyclomatique <= B (xenon)
- ISO 12207: Commits conventionnels (feat/fix/test/docs)
- ISO 42001: Citations obligatoires, 0% hallucination

## Principes de developpement

- **KISS (Keep It Simple, Stupid)**: Solutions simples et directes, eviter over-engineering
- **Code leverage**: Reutiliser modules existants (ex: token_utils.py) avant de creer du code duplique
- **DRY (Don't Repeat Yourself)**: Factoriser le code commun dans des modules partages

## Workflow

- Lire le fichier AVANT de le modifier
- Executer tests apres chaque modification
- Ne jamais reduire la couverture de tests
- Verifier les donnees contre les PDF sources (pas de texte invente)
- Utiliser les modules partages existants (token_utils, chunk_normalizer, etc.)

## References

- @docs/AI_POLICY.md: Politique anti-hallucination
- @docs/QUALITY_REQUIREMENTS.md: Exigences qualite
- @docs/specs/PHASE1A_SPECS.md: Specs Phase 1A actuelle
