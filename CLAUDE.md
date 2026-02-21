# Pocket Arbiter - Claude Code Memory

> Application RAG mobile pour arbitres d'echecs
> Normes: ISO 25010, 42001, 12207, 29119, 27001

## Bash commands

- `python -m pytest scripts/ -v`: Run all tests
- `python -m pytest scripts/ --cov=scripts --cov-config=.coveragerc --cov-fail-under=80`: Tests with 80% coverage gate
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

## Environment

- **Virtualenv**: `.venv/` (isolé, pip-audit clean)
- **Activation**: `.venv/Scripts/activate` (Windows)
- **Pre-commit hooks**: installés depuis le venv

## Current Work: GS Correction (PLAN-GS-CORR-002)

- **Plan source**: docs/plans/GS_CORRECTION_PLAN_V2.md
- **Phase courante**: GO/NO-GO A→B
- **Plans**: P1 [x] → P2 [x] → **GO/NO-GO A→B** → P3 [ ] → P4 [ ] → GO/NO-GO B→C,C→D → P5 [ ] → P6 [ ]
- **Derniere gate validee**: P2 (Phase A re-generation, all gates PASS)
- **Prochain go/no-go**: A→B (pret)
- **Fichier GS courant**: tests/data/gs_scratch_v1_step1.json (614Q, v1.1+P1+P2)
- **P2 resultats**: 80Q remplacées, 4 profils (20x HARD_APPLY, 20x HARD_ANALYZE, 20x MED_APPLY_INF, 20x MED_ANALYZE_COMP), gates A-G1 à A-G5 PASS

## References

- @docs/AI_POLICY.md: Politique anti-hallucination
- @docs/QUALITY_REQUIREMENTS.md: Exigences qualite
- @docs/specs/PHASE1A_SPECS.md: Specs Phase 1A actuelle
- @docs/plans/GS_CORRECTION_PLAN_V2.md: Plan correction GS (PLAN-GS-CORR-002)
