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
  iso/              # Validation ISO (100% coverage)
  pipeline/         # Extraction PDF, chunking, embeddings (Phase 1A)
  evaluation/
    annales/        # GS generation, correction, regression (Phase A)
    ares/           # ARES evaluation framework
  training/         # Fine-tuning embeddings (Phase 2)
corpus/
  fr/               # 29 PDF FFE
  intl/             # 1 PDF FIDE
tests/data/         # Questions gold standard (gs_scratch_v1_step1.json)
data/gs_generation/ # Artefacts generation (candidates, replacements, snapshots)
docs/               # Specs ISO, plans, CVE register
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

- **Virtualenv**: `.venv/` (required for pip-audit and dependency isolation)
- **Setup**:
  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate
  # Unix:    source .venv/bin/activate
  pip install -r requirements.txt
  pip install pre-commit xenon pip-audit  # dev tools
  python -m pre_commit install
  python -m pre_commit install --hook-type commit-msg --hook-type pre-push
  ```
- **CVE exceptions**: docs/CVE_EXCEPTIONS.md

## Current Work: GS Correction (PLAN-GS-CORR-002)

- **Plan source**: docs/plans/GS_CORRECTION_PLAN_V2.md
- **Phase courante**: Phase B - P3 en cours
- **Plans**: P1 [x] → P2 [x] → GO/NO-GO A→B [x] → **P3 [~]** → P4 [ ] → GO/NO-GO B→C,C→D → P5 [ ] → P6 [ ]
- **Derniere gate validee**: GO/NO-GO A→B (2026-02-22)
- **Prochain jalon**: P3a (remplacer 95 page-number Qs) + P3b (orchestrateur generation)
- **Fichier GS courant**: tests/data/gs_scratch_v1_step1.json (614Q, v1.1+P1+P2+recalib)
- **Recalibration script**: `scripts/evaluation/annales/recalibrate_full.py` (applique TOUTES les corrections depuis baseline)

### Phase B Prep (2026-02-26) - Analyse completee

**95 page-number questions** : 100% Remember/factual/medium/extractive, pattern "Quelle regle a la page X?". Decision : remplacer en P3a (IDs stables, meme workflow que P2).

**Stratification chunks non couverts** : 1501/1857 (19.2% coverage), 1288 generables (>=50 tok).
- P1_LA: 889, P2_Coupes: 73, P3_Champ: 185, P4_Admin: 185, P5_Other: 169
- Ordre : petits docs d'abord (diversite), puis LA (volume)
- Cible 80% = 1485 chunks, besoin 1129 de plus

### GO/NO-GO A→B (2026-02-22) - PASS conditionnel

**Level 1 - Gates**: A-G1 schema PASS, A-G2 hard 12.1% PASS, A-G3 4CL PASS, A-G4 chunk_match PASS, A-G5 regression PASS (37+5xfail)
**Level 2 - Tests**: 1508 passed, 1 skipped, 5 xfailed. Lint + coverage + complexity OK.
**Level 3 - LLM-as-Judge (2 rounds, 30Q chacun)**:
- question_type: kappa=0.948 PASS
- cognitive_level: kappa=0.748 PASS
- answer_type: kappa=0.634 PASS
- difficulty_bucketed: kappa=-0.284 FAIL (known limitation)

**Recalibration appliquee (614Q, pas seulement P2)**:
- answer_type: 34 changes (seuil kw overlap 0.45)
- cognitive_level: 205 Understand→Remember (pattern-based)
- question_type: 149 procedural/scenario→factual
- difficulty: 129 easy-caps + 11 hard replacements
- Distributions finales: Remember 65%, factual 70%, extractive 93%, hard 12.1%

**Issues identifies pour Phase B**:
- 95 questions "page-number" (Quelle regle a la page X?) = mauvaises pour RAG, a redesigner
- Difficulty non calibrable par kw overlap seul (necessite tests retrieval reels)
- Kappa difficulty structurellement faible (base rate skew)

## References

- @docs/AI_POLICY.md: Politique anti-hallucination
- @docs/QUALITY_REQUIREMENTS.md: Exigences qualite
- @docs/specs/PHASE1A_SPECS.md: Specs Phase 1A actuelle
- @docs/plans/GS_CORRECTION_PLAN_V2.md: Plan correction GS (PLAN-GS-CORR-002)
