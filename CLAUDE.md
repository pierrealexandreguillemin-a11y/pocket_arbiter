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
tests/data/         # Questions gold standard (gold_standard_annales_fr_v8_adversarial.json)
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

## Current Work: Triplet Generation (SPEC-TRIP-001)

- **Spec source**: docs/specs/TRIPLET_GENERATION_SPEC.md
- **GS primaire**: `tests/data/gold_standard_annales_fr_v8_adversarial.json` (482Q = 383 answerable + 99 adversarial)
- **Prochain jalon**: reclassifier answer_type (~300 MC -> extractive/abstractive) puis generer triplets

### Pivot v8 annales (2026-02-27) - Decision

**gs_scratch abandonne** : audit qualite 71.5% answerable = garbage (templates mecaniques). Scripts supprimes. P3a (95 page-number) annule.

**v8 annales adopte comme GS primaire**. Re-evaluation a montre que les 3 objections initiales etaient infondees :
- "post-hoc matching" → FAUX : 420/420 `manual_by_design`, chunk_match_score=100
- "237 chunk issues" → RESOLU : cascading fixes (152+101+25+290+238), final = 0 issues
- "100% MCQ" → REFORMULE : 335/340 reponses reformulees (194 chars moy, pas des lettres)

**Pourquoi v8** : 343 questions d'examen FFE reelles (10 sessions, 4 UV) + 40 human, success_rate 0.05-1.00, 28 docs couverts (vs 17 gs_scratch), 99 adversarial UAEval4RAG inclus. Qualite > quantite pour QLoRA.

### Nettoyage Q/choices mismatch (2026-02-28)

43 questions supprimees (Q text d'une UV avec choices d'une autre UV, bug d'extraction):
- jun2025: 20 mismatches (64->44)
- jun2021: 14 mismatches + 6 duplicats + 3 data quality (42->19)
- Archive: `data/benchmarks/removed_questions_qa_mismatch_2026-02-28.json`
- Audit: `data/benchmarks/audit_gs_v8_mcq_answers_2026-02-28.json`

### v8 annales — chiffres cles (post-cleanup)

| Dimension | Valeur |
|-----------|--------|
| Answerable | 383 (343 annales + 40 human) |
| Unanswerable | 99 (20%, 6 categories UAEval4RAG) |
| Testables (hors requires_context) | ~290 |
| Documents couverts | 28/28 |
| Chunks uniques | ~185 |
| answer_type: multiple_choice | ~300 (a reclassifier) |

### Travail restant avant triplets

1. **Reclassifier answer_type** : ~300 "multiple_choice" -> extractive/abstractive (reponses deja reformulees, label seul a corriger)
2. **Charger chunk_text** pour champ `positive` des triplets
3. **Hard negative mining** : sentence-transformers, margin 0.05 (NV-Retriever, SPEC-TRIP-001 Section 3)
4. **Split 80/20** : ~230 train GS + ~58 val GS (100% GS, zero synthetique en val)

### gs_scratch — archive

- **Fichier**: `tests/data/gs_scratch_v1_step1.json` (614Q, v1.1+P1+P2+recalib) — ARCHIVE, plus le GS primaire
- **Reutilisable**: ~113 questions non-garbage potentiellement cherry-pickables en Phase B (coverage supplement)
- **Scripts Phase A**: recalibrate_full.py, regenerate_targeted.py, verify_regression.py — conserves (reutilisables)
- **Plan**: docs/plans/GS_CORRECTION_PLAN_V2.md — historique, plus actif

## References

- @docs/AI_POLICY.md: Politique anti-hallucination
- @docs/QUALITY_REQUIREMENTS.md: Exigences qualite
- @docs/specs/PHASE1A_SPECS.md: Specs Phase 1A actuelle
- @docs/specs/TRIPLET_GENERATION_SPEC.md: Spec triplets (SPEC-TRIP-001, source de verite)
- @docs/plans/GS_CORRECTION_PLAN_V2.md: Plan correction GS (PLAN-GS-CORR-002, historique/archive)
