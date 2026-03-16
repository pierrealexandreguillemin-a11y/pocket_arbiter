# Chantier 1 : Menage — Design Spec

> **Date**: 2026-03-16
> **Statut**: Approuve
> **Objectif**: Nettoyer le repo pour repartir sur des bases saines — zero ambiguite sur l'etat du projet

---

## Contexte

574 commits, ~135 scripts Python (hors tests), ~64 docs markdown. Docs contredisent le code, code contredit les donnees, metadata GS sedimentees. Le pipeline a produit des chunks trop petits (109 tokens median), sans parents dans l'index, sans table summaries. Les specs pipeline ont ete modifiees apres la derniere generation sans relancer le pipeline.

**Principe** : tout ce qui n'est pas source de verite ou standard valide → archive. On reecrit le pipeline dans le chantier 2.

---

## 1. Archivage scripts

**Destination** : `scripts/archive/`

**Archiver tout `scripts/` sauf :**
- `scripts/iso/` (validation qualite, independant du pipeline)

Cela inclut :
- Tous les scripts core pipeline (chunker, embeddings, export, search)
- Tous les one-off (fix_*, migrate_*, add_*, p0-p4_*)
- Tous les tests (`scripts/**/tests/`) — ils importent des modules archives, donc cassent. Seront reecrits avec le pipeline dans le chantier 2.
- Les `__init__.py`, `README.md`, `conftest.py` de chaque module archive

Les scripts ont produit l'etat casse actuel. On reecrit dans le chantier 2 depuis les standards industrie.

**Post-archivage obligatoire** : mettre a jour `.coveragerc` (retirer les 40+ omit lines referençant des fichiers deplaces) et ruff config pour que les gates ISO passent sur le repo nettoye.

---

## 2. Archivage docs

**Destination** : `docs/archive/`

**Archiver tout `docs/` sauf** les fichiers listes en section 3.

Cela inclut notamment :
- Plans perimes : GS_CONFORMITY_PLAN_V1, GS_CORRECTION_PLAN_V2, GS_FROM_SCRATCH_BY_DESIGN
- Specs GS perimes : GS_BY_DESIGN_METHODOLOGY, GS_ANNALES_V7_OPTIMIZATION_SPEC, GS_SCHEMA_V2
- Specs pipeline sedimentees : CHUNKING_STRATEGY (19 commits), RETRIEVAL_PIPELINE (13 commits), PHASE1A_SPECS, CHUNK_SCHEMA
- Docs sedimentees : ISO_MODEL_DEPLOYMENT_ANALYSIS (6 commits), RECALL_OPTIMIZATION_PLAN (8 commits)
- Audits : tout `docs/audits/` (batch reports, validation reports, sincerity audit)
- Research perimes : AUDIT_GS_*, AUDIT_QUALITY_*, ISO_OPTIMIZATION_MAPPING
- Specs GS v6 : GOLD_STANDARD_V6_ANNALES, GS_CONFORMITY_CHECKLIST, GS_SCRATCH_V1_SPEC
- Divers : HARD_CASES_VERIFIED, PHASE1B_REMEDIATION_PLAN, RECALL_REMEDIATION (deja marquee ARCHIVEE), TABLE_SUMMARY_PROCEDURE_ISO
- `docs/schemas/` (triplet_schema.json)
- `docs/research/_archive/` existant → migre dans `docs/archive/research/`

---

## 3. Docs gardees telles quelles

### Fondations (ere 1, stables)
- `docs/VISION.md`
- `docs/ARCHITECTURE.md`
- `docs/PROJECT_ROADMAP.md`
- `docs/AI_POLICY.md`
- `docs/QUALITY_REQUIREMENTS.md`
- `docs/TEST_PLAN.md`
- `docs/README.md`

### Standards/research (snapshots fiables ou contenu utile)
- `docs/research/FINETUNING_LESSONS.md` (1 commit)
- `docs/research/RECALL_FAILURE_ANALYSIS_2026-01-20.md` (1 commit)
- `docs/research/OFFLINE_OPTIMIZATIONS_2026-01-20.md` (utile)
- `docs/research/FINETUNING_RESOURCES.md` (utile)
- `docs/research/annales_dec2024_UVR.md` (reference annales)
- `docs/specs/ADVERSARIAL_QUESTIONS_STRATEGY.md` (1 commit)
- `docs/specs/EMBEDDING_FALLBACK_STRATEGY.md` (1 commit)
- `docs/ISO_VECTOR_SOLUTIONS.md` (2 commits, research web)
- `docs/specs/UNIFIED_TRAINING_DATA_SPEC.md` (2 commits)
- `docs/specs/TRIPLET_GENERATION_SPEC.md` (utile)

### Governance
- `docs/INDEX.md`
- `docs/DOC_CONTROL.md`
- `docs/DEPENDENCIES.md`
- `docs/GS_VALIDATION.md`
- `docs/GOLD_STANDARD_SPECIFICATION.md`
- `docs/ANNALES_STRUCTURE.md`
- `docs/DVC_GUIDE.md`
- `docs/CVE_EXCEPTIONS.md`

### Plans recents (ere 3, pertinents pour chantier 3)
- `docs/plans/2026-02-27-baseline-recall-v8-design.md`
- `docs/plans/2026-02-27-baseline-recall-v8.md`

---

## 4. Donnees inchangees

- **Corpus** : `corpus/` (30 PDFs, chunks, parents, table summaries)
- **GS** : `tests/data/gold_standard_annales_fr_v8_adversarial.json`
- **Benchmarks** : `data/benchmarks/`
- **Models** : `models/model_card.json`

**Archiver** : `data/gs_generation/` (artefacts de generation GS casses) → `data/archive/gs_generation/`

---

## 5. Reecriture

### CLAUDE.md
From scratch. Contenu :
- Vision projet (1 ligne)
- Etat reel du corpus, GS, pipeline (faits, pas aspirations)
- Cap : fix pipeline → re-mesure recall → decision
- Commandes, code style, structure (repris de l'actuel, corriges)
- Reference `docs/PROJECT_HISTORY.md`

### docs/PROJECT_HISTORY.md
Chronologie factuelle des errements :
- Ere 1 (jan) : fondations, pipeline initial, specs
- Ere 2 (fev) : gs_scratch echec, corrections GS, 6 passes automatiques
- Ere 3 (fin fev) : pivot annales, baseline recall 37%, nettoyage 122 Q mismatch
- Ere 4 (mars) : menage, redemarrage

### Memoire
`MEMORY.md` + fichiers memoire : reecrits en coherence avec le nouveau CLAUDE.md.

---

## 6. Verification Gemma 3

Recherche web : modele plus recent que Gemma 3 270M IT pour RAG on-device ?
- Gemma 3n, Gemma 3.5, nouveaux modeles Google 2026
- Recommandations fabricant mises a jour
- **Scope** : noter les findings uniquement. Pas de modification model_card.json dans ce chantier — les implications touchent le pipeline (chantier 2).

---

## 7. Verification post-archivage

Smoke test obligatoire apres execution :
1. `python -m pytest scripts/iso/ -v` → PASS
2. `python -m ruff check scripts/iso/` → PASS
3. `python -m ruff check scripts/` → pas d'erreur sur fichiers inexistants
4. Aucun fichier present hors `archive/` n'est ambigu ou perime

---

## Ce qu'on ne touche PAS

- Le GS (chantier 2)
- Le pipeline / chunking / embeddings (chantier 2)
- Le recall baseline (chantier 3)

---

## Livrable

Un repo ou :
- Tout fichier present est soit correct soit explicitement dans `archive/`
- CLAUDE.md reflete l'etat reel
- Le cap est clair : fix pipeline → re-mesure → decision
- Zero ambiguite entre docs, code et donnees
