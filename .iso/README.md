# ISO Enforcement - Pocket Arbiter

> Système de conformité ISO automatisé pour le développement

## Vue d'ensemble

Ce dossier contient la configuration et les outils d'enforcement des normes ISO :

| Norme | Scope | Documentation |
|-------|-------|---------------|
| ISO/IEC 25010 | Qualité logicielle | `docs/QUALITY_REQUIREMENTS.md` |
| ISO/IEC 42001 | Gouvernance IA | `docs/AI_POLICY.md` |
| ISO/IEC 12207 | Cycle de vie | `docs/VISION.md` |
| ISO/IEC 29119 | Tests | `docs/TEST_PLAN.md` |

---

## Structure

```
.iso/
├── config.json              # Configuration des gates et phases
├── README.md                # Ce fichier
├── checklists/
│   ├── feature_template.md  # Template checklist feature standard
│   ├── ai_feature.md        # Template checklist feature IA (ISO 42001)
│   ├── phase1_pipeline.md   # Checklist specifique Phase 1
│   └── phase2_android_rag.md # Checklist Phase 2 - Android RAG Mid-Range
└── templates/
    └── spec_template.md     # Template de spécification

.githooks/
├── pre-commit               # Validation avant commit
└── commit-msg               # Validation message de commit
```

---

## Installation des hooks Git

Pour activer l'enforcement ISO local :

```bash
# Configurer Git pour utiliser nos hooks
git config core.hooksPath .githooks

# Rendre les hooks exécutables (Linux/Mac)
chmod +x .githooks/*
```

---

## Utilisation

### 1. Avant de coder une feature

1. Copier `.iso/templates/spec_template.md` vers `docs/specs/[feature].md`
2. Remplir la spécification
3. Copier `.iso/checklists/feature_template.md` (ou `ai_feature.md` pour l'IA)
4. Valider les pré-requis

### 2. Pendant le développement

Les hooks pre-commit vérifient automatiquement :
- Pas de TODO/FIXME critiques
- Pas de secrets hardcodés
- JSON valides
- Documentation ISO présente
- Patterns anti-hallucination (pour code IA)

### 3. Au commit

Le hook commit-msg vérifie :
- Format : `[type] Description`
- Types valides : `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`
- Longueur minimale
- Pas de WIP sur main

### 4. Validation manuelle

```bash
# Valider le projet complet
python scripts/iso/validate_project.py

# Valider une phase spécifique
python scripts/iso/validate_project.py --phase 1

# Mode verbose
python scripts/iso/validate_project.py --verbose
```

---

## Gates par phase

| Phase | Gates obligatoires |
|-------|-------------------|
| 0 - Fondations | `docs_exist`, `structure_valid`, `git_initialized` |
| 1 - Pipeline | `specs_exist`, `scripts_tested`, `corpus_processed` |
| 2 - Retrieval | `specs_exist`, `ui_implemented`, `retrieval_tested` |
| 3 - LLM | `specs_exist`, `grounding_verified`, `hallucination_tested` |
| 4 - Qualité | `performance_validated`, `coverage_met`, `lint_clean` |
| 5 - Beta | `user_tested`, `docs_complete`, `release_ready` |

---

## Métriques cibles

| Métrique | Cible | Bloquant |
|----------|-------|----------|
| Test pass rate | 100% | Oui |
| Code coverage | ≥ 60% | Non |
| Lint warnings critiques | 0 | Oui |
| Retrieval recall | ≥ 80% | Oui |
| Hallucination rate | 0% | Oui |
| Latence moyenne | < 5s | Non |

---

## CI/CD

Le pipeline GitHub Actions exécute automatiquement :

1. **iso-validation** : Gate obligatoire
   - Validation structure ISO
   - Vérification documentation
   - Check anti-hallucination

2. **test-python** : Tests scripts
3. **validate-test-data** : Validation JSON
4. **quality-check** : Métriques (après iso-validation)
5. **build-android** : Build APK (sur tag uniquement)

---

## Troubleshooting

### Hook pre-commit bloqué

```bash
# Voir les détails
git commit --verbose

# Bypass temporaire (NON RECOMMANDÉ)
git commit --no-verify -m "[type] Message"
```

### Validation ISO échoue

```bash
# Diagnostic détaillé
python scripts/iso/validate_project.py --verbose

# Voir les erreurs spécifiques
python scripts/iso/validate_project.py 2>&1 | grep "❌"
```

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-01-11 | Creation systeme ISO enforcement |
| 1.1 | 2026-01-14 | Ajout checklist Phase 2 Android RAG Mid-Range |
