# Procedure de Controle Documentaire

> **Document ID**: DOC-CTRL-001
> **ISO Reference**: ISO 9001:2015 Clause 7.5 - Information documentee
> **Version**: 1.1
> **Date**: 2026-01-14
> **Statut**: Approuve
> **Classification**: Interne
> **Auteur**: Claude Opus 4.5
> **Mots-cles**: controle, documentation, gestion, ISO 9001, procedures

---

## 1. Objet et perimetre

### 1.1 Objet
Cette procedure definit les regles de creation, identification, revision, approbation, distribution et archivage de l'information documentee du projet Pocket Arbiter, conformement a la clause 7.5 de l'ISO 9001:2015.

### 1.2 Perimetre
S'applique a :
- Tous les documents du repertoire `docs/`
- Les fichiers de specification dans `.iso/templates/`
- Les checklists dans `.iso/checklists/`
- Les prompts dans `prompts/`
- L'inventaire du corpus dans `corpus/INVENTORY.md`

### 1.3 Documents de reference
| Norme | Titre | Application |
|-------|-------|-------------|
| ISO 9001:2015 | Systemes de management de la qualite | Clause 7.5 |
| ISO 15489-1:2016 | Gestion des documents d'activite | Cycle de vie |
| ISO 82045-1:2001 | Gestion de documents | Metadonnees |
| ISO 999:1996 | Lignes directrices pour l'indexation | Index |
| ISO/IEC 26514:2022 | Documentation utilisateur | Structure |

---

## 2. Schema de numerotation des documents (ISO 82045)

### 2.1 Format d'identification
```
[CATEGORIE]-[TYPE]-[NUMERO]
```

### 2.2 Categories
| Code | Categorie | Description |
|------|-----------|-------------|
| DOC | Documentation | Documents normatifs et procedures |
| SPEC | Specifications | Exigences et specifications techniques |
| TEST | Tests | Plans et rapports de tests |
| PROM | Prompts | Prompts IA versiones |
| CORP | Corpus | Documents sources et inventaire |

### 2.3 Types
| Code | Type | Exemple |
|------|------|---------|
| CTRL | Controle | DOC-CTRL-001 |
| POL | Politique | DOC-POL-001 (AI_POLICY) |
| REQ | Exigences | SPEC-REQ-001 (QUALITY_REQUIREMENTS) |
| PLAN | Plan | TEST-PLAN-001 |
| REF | Reference | DOC-REF-001 (ISO_STANDARDS_REFERENCE) |
| VIS | Vision | SPEC-VIS-001 |
| INV | Inventaire | CORP-INV-001 |
| LOG | Changelog | PROM-LOG-001 |

### 2.4 Registre des documents

| ID Document | Titre | Fichier | Version |
|-------------|-------|---------|---------|
| DOC-CTRL-001 | Procedure de Controle Documentaire | docs/DOC_CONTROL.md | 1.1 |
| DOC-REF-001 | Reference des Normes ISO | docs/ISO_STANDARDS_REFERENCE.md | 1.1 |
| DOC-POL-001 | Politique IA Responsable | docs/AI_POLICY.md | 1.0 |
| SPEC-REQ-001 | Exigences Qualite | docs/QUALITY_REQUIREMENTS.md | 1.0 |
| SPEC-VIS-001 | Vision Projet | docs/VISION.md | 1.0 |
| TEST-PLAN-001 | Plan de Tests | docs/TEST_PLAN.md | 1.0 |
| DOC-IDX-001 | Index Principal | docs/INDEX.md | 1.0 |
| DOC-GUIDE-001 | Guide DVC | docs/DVC_GUIDE.md | 1.0 |
| DOC-ARCH-001 | Architecture Technique | docs/ARCHITECTURE.md | 1.0 |
| PLAN-RDM-001 | Roadmap Projet | docs/PROJECT_ROADMAP.md | 1.1 |
| SPEC-P1A-001 | Specifications Phase 1A | docs/specs/PHASE1A_SPECS.md | 1.0 |
| SPEC-CHK-001 | Schema JSON Chunks | docs/CHUNK_SCHEMA.md | 1.0 |
| CHK-PIPE-001 | Checklist Phase 1 Pipeline | .iso/checklists/phase1_pipeline.md | 1.1 |
| CHK-P2-001 | Checklist Phase 2 Android | .iso/checklists/phase2_android_rag.md | 1.0 |
| CORP-INV-001 | Inventaire Corpus | corpus/INVENTORY.md | 1.0 |
| PROM-LOG-001 | Changelog Prompts | prompts/CHANGELOG.md | 1.0 |

---

## 3. En-tete standard des documents (ISO 82045)

### 3.1 Metadonnees obligatoires
Tout document doit commencer par un bloc de metadonnees :

```markdown
# [Titre du document]

> **Document ID**: [CATEGORIE]-[TYPE]-[NUMERO]
> **ISO Reference**: [Norme(s) applicable(s)]
> **Version**: [X.Y]
> **Date**: [AAAA-MM-JJ]
> **Statut**: [Draft | En revue | Approuve | Obsolete]
> **Classification**: [Public | Interne | Confidentiel]
> **Auteur**: [Nom]
> **Mots-cles**: [mot1, mot2, mot3, ...]

---
```

### 3.2 Champs de metadonnees

| Champ | Obligatoire | Description |
|-------|-------------|-------------|
| Document ID | Oui | Identifiant unique selon schema |
| ISO Reference | Oui | Norme(s) ISO applicable(s) |
| Version | Oui | Format X.Y (majeure.mineure) |
| Date | Oui | Date de derniere modification |
| Statut | Oui | Etat du cycle de vie |
| Classification | Oui | Niveau de confidentialite |
| Auteur | Oui | Createur ou responsable |
| Mots-cles | Oui | Termes d'indexation (ISO 999) |

### 3.3 Statuts du cycle de vie (ISO 15489)

```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│  Draft  │───▶│ En revue │───▶│ Approuve  │───▶│ Obsolete │
└─────────┘    └──────────┘    └───────────┘    └──────────┘
                     │                │
                     ▼                │
               ┌───────────┐          │
               │  Rejete   │──────────┘
               └───────────┘
```

---

## 4. Procedures de controle (ISO 9001 Clause 7.5.3)

### 4.1 Creation de document
1. **Identifier** le besoin documentaire
2. **Attribuer** un ID selon le schema (section 2)
3. **Rediger** avec l'en-tete standard (section 3)
4. **Soumettre** en statut "Draft"
5. **Versionner** dans Git avec commit explicite

### 4.2 Revue et approbation
| Document | Revu par | Approuve par |
|----------|----------|--------------|
| Politique (POL) | Tech Lead | Product Owner |
| Exigences (REQ) | Tech Lead | Product Owner |
| Tests (PLAN) | Tech Lead | Tech Lead |
| Guides (GUIDE) | Developpeur | Tech Lead |
| Prompts (PROM) | Developpeur | Developpeur |

### 4.3 Modification de document
1. **Incrementer** la version (mineure pour corrections, majeure pour refonte)
2. **Mettre a jour** la date
3. **Documenter** le changement dans l'historique
4. **Committer** avec reference au document modifie

Format de commit :
```
docs: [DOC-ID] Description du changement

- Detail 1
- Detail 2

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### 4.4 Archivage et retention
| Type document | Retention | Archivage |
|---------------|-----------|-----------|
| Politique | Permanent | Git history |
| Specifications | 5 ans apres fin projet | Git history |
| Plans de test | 2 ans apres release | Git history |
| Rapports | 1 an | Archive externe |

### 4.5 Distribution et acces
- **Stockage** : Depot Git (unique source de verite)
- **Acces** : Selon classification
  - Public : Tous
  - Interne : Equipe projet
  - Confidentiel : Sur demande explicite
- **Format** : Markdown (portable, versionnable)

---

## 5. Index et recherche (ISO 999)

### 5.1 Principes d'indexation
- Chaque document doit avoir des **mots-cles** pertinents
- Les mots-cles doivent etre en **francais** (langue principale)
- Utiliser un **vocabulaire controle** (termes normalises)

### 5.2 Vocabulaire controle

| Terme normalise | Synonymes acceptes |
|-----------------|-------------------|
| qualite | qualite logicielle, SQuaRE |
| test | tests, testing, verification |
| IA | intelligence artificielle, AI, LLM |
| hallucination | fabrication, invention |
| retrieval | recherche, RAG |
| corpus | sources, documents, PDF |
| arbitre | arbitrage, echecs |

### 5.3 Index principal
Voir `docs/INDEX.md` pour l'index complet des documents.

---

## 6. Audits documentaires

### 6.1 Checklist de conformite
- [ ] En-tete standard present
- [ ] Document ID unique et valide
- [ ] Version coherente avec historique
- [ ] Date a jour
- [ ] Statut correct
- [ ] Mots-cles presents (minimum 3)
- [ ] Historique des modifications present
- [ ] Liens internes valides

### 6.2 Frequence d'audit
| Type | Frequence | Responsable |
|------|-----------|-------------|
| Auto-verification | Chaque commit | Developpeur |
| Revue de coherence | Fin de phase | Tech Lead |
| Audit complet | Release majeure | Equipe |

### 6.3 Non-conformites
| Niveau | Description | Action |
|--------|-------------|--------|
| Mineure | Metadonnee manquante | Corriger avant merge |
| Majeure | Structure non conforme | Bloquer, corriger |
| Critique | Document non identifie | Bloquer, escalade |

---

## 7. Outils et automatisation

### 7.1 Pre-commit hook
Le hook `.githooks/pre-commit.py` verifie :
- Syntaxe JSON valide
- Documents ISO requis presents
- Pas de secrets dans les fichiers

### 7.2 CI/CD
Le job `docs-quality` dans `.github/workflows/ci.yml` :
- Verifie l'existence des documents obligatoires
- Compte les lignes (alerte si < 10 lignes)
- Genere des metriques

### 7.3 Validation manuelle
Commande pour verifier la structure :
```bash
python scripts/iso/validate_project.py --verbose
```

---

## 8. Historique du document

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-11 | Claude Opus 4.5 | Creation initiale |
| 1.1 | 2026-01-14 | Claude Opus 4.5 | Ajout documents Phase 1A au registre |

---

## 9. Approbations

| Role | Nom | Date | Signature |
|------|-----|------|-----------|
| Redacteur | Claude Opus 4.5 | 2026-01-11 | Auto |
| Verificateur | | | |
| Approbateur | | | |

---

*Ce document est maintenu dans le cadre du systeme de gestion documentaire ISO du projet Pocket Arbiter.*
