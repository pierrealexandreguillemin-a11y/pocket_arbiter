# Index Principal de la Documentation

> **Document ID**: DOC-IDX-001
> **ISO Reference**: ISO 999:1996 - Lignes directrices pour l'indexation
> **Version**: 1.0
> **Date**: 2026-01-11
> **Statut**: Approuve
> **Classification**: Interne
> **Auteur**: Claude Opus 4.5
> **Mots-cles**: index, documentation, navigation, reference, ISO 999

---

## 1. Structure documentaire

```
pocket_arbiter/
├── docs/                          # Documentation principale
│   ├── INDEX.md                   # Ce document (DOC-IDX-001)
│   ├── DOC_CONTROL.md             # Procedure de controle (DOC-CTRL-001)
│   ├── ISO_STANDARDS_REFERENCE.md # Reference ISO (DOC-REF-001)
│   ├── AI_POLICY.md               # Politique IA (DOC-POL-001)
│   ├── QUALITY_REQUIREMENTS.md    # Exigences qualite (SPEC-REQ-001)
│   ├── VISION.md                  # Vision projet (SPEC-VIS-001)
│   ├── TEST_PLAN.md               # Plan de tests (TEST-PLAN-001)
│   └── DVC_GUIDE.md               # Guide DVC (DOC-GUIDE-001)
│
├── prompts/                       # Prompts IA versiones
│   ├── README.md                  # Description des prompts
│   ├── CHANGELOG.md               # Historique prompts (PROM-LOG-001)
│   └── CLAUDE_CODE_PHASE1.md      # Prompt Phase 1
│
├── corpus/                        # Documents sources
│   └── INVENTORY.md               # Inventaire corpus (CORP-INV-001)
│
├── .iso/                          # Configuration ISO
│   ├── README.md                  # Guide ISO
│   ├── config.json                # Configuration validation
│   ├── templates/                 # Modeles de documents
│   └── checklists/                # Checklists de validation
│
├── README.md                      # Presentation projet
└── CLAUDE_CODE_INSTRUCTIONS.md    # Instructions developpement
```

---

## 2. Index par categorie

### 2.1 Gouvernance et controle

| ID | Document | Description | Statut |
|----|----------|-------------|--------|
| DOC-CTRL-001 | [DOC_CONTROL.md](DOC_CONTROL.md) | Procedure de controle documentaire | Approuve |
| DOC-REF-001 | [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) | Reference des normes ISO applicables | Approuve |
| DOC-IDX-001 | [INDEX.md](INDEX.md) | Index principal (ce document) | Approuve |

### 2.2 Specifications et exigences

| ID | Document | Description | Statut |
|----|----------|-------------|--------|
| SPEC-VIS-001 | [VISION.md](VISION.md) | Vision et objectifs du projet | Draft |
| SPEC-REQ-001 | [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) | Exigences qualite ISO 25010 | Draft |

### 2.3 Intelligence artificielle

| ID | Document | Description | Statut |
|----|----------|-------------|--------|
| DOC-POL-001 | [AI_POLICY.md](AI_POLICY.md) | Politique IA responsable ISO 42001 | Draft |
| PROM-LOG-001 | [prompts/CHANGELOG.md](../prompts/CHANGELOG.md) | Historique des versions de prompts | Actif |

### 2.4 Tests et qualite

| ID | Document | Description | Statut |
|----|----------|-------------|--------|
| TEST-PLAN-001 | [TEST_PLAN.md](TEST_PLAN.md) | Plan de tests ISO 29119 | Draft |

### 2.5 Guides techniques

| ID | Document | Description | Statut |
|----|----------|-------------|--------|
| DOC-GUIDE-001 | [DVC_GUIDE.md](DVC_GUIDE.md) | Guide d'utilisation DVC | Approuve |

### 2.6 Corpus et donnees

| ID | Document | Description | Statut |
|----|----------|-------------|--------|
| CORP-INV-001 | [corpus/INVENTORY.md](../corpus/INVENTORY.md) | Inventaire des documents sources | Actif |

---

## 3. Index par sujet (ISO 999)

### A
- **Accessibilite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.4.4
- **AI (Intelligence Artificielle)** : [AI_POLICY.md](AI_POLICY.md)
- **Android** : [VISION.md](VISION.md) Section 5.1
- **Approbations** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 4.2
- **Archivage** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 4.4

### C
- **Citation** : [AI_POLICY.md](AI_POLICY.md) Section 7.2
- **Compatibilite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.3
- **Controle documentaire** : [DOC_CONTROL.md](DOC_CONTROL.md)
- **Corpus** : [corpus/INVENTORY.md](../corpus/INVENTORY.md)
- **Couverture tests** : [TEST_PLAN.md](TEST_PLAN.md) Section 3

### D
- **Disclaimer** : [AI_POLICY.md](AI_POLICY.md) Section 7.1
- **Documentation** : [DOC_CONTROL.md](DOC_CONTROL.md)
- **DVC** : [DVC_GUIDE.md](DVC_GUIDE.md)

### E
- **Embeddings** : [AI_POLICY.md](AI_POLICY.md) Section 5.1
- **Exigences qualite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md)

### F
- **FFE (Federation Francaise des Echecs)** : [VISION.md](VISION.md) Section 2.1
- **FIDE** : [VISION.md](VISION.md) Section 2.1
- **Fiabilite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.5

### G
- **Gates (phase gates)** : [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.3
- **Grounding** : [AI_POLICY.md](AI_POLICY.md) Section 3.2

### H
- **Hallucination** : [AI_POLICY.md](AI_POLICY.md) Section 3.2, [TEST_PLAN.md](TEST_PLAN.md) Section 3.3

### I
- **ISO 25010** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.1
- **ISO 29119** : [TEST_PLAN.md](TEST_PLAN.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.4
- **ISO 42001** : [AI_POLICY.md](AI_POLICY.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.2
- **ISO 12207** : [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.3
- **Index** : Ce document

### L
- **LLM** : [AI_POLICY.md](AI_POLICY.md) Section 5.2

### M
- **Maintenabilite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.7
- **Metadonnees** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 3

### N
- **Numerotation documents** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 2

### O
- **Offline** : [VISION.md](VISION.md) Section 5.1

### P
- **Performance** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.2
- **Phase gates** : [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.3
- **Prompts** : [prompts/](../prompts/), [prompts/CHANGELOG.md](../prompts/CHANGELOG.md)

### Q
- **Qualite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md)

### R
- **RAG** : [VISION.md](VISION.md) Section 3.1
- **Recall** : [TEST_PLAN.md](TEST_PLAN.md) Section 3.2
- **Retrieval** : [AI_POLICY.md](AI_POLICY.md) Section 3.2
- **Risques** : [VISION.md](VISION.md) Section 7, [AI_POLICY.md](AI_POLICY.md) Section 3

### S
- **Securite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.6
- **Structure projet** : [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.3

### T
- **Tests** : [TEST_PLAN.md](TEST_PLAN.md)
- **Tracabilite** : [DOC_CONTROL.md](DOC_CONTROL.md), [corpus/INVENTORY.md](../corpus/INVENTORY.md)

### U
- **Utilisabilite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.4

### V
- **Validation** : [TEST_PLAN.md](TEST_PLAN.md) Section 3.5
- **Version** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 4.3
- **Vision** : [VISION.md](VISION.md)

---

## 4. Index par norme ISO

| Norme | Documents applicables |
|-------|----------------------|
| ISO 9001:2015 | [DOC_CONTROL.md](DOC_CONTROL.md) |
| ISO 12207:2017 | [VISION.md](VISION.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) |
| ISO 25010:2023 | [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) |
| ISO 29119:2021 | [TEST_PLAN.md](TEST_PLAN.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) |
| ISO 42001:2023 | [AI_POLICY.md](AI_POLICY.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) |
| ISO 82045:2001 | [DOC_CONTROL.md](DOC_CONTROL.md) |
| ISO 999:1996 | Ce document |
| ISO 15489:2016 | [DOC_CONTROL.md](DOC_CONTROL.md) |

---

## 5. Matrice de tracabilite

| Exigence | Source | Implementation |
|----------|--------|----------------|
| Identification documents | ISO 9001 7.5.2 | [DOC_CONTROL.md](DOC_CONTROL.md) Section 2 |
| Controle des modifications | ISO 9001 7.5.3 | [DOC_CONTROL.md](DOC_CONTROL.md) Section 4.3 |
| Indexation | ISO 999 | Ce document |
| Metadonnees | ISO 82045 | [DOC_CONTROL.md](DOC_CONTROL.md) Section 3 |
| Retention | ISO 15489 | [DOC_CONTROL.md](DOC_CONTROL.md) Section 4.4 |
| Qualite logicielle | ISO 25010 | [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) |
| Gouvernance IA | ISO 42001 | [AI_POLICY.md](AI_POLICY.md) |
| Tests | ISO 29119 | [TEST_PLAN.md](TEST_PLAN.md) |
| Cycle de vie | ISO 12207 | [VISION.md](VISION.md), phases |

---

## 6. Recherche rapide

### Par phase projet
| Phase | Documents cles |
|-------|---------------|
| Phase 0 | VISION.md, AI_POLICY.md, ISO_STANDARDS_REFERENCE.md, DOC_CONTROL.md |
| Phase 1 | corpus/INVENTORY.md, prompts/CHANGELOG.md |
| Phase 2 | TEST_PLAN.md (Section 3.2) |
| Phase 3 | AI_POLICY.md (Section 3.2), TEST_PLAN.md (Section 3.3) |
| Phase 4 | QUALITY_REQUIREMENTS.md, TEST_PLAN.md (Section 3.4) |
| Phase 5 | TEST_PLAN.md (Section 3.5), guide utilisateur |

### Par role
| Role | Documents pertinents |
|------|---------------------|
| Developpeur | VISION.md, AI_POLICY.md, DVC_GUIDE.md, prompts/ |
| Testeur | TEST_PLAN.md, QUALITY_REQUIREMENTS.md |
| Product Owner | VISION.md, ISO_STANDARDS_REFERENCE.md |
| Auditeur | DOC_CONTROL.md, INDEX.md, ISO_STANDARDS_REFERENCE.md |

---

## 7. Historique du document

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-11 | Claude Opus 4.5 | Creation initiale |

---

*Index maintenu conformement a ISO 999:1996 dans le cadre du systeme de gestion documentaire du projet Pocket Arbiter.*
