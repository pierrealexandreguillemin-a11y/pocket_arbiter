# Index Principal de la Documentation

> **Document ID**: DOC-IDX-001
> **ISO Reference**: ISO 999:1996 - Lignes directrices pour l'indexation
> **Version**: 2.0
> **Date**: 2026-01-25
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
│   ├── ARCHITECTURE.md            # Architecture technique (SPEC-ARCH-001)
│   ├── TEST_PLAN.md               # Plan de tests (TEST-PLAN-001)
│   ├── CHUNKING_STRATEGY.md       # Strategie chunking (SPEC-CHUNK-001)
│   ├── ISO_MODEL_DEPLOYMENT_ANALYSIS.md  # Analyse modele (DOC-MODEL-001)
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
| SPEC-ARCH-001 | [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture technique | Draft |
| SPEC-REQ-001 | [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) | Exigences qualite ISO 25010 | En cours |
| SPEC-CHUNK-001 | [CHUNKING_STRATEGY.md](CHUNKING_STRATEGY.md) | Strategie chunking Parent-Child v6.0 | Approuve |
| SPEC-RETR-001 | [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) | Pipeline retrieval v4.0 dual-mode | Approuve |
| SPEC-SCH-001 | [CHUNK_SCHEMA.md](CHUNK_SCHEMA.md) | Schema JSON chunks v2.1 | Approuve |

### 2.3 Intelligence artificielle

| ID | Document | Description | Statut |
|----|----------|-------------|--------|
| DOC-POL-001 | [AI_POLICY.md](AI_POLICY.md) | Politique IA responsable ISO 42001 | Draft |
| DOC-MODEL-001 | [ISO_MODEL_DEPLOYMENT_ANALYSIS.md](ISO_MODEL_DEPLOYMENT_ANALYSIS.md) | Analyse deploiement modele | Approuve |
| RES-LORA-001 | [research/LORA_FINETUNING_GUIDE.md](research/LORA_FINETUNING_GUIDE.md) | Guide fine-tuning MRL+LoRA | Draft |
| RES-VECTOR-001 | [ISO_VECTOR_SOLUTIONS.md](ISO_VECTOR_SOLUTIONS.md) | Solutions vector-based RAG | En cours |
| PROM-LOG-001 | [prompts/CHANGELOG.md](../prompts/CHANGELOG.md) | Historique des versions de prompts | Actif |

### 2.4 Tests et qualite

| ID | Document | Description | Statut |
|----|----------|-------------|--------|
| TEST-PLAN-001 | [TEST_PLAN.md](TEST_PLAN.md) | Plan de tests ISO 29119 | Draft |
| SPEC-GS-001 | [GOLD_STANDARD_SPECIFICATION.md](GOLD_STANDARD_SPECIFICATION.md) | Specification Gold Standard | Approuve |
| SPEC-GS-V7 | [specs/GOLD_STANDARD_V6_ANNALES.md](specs/GOLD_STANDARD_V6_ANNALES.md) | Gold Standard v7.4.7 (420Q, 82.8% answerability) | Approuve |
| SPEC-GS-OPT | [specs/GS_ANNALES_V7_OPTIMIZATION_SPEC.md](specs/GS_ANNALES_V7_OPTIMIZATION_SPEC.md) | Optimisation GS v7 pour triplets | En cours |
| SPEC-ADV-V1 | [specs/ADVERSARIAL_QUESTIONS_STRATEGY.md](specs/ADVERSARIAL_QUESTIONS_STRATEGY.md) | Strategie questions adversariales SQuAD 2.0 | Approuve |
| SPEC-UTD-001 | [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) | Generation Donnees Unifiees | Draft |

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
- **Android** : [VISION.md](VISION.md) Section 5.1, [ARCHITECTURE.md](ARCHITECTURE.md) Section 5.2
- **Annales DNA** : [ANNALES_STRUCTURE.md](ANNALES_STRUCTURE.md), [specs/GOLD_STANDARD_V6_ANNALES.md](specs/GOLD_STANDARD_V6_ANNALES.md)
- **ARES (Evaluation RAG)** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 5
- **Architecture** : [ARCHITECTURE.md](ARCHITECTURE.md)
- **Approbations** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 4.2
- **Archivage** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 4.4

### B
- **BEIR (Benchmark)** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 3.4.3, [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4.3
- **BY DESIGN (Validation)** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 1.3-1.4, [specs/TRIPLET_GENERATION_SPEC.md](specs/TRIPLET_GENERATION_SPEC.md) Section 4.5

### C
- **Chunking** : [CHUNKING_STRATEGY.md](CHUNKING_STRATEGY.md)
- **Citation** : [AI_POLICY.md](AI_POLICY.md) Section 7.2
- **Compatibilite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.3
- **Context-Grounded Generation** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 1.4, [specs/TRIPLET_GENERATION_SPEC.md](specs/TRIPLET_GENERATION_SPEC.md) Section 4.5
- **Controle documentaire** : [DOC_CONTROL.md](DOC_CONTROL.md)
- **Corpus** : [corpus/INVENTORY.md](../corpus/INVENTORY.md)
- **Couverture tests** : [TEST_PLAN.md](TEST_PLAN.md) Section 3

### D
- **Deduplication (SoftDedup, SemHash)** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4.1, [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 2.3
- **Disclaimer** : [AI_POLICY.md](AI_POLICY.md) Section 7.1
- **Dual-RAG Architecture** : [VISION.md](VISION.md) Section 1.1-1.2, [GOLD_STANDARD_SPECIFICATION.md](GOLD_STANDARD_SPECIFICATION.md) Section 1.1, [CHUNKING_STRATEGY.md](CHUNKING_STRATEGY.md) Section 0, [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) Section 0
- **Docling** : [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) Section 1, [CHUNKING_STRATEGY.md](CHUNKING_STRATEGY.md)
- **Documentation** : [DOC_CONTROL.md](DOC_CONTROL.md)
- **Dual-mode chunking** : [CHUNKING_STRATEGY.md](CHUNKING_STRATEGY.md) Section 3, [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md)
- **DVC** : [DVC_GUIDE.md](DVC_GUIDE.md)

### E
- **Embeddings** : [AI_POLICY.md](AI_POLICY.md) Section 5.1, [research/LORA_FINETUNING_GUIDE.md](research/LORA_FINETUNING_GUIDE.md)
- **EmbeddingGemma** : [research/LORA_FINETUNING_GUIDE.md](research/LORA_FINETUNING_GUIDE.md), [ISO_VECTOR_SOLUTIONS.md](ISO_VECTOR_SOLUTIONS.md)
- **Exigences qualite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md)

### F
- **FFE (Federation Francaise des Echecs)** : [VISION.md](VISION.md) Section 2.1
- **FIDE** : [VISION.md](VISION.md) Section 2.1
- **Fiabilite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.5
- **Fine-tuning** : [research/LORA_FINETUNING_GUIDE.md](research/LORA_FINETUNING_GUIDE.md), [ISO_VECTOR_SOLUTIONS.md](ISO_VECTOR_SOLUTIONS.md)

### G
- **Gates (phase gates)** : [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.3
- **Gold Standard** : [GOLD_STANDARD_SPECIFICATION.md](GOLD_STANDARD_SPECIFICATION.md), [specs/GOLD_STANDARD_V6_ANNALES.md](specs/GOLD_STANDARD_V6_ANNALES.md)
- **Grounding** : [AI_POLICY.md](AI_POLICY.md) Section 3.2

### H
- **Hallucination** : [AI_POLICY.md](AI_POLICY.md) Section 3.2, [TEST_PLAN.md](TEST_PLAN.md) Section 3.3
- **Hard Negatives (NV-Embed-v2)** : [specs/TRIPLET_GENERATION_SPEC.md](specs/TRIPLET_GENERATION_SPEC.md) Section 11.3, [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 3.3, [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4.2
- **HybridChunker** : [CHUNKING_STRATEGY.md](CHUNKING_STRATEGY.md) Section 3.1, [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) Section 2

### I
- **ISO 25010** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.1
- **ISO 29119** : [TEST_PLAN.md](TEST_PLAN.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.4
- **ISO 42001** : [AI_POLICY.md](AI_POLICY.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.2
- **ISO 12207** : [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.3
- **Index** : Ce document

### L
- **LLM** : [AI_POLICY.md](AI_POLICY.md) Section 5.2
- **LoRA (Low-Rank Adaptation)** : [research/LORA_FINETUNING_GUIDE.md](research/LORA_FINETUNING_GUIDE.md)

### M
- **Maintenabilite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.7
- **Matryoshka (MRL)** : [research/LORA_FINETUNING_GUIDE.md](research/LORA_FINETUNING_GUIDE.md), [ISO_VECTOR_SOLUTIONS.md](ISO_VECTOR_SOLUTIONS.md)
- **Metadonnees** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 3
- **MMTEB (Multilingual)** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4.3 (INTL), [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 2.5
- **Model Collapse (Prevention)** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 2.3, [specs/TRIPLET_GENERATION_SPEC.md](specs/TRIPLET_GENERATION_SPEC.md) Section 11.4
- **MRL (Matryoshka Representation Learning)** : [research/LORA_FINETUNING_GUIDE.md](research/LORA_FINETUNING_GUIDE.md)
- **MTEB (Benchmark)** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4.3, [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 2.5, [GOLD_STANDARD_SPECIFICATION.md](GOLD_STANDARD_SPECIFICATION.md) Section 10.4

### N
- **Numerotation documents** : [DOC_CONTROL.md](DOC_CONTROL.md) Section 2
- **NV-Embed-v2** : [specs/TRIPLET_GENERATION_SPEC.md](specs/TRIPLET_GENERATION_SPEC.md) Section 11.3, [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4.2

### O
- **Offline** : [VISION.md](VISION.md) Section 5.1

### P
- **Page provenance** : [CHUNK_SCHEMA.md](CHUNK_SCHEMA.md) Section 2, [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) (ISO 42001 A.6.2.2)
- **Parent-Child chunking** : [CHUNKING_STRATEGY.md](CHUNKING_STRATEGY.md) Section 3, [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) Section 2
- **Performance** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.2
- **Phase gates** : [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.3
- **Pipeline** : [ARCHITECTURE.md](ARCHITECTURE.md) Section 5.1
- **Prompts** : [prompts/](../prompts/), [prompts/CHANGELOG.md](../prompts/CHANGELOG.md)

### Q
- **Qualite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md), [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md)
- **Quality Gates (Training Data)** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 2.4

### R
- **RAG** : [VISION.md](VISION.md) Section 3.1
- **RAGen (Context-Grounded)** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 1.4, [GOLD_STANDARD_SPECIFICATION.md](GOLD_STANDARD_SPECIFICATION.md) Section 11.2
- **RAGAS (Evaluation)** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 3.4.4, [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4.3
- **Recall** : [TEST_PLAN.md](TEST_PLAN.md) Section 3.2
- **Reformulation (Questions)** : [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 3.2
- **Retrieval** : [AI_POLICY.md](AI_POLICY.md) Section 3.2
- **Risques** : [VISION.md](VISION.md) Section 7, [AI_POLICY.md](AI_POLICY.md) Section 3

### S
- **Securite** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 2.6
- **SemHash (Deduplication)** : [specs/TRIPLET_GENERATION_SPEC.md](specs/TRIPLET_GENERATION_SPEC.md) Section 11.7, [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 11.5
- **SoftDedup (Deduplication)** : [specs/TRIPLET_GENERATION_SPEC.md](specs/TRIPLET_GENERATION_SPEC.md) Section 11.4, [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4.1
- **Standards Industrie (Training Data)** : [QUALITY_REQUIREMENTS.md](QUALITY_REQUIREMENTS.md) Section 4, [GOLD_STANDARD_SPECIFICATION.md](GOLD_STANDARD_SPECIFICATION.md) Section 10, [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md) Section 2.3
- **Structure projet** : [ISO_STANDARDS_REFERENCE.md](ISO_STANDARDS_REFERENCE.md) Section 1.3

### T
- **Tests** : [TEST_PLAN.md](TEST_PLAN.md)
- **Tracabilite** : [DOC_CONTROL.md](DOC_CONTROL.md), [corpus/INVENTORY.md](../corpus/INVENTORY.md)
- **Triplets (Training)** : [specs/TRIPLET_GENERATION_SPEC.md](specs/TRIPLET_GENERATION_SPEC.md), [specs/UNIFIED_TRAINING_DATA_SPEC.md](specs/UNIFIED_TRAINING_DATA_SPEC.md)

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
| Phase 0 | VISION.md, AI_POLICY.md, ISO_STANDARDS_REFERENCE.md, DOC_CONTROL.md, ARCHITECTURE.md |
| Phase 1 | CHUNKING_STRATEGY.md, ISO_MODEL_DEPLOYMENT_ANALYSIS.md, corpus/INVENTORY.md |
| Phase 2 | ARCHITECTURE.md (Section 5.2), TEST_PLAN.md (Section 3.2) |
| Phase 3 | AI_POLICY.md (Section 3.2), TEST_PLAN.md (Section 3.3) |
| Phase 4 | QUALITY_REQUIREMENTS.md, TEST_PLAN.md (Section 3.4) |
| Phase 5 | TEST_PLAN.md (Section 3.5), guide utilisateur |

### Par role
| Role | Documents pertinents |
|------|---------------------|
| Developpeur | VISION.md, ARCHITECTURE.md, AI_POLICY.md, DVC_GUIDE.md, prompts/ |
| Testeur | TEST_PLAN.md, QUALITY_REQUIREMENTS.md |
| Product Owner | VISION.md, ARCHITECTURE.md, ISO_STANDARDS_REFERENCE.md |
| Auditeur | DOC_CONTROL.md, INDEX.md, ISO_STANDARDS_REFERENCE.md |

---

## 7. Historique du document

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-11 | Claude Opus 4.5 | Creation initiale |
| 1.1 | 2026-01-18 | Claude Opus 4.5 | Ajout CHUNKING_STRATEGY.md, ISO_MODEL_DEPLOYMENT_ANALYSIS.md |
| 1.2 | 2026-01-21 | Claude Opus 4.5 | Ajout LORA_FINETUNING_GUIDE.md, ISO_VECTOR_SOLUTIONS.md, index LoRA/MRL |
| 1.3 | 2026-01-22 | Claude Opus 4.5 | Ajout RETRIEVAL_PIPELINE.md, CHUNK_SCHEMA.md, index Docling/HybridChunker/Parent-Child/Page provenance |
| 1.4 | 2026-01-23 | Claude Opus 4.5 | Benchmark chunking optimizations (dual-size -5.22%, semantic -4.05% vs baseline 86.94%) |
| 1.5 | 2026-01-24 | Claude Opus 4.5 | Ajout Gold Standard v6.5, ANNALES_STRUCTURE.md, index Annales DNA |
| 1.6 | 2026-01-24 | Claude Opus 4.5 | Ajout SPEC-UTD-001 (Unified Training Data), index ARES/BEIR/RAGAS/Triplets |
| 1.7 | 2026-01-24 | Claude Opus 4.5 | **Ajout index standards industrie**: BY DESIGN, Context-Grounded, Deduplication, Hard Negatives, MTEB, MMTEB (INTL), Model Collapse, NV-Embed-v2, Quality Gates, RAGen, SemHash, SoftDedup, Standards Industrie |
| 1.8 | 2026-01-24 | Claude Opus 4.5 | **Ajout Dual-RAG Architecture** (VISION v2.0): index separation FR/INTL, references croisees tous docs concernes |

---

*Index maintenu conformement a ISO 999:1996 dans le cadre du systeme de gestion documentaire du projet Pocket Arbiter.*
