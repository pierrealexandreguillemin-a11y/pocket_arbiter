# Gold Standard Specification - Pocket Arbiter

> **Document ID**: SPEC-GS-001
> **ISO Reference**: ISO 29119-3 (Test Documentation), ISO 25010, ISO 42001
> **Version**: 1.8
> **Date**: 2026-02-02
> **Statut**: Approuve
> **Classification**: Qualite
> **Auteur**: Claude Opus 4.5

---

## 1. Objet et Portee

Ce document definit les principes, exigences et normes appliquees au Gold Standard du projet Pocket Arbiter pour l'evaluation du systeme RAG.

### 1.1 Architecture Dual-RAG (VISION v2.0)

> **IMPORTANT**: Suite a VISION.md v2.0, les RAG FR et INTL sont **SEPARES**.
> Cause: Pollution mutuelle des corpus due a specificite metier et scopes differents.

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│     GOLD STANDARD FR            │    │     GOLD STANDARD INTL          │
│     (CE DOCUMENT)               │    │     (DOCUMENT SEPARE A CREER)   │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│ Status: ACTIF                   │    │ Status: OBSOLETE                │
│ GS FR v8.0: 420 questions       │    │ GS INTL v2.1: 93 questions      │
│ GS FR Annales unified           │    │ --> A REFAIRE from scratch      │
│ Corpus: 29 docs FFE             │    │ Corpus: INCOMPLET               │
│ Chunking: LangChain ✓           │    │ Chunking: OBSOLETE              │
│ Embeddings: EmbeddingGemma ✓    │    │ Embeddings: OBSOLETE            │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

### 1.2 Perimetre de ce Document

**FOCUS: RAG FRANCE uniquement**

| Gold Standard | Questions | Status | Database |
|---------------|-----------|--------|----------|
| GS FR Annales v8.0 | 420 (386 annales + 34 human) | Actif | corpus_mode_b_fr.db |
| **Total FR** | **420** | **ACTIF** | **SEPARE** |

**HORS PERIMETRE** (document separe a creer):

| Gold Standard | Questions | Status | Action |
|---------------|-----------|--------|--------|
| ~~GS INTL v2.1~~ | ~~93~~ | OBSOLETE | A refaire apres completion corpus FIDE |

### 1.3 Gold Standard FR Annales v8.0 (2026-02-02)

| Metrique | Valeur | Status |
|----------|--------|--------|
| Questions totales | 420 | ✅ |
| Documents couverts | 28/28 (100%) | ✅ |
| ID Schema | URN-like v1.0.0 | ✅ |
| requires_context | 92 (exclus) | ✅ |
| Testables | 328 | ✅ |
| Chunks alignes | 420/420 (100%) | ✅ |
| **Score humain** | **100%** | ✅ REFERENCE |

**Note importante v8.0**: Ce GS est un benchmark de **RAISONNEMENT**, pas de retrieval RAG classique.
Les reponses annales sont des reformulations, calculs et inferences, pas des extractions directes.
Le score humain de 100% confirme que toutes les reponses sont derivables du corpus.

**Distribution par namespace:**
| Namespace | Count |
|-----------|-------|
| ffe:annales:* | 386 |
| ffe:human:* | 34 |

**Distribution reasoning_class (Know Your RAG):**
| Classe | Count | % | Role |
|--------|-------|---|------|
| summary | 240 | 57% | Semantic bridge (LREM +19.2%) |
| fact_single | 162 | 39% | Ancrage factuel |
| arithmetic | 12 | 3% | Calcul/inference |
| reasoning | 6 | 1% | Multi-hop reasoning |

> **Justification scientifique (recherches web 2026-01-26)**:
> - LREM (Alibaba arXiv:2510.14321): Reasoning training = **+19.2% Q&A performance**
> - Know Your RAG (COLING 2025 arXiv:2411.19710): 95% datasets generes = fact_single (trap a eviter)
> - Notre distribution 57% summary = **AVANTAGE DELIBERE** pour fine-tuning on-device
> - Summary force comprehension semantique profonde vs patterns lexicaux superficiels

**ID Schema v1.0.0 (multi-corpus ready):**
```
{corpus}:{source}:{category}:{sequence}:{hash}

Exemples:
  ffe:annales:rules:001:a3f2b8c1
  ffe:human:rating:001:7d4e9f2a
  adversarial:squad-adv:rules:001:f1e2d3c4
```

> Voir: `docs/specs/GOLD_STANDARD_V6_ANNALES.md` pour specifications detaillees
> Voir: `tests/data/id_migration_map.json` pour mapping legacy IDs

---

## 2. References Normatives

| Norme | Application |
|-------|-------------|
| **ISO 29119-3** | Structure des donnees de test |
| **ISO 29119-4** | Techniques de test (coverage) |
| **ISO 25010** | Metriques qualite (Recall >= 90%) |
| **ISO 42001 A.6.2.2** | Provenance tracable (expected_chunk_id) |
| **ISO 42001 A.8.4** | Validation modeles IA |
| **arXiv:1806.03822** | SQuAD 2.0 - 25-35% unanswerable |
| **arXiv:2412.12300** | UAEval4RAG - 6 categories unanswerable |
| **arXiv:2004.14004** | SQuAD2-CR - 5 categories adversariales |

---

## 3. Principes du Gold Standard

### 3.1 Independance

> Les questions sont redigees **AVANT** de connaitre les resultats du retrieval.

- Questions basees sur expertise metier (arbitre FFE)
- Pas d'optimisation des questions pour le systeme
- `expected_pages` verifiees contre PDF source uniquement

### 3.2 Representativite

Distribution alignee sur usage reel:
- Questions simples (answerable): ~70%
- Questions complexes (multi-doc, multi-hop): ~15%
- Edge cases (hard cases): ~15%

### 3.3 Tracabilite (ISO 42001 A.6.2.2)

Chaque question ANSWERABLE contient:
- `id`: Identifiant unique (FR-Q001, INTL-Q001)
- `expected_docs`: Documents source
- `expected_pages`: Pages source verifiees
- `expected_chunk_id`: ID chunk Mode B (provenance exacte) **(depuis v5.30)**
- `validation.status`: Statut de validation
- `audit`: Date d'ajout/modification

> **Note**: `expected_chunk_id` reference le chunk dans `corpus_mode_b_*.db`

---

## 4. Categories de Questions (UAEval4RAG)

Basees sur [arXiv:2412.12300](https://arxiv.org/abs/2412.12300) - UAEval4RAG Framework.

### 4.1 Taxonomie des 6 Categories

| Cat | Type | Description | Objectif Test |
|-----|------|-------------|---------------|
| **1** | ANSWERABLE | Information presente et complete | Retrieval standard |
| **2** | PARTIAL_INFO | Information incomplete dans corpus | Graceful degradation |
| **3** | VOCABULARY_MISMATCH | Termes differents (synonymes) | Robustesse semantique |
| **4** | MULTI_HOP_IMPOSSIBLE | Necessite inference non disponible | Detection limite |
| **5** | FALSE_PREMISE | Question basee sur premisse fausse | Rejection appropriee |
| **6** | OUT_OF_SCOPE | Hors perimetre corpus | Graceful failure |

### 4.2 Distribution Cible vs Actuelle (v8.0)

| Categorie | % Cible | Actuel FR | Status |
|-----------|---------|-----------|--------|
| ANSWERABLE | 65-75% | 67.0% | ✅ CONFORME |
| PARTIAL_INFO | 1-3% | 1.3% | ✅ CONFORME |
| VOCABULARY_MISMATCH | 3-5% | 5.0% | ✅ CONFORME |
| MULTI_HOP_IMPOSSIBLE | 1-2% | 0.9% | ✅ CONFORME |
| FALSE_PREMISE | 3-5% | 4.7% | ✅ CONFORME |
| OUT_OF_SCOPE | 5-10% | 6.9% | ✅ CONFORME |
| **UNANSWERABLE TOTAL** | 25-35% | 33.0% | ✅ SQuAD 2.0 |

### 4.3 Exemples par Categorie

#### Cat 1: ANSWERABLE
```json
{
  "id": "FR-Q01",
  "question": "Quelle est la regle du toucher-jouer ?",
  "hard_type": "ANSWERABLE",
  "expected_pages": [41, 42]
}
```

#### Cat 2: PARTIAL_INFO
```json
{
  "id": "FR-Q135",
  "question": "Quel est le bareme complet d'indemnisation pour un arbitre AFE2 sur un Open de plus de 200 joueurs ?",
  "hard_type": "PARTIAL_INFO",
  "corpus_truth": "Bareme existe mais cas specifique 200+ joueurs non detaille"
}
```

#### Cat 3: VOCABULARY_MISMATCH
```json
{
  "id": "FR-Q77",
  "question": "Un joueur qui n'a joue aucune partie pendant 18 mois...",
  "hard_type": "VOCABULARY_MISMATCH",
  "corpus_truth": "periode d'un an (12 mois) - p.183 S7.2.2"
}
```

#### Cat 4: MULTI_HOP_IMPOSSIBLE
```json
{
  "id": "FR-Q136",
  "question": "Si un joueur fait 3 coups illegaux en blitz et que l'arbitre n'a pas vu les 2 premiers, quelle est la sanction cumulative ?",
  "hard_type": "MULTI_HOP_IMPOSSIBLE",
  "corpus_truth": "Regles existent separement mais inference combinee non explicite"
}
```

#### Cat 5: FALSE_PREMISE
```json
{
  "id": "FR-Q137",
  "question": "Selon l'article 15.3 du reglement FFE, quand peut-on reclamer une nulle technique ?",
  "hard_type": "FALSE_PREMISE",
  "corpus_truth": "Article 15.3 n'existe pas dans le reglement FFE"
}
```

#### Cat 6: OUT_OF_SCOPE
```json
{
  "id": "FR-Q138",
  "question": "Quelles sont les regles specifiques pour les tournois d'echecs en ligne sur Chess.com ?",
  "hard_type": "OUT_OF_SCOPE",
  "corpus_truth": "Chess.com hors perimetre corpus FFE/FIDE"
}
```

---

## 5. Format des Questions

### 5.1 Schema JSON

```json
{
  "id": "FR-Q{N}",
  "question": "string - formulation naturelle",
  "category": "enum[tournoi|arbitrage|regles_jeu|classement|...]",
  "expected_docs": ["fichier.pdf"],
  "keywords": ["keyword1", "keyword2"],
  "expected_pages": [page1, page2],
  "expected_chunk_id": "source.pdf-p{page}-parent{N}-child{N}",  // NOUVEAU v5.30
  "metadata": {
    "type": "enum[definition|regle|arbitrage|edge|admin]",
    "chapter": "string (ex: 6.1)",
    "hard_case": "boolean",
    "hard_type": "enum[ANSWERABLE|PARTIAL_INFO|VOCABULARY_MISMATCH|MULTI_HOP_IMPOSSIBLE|FALSE_PREMISE|OUT_OF_SCOPE]",
    "hard_reason": "string - explication si hard_case",
    "corpus_truth": "string - verite terrain verifiee",
    "test_purpose": "string - objectif du test"
  },
  "validation": {
    "status": "enum[VALIDATED|HARD_CASE|PENDING|FAILED_RETRIEVAL]",
    "method": "manual_verification",
    "recall_actual": "string (ex: 67%)",
    "audit_note": "string"
  },
  "difficulty": "enum[easy|medium|hard]",
  "audit": "added_YYYY-MM-DD"
}
```

### 5.2 Champs Obligatoires

| Champ | Requis | Description |
|-------|--------|-------------|
| `id` | Oui | Identifiant unique |
| `question` | Oui | Texte de la question |
| `expected_docs` | Oui* | Document(s) source |
| `expected_pages` | Oui* | Pages verifiees |
| `expected_chunk_id` | Oui* | Chunk Mode B (ISO 42001 A.6.2.2) |
| `metadata.hard_type` | Oui | Categorie UAEval4RAG |
| `validation.status` | Oui | Statut validation |

> *Non requis pour questions unanswerable (FALSE_PREMISE, OUT_OF_SCOPE, etc.)

---

## 6. Metriques et Seuils

### 6.1 Metriques ISO 25010

| Metrique | Formule | Seuil | Actuel |
|----------|---------|-------|--------|
| **Recall@5 FR** | Pages trouvees / Pages attendues | >= 90% | 91.56% |
| **Recall@5 INTL** | Pages trouvees / Pages attendues | >= 90% | 93.22% |
| **Tolerance** | +/- 2 pages adjacentes | Accepte | Oui |
| **Coverage** | Questions / Categories | 100% | 13/13 |

### 6.2 Etat Actuel (2026-02-02)

| Corpus | Version | Questions | requires_context | chunk_id | Status |
|--------|---------|-----------|-----------------|----------|--------|
| FR | v8.0 | 420 | 92 (22%) | 420/420 | ✅ CB-01 PASS (verbatim 100%) |
| INTL | v2.1 | 93 | 26 (28%) | 67/67 | ⚠️ OBSOLETE — a reconstruire |
| **Total** | | **513** | 118 | 487 | **FR: ISO CONFORME** |

> **Note FR v8.0**: Les 420 `expected_chunk_id` sont valides par verification verbatim (CB-01=100%).
> P4 audit fixes appliques: correct_answer, unified taxonomy, markdown cleanup, difficulty variance.

---

## 7. Processus de Validation

### 7.1 Ajout de Question

1. Rediger question en langage naturel
2. Identifier `expected_docs` et `expected_pages` dans PDF source
3. Classifier selon taxonomie UAEval4RAG (hard_type)
4. Ajouter `corpus_truth` si hard_case
5. Executer test recall
6. Mettre a jour `validation.status`

### 7.2 Audit de Question

```bash
# Validation recall
python -m scripts.pipeline.tests.test_recall --tolerance 2 -v

# Verification coverage
python -m scripts.pipeline.tests.test_recall --coverage
```

---

## 8. Historique et Audit Trail

### 8.1 Versions Gold Standard FR

| Version | Date | Questions | Recall | Changements |
|---------|------|-----------|--------|-------------|
| 5.0 | 2026-01-18 | 75 | 78% | Creation initiale |
| 5.7 | 2026-01-19 | 68 | 97.06% | Audit pages (23 corrections) |
| 5.16 | 2026-01-20 | 93 | 95.70% | Ajout questions Ch5-6 |
| 5.22 | 2026-01-20 | 134 | 91.17% | +41 hard cases (annales) |
| 5.23 | 2026-01-20 | 134 | 91.17% | Ajout corpus_truth, hard_type |
| 5.26 | 2026-01-20 | 150 | 91.56% | +16 questions, normalisation ISO |
| 5.29 | 2026-01-22 | 237 | TBD | +87 adversarial SQuAD2-CR/UAEval4RAG |
| **5.30** | 2026-01-23 | **318** | TBD | +81 adversarial, **+expected_chunk_id** |

### 8.1.1 Versions Gold Standard INTL

| Version | Date | Questions | Recall | Changements |
|---------|------|-----------|--------|-------------|
| 1.0 | 2026-01-19 | 25 | 80.00% | Creation initiale |
| 2.0 | 2026-01-20 | 43 | 93.22% | +18 hard cases UAEval4RAG, audit ISO 29119 |
| **2.1** | 2026-01-23 | **93** | TBD | +50 questions, **+expected_chunk_id** |

### 8.2 Corrections Majeures

| Question | Correction | Date | Raison |
|----------|------------|------|--------|
| FR-Q77 | hard_type=VOCABULARY_MISMATCH | 2026-01-20 | "18 mois" vs "un an" |
| FR-Q94 | corpus_truth ajoute | 2026-01-20 | Best practice arXiv:2412.12300 |
| FR-Q68 | expected_pages [29]->[141] | 2026-01-19 | p29 etait faux |
| FR-Q18 | expected_pages [58]->[57] | 2026-01-19 | Rapides vs Blitz |
| FR-Q145 | expected_pages supprime | 2026-01-23 | Incoherent avec FALSE_PREMISE |
| FR-Q04 | chunk_id revert | 2026-01-23 | Correction erronee revertee |

---

## 9. Documents Obsoletes (A Supprimer)

| Fichier | Raison | Action |
|---------|--------|--------|
| `docs/gold_standard_status_v1.md` | Remplace par ce document | Supprimer |
| `docs/GOLD_STANDARD_AUDIT_2026-01-19.md` | Fusionne dans S8 | Supprimer |
| `docs/GOLD_STANDARD_AUDIT_2026-01-20.md` | Fusionne dans S8 | Supprimer |

---

## 10. Minima Standards Industrie (OBLIGATOIRE)

> **Ref**: QUALITY_REQUIREMENTS.md Section 4

### 10.1 Taille et Distribution

| Metrique | Minimum | Standard Industrie | Actuel | Status |
|----------|---------|-------------------|--------|--------|
| Taille GS total | >= 200 | BEIR (300-50k) | 420 FR | ✅ |
| Questions adversariales | 25-30% | SQuAD 2.0 (33%) | 22% (92 rc) | ⚠️ A COMPLETER |
| Couverture categories | >= 80% | BEIR diversity | 100% (13 cat) | ✅ |
| Deduplication | < 5% sim | SoftDedup | TBD | ⚠️ A VERIFIER |

### 10.2 Qualite Semantique

| Metrique | Minimum | Standard Industrie | Actuel | Status |
|----------|---------|-------------------|--------|--------|
| Context-grounded | 100% | RAGen, Source2Synth | **100%** (CB-04 BY DESIGN) | ✅ |
| Validation humaine | 100% GS | Industrie | 100% | ✅ |
| expected_chunk_id | 100% answerable | ISO 42001 A.6.2.2 | **100%** (420/420) | ✅ |

### 10.3 Actions Requises

| Action | Priorite | Standard | Status |
|--------|----------|----------|--------|
| Reformulation BY DESIGN | P0 | RAGen | ✅ FAIT (P2 manual_by_design) |
| Deduplication SemHash | P1 | SoftDedup | ⚠️ A verifier |
| Mapping chunk_id 100% | P1 | ISO 42001 | ✅ FAIT (420/420) |
| MMTEB evaluation INTL | P2 | MMTEB | ❌ INTL obsolete |

### 10.4 Benchmarks Cibles

| Benchmark | Corpus | Cible | Standard |
|-----------|--------|-------|----------|
| MTEB Retrieval | FR | >= 0.70 NDCG@10 | MTEB leaderboard |
| MMTEB Retrieval | INTL | >= 0.65 NDCG@10 | MMTEB multilingual |
| BEIR custom | FR+INTL | >= 0.75 Recall@5 | BEIR zero-shot |
| ARES Context Relevance | FR+INTL | >= 0.80 | ARES PPI |

---

## 11. References

### 11.1 Standards Adversariaux

| Reference | Titre | Application |
|-----------|-------|-------------|
| [arXiv:2412.12300](https://arxiv.org/abs/2412.12300) | UAEval4RAG Framework | 6 categories unanswerable |
| [arXiv:1806.03822](https://arxiv.org/abs/1806.03822) | SQuAD 2.0 | 33% unanswerable ratio |
| [arXiv:2510.11956](https://arxiv.org/abs/2510.11956) | CRUMQs Evaluation | Question quality |
| [SPEC-ADV-V1](specs/ADVERSARIAL_QUESTIONS_STRATEGY.md) | Strategie Adversariales Pocket Arbiter | Taxonomie unifiee |

### 11.2 Standards Context-Grounded Generation

| Reference | Titre | Application |
|-----------|-------|-------------|
| [RAGen](https://arxiv.org/abs/2411.14831) | Semantically grounded QAC datasets | Validation BY DESIGN |
| [Source2Synth](https://arxiv.org/abs/2409.08239) | Grounded in real-world sources | Generation ancree |
| [FACTS Grounding](https://huggingface.co/datasets/google/facts-grounding) | Google benchmark | Grounded responses |

### 11.3 Standards Data Quality

| Reference | Titre | Application |
|-----------|-------|-------------|
| [SoftDedup](https://arxiv.org/abs/2407.06564) | Soft Deduplication | Deduplication fuzzy |
| [SemHash](https://github.com/MinishLab/semhash) | Semantic hashing | Fast dedup embeddings |
| [Lin et al.](https://arxiv.org/abs/2406.15126) | Synthetic Data Survey | Best practices |

### 11.4 Standards Benchmarks

| Reference | Titre | Application |
|-----------|-------|-------------|
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | Massive Text Embedding Benchmark | FR evaluation |
| [MMTEB](https://arxiv.org/abs/2502.13595) | Massive Multicultural TEB | INTL evaluation |
| [BEIR](https://github.com/beir-cellar/beir) | Benchmarking IR | Retrieval standard |

### 11.5 Standards ISO

| Standard | Titre | Application |
|----------|-------|-------------|
| [ISO 29119-3](https://www.iso.org/standard/79428.html) | Test Documentation | Structure donnees test |
| [ISO 25010](https://www.iso.org/standard/35733.html) | Software Quality | Metriques qualite |
| [ISO 42001](https://www.iso.org/standard/81230.html) | AI Management | Provenance, lineage |

---

*Ce document remplace gold_standard_status_v1.md et les audits 2026-01-19/20.*
*Derniere mise a jour: 2026-02-02 - GS Annales v8.0 (P4 audit fixes, 15 criteres bloquants PASS)*
