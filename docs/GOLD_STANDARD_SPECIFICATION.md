# Gold Standard Specification - Pocket Arbiter

> **Document ID**: SPEC-GS-001
> **ISO Reference**: ISO 29119-3 (Test Documentation), ISO 25010, ISO 42001
> **Version**: 1.1
> **Date**: 2026-01-20
> **Statut**: Approuve
> **Classification**: Qualite
> **Auteur**: Claude Opus 4.5

---

## 1. Objet et Portee

Ce document definit les principes, exigences et normes appliquees au Gold Standard du projet Pocket Arbiter pour l'evaluation du systeme RAG.

**Perimetre:**
- Gold Standard FR: 150 questions (corpus reglements FFE, 46 hard cases)
- Gold Standard INTL: 43 questions (corpus FIDE Arbiters Manual, 12 hard cases)
- **Total**: 193 questions (58 hard cases)

---

## 2. References Normatives

| Norme | Application |
|-------|-------------|
| **ISO 29119-3** | Structure des donnees de test |
| **ISO 29119-4** | Techniques de test (coverage) |
| **ISO 25010** | Metriques qualite (Recall >= 90%) |
| **ISO 42001 A.8.4** | Validation modeles IA |
| **arXiv:2412.12300** | UAEval4RAG - 6 categories unanswerable |

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

### 3.3 Tracabilite (ISO 42001)

Chaque question contient:
- `id`: Identifiant unique (FR-Q001, INTL-Q001)
- `expected_pages`: Pages source verifiees
- `validation.status`: Statut de validation
- `audit`: Date d'ajout/modification

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

### 4.2 Distribution Cible

| Categorie | % Cible | Actuel FR | Gap |
|-----------|---------|-----------|-----|
| ANSWERABLE | 70% | ~85% | -15% |
| PARTIAL_INFO | 5% | ~2% | +3% |
| VOCABULARY_MISMATCH | 10% | ~8% | +2% |
| MULTI_HOP_IMPOSSIBLE | 5% | ~2% | +3% |
| FALSE_PREMISE | 5% | ~1% | +4% |
| OUT_OF_SCOPE | 5% | ~2% | +3% |

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
| `expected_docs` | Oui | Document(s) source |
| `expected_pages` | Oui | Pages verifiees |
| `metadata.hard_type` | Oui | Categorie UAEval4RAG |
| `validation.status` | Oui | Statut validation |

---

## 6. Metriques et Seuils

### 6.1 Metriques ISO 25010

| Metrique | Formule | Seuil | Actuel |
|----------|---------|-------|--------|
| **Recall@5 FR** | Pages trouvees / Pages attendues | >= 90% | 91.56% |
| **Recall@5 INTL** | Pages trouvees / Pages attendues | >= 90% | 93.22% |
| **Tolerance** | +/- 2 pages adjacentes | Accepte | Oui |
| **Coverage** | Questions / Categories | 100% | 13/13 |

### 6.2 Etat Actuel (2026-01-20)

| Corpus | Version | Questions | Hard Cases | Recall | Status |
|--------|---------|-----------|------------|--------|--------|
| FR | v5.26 | 150 | 46 (31%) | 91.56% | PASS |
| INTL | v2.0 | 43 | 12 (28%) | 93.22% | PASS |
| **Total** | | **193** | 58 | - | **ISO PASS** |

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

### 8.1.1 Versions Gold Standard INTL

| Version | Date | Questions | Recall | Changements |
|---------|------|-----------|--------|-------------|
| 1.0 | 2026-01-19 | 25 | 80.00% | Creation initiale |
| 2.0 | 2026-01-20 | 43 | 93.22% | +18 hard cases UAEval4RAG, audit ISO 29119 |

### 8.2 Corrections Majeures

| Question | Correction | Date | Raison |
|----------|------------|------|--------|
| FR-Q77 | hard_type=VOCABULARY_MISMATCH | 2026-01-20 | "18 mois" vs "un an" |
| FR-Q94 | corpus_truth ajoute | 2026-01-20 | Best practice arXiv:2412.12300 |
| FR-Q68 | expected_pages [29]->[141] | 2026-01-19 | p29 etait faux |
| FR-Q18 | expected_pages [58]->[57] | 2026-01-19 | Rapides vs Blitz |

---

## 9. Documents Obsoletes (A Supprimer)

| Fichier | Raison | Action |
|---------|--------|--------|
| `docs/gold_standard_status_v1.md` | Remplace par ce document | Supprimer |
| `docs/GOLD_STANDARD_AUDIT_2026-01-19.md` | Fusionne dans S8 | Supprimer |
| `docs/GOLD_STANDARD_AUDIT_2026-01-20.md` | Fusionne dans S8 | Supprimer |

---

## 10. References

- [arXiv:2412.12300](https://arxiv.org/abs/2412.12300) - UAEval4RAG Framework
- [arXiv:2510.11956](https://arxiv.org/abs/2510.11956) - CRUMQs Evaluation
- [ISO 29119-3](https://www.iso.org/standard/79428.html) - Test Documentation
- [ISO 25010](https://www.iso.org/standard/35733.html) - Software Quality

---

*Ce document remplace gold_standard_status_v1.md et les audits 2026-01-19/20.*
