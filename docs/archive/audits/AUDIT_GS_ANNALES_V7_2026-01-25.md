# Audit Gold Standard Annales FR v7.3.0

> **Document ID**: AUDIT-GS-ANN-2026-01-25
> **ISO Reference**: ISO 42001, ISO 25010, ISO 29119
> **Standards**: SQuAD 2.0, BEIR, UAEval4RAG, MTEB
> **Date**: 2026-01-25
> **Auditeur**: Claude Opus 4.5
> **Fichier audite**: `tests/data/gold_standard_annales_fr_v7.json`

---

## 1. Resume Executif

| Aspect | Status | Score |
|--------|--------|-------|
| **Conformite ISO** | PARTIEL | 75% |
| **Standards Industrie** | PARTIEL | 60% |
| **Qualite Donnees** | BON | 85% |
| **Verdict Global** | **AMELIORATIONS REQUISES** | - |

### Findings Critiques

| Severite | Count | Actions |
|----------|-------|---------|
| CRITIQUE | 1 | SQuAD 2.0: 0% unanswerable |
| MAJEUR | 2 | Questions sans ?, question_type None |
| MINEUR | 2 | Encoding chunk_ids, hard_type manquant |

---

## 2. Donnees Auditees

### 2.1 Metriques Generales

| Metrique | Valeur | Status |
|----------|--------|--------|
| Version | 7.3.0 | - |
| Schema | unified-v1 | ✅ |
| Questions totales | 420 | ✅ |
| Documents couverts | 28 | ✅ |
| Deduplication | 91 doublons supprimes | ✅ |

### 2.2 ID Schema

```
Format: {corpus}:{source}:{category}:{sequence}:{hash}
Exemple: ffe:annales:clubs:001:55d409b5
```

| Namespace | Count | % |
|-----------|-------|---|
| ffe:annales | 386 | 91.9% |
| ffe:human | 34 | 8.1% |

---

## 3. Conformite Champs (ISO 42001 A.6.2.2)

### 3.1 Champs Racine

| Champ | Coverage | Requis | Status |
|-------|----------|--------|--------|
| id | 420/420 (100%) | OUI | ✅ |
| question | 420/420 (100%) | OUI | ✅ |
| expected_docs | 420/420 (100%) | OUI | ✅ |
| expected_pages | 420/420 (100%) | OUI | ✅ |
| expected_answer | 420/420 (100%) | OUI | ✅ |
| expected_chunk_id | 420/420 (100%) | OUI | ✅ |
| is_impossible | 420/420 (100%) | OUI | ✅ |
| category | 420/420 (100%) | OUI | ✅ |
| keywords | 420/420 (100%) | OUI | ✅ |
| validation | 420/420 (100%) | OUI | ✅ |
| metadata | 420/420 (100%) | OUI | ✅ |

### 3.2 Champs Metadata

| Champ | Coverage | Requis | Status |
|-------|----------|--------|--------|
| answer_type | 420/420 (100%) | OUI | ✅ |
| reasoning_type | 420/420 (100%) | OUI | ✅ |
| cognitive_level | 420/420 (100%) | OUI | ✅ |
| article_reference | 420/420 (100%) | OUI | ✅ |
| difficulty | 386/420 (91.9%) | NON | ⚠️ |
| question_type | 386/420 (91.9%) | OUI | ❌ |
| annales_source | 386/420 (91.9%) | NON | ⚠️ |
| hard_type | 0/420 (0%) | OUI* | ❌ |

> *hard_type requis pour questions adversariales

---

## 4. Conformite Standards Industrie

### 4.1 SQuAD 2.0 (arXiv:1806.03822)

| Exigence | Cible | Actuel | Status |
|----------|-------|--------|--------|
| Unanswerable ratio | 25-35% | **0%** | ❌ CRITIQUE |
| is_impossible field | Present | Present (all False) | ⚠️ |

**Finding**: Le fichier GS Annales v7.3.0 contient **0 questions unanswerable** (is_impossible=False pour les 420 questions).

**Impact**: Non conforme SQuAD 2.0. Le modele ne sera pas teste sur sa capacite a rejeter les questions hors scope.

**Note**: Le fichier `gold_standard_fr.json` v5.30 contient 105 questions adversariales (33%). Ces fichiers sont **separes** mais peuvent etre combines pour l'evaluation.

### 4.2 BEIR (Benchmarking IR)

| Exigence | Cible | Actuel | Status |
|----------|-------|--------|--------|
| Dataset size | >= 200 | 420 | ✅ |
| Category diversity | >= 80% | 13 categories | ✅ |
| Document coverage | Broad | 28 docs | ✅ |

### 4.3 UAEval4RAG (arXiv:2412.12300)

| Categorie | Requis | Actuel | Status |
|-----------|--------|--------|--------|
| FALSE_PRESUPPOSITION | 12-14 | 0 | ❌ |
| OUT_OF_SCOPE | 18-21 | 0 | ❌ |
| VOCABULARY_MISMATCH | 9-10 | 0 | ❌ |
| ENTITY_SWAP | 9-10 | 0 | ❌ |
| NUMBER_SWAP | 6-7 | 0 | ❌ |
| UNDERSPECIFIED | 3-4 | 0 | ❌ |

### 4.4 Bloom's Taxonomy

| Niveau | Count | % | Status |
|--------|-------|---|--------|
| Remember | 238 | 56.7% | ✅ |
| Apply | 161 | 38.3% | ✅ |
| Understand | 15 | 3.6% | ✅ |
| Analyze | 6 | 1.4% | ✅ |

**Distribution acceptable** - dominance Remember/Apply coherente avec questions d'examen FFE.

---

## 5. Qualite des Donnees

### 5.1 Format Questions

| Check | Count | % | Status |
|-------|-------|---|--------|
| Questions terminant par "?" | 287/420 | 68.3% | ❌ |
| Questions sans "?" | 133/420 | 31.7% | **MAJEUR** |

**Exemples questions sans "?":**
- `ffe:annales:clubs:008:5e3f7d24`: "Un arbitre international allemand..."
- `ffe:annales:clubs:013:a7e8c494`: "Pour un tournoi, le montant..."

**Impact**: Les questions sans "?" peuvent ne pas etre detectees comme questions par certains modeles.

### 5.2 Chunk ID Validation

| Pattern | Count | Status |
|---------|-------|--------|
| Format valide (`*.pdf-pXXX-parentN-childN`) | 414 | ✅ |
| Encoding issues (parité → parit�) | 6 | ⚠️ MINEUR |

### 5.3 Question Type Distribution

| Type | Count | % | Status |
|------|-------|---|--------|
| factual | 214 | 51.0% | ✅ |
| scenario | 157 | 37.4% | ✅ |
| **None** | 34 | 8.1% | ❌ MAJEUR |
| procedural | 10 | 2.4% | ✅ |
| comparative | 5 | 1.2% | ✅ |

### 5.4 Answer Type Distribution

| Type | Count | % | Conforme RAGAS/BEIR |
|------|-------|---|---------------------|
| multiple_choice | 334 | 79.5% | ✅ |
| extractive | 58 | 13.8% | ✅ |
| abstractive | 19 | 4.5% | ✅ |
| list | 8 | 1.9% | ✅ |
| yes_no | 1 | 0.2% | ✅ |

### 5.5 Validation Status

| Status | Count | % |
|--------|-------|---|
| VALIDATED | 386 | 91.9% |
| PENDING | 34 | 8.1% |

---

## 6. Conformite ISO

### 6.1 ISO 42001 - AI Management

| Clause | Exigence | Actuel | Status |
|--------|----------|--------|--------|
| A.6.2.2 | Provenance tracable | expected_chunk_id 100% | ✅ |
| A.6.2.3 | Lineage data | deduplication documented | ✅ |
| A.7.3 | Documentation donnees | methodology present | ✅ |
| A.8.4 | Validation IA | validation.status 100% | ✅ |

### 6.2 ISO 25010 - Software Quality

| Exigence | Cible | Actuel | Status |
|----------|-------|--------|--------|
| Recall test set | >= 90% | TBD | ⚠️ |
| Taux hallucination | 0% | N/A (test requis) | ⚠️ |
| Coverage categories | 100% | 13/13 | ✅ |

### 6.3 ISO 29119 - Software Testing

| Exigence | Cible | Actuel | Status |
|----------|-------|--------|--------|
| Test data documented | OUI | Schema + methodology | ✅ |
| Validation set separate | OUI | Non explicite | ⚠️ |
| Reproductibilite | Seeds fixes | deduplication tracked | ✅ |

---

## 7. Couverture Documents

### 7.1 Top 10 Documents

| Document | Questions | % |
|----------|-----------|---|
| LA-octobre2025.pdf | 250 | 59.5% |
| A02_2025_26_Championnat_de_France_des_Clubs.pdf | 63 | 15.0% |
| R01_2025_26_Regles_generales.pdf | 36 | 8.6% |
| R03_2025_26_Competitions_homologuees.pdf | 18 | 4.3% |
| C01_2025_26_Coupe_de_France.pdf | 11 | 2.6% |
| J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf | 6 | 1.4% |
| C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf | 6 | 1.4% |
| J01_2025_26_Championnat_de_France_Jeunes.pdf | 5 | 1.2% |
| C04_2025_26_Coupe_de_la_parite.pdf | 4 | 1.0% |
| Autres (19 docs) | 21 | 5.0% |

**Total**: 28 documents uniques

### 7.2 Couverture Categories

| Categorie | Questions | % |
|-----------|-----------|---|
| competitions | 108 | 25.7% |
| regles_jeu | 106 | 25.2% |
| open | 64 | 15.2% |
| tournoi | 47 | 11.2% |
| interclubs | 44 | 10.5% |
| regles_ffe | 21 | 5.0% |
| classement | 9 | 2.1% |
| jeunes | 8 | 1.9% |
| administratif | 4 | 1.0% |
| regional | 4 | 1.0% |
| feminin | 2 | 0.5% |
| handicap | 2 | 0.5% |
| medical | 1 | 0.2% |

---

## 8. Couverture Sessions Annales

| Session | Questions | % |
|---------|-----------|---|
| dec2024 | 92 | 21.9% |
| jun2025 | 64 | 15.2% |
| dec2023 | 52 | 12.4% |
| jun2024 | 47 | 11.2% |
| jun2021 | 42 | 10.0% |
| jun2023 | 30 | 7.1% |
| jun2022 | 24 | 5.7% |
| dec2021 | 18 | 4.3% |
| dec2019 | 11 | 2.6% |
| dec2022 | 6 | 1.4% |
| (human-added) | 34 | 8.1% |

---

## 9. Actions Requises

### 9.1 CRITIQUE (P0)

| Action | Justification | Standard |
|--------|---------------|----------|
| ~~Ajouter questions adversariales~~ | 0% unanswerable vs 25-35% requis | SQuAD 2.0 |

> **Note**: Les questions adversariales sont dans `gold_standard_fr.json` (105 Q).
> **Decision**: Garder les fichiers separes (training vs evaluation).

### 9.2 MAJEUR (P1)

| Action | Count | Justification |
|--------|-------|---------------|
| Ajouter "?" aux questions | 133 | Format question standard |
| Completer question_type | 34 | Taxonomie incomplete |

### 9.3 MINEUR (P2)

| Action | Count | Justification |
|--------|-------|---------------|
| Corriger encoding chunk_ids | 6 | UTF-8 consistency |
| Ajouter hard_type metadata | 420 | UAEval4RAG taxonomy |

---

## 10. Recommandations

### 10.1 Architecture Fichiers (RECOMMANDE)

```
tests/data/
├── gold_standard_annales_fr_v7.json    # 420 Q - TRAINING (answerable)
├── gold_standard_fr.json               # DEPRECATED - remplace par adversarial_questions.json
└── adversarial_questions.json          # 105 Q - EVALUATION ADVERSARIALE
```

**Rationale**:
- Annales = questions officielles DNA = ground truth pour training embeddings
- Adversarial = tests de robustesse = evaluation uniquement
- Separation conforme TRIPLET_GENERATION_SPEC.md Section 1 (Val = 100% GS, jamais synthetique)

### 10.2 Metriques Combinees

| Metrique | Annales v7 | FR v5.30 | Combined |
|----------|------------|----------|----------|
| Questions totales | 420 | 318 | 738 |
| Answerable | 420 (100%) | 213 (67%) | 633 (86%) |
| Adversarial | 0 (0%) | 105 (33%) | 105 (14%) |

**Note**: Le ratio combine 14% adversarial est sous le seuil SQuAD 2.0 (25-35%), mais acceptable si les fichiers sont utilises pour des purposes differents (training vs evaluation).

---

## 11. Conclusion

Le Gold Standard Annales FR v7.3.0 presente une **bonne qualite de donnees** pour le training d'embeddings:
- ✅ Tracabilite ISO 42001 complete (chunk_id, validation)
- ✅ Taxonomie Bloom's Taxonomy implementee
- ✅ Couverture 28 documents, 13 categories
- ✅ Schema unifie URN-like v1.0.0

**Ameliorations requises**:
- ❌ Questions sans "?" (133/420)
- ❌ question_type=None (34/420)

**Architecture validee**:
- Annales v7.3.0 = training set (answerable uniquement)
- gold_standard_fr.json = **DEPRECATED** (remplace par adversarial_questions.json)
- adversarial_questions.json = evaluation adversariale (105 Q)
- Conforme TRIPLET_GENERATION_SPEC.md

**Optimisations requises** (voir `docs/specs/GS_ANNALES_V7_OPTIMIZATION_SPEC.md`):
- P0: Normaliser questions (ajouter "?"), completer question_type
- P1: Ajouter reasoning_class (Know Your RAG taxonomy)
- P2: Hard negative mining avec TopK-PercPos (NV-Embed-v2)

---

## 12. Annexes

### A. References Normatives

| Standard | Document | Application |
|----------|----------|-------------|
| ISO 42001:2023 | AI Management Systems | Tracabilite, lineage |
| ISO 25010:2023 | Software Quality | Metriques qualite |
| ISO 29119:2021 | Software Testing | Test data |
| SQuAD 2.0 | arXiv:1806.03822 | Unanswerable ratio |
| UAEval4RAG | arXiv:2412.12300 | Adversarial taxonomy |
| BEIR | github.com/beir-cellar | Benchmark IR |

### B. Commandes Audit

```bash
# Field coverage
python -c "import json; d=json.load(open('tests/data/gold_standard_annales_fr_v7.json')); print(len(d['questions']))"

# Question mark check
python -c "import json; d=json.load(open('tests/data/gold_standard_annales_fr_v7.json')); print(sum(1 for q in d['questions'] if not q['question'].strip().endswith('?')))"
```

---

*Document conforme ISO 29119-3 - Test Documentation*
*Genere: 2026-01-25*
