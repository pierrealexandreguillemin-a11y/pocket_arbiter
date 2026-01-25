# Analyse Adversarial Questions - GS v7 Enrichment

> **Date**: 2026-01-25
> **Source**: tests/data/gold_standard_fr.json v5.30
> **Cible**: tests/data/gold_standard_annales_fr_v7.json v7.3.0

## 1. Extraction Complete

**105 questions adversariales extraites** vers `adversarial_review.json`

### Distribution actuelle:

| hard_type | Count | % |
|-----------|-------|---|
| OUT_OF_SCOPE | 22 | 21.0% |
| VOCABULARY_MISMATCH | 16 | 15.2% |
| FALSE_PRESUPPOSITION | 12 | 11.4% |
| ENTITY_SWAP | 10 | 9.5% |
| ANTONYM | 10 | 9.5% |
| NEGATION | 8 | 7.6% |
| NUMBER_SWAP | 7 | 6.7% |
| PARTIAL_INFO | 4 | 3.8% |
| SAFETY_CONCERNED | 4 | 3.8% |
| MULTI_HOP_IMPOSSIBLE | 3 | 2.9% |
| FALSE_PREMISE | 3 | 2.9% |
| UNDERSPECIFIED | 3 | 2.9% |
| MUTUAL_EXCLUSION | 3 | 2.9% |

## 2. Analyse vs SQuAD 2.0 (arXiv:1806.03822)

### Mapping categories vers SQuAD 2.0:

| SQuAD 2.0 Category | Notre Count | Notre % | Cible SQuAD | Status |
|--------------------|-------------|---------|-------------|--------|
| OTHER_NEUTRAL | 33 | 31.4% | 24% | EXCES +7% |
| ANTONYM | 26 | 24.8% | 20% | OK |
| IMPOSSIBLE_CONDITION | 18 | 17.1% | 4% | EXCES +13% |
| ENTITY_SWAP | 17 | 16.2% | 21% | OK |
| NEGATION | 8 | 7.6% | 9% | OK |
| MUTUAL_EXCLUSION | 3 | 2.9% | 15% | **MANQUE -12%** |

### Diagnostic:

1. **MUTUAL_EXCLUSION critique**: 3 vs 15 cible = besoin +12 questions
2. **IMPOSSIBLE_CONDITION sur-represente**: Nos FALSE_PREMISE/PRESUPPOSITION sont trop nombreuses
3. **OTHER_NEUTRAL leger exces**: OUT_OF_SCOPE peut etre reduit

## 3. Qualite ISO 42001 / ISO 29119

| Critere | Score | Conformite |
|---------|-------|------------|
| corpus_truth | 105/105 (100%) | CONFORME |
| keywords | 105/105 (100%) | CONFORME |
| expected_docs | 16/105 (15%) | **NON-CONFORME** |
| test_purpose | 105/105 (100%) | CONFORME |

### Actions requises:

- **expected_docs manquant** sur 89 questions OUT_OF_SCOPE (normal car hors corpus)
- Questions IN-SCOPE doivent avoir expected_docs

## 4. Ratio Cible GS v7

Per SQuAD 2.0 methodology:
- **Training**: 2:1 (answerable:unanswerable)
- **Evaluation**: 1:1

### Calcul:

| Metrique | Valeur |
|----------|--------|
| Answerable actuelles GS v7 | 420 |
| Ratio cible | 33% unanswerable |
| Unanswerable cible | ~140 |
| Disponibles | 105 |
| **A CREER** | **35** |

## 5. Recommandations Tri

### A. Questions a GARDER (Tier 1 - Import direct):

**Criteres**: corpus_truth + test_purpose + hard_type bien categorise

- VOCABULARY_MISMATCH (16) - test robustesse synonymes
- ENTITY_SWAP (10) - test confusion entites
- ANTONYM (10) - test sens inverse
- NEGATION (8) - test negation
- NUMBER_SWAP (7) - test confusion numerique
- **Total Tier 1: 51 questions**

### B. Questions a REVOIR (Tier 2 - Validation requise):

- FALSE_PRESUPPOSITION (12) - verifier vs corpus actuel
- FALSE_PREMISE (3) - idem
- PARTIAL_INFO (4) - peut etre answerable avec meilleur retrieval
- MULTI_HOP_IMPOSSIBLE (3) - verifier complexite
- **Total Tier 2: 22 questions**

### C. Questions a REDUIRE (Tier 3 - Trop represente):

- OUT_OF_SCOPE (22) - garder 10-12 max, supprimer doublons
- SAFETY_CONCERNED (4) - garder 2 max
- UNDERSPECIFIED (3) - fusionner avec PARTIAL_INFO
- **Total Tier 3: 29 -> reduire a ~15**

### D. Questions a CREER:

- MUTUAL_EXCLUSION: +12 questions (2.9% -> 15%)
- Exemples patterns:
  - "Peut-on etre a la fois arbitre et joueur dans le meme tournoi?"
  - "Un club peut-il avoir 2 equipes en N1?"
  - "Un joueur peut-il jouer en N2 ET N3 la meme saison?"

## 6. Plan Import GS v7

### Phase 1: Import Tier 1 (51 questions)
- Assigner nouveaux IDs: `adversarial:human:{category}:{seq}:{hash}`
- Ajouter `is_impossible: true`
- Valider expected_docs pour questions IN-SCOPE

### Phase 2: Review Tier 2 (22 questions)
- Verification manuelle corpus_truth vs PDFs actuels
- Reclassification si necessaire

### Phase 3: Create MUTUAL_EXCLUSION (+12 questions)
- Focus sur regles mutuellement exclusives FFE
- Pattern: "X et Y en meme temps?"

### Phase 4: Cleanup Tier 3
- OUT_OF_SCOPE: garder 1 par plateforme (Chess.com, Lichess, USCF, ECU, ICCF)
- Total reduit: ~12 questions

## 7. Distribution Cible Finale

| Category | Actuel | Cible | Action |
|----------|--------|-------|--------|
| VOCABULARY_MISMATCH | 16 | 16 | Keep all |
| ENTITY_SWAP | 10 | 15 | +5 |
| ANTONYM | 10 | 14 | +4 |
| NEGATION | 8 | 8 | Keep all |
| NUMBER_SWAP | 7 | 7 | Keep all |
| MUTUAL_EXCLUSION | 3 | 15 | **+12** |
| OUT_OF_SCOPE | 22 | 12 | -10 |
| FALSE_PRESUPPOSITION | 12 | 8 | -4 |
| PARTIAL_INFO | 4 | 6 | +2 |
| MULTI_HOP_IMPOSSIBLE | 3 | 5 | +2 |
| OTHER | 10 | 4 | -6 |
| **TOTAL** | **105** | **110** | +5 net |

Avec 35 nouvelles questions creees: **140 total** = 33% de 420+140 = 560 questions
