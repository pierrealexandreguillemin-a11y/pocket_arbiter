# Gold Standard Status Report v1

> **Pocket Arbiter** - RAG Retrieval Benchmark
> Date: 2026-01-20 | Version: 5.16 | Recall@5: 95.70%

---

## 1. Résumé Actuel du Gold Standard

### 1.1 Métriques Globales

| Métrique | Valeur | Cible ISO 25010 |
|----------|--------|-----------------|
| **Total questions** | 93 | >= 50 |
| **Recall@5 (tol=2)** | 95.70% | >= 80% |
| **Hard cases** | 4 (4.3%) | Documentés |
| **Questions passantes** | 89/93 | - |

### 1.2 Répartition par Catégorie

| Catégorie | Count | % |
|-----------|-------|---|
| tournoi | 18 | 19.4% |
| arbitrage | 13 | 14.0% |
| regles_jeu | 11 | 11.8% |
| administration | 10 | 10.8% |
| classement | 8 | 8.6% |
| jeunes | 7 | 7.5% |
| discipline | 6 | 6.5% |
| handicap | 5 | 5.4% |
| feminin | 4 | 4.3% |
| temps | 3 | 3.2% |
| cadences | 3 | 3.2% |
| titres | 3 | 3.2% |
| notation | 2 | 2.2% |

### 1.3 Répartition par Chapitre (LA-octobre2025)

| Chapitre | Count | Description |
|----------|-------|-------------|
| N/A (autres PDF) | 75 | Règlements spécifiques FFE |
| Ch 6.1 | 6 | Classement Elo Standard FIDE |
| Ch 5.5 | 5 | Barème indemnisation arbitres |
| Ch 6.3 | 3 | Titres FIDE |
| Ch 5.1-5.4 | 4 | Mission administrative arbitre |

### 1.4 Distribution des Sources

| Source | Count | % |
|--------|-------|---|
| LA-octobre2025.pdf | 44 | 47.3% |
| R01 (Règles générales) | 5 | 5.4% |
| Autres (27 PDFs) | 44 | 47.3% |

### 1.5 Statistiques des Questions

| Stat | Valeur |
|------|--------|
| Longueur min | 26 caractères |
| Longueur max | 178 caractères |
| Longueur moyenne | 67.2 caractères |
| Questions "hard_case" | 4 (4.3%) |
| Questions "gamées" (reformulées) | ~0% (revertées) |

---

## 2. Informations pour Génération de Hard Cases

### 2.1 Critères d'Évaluation du Gold Standard

| Critère | Description | Status Actuel |
|---------|-------------|---------------|
| **Qualité annotations** | expected_pages vérifiées vs PDF source | ✅ Validé manuellement |
| **Diversité catégories** | 13 catégories couvertes | ✅ Bon |
| **Diversité sources** | 28+ PDFs référencés | ✅ Bon |
| **Difficulté** | Mix easy/medium/hard | ⚠️ 95% easy |
| **Langage naturel** | Questions production-like | ⚠️ 4 hard_cases seulement |

### 2.2 Statistiques Précises

```
Questions totales:     93
Couverture chapitres:  Ch1-6 LA-octobre + 27 règlements FFE
Couverture pages:      ~180 pages uniques référencées
Synonymes testés:      ~5% (estimé)
Fautes d'ortho:        0%
Négations:             ~2% (estimé)
Questions multi-pages: ~30%
```

### 2.3 Exemples Concrets de Questions (10)

#### Exemple 1 - Règle classique (EASY)
```json
{
  "id": "FR-Q01",
  "question": "Quelle est la règle du toucher-jouer ?",
  "category": "regles_jeu",
  "expected_docs": ["LA-octobre2025.pdf"],
  "expected_pages": [41, 42],
  "hard_case": false
}
```

#### Exemple 2 - Règle précise (EASY)
```json
{
  "id": "FR-Q10",
  "question": "Quelles sont les règles de promotion du pion ?",
  "category": "regles_jeu",
  "expected_docs": ["LA-octobre2025.pdf"],
  "expected_pages": [39, 40, 42],
  "hard_case": false
}
```

#### Exemple 3 - Tournoi spécifique (MEDIUM)
```json
{
  "id": "FR-Q25",
  "question": "Quelles sont les règles des interclubs ?",
  "category": "tournoi",
  "expected_docs": ["A02_2025_26_Championnat_de_France_des_Clubs.pdf"],
  "expected_pages": [1, 2, 3],
  "hard_case": false
}
```

#### Exemple 4 - Définition glossaire (EASY)
```json
{
  "id": "FR-Q70",
  "question": "Que signifie j'adoube et quand peut-on l'utiliser ?",
  "category": "regles_jeu",
  "expected_docs": ["LA-octobre2025.pdf"],
  "expected_pages": [38, 41],
  "hard_case": false
}
```

#### Exemple 5 - HARD CASE (keyword mismatch)
```json
{
  "id": "FR-Q77",
  "question": "Un joueur qui n'a joué aucune partie pendant 18 mois voit-il son classement Elo Standard supprimé ?",
  "category": "classement",
  "expected_docs": ["LA-octobre2025.pdf"],
  "expected_pages": [183, 188],
  "hard_case": true,
  "hard_reason": "Keyword '18 mois' absent du corpus - règle parle de '1 an' d'inactivité"
}
```

#### Exemple 6 - HARD CASE (terme absent)
```json
{
  "id": "FR-Q85",
  "question": "Dans quel délai un arbitre doit-il transmettre le rapport d'un match individuel à la DNA ?",
  "category": "arbitrage",
  "expected_docs": ["LA-octobre2025.pdf"],
  "expected_pages": [165],
  "hard_case": true,
  "hard_reason": "Keyword 'délai transmission DNA' absent de Ch5.1"
}
```

#### Exemple 7 - HARD CASE (formulation naturelle)
```json
{
  "id": "FR-Q86",
  "question": "Quelles sont les étapes pour la gestion administrative en cas d'absence d'un joueur à la première ronde ?",
  "category": "arbitrage",
  "expected_docs": ["LA-octobre2025.pdf"],
  "expected_pages": [169, 170, 171],
  "hard_case": true,
  "hard_reason": "Keyword 'absence premiere ronde' absent de Ch5.3"
}
```

#### Exemple 8 - HARD CASE (synonyme)
```json
{
  "id": "FR-Q87",
  "question": "Pour obtenir un premier classement Elo Standard FIDE, quelles sont les conditions minimales ?",
  "category": "classement",
  "expected_docs": ["LA-octobre2025.pdf"],
  "expected_pages": [182, 183],
  "hard_case": true,
  "hard_reason": "Keyword mismatch: 'premier classement' vs 'classement initial' dans corpus"
}
```

### 2.4 Patterns de Retrieval (Biais Identifiés)

#### Definition Query Patterns (auto glossary boost x3.5)
```python
DEFINITION_QUERY_PATTERNS = [
    "qu'est-ce que",
    "qu'est ce que",
    "c'est quoi",
    "définition de",
    "que signifie",
    "que veut dire",
    "comment définir",
    "selon les règles",
    "selon le règlement",
    "terme officiel",
    "what is",  # INTL
]
```

#### Source Filter Patterns (auto source routing)
```python
SOURCE_FILTER_PATTERNS = {
    "selon les statuts": "Statuts",
    "cadences officielles": "LA-octobre",
    "règles du jeu": "LA-octobre",
    "toucher-jouer": "LA-octobre",
    "pièce touchée": "LA-octobre",
    "coup illégal": "LA-octobre",
    "éthique": "Code_ethique",
}
```

#### Biais Identifiés

| Biais | Description | Impact |
|-------|-------------|--------|
| **Glossary boost** | Questions "qu'est-ce que X" surperforment | Recall gonflé sur définitions |
| **Source filter** | Keywords techniques auto-routés | Faux positif si multi-source |
| **Keyword matching** | EmbeddingGemma dépend des termes exacts | Échec sur synonymes/paraphrases |

---

## 3. Template pour Nouvelles Questions

### 3.1 Format JSON Standard

```json
{
  "id": "FR-Q{N}",
  "question": "[Texte naturel, production-like]",
  "category": "[tournoi|arbitrage|regles_jeu|classement|administration|jeunes|discipline|handicap|feminin|temps|cadences|titres|notation]",
  "expected_docs": ["[nom_fichier.pdf]"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "expected_pages": [page1, page2],
  "metadata": {
    "type": "[definition|regle|arbitrage|edge|admin]",
    "chapter": "[ex: 6.1]",
    "hard_case": false,
    "hard_reason": "[si hard_case=true, explication]"
  },
  "validation": {
    "status": "VALIDATED",
    "method": "manual_verification",
    "details": []
  },
  "difficulty": "[easy|medium|hard]",
  "notes": "[ex: test synonymes, négation, langage naturel]",
  "audit": "added_YYYY-MM-DD"
}
```

### 3.2 Format CSV Simplifié (pour bulk import)

```csv
id,query,type,expected_source,expected_pages,difficulty,notes,hard_flag
FR-Q94,"Question texte naturel",regle,LA-octobre2025.pdf,"[182,183]",hard,"test synonyme",true
FR-Q95,"Autre question",definition,LA-octobre2025.pdf,"[67,68]",easy,"glossaire",false
```

### 3.3 Checklist Validation

- [ ] Question formulée en langage naturel (pas de keywords forcés)
- [ ] expected_pages vérifié contre PDF source
- [ ] Catégorie cohérente avec le contenu
- [ ] Si hard_case, hard_reason documenté
- [ ] Pas de duplication avec questions existantes

---

## 4. Analyse des Gaps et Propositions Hard Cases

### 4.1 Zones Sous-Testées

| Zone | Coverage | Gap Identifié |
|------|----------|---------------|
| **Négations** | ~2% | "Quand ne peut-on PAS..." absent |
| **Langage oral** | ~5% | Formulations courtes, familières |
| **Fautes d'ortho** | 0% | Robustesse non testée |
| **Questions multi-parties** | ~10% | "A ou B" peu testées |
| **Synonymes** | ~5% | "classement initial" vs "premier Elo" |
| **Chiffres spécifiques** | ~15% | "18 mois", "5 parties" souvent absents corpus |

### 4.2 Propositions de Hard Cases (5 thèmes)

#### Theme 1: Négations
**Pourquoi hard**: EmbeddingGemma ignore souvent la négation, retrieve le contraire.
```
Idée: "Dans quel cas un arbitre n'a-t-il PAS le droit de refuser une nulle ?"
```

#### Theme 2: Langage très oral / SMS-like
**Pourquoi hard**: Formulations courtes sans structure grammaticale complète.
```
Idée: "pat cest quoi deja ?" / "règle du roi noyé ?"
```

#### Theme 3: Synonymes non-présents dans corpus
**Pourquoi hard**: Terme utilisateur ≠ terme officiel corpus.
```
Idée: "C'est quoi le temps de réflexion Fischer ?" (corpus = "incrément")
Idée: "Règle des 50 coups sans prise" (corpus = "règle des 50 coups")
```

#### Theme 4: Questions numériques spécifiques
**Pourquoi hard**: Chiffres souvent légèrement différents dans corpus.
```
Idée: "Après combien de mois sans jouer perd-on son Elo ?" (corpus = "1 an" pas "12 mois")
Idée: "Faut-il 5 ou 9 parties pour un premier classement ?" (test précision)
```

#### Theme 5: Questions composées (A et B)
**Pourquoi hard**: Retriever doit trouver info sur 2 sujets, souvent pages différentes.
```
Idée: "Quelles sont les règles du pat ET de la nulle par répétition ?"
Idée: "Différence entre forfait et abandon en tournoi ?"
```

---

## 5. Recommandations

### 5.1 Actions Prioritaires

1. **Ajouter 10-15 hard cases** couvrant les 5 thèmes identifiés
2. **Tester robustesse orthographique** avec 5 questions avec fautes volontaires
3. **Augmenter couverture Ch5-6** du LA-octobre (actuellement 18 questions)
4. **Documenter tous les hard_case** avec hard_reason explicite

### 5.2 Métriques Cibles

| Métrique | Actuel | Cible v2 |
|----------|--------|----------|
| Total questions | 93 | 120+ |
| Hard cases | 4 (4.3%) | 20+ (15%) |
| Recall hard cases | 0% | 50%+ |
| Couverture négations | ~2% | 10% |
| Couverture synonymes | ~5% | 15% |

---

## Historique

| Version | Date | Changements |
|---------|------|-------------|
| 5.16 | 2026-01-20 | 93 questions, 4 hard_cases, Recall 95.70% |
| 5.14 | 2026-01-19 | 86 questions, 3 hard_cases revertées |
| 5.0 | 2026-01-18 | 75 questions initiales |

---

*Document généré pour analyse par LLM externe (Grok/GPT) - Pocket Arbiter Project*
