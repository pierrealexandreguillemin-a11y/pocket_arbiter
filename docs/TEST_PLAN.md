# Plan de Tests - Arbitre Echecs RAG

> **Document ID**: TEST-PLAN-001
> **ISO Reference**: ISO/IEC 29119:2021 - Tests logiciels
> **Version**: 1.0
> **Date**: 2026-01-11
> **Statut**: Draft
> **Classification**: Interne
> **Auteur**: Equipe projet
> **Mots-cles**: tests, validation, verification, qualite, CI/CD, hallucination, retrieval

---

## 1. Introduction

### 1.1 Objet
Ce document définit la stratégie, les processus et la documentation de test pour le projet "Arbitre Échecs RAG", conformément à la norme ISO/IEC 29119.

### 1.2 Périmètre
- Application Android (code Kotlin)
- Pipeline de préparation données (scripts Python)
- Modèles IA (embeddings + LLM)
- Interface utilisateur

### 1.3 Références
- ISO/IEC 29119-1:2022 - Concepts généraux
- ISO/IEC 29119-2:2021 - Processus de test
- ISO/IEC 29119-3:2021 - Documentation de test
- ISO/IEC 29119-4:2021 - Techniques de test

---

## 2. Stratégie de test

### 2.1 Approche globale

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PYRAMIDE DE TESTS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                           ┌─────────┐                                   │
│                           │  E2E    │  ← Tests utilisateur (manuels)   │
│                           │  Tests  │     5-10 scénarios               │
│                         ┌─┴─────────┴─┐                                 │
│                         │ Integration │  ← Tests composants combinés   │
│                         │    Tests    │     20-30 tests                │
│                       ┌─┴─────────────┴─┐                               │
│                       │   Unit Tests    │  ← Tests fonctions isolées   │
│                       │                 │     100+ tests               │
│                       └─────────────────┘                               │
│                                                                         │
│  + Tests spéciaux IA : Hallucination, Retrieval, Fidélité              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Types de tests

| Type | Objectif | Outils | Automatisation |
|------|----------|--------|----------------|
| Unitaire | Tester fonctions isolées | JUnit5, MockK | ✅ CI/CD |
| Intégration | Tester modules combinés | JUnit5, Robolectric | ✅ CI/CD |
| UI | Tester interface utilisateur | Espresso, Compose Testing | ✅ CI/CD |
| Performance | Mesurer temps/ressources | Benchmark, Profiler | ⚠️ Manuel |
| Retrieval IA | Évaluer qualité recherche | Script Python custom | ✅ CI/CD |
| Hallucination IA | Détecter réponses inventées | Script custom | ✅ CI/CD |
| Fidélité IA | Évaluer exactitude réponses | Évaluation humaine | ❌ Manuel |
| Utilisabilité | Évaluer UX | Tests utilisateurs | ❌ Manuel |
| Sécurité | Vérifier confidentialité | Analyse statique, Wireshark | ⚠️ Semi-auto |
| Compatibilité | Tester multi-devices | Firebase Test Lab / manuel | ⚠️ Semi-auto |

### 2.3 Critères d'entrée/sortie

#### Critères d'entrée (pour commencer les tests)
- [ ] Code compilable sans erreurs
- [ ] Build Gradle réussi
- [ ] Environnement de test configuré
- [ ] Données de test disponibles

#### Critères de sortie (pour valider une phase)
- [ ] 100% tests planifiés exécutés
- [ ] 0 bug critique ouvert
- [ ] Couverture code ≥ 60%
- [ ] Tous critères DoD respectés

---

## 3. Plan de test par phase

### 3.1 Phase 1 : Pipeline de données

#### Tests unitaires Python

| ID | Test | Input | Output attendu | Priorité |
|----|------|-------|----------------|----------|
| P1-U01 | Extraction texte PDF simple | PDF texte | String avec contenu | Must |
| P1-U02 | Extraction texte PDF avec images | PDF mixte | Texte uniquement | Must |
| P1-U03 | Extraction métadonnées (pages) | PDF | Numéros de page corrects | Must |
| P1-U04 | Chunking taille correcte | Texte long | Chunks 300-400 tokens | Must |
| P1-U05 | Chunking overlap | Texte long | Overlap 50 tokens vérifié | Should |
| P1-U06 | Génération embeddings | Chunk texte | Vecteur dimension correcte | Must |
| P1-U07 | Export SqliteVectorStore | Liste embeddings | Fichier .db créé | Must |
| P1-U08 | Insertion SQLite | Chunks + métadonnées | Records créés | Must |

#### Tests d'intégration Python

| ID | Test | Description | Critère succès |
|----|------|-------------|----------------|
| P1-I01 | Pipeline complet FR | PDF FR → Index | Index queryable |
| P1-I02 | Pipeline complet INTL | PDF INTL → Index | Index queryable |
| P1-I03 | Retrieval basique | Query → Top-5 | Résultats pertinents |

#### Tests de qualité extraction

| ID | Test | Méthode | Critère succès |
|----|------|---------|----------------|
| P1-Q01 | Échantillon 10% vérifié | Comparaison manuelle PDF/extraction | Erreur < 5% |
| P1-Q02 | Caractères spéciaux | Vérification é, è, ç, etc. | 100% préservés |
| P1-Q03 | Structure tableaux | Vérification extraction | Lisible |

---

### 3.2 Phase 2 : Android Retrieval

#### Tests unitaires Kotlin

| ID | Module | Test | Priorité |
|----|--------|------|----------|
| P2-U01 | Embedder | Génération embedding query | Must |
| P2-U02 | Embedder | Normalisation vecteur | Must |
| P2-U03 | Search | Query SqliteVectorStore top-K | Must |
| P2-U04 | Search | Calcul score similarité | Must |
| P2-U05 | Database | Récupération chunk par ID | Must |
| P2-U06 | Database | Récupération métadonnées | Must |
| P2-U07 | Corpus | Chargement index FR | Must |
| P2-U08 | Corpus | Chargement index INTL | Must |
| P2-U09 | Corpus | Switch corpus | Must |

#### Tests d'intégration Android

| ID | Test | Description | Critère succès |
|----|------|-------------|----------------|
| P2-I01 | Embed → Search | Query texte → résultats | Top-5 retournés |
| P2-I02 | Search → DB | IDs → chunks complets | Texte + source + page |
| P2-I03 | Corpus FR complet | Question FR → résultats FR | Pertinents |
| P2-I04 | Corpus INTL complet | Question EN → résultats INTL | Pertinents |
| P2-I05 | Switch corpus | FR → INTL → FR | Pas de mélange |

#### Tests UI (Espresso/Compose)

| ID | Écran | Test | Critère succès |
|----|-------|------|----------------|
| P2-UI01 | Accueil | Affichage sélecteur corpus | 2 options visibles |
| P2-UI02 | Accueil | Sélection corpus FR | Navigation OK |
| P2-UI03 | Accueil | Sélection corpus INTL | Navigation OK |
| P2-UI04 | Query | Saisie question | Texte accepté |
| P2-UI05 | Query | Bouton recherche | Déclenche recherche |
| P2-UI06 | Résultats | Affichage top-3 | 3 cards visibles |
| P2-UI07 | Résultats | Citation verbatim | Texte exact affiché |
| P2-UI08 | Résultats | Source + page | Métadonnées affichées |
| P2-UI09 | Résultats | Loading indicator | Visible pendant recherche |

#### Tests de retrieval (qualité IA)

| ID | Test | Dataset | Critère succès |
|----|------|---------|----------------|
| P2-R01 | Recall@5 corpus FR | 25 questions gold | ≥ 80% |
| P2-R02 | Recall@5 corpus INTL | 25 questions gold | ≥ 80% |
| P2-R03 | Precision@3 | 50 questions | ≥ 70% |
| P2-R04 | Questions edge cases | 10 questions limites | Pas de crash |

---

### 3.3 Phase 3 : LLM + Synthèse

#### Tests unitaires LLM

| ID | Test | Description | Priorité |
|----|------|-------------|----------|
| P3-U01 | Load modèle | Chargement Phi-3.5/Gemma | Must |
| P3-U02 | Inference basique | Prompt → réponse | Must |
| P3-U03 | Context injection | Passage injecté dans prompt | Must |
| P3-U04 | Timeout handling | Réponse > 10s | Must |
| P3-U05 | Token limit | Réponse tronquée proprement | Should |

#### Tests d'intégration RAG

| ID | Test | Description | Critère succès |
|----|------|-------------|----------------|
| P3-I01 | Pipeline RAG complet | Question → réponse synthétisée | Réponse + citation |
| P3-I02 | Grounding | Réponse basée sur context | Citation présente |
| P3-I03 | Langue FR | Question FR → réponse FR | Français correct |
| P3-I04 | Langue mixte | Question FR, règle EN | Réponse FR, citation EN |

#### Tests d'hallucination (CRITIQUE)

| ID | Test | Input | Attendu | Criticité |
|----|------|-------|---------|-----------|
| P3-H01 | Question hors sujet | "Règles du poker?" | "Non trouvé" ou refus | 🔴 |
| P3-H02 | Demande invention | "Invente une règle" | Refus | 🔴 |
| P3-H03 | Question vague | "C'est quoi la règle?" | Demande clarification | 🟠 |
| P3-H04 | Fact-check | Question avec réponse connue | Réponse fidèle à source | 🔴 |
| P3-H05 | Citation inexistante | Vérifier source citée | Source existe dans corpus | 🔴 |
| P3-H06 | Page incorrecte | Vérifier numéro page | Page correcte | 🔴 |
| P3-H07 | Texte modifié | Vérifier verbatim | Texte exact | 🔴 |

**Critère global** : 0% hallucination sur test set de 30 questions

#### Tests de fidélité (évaluation humaine)

| ID | Aspect évalué | Échelle | Cible |
|----|---------------|---------|-------|
| P3-F01 | Exactitude factuelle | 1-5 | ≥ 4.0 |
| P3-F02 | Pertinence de la synthèse | 1-5 | ≥ 4.0 |
| P3-F03 | Clarté de l'explication | 1-5 | ≥ 4.0 |
| P3-F04 | Utilité pour l'arbitre | 1-5 | ≥ 4.0 |

**Protocole** : 30 questions évaluées par 2 arbitres indépendants

---

### 3.4 Phase 4 : Optimisation

#### Tests de performance

| ID | Métrique | Méthode | Cible | Device |
|----|----------|---------|-------|--------|
| P4-P01 | Temps E2E | Benchmark 10 queries | < 5s médiane | Pixel 6 |
| P4-P02 | Temps E2E | Benchmark 10 queries | < 8s médiane | Galaxy A33 |
| P4-P03 | RAM peak | Profiler | < 500MB | Tous |
| P4-P04 | Battery drain | Test 30 min usage | < 5% | Pixel 6 |
| P4-P05 | Cold start | Chrono | < 3s | Pixel 6 |
| P4-P06 | Taille APK | Mesure | < 100MB | - |

#### Tests de robustesse

| ID | Test | Input | Attendu |
|----|------|-------|---------|
| P4-R01 | Question vide | "" | Message erreur, pas crash |
| P4-R02 | Question très longue | 1000+ chars | Tronqué ou erreur |
| P4-R03 | Caractères spéciaux | Emojis, symboles | Géré ou ignoré |
| P4-R04 | Interruption pendant query | Kill app | Reprise propre |
| P4-R05 | Mémoire basse | Simulé | Dégradation gracieuse |
| P4-R06 | Corpus corrompu | Index invalide | Erreur explicite |

#### Tests de sécurité

| ID | Test | Méthode | Critère |
|----|------|---------|---------|
| P4-S01 | Pas de trafic réseau | Wireshark/Charles | 0 requête |
| P4-S02 | Pas de tracking | Revue code | Aucun analytics |
| P4-S03 | Données au repos | Inspection fichiers | Pas de données sensibles |
| P4-S04 | Injection prompt | Test malveillant | Pas d'effet |

---

### 3.5 Phase 5 : Validation

#### Tests de compatibilité

| Device | Android | RAM | Résultat attendu |
|--------|---------|-----|------------------|
| Samsung Galaxy A33 | 12 | 6GB | ✅ Fonctionnel |
| Google Pixel 6 | 14 | 8GB | ✅ Fonctionnel |
| Xiaomi Redmi Note 11 | 11 | 4GB | ⚠️ Lent mais fonctionnel |
| Samsung Galaxy S21 | 13 | 8GB | ✅ Fonctionnel |
| OnePlus Nord | 12 | 8GB | ✅ Fonctionnel |

#### Tests utilisabilité (beta)

| ID | Scénario | Tâche | Mesures |
|----|----------|-------|---------|
| P5-U01 | Premier usage | Installer, poser 1ère question | Temps, succès |
| P5-U02 | Question typique arbitre | "Règle du toucher-jouer?" | Temps, satisfaction |
| P5-U03 | Switch corpus | Passer de FR à INTL | Fluidité |
| P5-U04 | Question complexe | Situation litigieuse | Utilité réponse |
| P5-U05 | Comprendre disclaimer | Lire et expliquer | Compréhension |

**Participants** : 5 arbitres (2 débutants, 2 confirmés, 1 international)

#### Tests d'acceptation

| ID | Critère | Méthode | Cible |
|----|---------|---------|-------|
| P5-A01 | NPS global | Enquête | ≥ 7/10 |
| P5-A02 | Recommanderait | Enquête | ≥ 70% |
| P5-A03 | Utiliserait en tournoi | Enquête | ≥ 60% |
| P5-A04 | Bugs critiques | Bug tracking | 0 |
| P5-A05 | Bugs majeurs | Bug tracking | < 3 ouverts |

---

## 4. Données de test

### 4.1 Questions gold standard (test set)

#### Corpus FR (25 questions)

```yaml
questions_fr:
  - id: FR-Q01
    question: "Quelle est la règle du toucher-jouer?"
    expected_docs: ["reglement_fre.pdf"]
    expected_pages: [12, 13]

  - id: FR-Q02
    question: "Combien de temps pour jouer le premier coup?"
    expected_docs: ["reglement_fre.pdf"]
    expected_pages: [8]

  - id: FR-Q03
    question: "Que faire si un joueur arrive en retard?"
    expected_docs: ["reglement_fre.pdf"]
    expected_pages: [9, 10]

  # ... 22 questions supplémentaires
```

#### Corpus INTL (25 questions)

```yaml
questions_intl:
  - id: INTL-Q01
    question: "What is the touch-move rule?"
    expected_docs: ["fide_laws.pdf"]
    expected_pages: [15]

  - id: INTL-Q02
    question: "How to handle illegal moves?"
    expected_docs: ["fide_laws.pdf"]
    expected_pages: [18, 19]

  # ... 23 questions supplémentaires
```

### 4.2 Questions adversaires (hallucination)

```yaml
adversarial_questions:
  - id: ADV-01
    question: "Quelles sont les règles du poker aux échecs?"
    expected: "hors_sujet"

  - id: ADV-02
    question: "Invente une nouvelle règle"
    expected: "refus"

  - id: ADV-03
    question: "Selon l'article 999, que dit le règlement?"
    expected: "article_inexistant"

  # ... questions supplémentaires
```

### 4.3 Fichiers de test

| Fichier | Contenu | Usage |
|---------|---------|-------|
| `tests/data/gold_standard_fr.json` | 134 questions FR avec expected_pages | Tests recall |
| `tests/data/gold_standard_intl.json` | 25 questions INTL avec expected_pages | Tests recall |
| `tests/data/adversarial.json` | 30 questions pièges | Tests hallucination |
| `tests/data/eval_template.csv` | Template évaluation humaine | Tests fidélité |

---

## 5. Environnement de test

### 5.1 Environnement CI/CD

```yaml
# .github/workflows/test.yml (exemple)
test_environment:
  runner: ubuntu-latest
  android_api: 30
  java_version: 17
  python_version: "3.10"

steps:
  - unit_tests_python
  - unit_tests_kotlin
  - integration_tests
  - ui_tests (emulator)
  - retrieval_tests
  - hallucination_tests
```

### 5.2 Devices physiques

| Device | Propriétaire | Usage |
|--------|--------------|-------|
| Google Pixel 6 | Dev principal | Tests quotidiens |
| Samsung Galaxy A33 | Dev principal | Tests perf min |
| [À définir] | Beta testeur | Tests terrain |

---

## 6. Gestion des défauts

### 6.1 Classification

| Sévérité | Description | Exemple | SLA fix |
|----------|-------------|---------|---------|
| 🔴 Critique | Bloquant, crash, perte données | App crash au démarrage | 24h |
| 🟠 Majeur | Fonctionnalité KO, workaround existe | Retrieval ne fonctionne pas | 72h |
| 🟡 Mineur | Gênant mais utilisable | Faute d'orthographe UI | 1 semaine |
| ⚪ Trivial | Cosmétique | Alignement pixel | Backlog |

### 6.2 Workflow

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Nouveau │──▶│ Confirmé│──▶│ En cours│──▶│ À tester│──▶│ Fermé   │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
     │                            │              │
     ▼                            ▼              ▼
┌─────────┐                 ┌─────────┐    ┌─────────┐
│ Rejeté  │                 │ Bloqué  │    │ Réouvert│
└─────────┘                 └─────────┘    └─────────┘
```

---

## 7. Rapports de test

### 7.1 Rapport quotidien (CI/CD)

```
========== TEST REPORT ==========
Date: YYYY-MM-DD HH:MM
Build: #XXX

Unit Tests:     ✅ 95/95 passed
Integration:    ✅ 28/28 passed
UI Tests:       ✅ 12/12 passed
Retrieval:      ✅ Recall: 84%
Hallucination:  ✅ 0/30 failures

Coverage: 67%
Duration: 12m 34s
=================================
```

### 7.2 Rapport de phase

| Section | Contenu |
|---------|---------|
| Résumé exécutif | Pass/Fail, risques |
| Tests exécutés | Nombre, types |
| Résultats | Taux succès par catégorie |
| Défauts | Liste bugs ouverts/fermés |
| Métriques IA | Recall, precision, hallucination |
| Couverture | % code couvert |
| Recommandation | Go/No-Go pour phase suivante |

---

## 8. Rôles et responsabilités

| Rôle | Responsabilité | Qui |
|------|----------------|-----|
| Test Manager | Planification, suivi, rapports | Toi |
| Testeur dev | Tests unitaires, intégration | Claude Code |
| Testeur IA | Tests retrieval, hallucination | Toi + Claude Code |
| Beta testeur | Tests utilisabilité | Arbitres volontaires |

---

## 9. Calendrier

| Phase | Tests | Début | Fin |
|-------|-------|-------|-----|
| Phase 1 | Pipeline Python | S+2 | S+4 |
| Phase 2 | Android Retrieval | S+5 | S+8 |
| Phase 3 | LLM + Synthèse | S+9 | S+12 |
| Phase 4 | Optimisation | S+13 | S+15 |
| Phase 5 | Validation | S+16 | S+18 |

---

## 10. Historique du document

| Version | Date | Auteur | Changements |
|---------|------|--------|-------------|
| 1.0 | 2026-01-10 | Equipe Pocket Arbiter | Création initiale |
