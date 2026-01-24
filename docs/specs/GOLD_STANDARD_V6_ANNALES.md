# Gold Standard v6 "Annales-Based" - Spécifications

> ISO 42001 A.7.3 - Documentation des données d'entraînement/évaluation

## 1. Intentions

### 1.1 Objectif principal

Créer un Gold Standard de haute qualité basé sur les **questions officielles des examens d'arbitres FFE** (annales DNA), offrant :

- **Validité garantie** : Questions validées par le jury national des examens
- **Traçabilité ISO 42001** : Chaque question liée à un article source précis
- **Couverture exhaustive** : 500-1500+ questions couvrant toutes les UV (UVR, UVC, UVO, UVT)
- **Difficulté calibrée** : Taux de réussite réel = proxy de difficulté

### 1.2 Problème résolu

| Limitation GS actuel | Solution Annales |
|---------------------|------------------|
| Questions rédigées manuellement | Questions officielles DNA bulletproof |
| Validation par expert unique | Validées par jury national + statistiques |
| Réponses implicites (chunks) | Réponses explicites + article source |
| Difficulté estimée subjectivement | Difficulté mesurée (taux réussite candidats) |
| 237 questions | 500-1500+ questions potentielles |
| Langage technique arbitre | Reformulation langage courant utilisateur |

### 1.3 Cas d'usage

1. **Évaluation RAG** : Mesurer la qualité du retrieval avec des questions validées
2. **Audit chunking** : Identifier les chunks manquants ou mal découpés
3. **ARES multi-métriques** : Context Relevance + Answer Faithfulness
4. **Benchmark difficulté** : Stratifier les tests par niveau de difficulté

## 2. Modus Operandi

### 2.1 Sources de données

```
ANNALES DNA (https://dna.ffechecs.fr/devenir-arbitre/examens/)
├── Sessions disponibles : 2017-2025 (~16 sessions)
├── Format : PDF avec questions + corrections détaillées
├── UV couvertes : UVR, UVC, UVO, UVT
└── Données par question :
    ├── Texte complet + choix A/B/C/D
    ├── Réponse correcte
    ├── Article de référence (ex: "Article 1.3 des règles du jeu")
    └── Taux de réussite des candidats
```

### 2.2 Pipeline de transformation

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: EXTRACTION (Docling)                                         │
│  ─────────────────────────────                                         │
│  Entrée  : Annales PDF                                                 │
│  Sortie  : JSON structuré (markdown + tables)                          │
│  Script  : scripts/pipeline/extract_docling.py (existant)              │
│  Statut  : ✅ Déc 2024 + Juin 2025 extraits (corpus/processed/)        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: PARSING (Nouveau)                                            │
│  ──────────────────────────                                            │
│  Entrée  : JSON Docling (markdown + tables corrections)                │
│  Sortie  : Liste Q/A structurées avec métadonnées                      │
│  Script  : scripts/evaluation/annales/parse_annales.py                 │
│                                                                         │
│  Extraction :                                                           │
│  • Questions depuis markdown (regex "Question N : ...")                │
│  • Choix A/B/C/D                                                       │
│  • Corrections depuis tables (réponse, article, taux)                  │
│  • Association question ↔ correction par numéro                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: MAPPING ARTICLES → CORPUS (Nouveau)                          │
│  ────────────────────────────────────────────                          │
│  Entrée  : Articles de référence (ex: "Article 1.3 des règles du jeu") │
│  Sortie  : Document corpus + pages correspondantes                     │
│  Script  : scripts/evaluation/annales/map_articles_to_corpus.py        │
│                                                                         │
│  Mapping :                                                              │
│  • "Article X.X des règles du jeu" → LA-octobre2025.pdf                │
│  • "R01 - Article Y" → R01_2025_26_Regles_generales.pdf                │
│  • "Chapitre Z du LA" → LA-octobre2025.pdf section Z                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: REFORMULATION (Semi-automatique)                             │
│  ─────────────────────────────────────────                             │
│  Entrée  : Question officielle format examen                           │
│  Sortie  : Question reformulée langage courant                         │
│  Script  : scripts/evaluation/annales/reformulate_questions.py         │
│                                                                         │
│  Exemples :                                                             │
│  • Officiel: "Quand dit-on qu'un joueur a le trait ?"                  │
│  • Courant:  "C'est à qui de jouer après l'appui sur la pendule ?"     │
│                                                                         │
│  Méthode : LLM avec prompt spécialisé + validation humaine             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: VALIDATION RÉPONSES (Critique)                               │
│  ───────────────────────────────────────                               │
│  Entrée  : Réponse officielle annales + article source                 │
│  Sortie  : Réponse validée contre règlements actuels                   │
│  Script  : scripts/evaluation/annales/validate_answers.py              │
│                                                                         │
│  Processus :                                                            │
│  1. Localiser l'article dans le corpus actuel                          │
│  2. Vérifier si la réponse est toujours valide                         │
│  3. Marquer les questions obsolètes (règlement modifié)                │
│  4. Générer rapport de validation                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: GÉNÉRATION GOLD STANDARD v6                                  │
│  ────────────────────────────────────                                  │
│  Entrée  : Questions validées + reformulées + mappées                  │
│  Sortie  : tests/data/gold_standard_annales_fr.json                    │
│  Script  : scripts/evaluation/annales/generate_gold_standard.py        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Structure des données intermédiaires

#### 2.3.1 Sortie parse_annales.py

```json
{
  "session": "dec2024",
  "source_file": "Annales-Decembre-2024.pdf",
  "extraction_date": "2025-01-24",
  "units": [
    {
      "uv": "UVR",
      "questions": [
        {
          "num": 1,
          "text": "Quand dit-on qu'un joueur a le trait ?",
          "choices": {
            "a": "Quand l'arbitre lance le début de la ronde",
            "b": "Quand son adversaire a joué son coup",
            "c": "Quand son adversaire a joué son coup et l'a validé par l'appui sur la pendule",
            "d": "Quand le joueur a noté le coup de son adversaire sur sa feuille de partie"
          },
          "correct_answer": "B",
          "article_reference": "Article 1.3 des règles du jeu",
          "success_rate": 0.84,
          "difficulty": 0.16
        }
      ],
      "statistics": {
        "total_questions": 30,
        "candidates_present": 221,
        "pass_rate": 0.64
      }
    }
  ]
}
```

#### 2.3.2 Format Gold Standard v6 final

```json
{
  "version": "6.0",
  "description": "Gold Standard v6 - Annales-based with official DNA questions",
  "methodology": {
    "source": "Annales examens arbitres FFE (DNA)",
    "validation": "Questions officielles validées par jury national",
    "reformulation": "Langage courant pour couverture requêtes utilisateur",
    "answer_verification": "Vérifiée contre règlements en vigueur",
    "iso_reference": "ISO 42001 A.7.3, ISO 25010 FA-01"
  },
  "taxonomy_standards": {
    "question_type": ["factual", "procedural", "scenario", "comparative"],
    "cognitive_level": ["RECALL", "UNDERSTAND", "APPLY", "ANALYZE"],
    "reasoning_type": ["single-hop", "multi-hop", "temporal"],
    "answer_type": ["extractive", "abstractive", "yes_no", "list", "multiple_choice"],
    "references": [
      "Bloom's Taxonomy for cognitive levels",
      "RAGAS/BEIR standards for question types",
      "Google Cloud RAG Best Practices",
      "Evidently AI RAG Evaluation Guide"
    ]
  },
  "questions": [
    {
      "id": "FR-ANN-UVR-001",
      "question": "À quel moment un joueur a-t-il le trait ?",
      "original_annales": "Quand dit-on qu'un joueur a le trait ?",
      "category": "regles_jeu",
      "expected_docs": ["LA-octobre2025.pdf"],
      "expected_pages": [37],
      "expected_answer": "Quand son adversaire a joué son coup et l'a validé par l'appui sur la pendule",
      "article_reference": "Article 1.3 des règles du jeu",
      "keywords": ["trait", "pendule", "appui", "jouer", "adversaire"],
      "difficulty": 0.16,
      "question_type": "factual",
      "cognitive_level": "RECALL",
      "reasoning_type": "single-hop",
      "answer_type": "multiple_choice",
      "annales_source": {
        "session": "dec2024",
        "uv": "UVR",
        "question_num": 1,
        "success_rate": 0.84
      },
      "validation": {
        "status": "VALIDATED",
        "method": "annales_official",
        "answer_current": true,
        "verified_date": "2025-01-24"
      }
    }
  ]
}
```

## 3. Résultats Attendus

### 3.1 Métriques quantitatives

| Métrique | Cible | Justification |
|----------|-------|---------------|
| **Nombre de questions** | 500+ (phase 1), 1500+ (complet) | 2 sessions × ~110 Q = 220 Q immédiates |
| **Couverture UV** | 100% (UVR, UVC, UVO, UVT) | Toutes les UV des examens |
| **Taux de mapping article→corpus** | ≥95% | Articles stables malgré évolutions |
| **Taux de réponses valides** | ≥90% | Règlements évoluent peu sur le fond |
| **Questions reformulées** | 100% | Couverture langage courant |

### 3.2 Qualité attendue

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GARANTIES DE QUALITÉ                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ✓ Questions validées par jury national DNA/FFE                        │
│  ✓ Réponses officielles avec justification article                     │
│  ✓ Difficulté mesurée objectivement (statistiques candidats)           │
│  ✓ Traçabilité ISO 42001 complète (source → article → corpus)          │
│  ✓ Zéro hallucination (données officielles uniquement)                 │
│  ✓ Couverture langage courant (reformulation)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Livrables

| Livrable | Chemin | Description |
|----------|--------|-------------|
| Script parsing | `scripts/evaluation/annales/parse_annales.py` | Parse JSON Docling → Q/A |
| Script mapping | `scripts/evaluation/annales/map_articles_to_corpus.py` | Articles → documents |
| Script reformulation | `scripts/evaluation/annales/reformulate_questions.py` | Langage courant |
| Script validation | `scripts/evaluation/annales/validate_answers.py` | Vérifie réponses |
| Script génération | `scripts/evaluation/annales/generate_gold_standard.py` | Assemble GS v6 |
| Gold Standard v6 | `tests/data/gold_standard_annales_fr.json` | Fichier final |
| Rapport extraction | `data/evaluation/annales/extraction_report.json` | Statistiques |

### 3.4 Utilisation avec ARES

```
GOLD STANDARD v6 ──────────────────────────────────────────────────────┐
       │                                                               │
       ▼                                                               │
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│ Context         │     │ Answer          │     │ Answer          │   │
│ Relevance       │     │ Faithfulness    │     │ Relevance       │   │
│ ────────────    │     │ ────────────    │     │ ────────────    │   │
│ Question seule  │     │ Q + Réponse     │     │ Q + Réponse     │   │
│ → Chunk trouvé? │     │ → Fidèle ctx?   │     │ → Répond Q?     │   │
└─────────────────┘     └─────────────────┘     └─────────────────┘   │
       │                        │                        │             │
       └────────────────────────┴────────────────────────┘             │
                                │                                      │
                                ▼                                      │
                    ┌─────────────────────────┐                        │
                    │  AUDIT CHUNKING         │◀───────────────────────┘
                    │  Questions difficiles   │
                    │  (taux < 50%)           │
                    │  → Gaps identifiés      │
                    └─────────────────────────┘
```

## 4. Contraintes et Risques

### 4.1 Contraintes

- **Encodage** : PDF français avec accents → UTF-8 normalization requise
- **Format variable** : Structure des annales peut varier selon les sessions
- **Évolution règlements** : Réponses anciennes peuvent être obsolètes

### 4.2 Risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Question mal parsée | Moyen | Faible | Validation manuelle échantillon |
| Article non mappé | Faible | Moyen | Mapping manuel fallback |
| Réponse obsolète | Moyen | Élevé | Flag + vérification systématique |
| Doublon inter-sessions | Élevé | Faible | Déduplication par similarité |

## 5. Planification

### Phase 1 (Immédiate) : Annales disponibles
- [x] Extraction Docling Déc 2024 + Juin 2025
- [ ] parse_annales.py
- [ ] Extraction ~220 questions

### Phase 2 (Court terme) : Enrichissement
- [ ] Téléchargement annales 2017-2024
- [ ] Extraction complète (~1500 questions)
- [ ] Déduplication

### Phase 3 (Moyen terme) : Validation
- [ ] Mapping articles → corpus
- [ ] Validation réponses actuelles
- [ ] Reformulation langage courant

### Phase 4 (Final) : Gold Standard v6
- [ ] Génération GS v6
- [ ] Intégration pipeline ARES
- [ ] Documentation finale
