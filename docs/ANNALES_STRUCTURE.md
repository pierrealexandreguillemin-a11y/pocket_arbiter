# Annales DNA - Structure des documents

> **Couverture**: 8 ans d'examens (2018-2025, ~16 sessions)
> **Format**: Structure QUASI-IDENTIQUE entre sessions, meme logique

## Format des fichiers PDF annales

Chaque fichier annales contient **4 examens** (UV = Unite de Valeur):

| UV | Contenu | Format |
|----|---------|--------|
| UVR | Regles du jeu d'echecs | QCM |
| UVC | Competitions federales | QCM ou questions ouvertes |
| UVO | Organisation tournois | QCM ou questions ouvertes |
| UVT | Travaux pratiques (PAPI) | Questions ouvertes + exercice |

## Structure de chaque UV dans le PDF

Pour chaque UV, le document contient 3 sections dans l'ordre:

1. **Sujet sans reponse** - Questions d'examen brutes
   - Questions numerotees
   - Choix A/B/C/D pour QCM
   - Enonces pour questions ouvertes

2. **Grille des reponses** - Tableau de correction
   - Numero question
   - Reponse correcte (lettre ou texte court)
   - Article de reference
   - Taux de reussite (%)

3. **Corrige detaille** - Explications completes
   - Question + Reponse attendue
   - Citation de l'article du reglement
   - Commentaires/explications

## Extraction Docling

Docling extrait dans le JSON:
- `markdown`: Texte complet du PDF
- `tables`: Tableaux structures (grilles de reponses)

## Sources des donnees Gold Standard

| Champ GS | Source dans annales |
|----------|---------------------|
| `question` | Sujet sans reponse |
| `answer_text` | Corrige detaille OU choices[lettre] |
| `article_reference` | Grille des reponses / Corrige |
| `expected_pages` | Derive de article_reference |
| `success_rate` | Grille des reponses |
| `choices` | Sujet sans reponse (QCM) |

## Sessions disponibles

- 2018: dec (session_2018)
- 2019: jun, dec
- 2021: jun, dec
- 2022: jun, dec
- 2023: jun, dec
- 2024: jun, dec
- 2025: jun

## Variations mineures entre sessions

| Variation | Sessions concernees | Impact |
|-----------|---------------------|--------|
| Format choix `- a)` | 2022-2025 | Standard |
| Format choix `- A -` | 2021 | Pattern supplementaire |
| Format choix `- ATexte` | 2018 | Pattern bare |
| Questions ouvertes | UVO/UVT anciennes | Extraction corrige |
| Images dans questions | Toutes sessions | Non extractible |

## Logique constante

Malgre les variations de formatage, la logique reste identique:

```
┌─────────────────────────────────────────────────────────────────────┐
│  SUJET (Questions brutes)                                           │
│  ─────────────────────────                                          │
│  • Numerotation: "Question N :" ou "QUESTION N :"                   │
│  • Choix QCM: - a), - A -, - A, A -, etc.                          │
│  • Questions ouvertes: enonce sans choix                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GRILLE DES REPONSES (Tableau)                                      │
│  ──────────────────────────────                                     │
│  Colonnes: Question | Reponse | Articles | Taux Reussite            │
│  Exemple:  1        | D       | Art 9.1.2| 82%                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CORRIGE DETAILLE                                                   │
│  ─────────────────                                                  │
│  • Reponse complete avec justification                              │
│  • Citation article du reglement                                    │
│  • Commentaires pedagogiques                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Notes

- Les reponses sont basees sur les reglements en vigueur AU MOMENT de l'examen
- Pour le Gold Standard, verifier contre le corpus actuel (2024-2025)
- Certaines questions contiennent des images (non extractibles)
- Structure identique permet parsing unifie malgre variations formatage
