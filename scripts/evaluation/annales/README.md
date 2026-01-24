# Pipeline Gold Standard v6 - Annales DNA

> **ISO 42001 A.7.3** - Tracabilite des donnees d'evaluation
> **Version**: 6.5.0
> **Date**: 2026-01-24
> **Couverture answer_text**: 90.4%

## Vue d'ensemble

Pipeline d'extraction et de generation du Gold Standard v6 a partir des
annales officielles des examens d'arbitres FFE (DNA).

```
PDF Annales (2019-2025)
        |
        v  [Docling]
JSON Markdown + Tables
        |
        v  [parse_annales.py]
Questions structurees
        |
        +---> [cleanup_gold_standard.py] ---> answer_text depuis choix QCM
        |
        +---> [extract_corrige_answers.py] ---> answer_text depuis corrige detaille
        |
        v  [map_articles_to_corpus.py]
Questions avec mapping article -> document
        |
        v  [generate_gold_standard.py]
Gold Standard v6.5.0
```

## Scripts

| Script | Description | Statut |
|--------|-------------|--------|
| `parse_annales.py` | Parse JSON Docling -> questions structurees | Actif |
| `cleanup_gold_standard.py` | Derive answer_text depuis choix QCM | Actif |
| `extract_corrige_answers.py` | Extrait answer_text depuis corrige detaille | Actif |
| `map_articles_to_corpus.py` | Mappe article_reference -> document corpus | Actif |
| `generate_gold_standard.py` | Assemble le Gold Standard final | Actif |
| `reformulate_questions.py` | Reformulation langage courant | Actif |
| `validate_answers.py` | Validation pages + articles vs corpus | Actif |
| `upgrade_schema.py` | Migration schema (one-shot v6.0->v6.2) | Archive |

## Usage

### Extraction complete (depuis zero)

```bash
# 1. Parser les annales Docling
python -m scripts.evaluation.annales.parse_annales \
    --docling-dir corpus/processed/annales_all \
    --output data/evaluation/annales/parsed

# 2. Mapper vers le corpus
python -m scripts.evaluation.annales.map_articles_to_corpus \
    --input data/evaluation/annales/parsed \
    --output data/evaluation/annales/mapped

# 3. Generer le Gold Standard
python -m scripts.evaluation.annales.generate_gold_standard \
    --input data/evaluation/annales/mapped \
    --output tests/data/gold_standard_annales_fr.json

# 4. Enrichir answer_text depuis corrige detaille
python scripts/evaluation/annales/extract_corrige_answers.py
```

### Mise a jour incrementale

```bash
# Extraire answer_text depuis corrige (idempotent)
python scripts/evaluation/annales/extract_corrige_answers.py

# Dry-run pour voir les extractions
python scripts/evaluation/annales/extract_corrige_answers.py --dry-run
```

## Couverture v6.5.0

| Session | Questions | answer_text | Couverture |
|---------|-----------|-------------|------------|
| dec2019 | 11 | 11 | 100% |
| dec2021 | 27 | 22 | 81% |
| dec2022 | 7 | 7 | 100% |
| dec2023 | 69 | 67 | 97% |
| dec2024 | 104 | 94 | 90% |
| jun2021 | 50 | 32 | 64% |
| jun2022 | 41 | 40 | 98% |
| jun2023 | 30 | 26 | 87% |
| jun2024 | 47 | 44 | 94% |
| jun2025 | 91 | 88 | 97% |
| **Total** | **477** | **431** | **90.4%** |

## Structure des donnees

### Source: Annales PDF

Chaque fichier annales contient 4 UV (Unites de Valeur):
- **UVR**: Regles du jeu d'echecs (QCM)
- **UVC**: Competitions federales (QCM)
- **UVO**: Organisation tournois (QCM/ouvertes)
- **UVT**: Travaux pratiques PAPI (ouvertes)

Structure de chaque UV:
1. Sujet sans reponse (questions)
2. Grille des reponses (tableau: reponse, article, taux)
3. Corrige detaille (explications officielles)

### Sortie: Gold Standard JSON

```json
{
  "version": "6.5.0",
  "questions": [
    {
      "id": "FR-ANN-UVR-001",
      "question": "A quel moment un joueur a-t-il le trait ?",
      "answer_text": "Quand son adversaire a joue et valide par l'appui sur la pendule",
      "answer_source": "corrige_detaille",
      "article_reference": "Article 1.3 des regles du jeu",
      "mcq_answer": "C",
      "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "annales_source": {
        "session": "dec2024",
        "uv": "UVR",
        "question_num": 1,
        "success_rate": 0.84
      }
    }
  ]
}
```

## Conformite ISO

| Norme | Exigence | Implementation |
|-------|----------|----------------|
| ISO 42001 A.7.3 | Tracabilite donnees | annales_source + article_reference |
| ISO 29119-3 | Documentation tests | Ce README + specs |
| ISO 25010 | Qualite fonctionnelle | 90.4% couverture reponses |

## Limitations connues

1. **Session jun2021 (64%)**: Format different (## Corrige : inline)
2. **Questions images**: Non extractibles depuis PDF
3. **Session 2018**: Exclue (format trop variable, principe Pareto)

## Historique

| Version | Date | Questions | answer_text% | Changements |
|---------|------|-----------|--------------|-------------|
| 6.0.0 | 2026-01-23 | 518 | 64% | Creation initiale |
| 6.3.0 | 2026-01-24 | 518 | 76% | Cleanup, validation |
| 6.4.0 | 2026-01-24 | 477 | 77% | Exclusion session 2018 |
| **6.5.0** | **2026-01-24** | **477** | **90.4%** | **Extraction corrige detaille** |

---

*Documentation ISO 29119-3 pour le pipeline Gold Standard Annales.*
