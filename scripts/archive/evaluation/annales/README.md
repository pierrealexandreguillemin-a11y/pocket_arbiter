# Pipeline Gold Standard v6 - Annales DNA

> **ISO 42001 A.7.3** - Tracabilite des donnees d'evaluation
> **Version**: 7.0.0
> **Date**: 2026-02-04
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
| `parse_docling_full.py` | Parse docling_document.texts -> explications | **Nouveau** |
| `extract_by_color.py` | Extraction par couleur (PyMuPDF) | **Nouveau** |
| `create_jun2025_final.py` | Fusion grilles + Docling + couleur | **Nouveau** |
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
| jun2023 | 170* | 20 | 12% | *Structure Metropole/DOM-TOM, voir section dediee |
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

### Sortie: Gold Standard JSON (Schema v2)

Schema v2.0 avec 46 champs en 8 groupes (voir `docs/specs/GS_SCHEMA_V2.md`):

| Groupe | Champs | Description |
|--------|--------|-------------|
| Racine | 2 | id, legacy_id |
| content | 3 | question, expected_answer, is_impossible |
| mcq | 5 | original_question, choices, mcq_answer, correct_answer, original_answer |
| provenance | 7 | chunk_id, docs, pages, article_reference, answer_explanation, annales_source |
| classification | 8 | category, keywords, difficulty, question_type, cognitive_level, etc. |
| validation | 7 | status, method, reviewer, answer_current, verified_date, etc. |
| processing | 7 | chunk_match_score, chunk_match_method, triplet_ready, etc. |
| audit | 3 | history, qat_revalidation, requires_inference |

8 contraintes de coherence obligatoires (C1-C8) documentees dans le schema.

Exemple simplifie:
```json
{
  "id": "ffe:annales:regles:001:abc123",
  "legacy_id": "FR-ANN-UVR-001",
  "content": {
    "question": "A quel moment un joueur a-t-il le trait ?",
    "expected_answer": "Quand son adversaire a joue et valide...",
    "is_impossible": false
  },
  "mcq": {
    "original_question": "...",
    "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "mcq_answer": "C",
    "correct_answer": "...",
    "original_answer": "..."
  },
  "provenance": {
    "chunk_id": "LA-octobre2025.pdf-p010-...",
    "article_reference": "Article 1.3 des regles du jeu",
    "answer_explanation": "Explication du corrige detaille...",
    "annales_source": {"session": "dec2024", "uv": "regles", "question_num": 1, "success_rate": 0.84}
  }
}
```

## Conformite ISO

| Norme | Exigence | Implementation |
|-------|----------|----------------|
| ISO 42001 A.7.3 | Tracabilite donnees | annales_source + article_reference |
| ISO 29119-3 | Documentation tests | Ce README + specs |
| ISO 25010 | Qualite fonctionnelle | 90.4% couverture reponses |

## Structures speciales par session

### jun2025 - Structure Standard Complete (110 questions)

Session de reference avec structure complete documentee:

| UV | Questions | Pages Sujet | Page Grille | Pages Corrige |
|----|-----------|-------------|-------------|---------------|
| UVR | 30 | 3-10 | 11 | 12-19 |
| UVC | 30 | 20-25 | 26 | 27-34 |
| UVO | 20 | 35-39 | 40 | 41-46 |
| UVT | 30 | 47-56 | 57 | 58-72 |

**Total jun2025: 110 questions**

Structure PDF type (72 pages):
- Page 1: Couverture
- Page 2: Sommaire
- Pages 3-72: 4 UV × 3 sections (Sujet + Grille + Corrige)

Fichiers de donnees jun2025:
- `tests/data/jun2025_grilles.json` - Grilles des 4 UV (110 questions)
- `tests/data/jun2025_final.json` - **Extraction complete: 103/110 (93.6%) avec explications**
- `tests/data/jun2025_color_extracted.json` - Extraction par couleur (PyMuPDF)
- `tests/data/jun2025_docling_extracted.json` - Extraction Docling texts

### jun2023 - Structure Metropole/DOM-TOM

Session unique avec examens separes par region:
- **UVR Metropole** (30 questions) - pages 13-26
- **UVC Metropole** (30 questions) - pages 28-42
- **UVR DOM-TOM** (30 questions) - pages 44-57
- **UVC DOM-TOM** (30 questions) - pages 59-70
- **UVO** (20 questions) - pages 72-77
- **UVT** (30 questions) - pages 88-100

**Total jun2023: 170 questions** (vs 30 dans pipeline automatique)

Fichiers Gold Standard jun2023:
- `tests/data/gs_uvo_jun2023.json` - UVO complet (20q) via DevTools
- `tests/data/gs_jun2023_manual.json` - Grilles UVR/UVC (120q)

## Methode DevTools (extraction manuelle)

Pour sessions avec formats non-standards ou verification PDF:

```bash
# 1. Ouvrir PDF dans Chrome via MCP chrome-devtools
# 2. Naviguer vers page (fill page number + Enter)
# 3. Ajuster zoom (50-75% pour vue complete)
# 4. take_screenshot pour capturer
# 5. Extraire manuellement: question, choix, reponse (violet), explication
```

Avantages:
- Verification directe contre PDF source
- Capture couleurs (reponses correctes en violet)
- Extraction explications completes

## Analyse Docling vs DevTools

### Extraction Docling (corpus/processed/annales_juin_2025/)

Analyse sur jun2025 (Annales-Juin-2025-VF2.json):

| Element | Extrait | Qualite |
|---------|---------|---------|
| Markdown | 192,662 chars | Complet |
| Tables | 22 (dont 4 grilles) | Complet |
| Questions | 218 occurrences | Complet (30q × 4UV × ~2) |
| Choix QCM | 784 markers | Complet |
| Grilles | 4 tables structurees | **Complet** |
| Article refs | 88 references | Complet |
| docling_document.texts | 1,981 blocs | **Complet** |
| docling_document.pages | 72 pages | **Complet** |
| Explications corrige | 102/109 | **94%** (via texts) |

#### Ce que Docling extrait bien:
- **Grilles des reponses**: Tables structurees avec Question, Reponse, Article, Taux
- **Questions QCM**: Texte complet avec les 4 choix (a/b/c/d)
- **References articles**: Dans les grilles et le texte

#### Ce que Docling perd (extraction naive):
- **Extraction naive du markdown**: Seules 4 phrases d'explication explicites
- **Contexte corrige**: Le texte explicatif entre question et article souvent perdu

#### Solution: utiliser docling_document.texts

L'extraction complete via `docling_document.texts` capture **102/109 questions avec explications**:

```python
# Extraction complete (pas juste le markdown)
doc = result["docling_document"]
texts = doc["texts"]  # 1981 blocs de texte structures
tables = doc["tables"]  # 22 tables avec structure
pages = doc["pages"]  # 72 pages avec provenance
```

Script: `scripts/evaluation/annales/parse_docling_full.py`

#### Extraction par couleur (PyMuPDF)

Les corriges utilisent des couleurs semantiques:

| Couleur | Code hex | Signification |
|---------|----------|---------------|
| Vert | `#00b050`, `#00cc00` | Reponse correcte |
| Violet | `#7030a0`, `#6600ff` | Reference article |
| Bleu | `#0070c0`, `#0000ff`, `#4472c4` | Explication |
| Noir | `#000000` | Texte normal |

PyMuPDF extrait ces couleurs, permettant l'identification automatique:

```python
import fitz
page = doc[page_num]
for span in page.get_text('dict')['blocks'][0]['lines'][0]['spans']:
    color = span['color']  # Integer RGB
    text = span['text']
```

Script: `scripts/evaluation/annales/extract_by_color.py`

**Note**: Docling ne supporte PAS l'extraction des couleurs ([issue #195](https://github.com/docling-project/docling-parse/issues/195)).

#### Verification manuelle DevTools (7 questions sans explication)

| Question | Page | Verdict | Explication |
|----------|------|---------|-------------|
| UVC Q3 | 27 | ✓ OK | Article = reponse (Stephane GOBERT) |
| UVC Q13 | 30 | ✗ Ratee | Explication bleue presente |
| UVO Q6 | 42 | ✗ Ratee | Explication bleue presente |
| UVT Q1 | 58 | ✓ OK | Article = reponse (L.A. Chap 5.2) |
| UVT Q4 | 59 | ✗ Ratee | Explication bleue presente |
| UVT Q11 | 63 | ✗ Ratee | Explication bleue presente |
| UVT Q18 | 66 | ✗ Ratee | Explication bleue presente |

**Resultat**: 2/7 sans explication reelle, 5/7 ratees par extraction (limites de question mal detectees).

**Couverture reelle jun2025**: 108/110 questions avec explication (98.2%)

### Recommandation

| Cas d'usage | Methode recommandee |
|-------------|---------------------|
| Questions + Reponses + Articles + Taux | **Docling grilles** (tables structurees) |
| Explications corrige detaille | **Docling texts** (102/109 = 94%) |
| Verification manuelle | **DevTools** (7 questions sans explication) |
| Schema v2 complet (46 champs) | **Docling full** + enrichissement grilles |

### Fichiers Docling jun2025

```
corpus/processed/annales_juin_2025/
├── Annales-Juin-2025-VF2.json       # 192k markdown, 22 tables (extraction initiale)
├── Annales-Juin-2025-VF2_full.json  # + docling_document complet (1981 texts, 72 pages)
└── Annales-Decembre-2024.json       # Session precedente

tests/data/
├── jun2025_grilles.json             # Grilles des 4 UV (110 questions)
├── jun2025_final.json               # Extraction fusionnee (grilles + docling + couleur)
├── jun2025_docling_extracted.json   # Extraction Docling texts (109 questions)
├── jun2025_color_extracted.json     # Extraction PyMuPDF couleur
└── jun2025_gs_v2.json               # **Gold Standard Schema v2 (46 champs, 8 groupes)**
```

### Schema v2 (46 champs)

Structure conforme a `docs/specs/GS_SCHEMA_V2.md`:

| Groupe | Champs | Description |
|--------|--------|-------------|
| Racine | 2 | id, legacy_id |
| content | 3 | question, expected_answer, is_impossible |
| mcq | 5 | original_question, choices, mcq_answer, correct_answer, original_answer |
| provenance | 6+4 | chunk_id, docs, pages, article_reference, answer_explanation, annales_source (4 sous-champs) |
| classification | 8 | category, keywords, difficulty, question_type, cognitive_level, reasoning_type, reasoning_class, answer_type |
| validation | 7 | status, method, reviewer, answer_current, verified_date, pages_verified, batch |
| processing | 7 | chunk_match_score, chunk_match_method, reasoning_class_method, triplet_ready, extraction_flags, answer_source, quality_score |
| audit | 3 | history, qat_revalidation, requires_inference |

Script: `scripts/evaluation/annales/convert_to_gs_v2.py`

## Limitations connues

1. **Session jun2021 (64%)**: Format different (## Corrige : inline)
2. **Questions images**: Non extractibles depuis PDF
3. **Session 2018**: Exclue (format trop variable, principe Pareto)
4. **jun2023 pipeline**: Extraction automatique incomplete (30/170 questions)
5. **Explications corrige**: Docling n'extrait pas toutes les explications

## Historique

| Version | Date | Questions | answer_text% | Changements |
|---------|------|-----------|--------------|-------------|
| 6.0.0 | 2026-01-23 | 518 | 64% | Creation initiale |
| 6.3.0 | 2026-01-24 | 518 | 76% | Cleanup, validation |
| 6.4.0 | 2026-01-24 | 477 | 77% | Exclusion session 2018 |
| 6.5.0 | 2026-01-24 | 477 | 90.4% | Extraction corrige detaille |
| 6.6.0 | 2026-02-04 | 477 | 90.4% | Documentation jun2025, analyse Docling |
| 6.7.0 | 2026-02-04 | 477 | 93.6% | Extraction couleur PyMuPDF |
| 6.8.0 | 2026-02-04 | 477 | 98.2% | Verification manuelle DevTools, jun2025: 108/110 |
| **7.0.0** | **2026-02-04** | **587** | **98.2%** | **Schema v2 (46 champs, 8 groupes), jun2025_gs_v2.json** |

---

*Documentation ISO 29119-3 pour le pipeline Gold Standard Annales.*
