# Pocket Arbiter

> Application Android 100% offline pour arbitres d'echecs — Q&A sur les reglements avec IA

[![ISO 25010](https://img.shields.io/badge/ISO-25010-blue)](docs/QUALITY_REQUIREMENTS.md)
[![ISO 42001](https://img.shields.io/badge/ISO-42001-green)](docs/AI_POLICY.md)
[![Android](https://img.shields.io/badge/Android-10%2B-brightgreen)]()

---

## Objectif

Permettre aux arbitres d'echecs de trouver rapidement les informations reglementaires en posant des questions en langage naturel. L'application fonctionne **100% hors ligne** et cite toujours ses sources.

### Fonctionnalites cles

- **2 corpus** : Reglements francais (FFE, 28 PDFs) et internationaux (FIDE, a construire)
- **Recherche semantique** : Hybrid cosine + BM25 FTS5, 5-way RRF, adaptive-k
- **Synthese IA** : Generation avec citations verbatim (source + page)
- **100% offline** : Aucune connexion requise
- **Vie privee** : Aucune donnee collectee

---

## Architecture

```
PRE-BUILD (Python, une fois)              RUNTIME (Android, offline)

PDF (28 docs FFE)                         Question utilisateur
    |                                         |
    v                                         v
[docling] -> Extraction                   [EmbeddingGemma-300M] -> Query embedding
    |                                         |
    v                                         v
[LangChain] -> Chunks 450/50              [SQLite] -> Hybrid search
    |                                     (cosine + BM25 FTS5 + structured cells
    v                                      + narrative rows + targeted rows)
[EmbeddingGemma-300M] -> Embeddings           |
    |                                         v
    v                                     [5-way RRF k=60] -> adaptive-k
[SQLite DB] -> corpus_v2_fr.db                |
  (children, parents, table_summaries,        v
   structured_cells, targeted_rows,       [LLM] -> Reponse + citations
   narrative_rows, FTS5 index)
```

---

## Etat du projet (avril 2026)

| Composant | Statut | Detail |
|-----------|--------|--------|
| **Corpus extraction** | Done | 28 PDFs FFE, docling + hierarchical-pdf |
| **Chunking** | Done | 1117 children, 273 parents, 117 table summaries |
| **Retrieval** | Done | recall@5 = 55.4% (max_k=5, production) |
| **Enrichment** | Done | Contextual retrieval, abbreviations, targeted rows, gradient intent |
| **Generation 270M** | Abandonne | ADR-001 gate : 0/34 reponses utiles |
| **Generation 1B** | Disqualifie | Hallucinations massives malgre contextes corrects |
| **Generation next** | En evaluation | Candidat : Gemma 4 E2B (2.3B eff, 128K ctx) |
| **Android** | A faire | LiteRT + LiteRT-LM |
| **Tests** | 334 tests | 68% coverage |

---

## Stack technique

### Pipeline de donnees (Python)

| Couche | Technologie |
|--------|-------------|
| Extraction PDF | docling + docling-hierarchical-pdf |
| Chunking | LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (EmbeddingGemma-300M, 768-dim) |
| Index | SQLite custom (cosine + BM25 FTS5 + structured cells) |
| Enrichment | Contextual retrieval (Anthropic 2024), abbreviations, CCH, chapter overrides |
| Stemming | Snowball FR + 70 synonymes chess |
| Qualite | Ruff, MyPy, pytest, pre-commit hooks |

### Application Android (a venir)

| Couche | Technologie |
|--------|-------------|
| Langage | Kotlin |
| UI | Jetpack Compose |
| Embeddings | LiteRT (.tflite) |
| LLM | LiteRT-LM (.litertlm) |
| Min SDK | Android 10 (API 29) |

---

## Structure du projet

```
pocket_arbiter/
├── scripts/
│   ├── pipeline/         # Pipeline actif (chunker, indexer, search, enrichment, recall)
│   ├── iso/              # Validation ISO (125 tests)
│   └── archive/          # Scripts archives
├── corpus/
│   ├── fr/               # 28 PDF FFE
│   ├── intl/             # 1 PDF FIDE
│   └── processed/        # Chunks, parents, DB (DVC tracked)
├── tests/data/           # Gold Standard (403 Q, 298 testables)
├── data/benchmarks/      # Recall baselines, experiments
├── docs/                 # Specs, plans, postmortems, ISO (65+ docs)
├── models/               # model_card.json
└── kaggle/               # Kernels training/eval
```

---

## Demarrage rapide

```bash
git clone https://github.com/pierrealexandreguillemin-a11y/pocket_arbiter.git
cd pocket_arbiter
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
pip install pre-commit xenon pip-audit
python -m pre_commit install
python -m pre_commit install --hook-type commit-msg --hook-type pre-push
```

### Commandes

```bash
# Tests (sans extraction PDF)
python -m pytest scripts/iso/ scripts/pipeline/tests/ -m "not slow" -v

# Tests complets (inclut extraction PDF ~1h)
python -m pytest scripts/iso/ scripts/pipeline/tests/ -v

# Quality hooks
python -m pre_commit run --all-files

# Lint
python -m ruff check scripts/

# Re-extraire corpus (~1h)
python scripts/pipeline/extract.py

# Push DB versionnee
python -m dvc push
```

---

## Avertissement IA

Cette application utilise l'intelligence artificielle pour aider a trouver des informations dans les reglements officiels.

- Les reponses sont des **interpretations indicatives**
- Referez-vous **toujours** au texte officiel cite
- L'arbitre reste **seul responsable** de ses decisions
- **Aucune donnee** n'est collectee ni transmise

---

## Documentation

| Document | Description |
|----------|-------------|
| [VISION.md](docs/VISION.md) | Vision et objectifs (Dual-RAG) |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architecture technique |
| [AI_POLICY.md](docs/AI_POLICY.md) | Politique IA responsable (ISO 42001) |
| [QUALITY_REQUIREMENTS.md](docs/QUALITY_REQUIREMENTS.md) | Exigences qualite (ISO 25010) |
| [PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md) | Chronologie des decisions |
| [GENERATION_EVAL_METHODOLOGY.md](docs/GENERATION_EVAL_METHODOLOGY.md) | Methodologie eval generation |
