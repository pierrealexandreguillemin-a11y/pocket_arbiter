# Task 3 : Indexer — Plan detaille

> **Prerequis** : Task 2 complete (1154 children, 304 parents, 28 PDFs extraits)
> **Spec** : `docs/superpowers/specs/2026-03-16-fix-pipeline-design.md` sections 2-3

---

## Objectif

Embedder les 1154 children + 111 table summaries avec EmbeddingGemma-300M (QAT) + Contextual Chunk Headers, stocker dans SQLite avec parents.

---

## Configuration modele (depuis archives projet)

| Parametre | Valeur | Source |
|-----------|--------|--------|
| Model ID | `google/embeddinggemma-300m-qat-q4_0-unquantized` | embeddings_config.py:21 |
| Dimensions | 768 | embeddings_config.py:38 |
| Batch size | 128 | embeddings_config.py:41 |
| Normalisation | L2 (norme=1) | embeddings.py:137 |
| Precision | float32 (pas float16) | embeddings_config.py:55 |
| Prompt query | `"task: search result \| query: "` | embeddings_config.py:53 |
| Prompt document | `"title: {title} \| text: "` | embeddings_config.py:54 |
| Fallback | `google/embeddinggemma-300m` | embeddings_config.py:30 |
| Temps estime | ~383ms/chunk → ~8 min pour 1265 vecteurs | report precedent |

**CRITIQUE** : utiliser le QAT, pas le full precision. Distribution shift sinon.

---

## CCH + Prompt Google

Le Contextual Chunk Header s'integre dans le **prompt document Google** :

```
title: Regles generales competitions FFE 2025-26 | 3.2. Forfait isole | text: 3.2.1. Est consideree comme etant forfait ...
```

Format : `title: {source_title} | {section} | text: {chunk_text}`

Pour les queries : `task: search result | query: {question}`

---

## SQLite schema

```sql
CREATE TABLE children (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,     -- 768 float32, L2 normalized
    parent_id TEXT NOT NULL,
    source TEXT NOT NULL,
    page INTEGER,
    article_num TEXT,
    section TEXT,
    tokens INTEGER
);

CREATE TABLE parents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    section TEXT,
    tokens INTEGER,
    page INTEGER
);

CREATE TABLE table_summaries (
    id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,
    raw_table_text TEXT NOT NULL,
    embedding BLOB NOT NULL,     -- 768 float32, L2 normalized
    source TEXT NOT NULL,
    page INTEGER,
    tokens INTEGER
);
```

---

## Etapes

### Step 1 : Ecrire les tests

Tests pour :
- `contextualize_for_embedding(text, source, section)` → format prompt Google + CCH
- `embed_query(query)` → utilise prompt query Google
- `create_db(path)` → schema correct, 3 tables
- `insert_children(db, children, embeddings)` → roundtrip embedding blob
- `insert_parents(db, parents)` → insertion + lecture
- `insert_table_summaries(db, summaries, embeddings)` → insertion + lecture
- `load_table_summaries(path)` → charge les 111 summaries, IDs compatibles

### Step 2 : Implementer indexer.py

Fonctions :
- `contextualize_for_embedding(text, source, section, source_titles)` → `"title: {title} | {section} | text: {text}"`
- `embed_texts(texts, model_id, is_query=False)` → np.ndarray (768,), L2 normalized
- `embed_query(query, model_id)` → np.ndarray (768,), avec prompt query
- `create_db(path)` → execute schema
- `insert_children(db, children, embeddings)` → blob storage
- `insert_parents(db, parents)` → text storage
- `insert_table_summaries(db, summaries, embeddings)` → blob storage
- `load_table_summaries(path)` → parse JSON existant
- `SOURCE_TITLES` : dict source filename → titre lisible (28 entrees)

### Step 3 : Verifier sur donnees reelles

- Charger les 1154 children depuis le chunker
- Contextualiser chaque child
- Verifier que le texte contextualise est < 2048 tokens (max EmbeddingGemma)
- Embedder un echantillon de 10 children, verifier dimensions (768,), norme (~1.0)

### Step 4 : Embedder le corpus complet

- 1154 children contextualises
- 111 table summaries contextualisees
- Batch size 128, float32
- Temps estime : ~8 min

### Step 5 : Construire la DB

- Inserer parents (304)
- Inserer children avec embeddings (1154)
- Inserer table summaries avec embeddings (111)
- Verifier counts dans chaque table

### Step 6 : Quality gates

| Gate | Critere | Comment verifier |
|------|---------|------------------|
| G1 | children count = 1154 | `SELECT COUNT(*) FROM children` |
| G2 | parents count >= 300 | `SELECT COUNT(*) FROM parents` |
| G3 | table_summaries count = 111 | `SELECT COUNT(*) FROM table_summaries` |
| G4 | embedding dim = 768 | `len(np.frombuffer(row, float32))` sur 5 samples |
| G5 | L2 norme ~1.0 | `np.linalg.norm()` sur 5 samples, tolerance 0.01 |
| G6 | toutes pages non-null pour children | `SELECT COUNT(*) WHERE page IS NULL` = 0 |
| G7 | tous parent_id valides | `SELECT ... WHERE parent_id NOT IN (SELECT id FROM parents)` = 0 |
| G8 | GS text retrouvable (297/298) | matching textuel normalise |
| G9 | search test : "composition jury appel" → resultat pertinent | verification manuelle |

---

## DoD (Definition of Done)

- [ ] `scripts/pipeline/indexer.py` ecrit et teste
- [ ] `corpus/processed/corpus_v2_fr.db` genere
- [ ] 9 quality gates PASS
- [ ] Tests unitaires PASS
- [ ] Commit conventionnel
- [ ] CLAUDE.md et memoire a jour
