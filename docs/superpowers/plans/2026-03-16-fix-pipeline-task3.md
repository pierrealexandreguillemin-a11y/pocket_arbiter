# Task 3 : Indexer — Plan detaille

> **Prerequis** : Task 2 complete (1154 children, 304 parents, 28 PDFs extraits)
> **Spec** : `docs/superpowers/specs/2026-03-16-fix-pipeline-design.md` sections 2-3

---

## Objectif

Embedder les 1154 children + 111 table summaries avec EmbeddingGemma-300M (QAT) + Contextual Chunk Headers, stocker dans SQLite avec parents.

---

## Configuration modele

| Parametre | Valeur | Source |
|-----------|--------|--------|
| Model ID | `google/embeddinggemma-300m-qat-q4_0-unquantized` | archives projet + decision coherence mobile |
| Dimensions | 768 | HuggingFace model card |
| Batch size | 128 | Google recommendation |
| Normalisation | L2 (norme=1) | HuggingFace model card |
| Precision | float32 ou bfloat16 (PAS float16) | Google docs, verifie mars 2026 |
| Max sequence | 2048 tokens | HuggingFace model card |
| Framework | sentence-transformers >= 5.2.0 | verifie mars 2026, v5.3.0 latest |
| Temps estime | ~383ms/chunk → ~8 min pour 1265 vecteurs | report precedent |

### Choix QAT vs full precision

- `google/embeddinggemma-300m` : full precision, meilleure qualite (~0.5% NDCG de plus)
- `google/embeddinggemma-300m-qat-q4_0-unquantized` : QAT, concu pour deploiement mobile

**Decision : QAT pour indexation ET inference mobile.** Raison : coherence embeddings.
Si on indexe avec full et que l'app Android query avec QAT, distribution shift.
La perte de 0.5% est negligeable vs le risque de mismatch.
(Decision coherente avec archives projet : embeddings_config.py:25)

---

## CCH + Prompts Google

sentence-transformers v5+ a `model.encode_query()` et `model.encode_document()` qui appliquent les prompts automatiquement. Le CCH s'integre via le parametre `title` du prompt document.

### Encoding documents (indexation)

```python
model = SentenceTransformer("google/embeddinggemma-300m-qat-q4_0-unquantized")
# CCH = title parameter
title = f"{source_display_title} | {section}"
# encode_document applique: "title: {title} | text: {content}"
emb = model.encode_document([chunk_text], titles=[title])
```

### Encoding queries (search)

```python
# encode_query applique: "task: search result | query: {content}"
emb = model.encode_query([question])
```

### Fallback (si encode_query/encode_document pas disponible)

Prependre manuellement :
- Documents : `f"title: {title} | text: {chunk_text}"`
- Queries : `f"task: search result | query: {question}"`

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
- `make_cch_title(source, section, source_titles)` → `"{display_title} | {section}"` pour le parametre title
- `embed_documents(texts, titles, model_id)` → np.ndarray (N, 768), L2 normalized, via `model.encode_document()`
- `embed_queries(queries, model_id)` → np.ndarray (N, 768), via `model.encode_query()`
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
