# Structured Table Lookup (Level 3) — Design Spec

> **Date**: 2026-03-19
> **Statut**: En cours
> **Prerequis**: Level 1 (summary embedding) DONE, Level 2 (row-as-chunk) REVERTED
> **Standard**: TableRAG NeurIPS 2024, TAG Berkeley 2024, LangChain multi-vector

---

## Architecture

```
Query utilisateur
    |
    v
[Keyword triggers + regex numerique]
    |
    ├─ no triggers → cosine + BM25 (existant, inchange)
    |
    └─ triggers detectes → structured_cell_search()
                              |
                              v
                    SELECT FROM structured_cells
                    WHERE col_name/cell_value MATCH query terms
                              |
                              v
                    table_summary_id(s) trouves
                              |
                              v
                    Three-way RRF: cosine + BM25 + structured (boost 1.5x)
                              |
                              v
                    adaptive_k + build_context (existant)
```

## Implementation — 3 fichiers

### 1. enrichment.py: parse_structured_cells()

Parse les 111 raw_table_text markdown en tuples (table_id, row_idx, col_name, cell_value).
Nettoyage: dot-padding, whitespace, abbreviation expansion.

### 2. indexer.py: wire insert_structured_cells()

Apres le build (pas d'embedding), inserer les cells dans la DB.
Table `structured_cells` deja dans le schema (indexer_db.py).

### 3. search.py: structured_cell_search() + keyword triggers

Keyword triggers stemmees → structured lookup → three-way RRF.
Pas de classifieur, pas de LLM dans la boucle query.

## Schema DB

```sql
CREATE TABLE structured_cells (
    table_id TEXT NOT NULL REFERENCES table_summaries(id),
    row_idx INTEGER NOT NULL,
    col_name TEXT NOT NULL,
    cell_value TEXT NOT NULL,
    source TEXT NOT NULL,
    page INTEGER,
    PRIMARY KEY (table_id, row_idx, col_name)
);
CREATE INDEX idx_cells_col_value ON structured_cells(col_name, cell_value);
```

Estimation: 111 tables × ~5 cols × ~12 rows = ~7K cells. Trivial pour SQLite.

## Keyword triggers

```python
TABLE_TRIGGERS = {
    "berger", "grille", "toutes rondes", "appariement",
    "scheveningen", "equipe", "echiquier",
    "elo", "classement", "niveau", "norme", "titre",
    "bareme", "frais", "deplacement", "distance",
    "cadence", "departage", "coefficient",
    "glossaire", "definition",
    "categorie", "age", "poussin", "pupille", "benjamin",
    "minime", "cadet", "junior",
}
```

Activation: 2+ triggers stemmees dans la query, OU regex numerique `\b\d{3,4}\b` (Elo).

## Three-way RRF

```python
for rank, (doc_id, _) in enumerate(structured_results):
    scores[doc_id] += structured_weight / (k + rank + 1)  # weight=1.5
```

Structured results = table_summary IDs (pas cell IDs). Le contexte retourne au LLM
est le raw_table_text complet (pattern multi-vector LangChain).

---

## Axes d'amelioration (documentes pour apres)

### A. Court terme (cette session ou prochaine)

1. **FTS5 sur structured_cells** — ajouter `structured_cells_fts` pour fuzzy matching
   sur cell_value. Utile quand l'utilisateur ecrit "catégorie pupille" mais la cellule
   dit "Pupilles". Le stemmer Snowball FR couvre deja une partie.

2. **Priority boost par table** — les 6 tables prioritaires (doc task3 lignes 236-245)
   pourraient avoir un boost multiplicateur > 1.5 dans le RRF. Les tables de categories
   d'age et cadences impactent le plus de questions GS.

3. **Validation structured_cells** — apres parsing, verifier que chaque table prioritaire
   est bien parsee. Certaines tables markdown ont des merges de cellules ou du formatage
   non-standard qui casse le parsing pipe-delimited.

### B. Moyen terme (chantier suivant)

4. **Intent detection affinee** — remplacer le keyword trigger par un scoring de confiance
   (nombre de triggers × poids par trigger). Plus de triggers = plus de boost structured.
   Permet un gradient au lieu d'un binaire on/off.

5. **Column-name normalization** — les headers de colonnes varient entre tables (majuscules,
   accents, abreviations). Un mapping de normalisation par table permettrait des queries
   cross-table ("quel est l'age pour [categorie]?" → chercher col "Age" dans toutes les
   tables qui ont cette colonne).

6. **Aggregation queries** — TAG (2024) montre que "combien de categories ont plus de X ans?"
   ne peut pas etre resolu par retrieval seul. Un mini-moteur d'aggregation SQL (COUNT, MIN,
   MAX, GROUP BY) sur structured_cells couvrirait ces cas. Scope Android natif.

### C. Long terme (post-MVP)

7. **Text-to-SQL leger** — Gemma 3n pourrait generer des requetes SQL simples a partir de
   questions en langage naturel. Le schema structured_cells est assez simple pour que meme
   un petit LLM produise des SELECT corrects. Mais necessite inference LLM dans la boucle
   query = latence supplementaire.

8. **Table versioning** — les reglements changent chaque saison. Les tables structurees
   doivent etre re-parsees a chaque mise a jour de corpus. Un hash de raw_table_text
   permettrait de detecter les changements et ne re-parser que les tables modifiees.

9. **Multi-table joins** — certaines questions croisent deux tables (ex: "cadence pour
   un tournoi de categorie pupille"). Requiert un join entre la table categories d'age
   et la table cadences. Architecture: graph de relations entre tables, ou simplement
   retourner les deux tables au LLM et le laisser croiser.

10. **Row-as-chunk rehabilite** — l'experience row-as-chunk a echoue avec 1355 rows
    dans le search general. Mais un row-as-chunk CIBLE (seulement les 6 tables prioritaires,
    ~200 rows) pourrait fonctionner sans polluer le top-k. A tester apres level 3.

---

## Sources

- [TableRAG (NeurIPS 2024)](https://arxiv.org/abs/2410.04739) — schema + cell retrieval
- [TAG (Berkeley 2024)](https://arxiv.org/abs/2408.14717) — text-to-SQL vs RAG for tables
- [LangChain Multi-Vector](https://blog.langchain.com/semi-structured-multi-modal-rag/) — summary retrieval, raw synthesis
- [Ragie Table Chunking](https://www.ragie.ai/blog/our-approach-to-table-chunking) — 3-level strategy
- [Row-as-chunk experiment](../../../data/benchmarks/row_as_chunk_experiment.md) — REVERTED, -6pp R@5
- fix-pipeline-task3.md lignes 191-264 — architecture + 6 tables prioritaires
