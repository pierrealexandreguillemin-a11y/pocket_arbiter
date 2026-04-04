# Targeted Table Retrieval — Design Spec

> **Date**: 2026-04-04
> **Statut**: En revue
> **Baseline**: recall@5 = 63.4% (189/298 GS annales), recall@5 human = 94.1% (32/34)
> **Target**: recall@5 >= 70% (Gate R1)
> **Contrainte**: un seul rebuild (~12 min embedding)

---

## Contexte

### Le problème

Le retrieval pipeline trouve 63.4% des chunks au rang 5. Le modèle SFT v5
(60.1% citations, gate PASS) est prêt à exploiter du meilleur contexte.
Le bottleneck est le retrieval, spécifiquement les tables.

### Diagnostic data-driven

**GS annales (264 Q examen) vs LLM (2232 Q chunk-generated) :**

| Segment | GS recall@5 | LLM recall@5 | Gap |
|---------|-------------|--------------|-----|
| Global | 63.4% | 79.3% | +15.9pp |
| Pages tables | 47.8% | 77.5% | **+29.7pp** |

Le contenu EST dans la DB. 73.8% des questions LLM ciblant les pages GS-manquées
sont retrouvées. Le problème est le **mismatch vocabulaire** entre les questions
d'examen FFE et les embeddings tabulaires.

**42 misses tabulaires (rank 0) se décomposent en :**
- 26 : table EXISTE dans la DB, embedding mismatch (query ≠ table summary)
- 16 : pas de table nearby (coverage gap, hors scope)
- 5 : rank 6-10 (presque top-5, weight tuning)

**Greedy set cover sur les 26 misses :**

| # | Table ID | Page | Rows | Misses couverts | Cumul | Contenu |
|---|----------|------|------|-----------------|-------|---------|
| 1 | LA-table2 | p.47 | 9 | 8 | 31% | Résultats fin de partie |
| 2 | R01-table0 | p.2 | 9 | 6 | 54% | Catégories d'âge FFE |
| 3 | LA-table73 | p.184 | 17 | 3 | 65% | Conversion Elo dp→prob |
| 4 | LA-table68 | p.177 | 3 | 3 | 77% | Checklist arbitre |
| 5 | R01-table1 | p.5 | 4 | 2 | 85% | Équivalences cadences |
| 6 | LA-table63 | p.162 | 3 | 1 | 88% | Conduite arbitre |

**6 tables, 45 rows, 23/26 misses couverts (88%).**

**Human questions (34 Q) :** 94.1% recall@5, 2 misses — les deux pointent vers
des tables cibles (conversion Elo p.185, composition équipes p.7).

### Impact estimé

| Scénario | recall@5 | Delta |
|----------|----------|-------|
| Actuel | 63.4% (189/298) | baseline |
| Best case (23/26 récupérés) | 71.1% (212/298) | +7.7pp |
| Conservateur (50% success) | 67.3% (201/298) | +3.9pp |

---

## Architecture

### Vue d'ensemble

```
Rebuild (build-time, une fois)
  enrichment.py
    -> parse_targeted_rows(6 tables) -> 45 row-chunks
    -> normalize_column_names(all cells) -> mapping canonique
  indexer.py
    -> embed 45 row-chunks (EmbeddingGemma-300M)
    -> insert normalized cells in structured_cells
    -> insert row-chunks in targeted_rows table

Search (query-time)
  search.py
    -> gradient_intent_score(query) -> float 0.0-3.0
    -> cosine + BM25 (existant, inchangé)
    -> structured_cell_search (existant, boost scalé par intent)
    -> targeted_row_cosine (NOUVEAU canal 5)
    -> 5-way RRF fusion (k=60)
    -> adaptive_k + build_context (existant)
```

### Canaux RRF après implémentation

| Canal | Source | Weight | Statut |
|-------|--------|--------|--------|
| 1 | Cosine children + table_summaries | 1.0 | Existant |
| 2 | BM25 FTS5 children + table_summaries | 1.0 | Existant |
| 3 | Structured cell lookup | intent_score × 1.5 | Modifié (B.4) |
| 4 | Narrative rows cosine | 0.5 | Existant |
| 5 | **Targeted row-chunks cosine** | **intent_score × 1.0** | **Nouveau (C.10)** |

---

## Volet 1 : Targeted Row-as-Chunk (C.10)

### Principe

Indexer chaque ligne des 6 tables prioritaires comme un chunk individuel.
Format : `[col1: val1] [col2: val2] ... [colN: valN]` (pas de header repetition).

L'expérience précédente (1355 rows, -6pp) a échoué car :
1. 1355 rows courtes noyaient les children prose dans le top-k
2. Header repetition dégradait BM25
3. Embeddings dominés par les headers, pas les valeurs

Le ciblage résout les 3 problèmes :
1. 45 rows (vs 1355) = pollution négligeable
2. Format `[col: val]` sans header repetition
3. Canal RRF séparé (canal 5) = pas de competition avec la prose

### Tables cibles

```python
TARGETED_TABLES = {
    "LA-octobre2025-table2",                    # 9 rows, 8 Q
    "R01_2025_26_Regles_generales-table0",      # 9 rows, 6 Q
    "LA-octobre2025-table73",                   # 17 rows, 3 Q
    "LA-octobre2025-table68",                   # 3 rows, 3 Q
    "R01_2025_26_Regles_generales-table1",      # 4 rows, 2 Q
    "LA-octobre2025-table63",                   # 3 rows, 1 Q
}
```

### Format row-chunk

Chaque row-chunk est préfixé du nom de la table (Point 1 — contexte orphelin)
et enrichi par forward-fill + unités explicites (Points 2-3).

```
# Exemple LA-table2, row 3 :
"Résultats fin de partie (forfait temps) | [Situation: Roi + Tour contre Roi] [Résultat: Gain] [Durée: Fin normale]"

# Exemple R01-table0, row 1 :
"Catégories d'âge FFE | [Catégorie: Pupille] [Abréviation: Pup] [Âge: 10-11 ans] [Année naissance: 2015-2016]"

# Exemple LA-table73, row 5 :
"Conversion Elo dp vers probabilité | [dp: 92 points] [P: 0.63] [Probabilité: 63%]"
```

**Forward fill (Point 2)** : si une cellule est vide ou contient "id."/"idem",
hériter de la valeur de la ligne précédente AVANT de formatter le row-chunk.
Les tables FFE utilisent ce pattern (catégorie d'âge sur la première ligne seulement).

```python
def forward_fill_rows(rows: list[dict]) -> list[dict]:
    """Fill empty cells with previous row's value."""
    prev = {}
    filled = []
    for row in rows:
        new_row = {}
        for col, val in row.items():
            val_stripped = val.strip().lower() if val else ""
            if not val_stripped or val_stripped in ("", "id.", "id", "idem"):
                new_row[col] = prev.get(col, val)
            else:
                new_row[col] = val
        prev = new_row
        filled.append(new_row)
    return filled
```

**Unités explicites (Point 3)** : suffixer les valeurs numériques ambiguës
quand le col_name le permet.

```python
UNIT_SUFFIXES = {
    "elo": "points",
    "classement": "points",
    "k": "coefficient",
    "cadence": "min",
    "temps": "min",
    "durée": "min",
    "âge": "ans",
    "age": "ans",
    "dp": "points",
}
```

Pas de repetition du header markdown. Chaque row est auto-descriptive.
Double ancrage : `title:` contient le CCH structurel, `text:` contient
le nom de table + les données formatées.

### Embedding

- Format document : `title: {source > section} | text: {row_text}`
- Modèle : EmbeddingGemma-300M base (identique aux children)
- Stockage : nouvelle table `targeted_rows` (distinct des 1355 `table_rows` désactivées)

### Schema DB

```sql
CREATE TABLE targeted_rows (
    id TEXT PRIMARY KEY,          -- "LA-octobre2025-table2-row3"
    table_id TEXT NOT NULL,       -- FK table_summaries.id
    row_idx INTEGER NOT NULL,
    text TEXT NOT NULL,           -- "[col: val] [col: val] ..."
    source TEXT NOT NULL,
    page INTEGER,
    embedding BLOB NOT NULL       -- 768-dim float32
);
CREATE INDEX idx_targeted_rows_table ON targeted_rows(table_id);
```

### Search integration

```python
def targeted_row_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Cosine search on targeted_rows. Returns table_summary_ids."""
    rows = conn.execute(
        "SELECT id, table_id, embedding FROM targeted_rows"
    ).fetchall()
    results = []
    for row_id, table_id, blob in rows:
        emb = blob_to_embedding(blob)
        sim = cosine_similarity(query_embedding, emb)
        results.append((table_id, sim))
    # Dedup: keep max score per table_id
    best = {}
    for table_id, score in results:
        if table_id not in best or score > best[table_id]:
            best[table_id] = score
    return sorted(best.items(), key=lambda x: -x[1])[:top_k]
```

Le résultat retourne des `table_summary_id`, pas des `row_id`.
Le contexte LLM reçoit le `raw_table_text` complet (pattern multi-vector LangChain).

---

## Volet 2 : Gradient Intent Detection (B.4)

### Principe

Remplacer le trigger binaire (on/off) par un score d'intention continu.
Le score module le boost des canaux 3 (structured cells) et 5 (targeted rows)
dans le RRF.

### Weights par trigger

```python
INTENT_WEIGHTS = {
    # Strong triggers (table data likely)
    "berger": 1.0, "grille": 0.9, "scheveningen": 1.0,
    "bareme": 0.8, "barème": 0.8,
    "categorie": 0.7, "catégorie": 0.7,
    "cadence": 0.7,
    "elo": 0.8, "classement": 0.6,
    "titre": 0.6, "norme": 0.7,
    "departage": 0.7, "départage": 0.7,
    "frais": 0.6, "deplacement": 0.5,
    "coefficient": 0.7,
    # Weak triggers (contextual)
    "age": 0.4, "âge": 0.4,
    "poussin": 0.5, "pupille": 0.5, "benjamin": 0.5,
    "minime": 0.5, "cadet": 0.5, "junior": 0.5,
    "glossaire": 0.3, "definition": 0.3,
}
```

### Calcul intent score

```python
def gradient_intent_score(query: str) -> float:
    """Compute intent score from query terms. Returns 0.0-3.0."""
    terms = set(stem(w) for w in query.lower().split())
    score = 0.0
    for trigger, weight in INTENT_WEIGHTS.items():
        if stem(trigger) in terms:
            score += weight
    # Numeric pattern boost (3-4 digit = Elo)
    if re.search(r"\b\d{3,4}\b", query):
        score += 0.5
    return min(score, 3.0)  # cap
```

### Application dans le RRF

```python
intent = gradient_intent_score(query)

# Canal 3 : structured cells
# intent=0 → weight=0 (désactivé, pas de pollution prose — Point 4)
# intent=1.0 → weight=1.5 (standard)
# intent=2.0 → weight=3.0 (fort signal tabulaire)
structured_weight = intent * 1.5

# Canal 5 : targeted rows
# intent=0 → weight=0 (désactivé)
# intent=1.0 → weight=1.0 (standard)
# intent=2.0 → weight=2.0 (fort signal tabulaire)
targeted_weight = intent * 1.0
```

**Point 4 (shadowing) résolu** : quand intent=0 (query narrative/disciplinaire),
les canaux 3 et 5 sont à weight=0 — aucune pollution tabulaire possible.
Les 45 targeted rows ne sont PAS dans le FTS5 index (cosine-only canal 5),
donc le BM25 ne peut pas les sur-scorer.

Le binaire actuel (`_has_table_triggers` >= 2 triggers) est remplacé.
Le structured_cell_search s'active dès que intent > 0, avec boost proportionnel.

---

## Volet 3 : Normalisation noms de colonnes (B.5)

### Principe

Mapper les headers de colonnes bruts vers des noms canoniques.
Appliqué au moment de l'indexation dans `structured_cells`.
Renforce le matching FTS5 quand la question utilise un terme différent du header.

### Mapping

```python
COLUMN_NORMALIZATION = {
    # Variantes -> canonical
    "cat.": "Catégorie",
    "cat": "Catégorie",
    "categ.": "Catégorie",
    "niv.": "Niveau",
    "niv": "Niveau",
    "nb": "Nombre",
    "nb.": "Nombre",
    "tps": "Temps",
    "tps/ronde": "Temps par ronde",
    "dur.": "Durée",
    "age": "Âge",
    "min": "Minimum",
    "max": "Maximum",
    "dept": "Département",
    "dep": "Département",
    "pts": "Points",
    "class.": "Classement",
    "rk": "Rang",
}
```

### Application

Au moment de `insert_structured_cells` dans l'indexer :
1. Lire le `col_name` brut
2. Chercher dans `COLUMN_NORMALIZATION` (case-insensitive, stripped)
3. Si trouvé : stocker le nom canonique dans `col_name`
4. Sinon : garder le nom brut

Le FTS5 indexe le nom canonique, pas le brut.

---

## Fichiers modifiés

| Fichier | Action | Lignes estimées |
|---------|--------|-----------------|
| `scripts/pipeline/enrichment.py` | MODIFY — `format_targeted_rows()`, `normalize_column_name()` | +60 |
| `scripts/pipeline/indexer.py` | MODIFY — insert targeted_rows, normalize cells | +40 |
| `scripts/pipeline/indexer_db.py` | MODIFY — CREATE TABLE targeted_rows | +10 |
| `scripts/pipeline/search.py` | MODIFY — `targeted_row_search()`, `gradient_intent_score()`, RRF 5-way | +80 |
| `scripts/pipeline/tests/test_enrichment.py` | MODIFY — tests targeted rows + normalization | +50 |
| `scripts/pipeline/tests/test_search.py` | MODIFY — tests gradient intent + targeted search | +60 |

---

## Quality Gates

### Avant rebuild

| Gate | Vérification |
|------|-------------|
| T1 | 6 tables cibles parsées : 45 rows total, aucune vide |
| T2 | Format row-chunk : chaque row contient [col: val] pour toutes colonnes |
| T3 | Column normalization : COLUMN_NORMALIZATION appliqué sur structured_cells |
| T4 | Gradient intent : score > 0 pour les 26 queries tabulaires manquées |
| T5 | Gradient intent : score = 0 pour 10 queries prose pures (pas de faux positifs) |

### Après rebuild (I1-I9 existants + nouveaux)

| Gate | Vérification |
|------|-------------|
| I10 | targeted_rows : 45 entries, toutes avec embedding 768-dim |
| I11 | structured_cells : col_names normalisés où applicable |

### Après recall measurement

| Gate | Vérification |
|------|-------------|
| R1 | recall@5 >= 70% GS annales (gate projet) |
| R2 | recall@5 human >= 94.1% (pas de régression) |
| R3 | Aucune régression > 2 questions sur les 189 qui passaient |
| R4 | Tabular recall@5 > 61.5% (amélioration significative) |

---

## Ordre d'exécution

1. **Volet 3 (B.5)** — Normalisation colonnes dans enrichment.py (pas de rebuild)
2. **Volet 1 (C.10)** — Targeted row-chunks : format, embed, insert
3. **Volet 2 (B.4)** — Gradient intent dans search.py
4. **Un seul rebuild** — 12 min (embed 45 rows + reindex cells normalisées)
5. **Tests unitaires** — T1-T5 gates
6. **Recall measurement** — GS annales + human + LLM
7. **Gate check** — R1-R4

---

## Pièges row-chunking adressés (review utilisateur)

| # | Piège | Cause | Parade |
|---|-------|-------|--------|
| 1 | Orphelin (contexte manquant) | Row `[col: val]` sans lien à la table | Préfixe nom de table dans `text:` + CCH dans `title:` |
| 2 | Forward fill (cellules vides) | Tables FFE : catégorie sur ligne 1 seulement | `forward_fill_rows()` avant formatting |
| 3 | Ambiguïté numérique | "1500" = Elo ou nb licenciés ? | `UNIT_SUFFIXES` : Elo → "points", cadence → "min" |
| 4 | Shadowing FTS5 | 45 rows denses sur-scorées par BM25 | Canal 5 = cosine-only (pas FTS5). Intent=0 → weight=0 |

---

## Ce qu'on ne fait PAS

- Pas de row-as-chunk sur les 111 tables (seulement 6 ciblées, 45 rows)
- Pas de cross-encoder reranker (600 MB RAM runtime)
- Pas de Text-to-SQL (latence LLM dans la boucle query)
- Pas de multi-table joins (complexité vs gain incertain)
- Pas de ré-activation des 1355 table_rows existantes (pollution prouvée)

---

## Standards

| Standard | Application |
|----------|-------------|
| TableRAG NeurIPS 2024 | Schema + cell retrieval (volets 1+3) |
| TAG Berkeley 2024 | Table-aware generation (volet 1) |
| LangChain multi-vector | Summary retrieval, raw synthesis (pattern existant) |
| EMNLP 2025 adaptive-k | Largest-gap (existant, inchangé) |
| Ragie table chunking | 3-level strategy (level 3 existant + C.10 ciblé) |
| ISO 29119 | TDD, quality gates |
| ISO 25010 | Fichiers <= 300 lignes |

---

## Sources

- [TableRAG NeurIPS 2024](https://arxiv.org/abs/2410.04739) — schema + cell retrieval
- [TAG Berkeley 2024](https://arxiv.org/abs/2408.14717) — text-to-SQL vs RAG for tables
- [LangChain Multi-Vector](https://blog.langchain.com/semi-structured-multi-modal-rag/) — summary retrieval
- [Ragie Table Chunking](https://www.ragie.ai/blog/our-approach-to-table-chunking) — 3-level strategy
- data/benchmarks/canal4_vs_phase3_audit.md — Canal 4 audit (42 misses breakdown)
- data/benchmarks/row_as_chunk_experiment.md — row-as-chunk REVERTED (-6pp)
- docs/superpowers/specs/2026-03-19-structured-tables-design.md — original design
