# Task 4 : Search — Design Spec

> **Date**: 2026-03-17
> **Statut**: Approuve
> **Prerequis**: Task 3 complete (corpus_v2_fr.db: 1253 children, 332 parents, 111 table summaries)
> **Spec parente**: `docs/superpowers/specs/2026-03-16-fix-pipeline-design.md` section 4

---

## Objectif

Implementer la recherche hybride (cosine + BM25) avec adaptive k, parent lookup dedup, et query expansion chess FR. Produit le module `search.py` pret pour la mesure recall (chantier 3).

---

## Architecture

```
Query utilisateur
    |
    v
[1. Query processing]
    +-- Snowball stem (FR)
    +-- Synonym expansion (dict chess ~50 termes)
    +-- EmbeddingGemma encode (format_query)
    |
    v
[2. Dual retrieval]
    +-- Cosine brute-force sur children + table_summaries embeddings
    +-- FTS5 BM25 sur children_fts + table_summaries_fts (texte stemme)
    |
    v
[3. Fusion RRF]
    Reciprocal Rank Fusion des deux listes -> score unifie
    |
    v
[4. Adaptive k]
    min_score, max_gap, max_k -> top-k filtres
    |
    v
[5. Parent lookup + dedup]
    children matches -> parent_id -> parents dedupliques
    table_summaries matchees -> raw_table_text
    |
    v
[6. Context assembly]
    Parents dedupliques + raw tables + metadata (source, page, section)
    -> pret pour le LLM
```

---

## 1. Query processing

### Stemming

Snowball FR (`snowballstemmer`, deja en dependance). Applique au build-time sur le corpus et au query-time sur la query.

Exemples :
- "arbitrage" -> "arbitr"
- "competitions" -> "competit"
- "forfaits" -> "forfait"

### Synonym expansion

Dict bidirectionnel ~50 termes chess/arbitrage FR. Si la query contient un terme, ses synonymes sont ajoutes a la query BM25.

```python
CHESS_SYNONYMS = {
    "cadence": ["temps", "rythme", "controle"],
    "elo": ["classement", "rating"],
    "forfait": ["absence", "defaut"],
    "mat": ["echec et mat"],
    "pendule": ["horloge", "montre"],
    "nul": ["nulle", "partie nulle", "remise"],
    "appariement": ["pairage", "tirage"],
    "homologation": ["validation", "officialisation"],
    "departage": ["tie-break", "barrage"],
    "roque": ["grand roque", "petit roque"],
    "mutation": ["transfert", "changement de club"],
    "licence": ["inscription", "affiliation"],
    "arbitre": ["juge", "directeur de tournoi"],
    "blitz": ["parties eclair", "cadence rapide"],
    "rapide": ["semi-rapide", "parties rapides"],
    # ~35 entrees supplementaires a completer
}
```

Bidirectionnel : "temps de jeu" -> ajoute "cadence". "cadence" -> ajoute "temps".

Pas de poids par synonyme. BM25 pondere naturellement par frequence corpus.

### Embedding

`format_query(query)` -> `"task: search result | query: {query}"` -> `model.encode()`.

Meme modele et prompt que l'indexation (coherence QAT).

---

## 2. Dual retrieval

### Cosine brute-force

Charge tous les embeddings children + table_summaries en memoire. Dot product (embeddings L2 normalises = cosine). Tri par score desc.

- 1253 children + 111 summaries = 1364 vecteurs x 768D
- Estimation : sub-10ms sur CPU (valide par benchmarks 500ms/100K vecteurs)
- Pas d'index ANN (HNSW/IVF) — inutile sous 10K vecteurs

### BM25 via FTS5

SQLite FTS5 avec texte pre-stemme (Snowball FR au build-time). La query est aussi stemmee + synonymes expandes avant MATCH.

```sql
SELECT id, bm25(children_fts) AS score
FROM children_fts
WHERE children_fts MATCH ?
ORDER BY score
LIMIT ?
```

FTS5 `unicode61 remove_diacritics 2` pour normaliser accents (e=e, a=a).

---

## 3. Fusion RRF

Reciprocal Rank Fusion combine les deux listes :

```python
def reciprocal_rank_fusion(
    cosine_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores = {}
    for rank, (doc_id, _) in enumerate(cosine_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

RRF choisi plutot que weighted linear combination car :
- Pas besoin de normaliser les scores (cosine [0,1] vs BM25 [0, +inf])
- Robuste sans tuning de alpha
- Standard industrie (Weaviate, Elasticsearch, Qdrant)

---

## 4. Adaptive k

Parametres exposes avec valeurs conservatrices par defaut :

| Parametre | Default | Role |
|-----------|---------|------|
| `min_score` | 0.005 | Score RRF minimum pour etre inclus (RRF scores ~0.01-0.03) |
| `max_gap` | 0.01 | Ecart max entre resultat N et N+1 (coupe si depasse) |
| `max_k` | 10 | Nombre max de resultats |

**A calibrer au chantier 3** sur la distribution des scores des 298 questions GS.

Logique :
1. Prendre les top-max_k resultats
2. Exclure ceux sous min_score
3. Scanner les gaps : si score[i] - score[i+1] > max_gap, couper a i
4. Retourner les survivants

---

## 5. Parent lookup + dedup

Pour chaque child dans les resultats filtres :
1. Recuperer `parent_id`
2. Charger `parent.text` depuis la table `parents`
3. Deduplication : si plusieurs children pointent vers le meme parent, garder le parent une seule fois
4. Ordonner les parents par le meilleur score child qui les reference

Pour les table_summaries matchees :
- Retourner `raw_table_text` (tableau markdown complet, pas le resume)

---

## 6. Context assembly

Retour de `search()` :

```python
@dataclass
class SearchResult:
    contexts: list[Context]      # Parents dedupliques + tables
    total_children_matched: int  # Nombre de children avant dedup
    scores: dict[str, float]     # id -> score RRF pour tracabilite

@dataclass
class Context:
    text: str           # Parent text ou raw_table_text
    source: str         # Nom PDF
    page: int | None    # Page source
    section: str        # Section heading
    context_type: str   # "parent" ou "table"
    score: float        # Meilleur score child/summary
    children_matched: list[str]  # IDs des children qui ont matche
```

Budget token estime : 2-3 parents (median 863 tok) + 0-2 raw tables = ~2500-3000 tokens. Largement sous les 32K de Gemma 3n.

---

## 7. Schema SQLite (ajouts au build)

```sql
-- Texte stemme des children pour BM25
CREATE VIRTUAL TABLE IF NOT EXISTS children_fts USING fts5(
    id UNINDEXED,
    text_stemmed,
    tokenize='unicode61 remove_diacritics 2'
);

-- Texte stemme des table summaries pour BM25
CREATE VIRTUAL TABLE IF NOT EXISTS table_summaries_fts USING fts5(
    id UNINDEXED,
    text_stemmed,
    tokenize='unicode61 remove_diacritics 2'
);
```

Population au build-time dans `indexer.py` :
1. Pour chaque child : `stem_text(child["text"])` -> INSERT dans `children_fts`
2. Pour chaque table summary : `stem_text(summary["summary_text"])` -> INSERT dans `table_summaries_fts`

---

## 8. ADR : Stemmer

| Decision | Choix | Alternatives rejetees | Raison |
|----------|-------|----------------------|--------|
| Stemmer | Snowball FR (`snowballstemmer`) | spaCy (trop lourd), Porter (anglais seul), fts5-snowball extension (cross-compilation JNI) | Deja en dependance, pure Python, FR natif, offline |
| Strategie | Pre-stem Python au build, dict stemme Kotlin au runtime | Runtime stemmer Android, FTS5 extension native | Pas de portage necessaire, dict hardcode ~50 termes suffit |

---

## 9. ADR : Vector search Android

| Decision | Choix | Alternatives rejetees | Raison |
|----------|-------|----------------------|--------|
| Vector search | Pure Kotlin brute-force dot product | sqlite-vec (SIMD natif), ObjectBox (vector DB) | Sub-10ms pour 1364 vecteurs, zero dependance, confirme Denisov 2026 |

Source : [On-Device RAG (Denisov, fev. 2026)](https://medium.com/google-developer-experts/on-device-rag-for-app-developers-embeddings-vector-search-and-beyond-47127e954c24)

---

## 10. ADR : Hybrid search

| Decision | Choix | Alternatives rejetees | Raison |
|----------|-------|----------------------|--------|
| Retrieval | Cosine + BM25 hybrid avec RRF | Cosine seul (plan initial P0), BM25 seul | +10-15% recall sur corpus reglementaire, FTS5 natif Android, <3ms surcout |
| Fusion | RRF (k=60) | Weighted linear (alpha tuning), DBSF | Pas de normalisation, robuste sans tuning, standard industrie |

Sources :
- [Hybrid Search Done Right (Ashutosh 2026)](https://ashutoshkumars1ngh.medium.com/hybrid-search-done-right-fixing-rag-retrieval-failures-using-bm25-hnsw-reciprocal-rank-fusion-a73596652d22)
- [ZeroClaw: SQLite Vector + FTS5](https://zeroclaws.io/blog/zeroclaw-hybrid-memory-sqlite-vector-fts5/)
- [Weaviate Hybrid Search](https://weaviate.io/blog/hybrid-search-explained)

---

## 11. Build-time vs query-time

| Operation | Quand | Ou |
|-----------|-------|-----|
| Stemmer Snowball FR sur corpus | Build-time | `indexer.py` -> INSERT FTS5 |
| Embeddings children + summaries | Build-time | `indexer.py` (deja fait) |
| Synonymes dict definition | Build-time | `synonyms.py` |
| EmbeddingGemma encode query | Query-time | `search.py` |
| Cosine brute-force | Query-time | `search.py` |
| FTS5 BM25 MATCH | Query-time | `search.py` |
| Synonym expansion query | Query-time | `search.py` |
| Stem query | Query-time | `search.py` |
| RRF fusion | Query-time | `search.py` |
| Adaptive k | Query-time | `search.py` |
| Parent lookup + dedup | Query-time | `search.py` |

---

## 12. Fichiers

| Fichier | Responsabilite |
|---------|---------------|
| `scripts/pipeline/search.py` | Fonctions search (cosine, bm25, rrf, adaptive_k, build_context, search) |
| `scripts/pipeline/synonyms.py` | CHESS_SYNONYMS dict + stem_text + expand_query |
| `scripts/pipeline/tests/test_search.py` | Tests search |
| `scripts/pipeline/tests/test_synonyms.py` | Tests synonymes/stemming |
| `scripts/pipeline/indexer.py` | Modification : ajout build FTS5 tables stemmees |

---

## 13. Quality gates

| Gate | Critere | Verification |
|------|---------|--------------|
| S1 | `search("composition jury appel")` top-1 contient "jury d'appel" | Automatise |
| S2 | `search("cadence Fischer equivalente")` top-3 contient table cadences | Automatise |
| S3 | `search("categorie U12")` top-3 contient table categories d'age | Automatise |
| S4 | BM25 seul retrouve "forfait" dans articles forfaits | Automatise |
| S5 | Hybrid recall >= cosine seul sur 10 queries echantillon | Automatise |
| S6 | Adaptive k retourne 1 <= n <= max_k (jamais 0 pour query valide) | Automatise |
| S7 | Parents dedupliques : jamais de parent en double | Automatise |
| S8 | FTS5 tables contiennent 1253 + 111 rows stemmees | SQL COUNT |

---

## 14. Ce qu'on ne fait PAS

- Fine-tuning embeddings (decision chantier 3, post recall measurement)
- Cross-encoder reranker (P2, conditionnel)
- HyDE (latence mobile)
- ColBERT (200x stockage)
- Metadata filtering (incompatible cascade reglementaire FFE)
- Row-as-chunk pour tables (Task 5 ou sous-tache dediee)
- SQLite structured lookup pour tables (Task 5 ou sous-tache dediee)
- Intent detection table vs prose (Task 5 ou sous-tache dediee)

---

## 15. Sources

- [Anthropic Contextual Retrieval (2025)](https://www.anthropic.com/news/contextual-retrieval)
- [Hybrid Search Done Right (Ashutosh 2026)](https://ashutoshkumars1ngh.medium.com/hybrid-search-done-right-fixing-rag-retrieval-failures-using-bm25-hnsw-reciprocal-rank-fusion-a73596652d22)
- [ZeroClaw: SQLite Vector + FTS5](https://zeroclaws.io/blog/zeroclaw-hybrid-memory-sqlite-vector-fts5/)
- [Weaviate Hybrid Search](https://weaviate.io/blog/hybrid-search-explained)
- [On-Device RAG (Denisov 2026)](https://medium.com/google-developer-experts/on-device-rag-for-app-developers-embeddings-vector-search-and-beyond-47127e954c24)
- [Snowball French stemmer](http://snowball.tartarus.org/algorithms/french/stemmer.html)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [Ragie: Table Chunking](https://www.ragie.ai/blog/our-approach-to-table-chunking)
- [ACL 2025: Terminology Enhanced RAG Legal Corpora](https://aclanthology.org/2025.ldk-1.16/)
- [Parent-Child Chunking LangChain](https://medium.com/@seahorse.technologies.sl/parent-child-chunking-in-langchain-for-advanced-rag-e7c37171995a)
- [DZone: Parent Document Retrieval](https://dzone.com/articles/parent-document-retrieval-useful-technique-in-rag)
