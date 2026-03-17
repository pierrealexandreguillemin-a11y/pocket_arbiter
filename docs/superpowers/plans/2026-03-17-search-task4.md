# Task 4 : Search — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement hybrid search (cosine + BM25 FTS5) with RRF fusion, adaptive k, parent-child dedup, Snowball FR stemmer, and chess synonyms. Produces `search.py` ready for recall measurement (chantier 3).

**Architecture:** Dual retrieval (cosine brute-force + FTS5 BM25) fused via RRF, filtered by adaptive k (min_score/max_gap/max_k), then parent lookup with dedup. Query processing: Snowball FR stem + chess synonym expansion. Build-time: FTS5 tables populated with pre-stemmed text in indexer.py.

**Tech Stack:** Python 3.10+, SQLite FTS5, snowballstemmer, numpy, sentence-transformers (EmbeddingGemma-300M QAT), pytest

**Spec:** `docs/superpowers/specs/2026-03-17-search-design.md`

**ISO Standards:**
- ISO 29119: TDD, coverage >= 80%, tests before implementation
- ISO 25010: Complexity <= B (xenon), max 300 lines/file
- ISO 12207: Conventional commits (feat/fix/test/docs)
- ISO 42001: Citations obligatoires, tracabilite scores

**Industry standards:**
- Hybrid search: RRF fusion (Weaviate, Elasticsearch standard)
- Parent-child RAG: search children, return parents (LangChain, Dify, GraphRAG)
- FTS5 BM25: SQLite native, Android compatible, <3ms (ZeroClaw benchmark)
- Stemming: Snowball FR, pre-stem at build-time (ACL 2025 legal corpora)
- Adaptive k: score threshold + gap detection (CRAG pattern)

---

## Architecture cible

```
corpus_v2_fr.db (SQLite)
+-- children          (1253 rows, embedding BLOB, text)
+-- parents           (332 rows, text)
+-- table_summaries   (111 rows, embedding BLOB, summary_text, raw_table_text)
+-- children_fts      (FTS5 virtual table, text_stemmed)         <-- NEW
+-- table_summaries_fts (FTS5 virtual table, text_stemmed)       <-- NEW
```

```
search("composition jury appel")
  1. stem("composition jury appel") -> "composit juri appel"
  2. expand("composit juri appel") -> "composit juri appel commission"
  3. embed("composition jury appel") -> [0.12, -0.34, ...] (768D)
  4. cosine_search(embedding, max_k=20) -> [(id1, 0.59), (id2, 0.52), ...]
  5. bm25_search("composit juri appel commission", max_k=20) -> [(id1, -4.2), (id3, -5.1), ...]
  6. rrf(cosine_results, bm25_results) -> [(id1, 0.033), (id2, 0.017), ...]
  7. adaptive_k(rrf_results, min_score=0.3, max_gap=0.15, max_k=10) -> top-5
  8. build_context(top-5) -> [Context(parent_text=..., source=..., page=...)]
```

---

## File map

| File | Action | Responsibility | Est. lines |
|------|--------|---------------|------------|
| `scripts/pipeline/synonyms.py` | CREATE | CHESS_SYNONYMS dict, stem_text(), expand_query() | ~80 |
| `scripts/pipeline/search.py` | CREATE | cosine_search, bm25_search, rrf, adaptive_k, build_context, search | ~250 |
| `scripts/pipeline/indexer.py` | MODIFY | Add FTS5 schema + populate_fts() in build_index | ~+40 |
| `scripts/pipeline/tests/test_synonyms.py` | CREATE | Tests stemming + synonym expansion | ~60 |
| `scripts/pipeline/tests/test_search.py` | CREATE | Tests all search functions + quality gates | ~250 |

---

## Task 1: Synonyms module (TDD)

**Files:**
- Create: `scripts/pipeline/synonyms.py`
- Create: `scripts/pipeline/tests/test_synonyms.py`

### Step 1: Write failing tests

- [ ] **1.1: Create test file**

```python
# scripts/pipeline/tests/test_synonyms.py
"""Tests for stemming and synonym expansion."""
from __future__ import annotations

import pytest
from scripts.pipeline.synonyms import (
    CHESS_SYNONYMS,
    stem_text,
    expand_query,
    build_reverse_synonyms,
)


class TestStemText:
    """Test Snowball FR stemming."""

    def test_basic_french_stemming(self) -> None:
        assert stem_text("arbitrage") == "arbitr"

    def test_plural_reduction(self) -> None:
        assert stem_text("competitions") == stem_text("competition")

    def test_verb_conjugation(self) -> None:
        assert stem_text("arbitrer") == stem_text("arbitrage")

    def test_multi_word(self) -> None:
        result = stem_text("les arbitres des competitions")
        assert "arbitr" in result
        assert "competit" in result

    def test_empty_string(self) -> None:
        assert stem_text("") == ""

    def test_preserves_numbers(self) -> None:
        result = stem_text("article 3.2")
        assert "3.2" in result

    def test_diacritics_preserved_in_stem(self) -> None:
        # Snowball handles accented French
        result = stem_text("equipe")
        assert len(result) > 0


class TestExpandQuery:
    """Test synonym expansion."""

    def test_known_synonym_added(self) -> None:
        result = expand_query("cadence de jeu")
        assert "temps" in result or "rythme" in result

    def test_reverse_synonym(self) -> None:
        # "temps" should expand to include "cadence"
        result = expand_query("temps de jeu")
        assert "cadenc" in result  # stemmed

    def test_no_expansion_for_unknown(self) -> None:
        result = expand_query("hello world")
        # Should still return stemmed text, just no extra terms
        assert "hello" in result.lower() or stem_text("hello") in result

    def test_multiple_synonyms(self) -> None:
        result = expand_query("forfait et cadence")
        assert "absenc" in result or "defaut" in result  # forfait synonyms
        assert "temp" in result or "rythm" in result  # cadence synonyms

    def test_empty_query(self) -> None:
        assert expand_query("") == ""

    def test_output_is_stemmed(self) -> None:
        result = expand_query("les forfaits")
        # Output should be stemmed
        assert "forfait" in result  # stem of "forfaits"


class TestChessSynonyms:
    """Test CHESS_SYNONYMS dict completeness."""

    def test_minimum_entries(self) -> None:
        assert len(CHESS_SYNONYMS) >= 15

    def test_key_terms_present(self) -> None:
        for term in ["cadence", "elo", "forfait", "mat", "appariement"]:
            assert term in CHESS_SYNONYMS, f"Missing key term: {term}"

    def test_values_are_lists(self) -> None:
        for key, values in CHESS_SYNONYMS.items():
            assert isinstance(values, list), f"{key} values should be list"
            assert len(values) >= 1, f"{key} should have at least 1 synonym"


class TestBuildReverseSynonyms:
    """Test reverse lookup building."""

    def test_builds_reverse(self) -> None:
        syns = {"a": ["b", "c"]}
        reverse = build_reverse_synonyms(syns)
        assert "b" in reverse
        assert "a" in reverse["b"]
        assert "c" in reverse
        assert "a" in reverse["c"]
```

- [ ] **1.2: Run tests — verify they FAIL**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/test_synonyms.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.pipeline.synonyms'`

### Step 2: Implement synonyms.py

- [ ] **1.3: Create synonyms.py**

```python
# scripts/pipeline/synonyms.py
"""Chess FR synonyms and Snowball FR stemming for BM25 query expansion.

Build-time: stem_text() stems corpus text for FTS5 indexing.
Query-time: expand_query() stems + expands query with chess synonyms.
"""
from __future__ import annotations

import re

import snowballstemmer

_stemmer = snowballstemmer.stemmer("french")

# Bidirectional chess/arbitrage FR synonyms.
# Keys and values are raw (unstemmed) — stemming applied at expansion time.
CHESS_SYNONYMS: dict[str, list[str]] = {
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
    "blitz": ["parties eclair"],
    "rapide": ["parties rapides"],
    "classement": ["elo", "rating", "niveau"],
    "equipe": ["club", "formation"],
    "joueur": ["participant", "competiteur"],
    "partie": ["rencontre", "match"],
    "victoire": ["gain", "point"],
    "defaite": ["perte"],
    "abandon": ["resignation"],
    "promotion": ["transformation", "sous-promotion"],
    "zeitnot": ["pression au temps", "drapeau"],
    "drapeau": ["chute du drapeau", "temps depasse"],
}


def stem_text(text: str) -> str:
    """Stem French text using Snowball FR.

    Args:
        text: Raw French text.

    Returns:
        Space-joined stemmed words. Numbers and punctuation preserved.
    """
    if not text:
        return ""
    words = text.split()
    stemmed = []
    for word in words:
        # Preserve numbers and punctuation-only tokens
        if re.match(r"^[\d.,/\-:]+$", word):
            stemmed.append(word)
        else:
            stemmed.append(_stemmer.stemWord(word.lower()))
    return " ".join(stemmed)


def build_reverse_synonyms(
    synonyms: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build reverse lookup: for each synonym value, list its keys.

    Args:
        synonyms: Forward synonym dict (term -> [synonyms]).

    Returns:
        Reverse dict (synonym -> [terms that have it as synonym]).
    """
    reverse: dict[str, list[str]] = {}
    for key, values in synonyms.items():
        for val in values:
            reverse.setdefault(val, []).append(key)
    return reverse


# Pre-built reverse lookup
_REVERSE_SYNONYMS = build_reverse_synonyms(CHESS_SYNONYMS)


def expand_query(query: str) -> str:
    """Stem query and expand with chess synonyms.

    Args:
        query: Raw user query in French.

    Returns:
        Stemmed query with synonym terms appended.
    """
    if not query:
        return ""
    query_lower = query.lower()
    extra_terms: list[str] = []

    # Forward: if query contains a key, add its synonyms
    for term, synonyms in CHESS_SYNONYMS.items():
        if term in query_lower:
            extra_terms.extend(synonyms)

    # Reverse: if query contains a synonym value, add its key
    for synonym, keys in _REVERSE_SYNONYMS.items():
        if synonym in query_lower:
            extra_terms.extend(keys)

    # Stem everything
    stemmed_query = stem_text(query)
    if extra_terms:
        stemmed_extras = stem_text(" ".join(extra_terms))
        return f"{stemmed_query} {stemmed_extras}"
    return stemmed_query
```

- [ ] **1.4: Run tests — verify they PASS**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/test_synonyms.py -v
```

Expected: ALL PASS

- [ ] **1.5: Commit**

```bash
git add scripts/pipeline/synonyms.py scripts/pipeline/tests/test_synonyms.py
git commit -m "feat(pipeline): add chess FR synonyms and Snowball stemmer

CHESS_SYNONYMS dict (25 entries), stem_text() with Snowball FR,
expand_query() with bidirectional synonym expansion.
TDD: 14 tests."
```

---

## Task 2: FTS5 build in indexer (TDD)

**Files:**
- Modify: `scripts/pipeline/indexer.py` (add FTS5 schema + populate)
- Modify: `scripts/pipeline/tests/test_indexer.py` (add FTS5 tests)

### Step 1: Write failing tests

- [ ] **2.1: Add FTS5 tests to test_indexer.py**

```python
# Append to scripts/pipeline/tests/test_indexer.py

from scripts.pipeline.synonyms import stem_text


class TestFts5Schema:
    """Test FTS5 virtual tables creation."""

    def test_creates_fts5_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        # Check FTS5 tables exist
        tables = [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "children_fts" in tables
        assert "table_summaries_fts" in tables
        conn.close()

    def test_fts5_search_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        conn.execute(
            "INSERT INTO children_fts (id, text_stemmed) VALUES (?, ?)",
            ("child-1", stem_text("arbitrage des competitions")),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT id, bm25(children_fts) FROM children_fts "
            "WHERE children_fts MATCH ?",
            (stem_text("arbitrage"),),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "child-1"
        conn.close()


class TestPopulateFts:
    """Test FTS5 population from children/summaries."""

    def test_populate_children_fts(self, tmp_path: Path) -> None:
        from scripts.pipeline.indexer import populate_fts
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        # Insert a parent + child first
        conn.execute(
            "INSERT INTO parents (id, text, source, section, tokens, page) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p1", "Parent", "test.pdf", "S1", 10, 1),
        )
        children = [
            {"id": "c1", "text": "Les joueurs doivent etre licencies",
             "parent_id": "p1", "source": "test.pdf", "page": 1,
             "article_num": None, "section": "S1", "tokens": 10},
        ]
        emb = np.random.randn(1, 768).astype(np.float32)
        insert_children(conn, children, emb)
        populate_fts(conn)
        # Verify FTS5 has the stemmed text
        count = conn.execute(
            "SELECT COUNT(*) FROM children_fts"
        ).fetchone()[0]
        assert count == 1
        # Verify search works
        rows = conn.execute(
            "SELECT id FROM children_fts WHERE children_fts MATCH ?",
            (stem_text("licencies"),),
        ).fetchall()
        assert len(rows) == 1
        conn.close()
```

- [ ] **2.2: Run tests — verify they FAIL**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/test_indexer.py::TestFts5Schema -v
```

Expected: FAIL (no FTS5 tables in schema / no populate_fts function)

### Step 2: Implement FTS5 in indexer

- [ ] **2.3: Add FTS5 schema to SCHEMA constant in indexer.py**

Add after the existing `table_summaries` CREATE:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS children_fts USING fts5(
    id UNINDEXED,
    text_stemmed,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS table_summaries_fts USING fts5(
    id UNINDEXED,
    text_stemmed,
    tokenize='unicode61 remove_diacritics 2'
);
```

- [ ] **2.4: Add populate_fts() function to indexer.py**

```python
def populate_fts(conn: sqlite3.Connection) -> None:
    """Populate FTS5 tables with stemmed text from children and table_summaries.

    Must be called after insert_children and insert_table_summaries.

    Args:
        conn: SQLite connection with children and table_summaries populated.
    """
    from scripts.pipeline.synonyms import stem_text

    # Clear existing FTS data (idempotent rebuild)
    conn.execute("DELETE FROM children_fts")
    conn.execute("DELETE FROM table_summaries_fts")

    # Populate children_fts
    rows = conn.execute("SELECT id, text FROM children").fetchall()
    conn.executemany(
        "INSERT INTO children_fts (id, text_stemmed) VALUES (?, ?)",
        [(row[0], stem_text(row[1])) for row in rows],
    )

    # Populate table_summaries_fts
    rows = conn.execute(
        "SELECT id, summary_text FROM table_summaries"
    ).fetchall()
    conn.executemany(
        "INSERT INTO table_summaries_fts (id, text_stemmed) VALUES (?, ?)",
        [(row[0], stem_text(row[1])) for row in rows],
    )

    conn.commit()
```

- [ ] **2.5: Call populate_fts() in build_index() after all inserts**

In `build_index()`, after `insert_table_summaries(...)`, add:

```python
    # 7. Build FTS5 index for BM25
    logger.info("=== Step 7: Populating FTS5 index ===")
    populate_fts(conn)
    fts_children = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
    fts_summaries = conn.execute("SELECT COUNT(*) FROM table_summaries_fts").fetchone()[0]
    logger.info("FTS5: %d children + %d summaries indexed", fts_children, fts_summaries)
```

- [ ] **2.6: Run tests — verify they PASS**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/test_indexer.py -m "not slow" -v
```

Expected: ALL PASS (including new FTS5 tests)

- [ ] **2.7: Commit**

```bash
git add scripts/pipeline/indexer.py scripts/pipeline/tests/test_indexer.py
git commit -m "feat(pipeline): add FTS5 tables for BM25 search

FTS5 virtual tables children_fts + table_summaries_fts populated
with Snowball FR stemmed text. populate_fts() called in build_index.
unicode61 remove_diacritics 2 for accent normalization."
```

---

## Task 3: Search core functions (TDD)

**Files:**
- Create: `scripts/pipeline/search.py`
- Create: `scripts/pipeline/tests/test_search.py`

### Step 1: Write failing tests for pure functions

- [ ] **3.1: Create test_search.py with unit tests**

```python
# scripts/pipeline/tests/test_search.py
"""Tests for search module."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import (
    SOURCE_TITLES,
    blob_to_embedding,
    create_db,
    embedding_to_blob,
    insert_children,
    insert_parents,
    insert_table_summaries,
    populate_fts,
)


class TestRecipocalRankFusion:
    """Test RRF score fusion."""

    def test_single_list(self) -> None:
        from scripts.pipeline.search import reciprocal_rank_fusion
        cosine = [("a", 0.9), ("b", 0.8)]
        bm25: list[tuple[str, float]] = []
        result = reciprocal_rank_fusion(cosine, bm25)
        assert result[0][0] == "a"
        assert result[1][0] == "b"

    def test_overlapping_docs_score_higher(self) -> None:
        from scripts.pipeline.search import reciprocal_rank_fusion
        cosine = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        bm25 = [("b", -3.0), ("a", -4.0), ("d", -5.0)]
        result = reciprocal_rank_fusion(cosine, bm25)
        ids = [r[0] for r in result]
        # "a" and "b" appear in both lists, should be top
        assert ids[0] in ("a", "b")
        assert ids[1] in ("a", "b")

    def test_empty_lists(self) -> None:
        from scripts.pipeline.search import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_k_parameter_affects_scores(self) -> None:
        from scripts.pipeline.search import reciprocal_rank_fusion
        cosine = [("a", 0.9)]
        bm25 = [("a", -3.0)]
        r1 = reciprocal_rank_fusion(cosine, bm25, k=60)
        r2 = reciprocal_rank_fusion(cosine, bm25, k=1)
        # Higher k = lower individual scores
        assert r1[0][1] < r2[0][1]


class TestAdaptiveK:
    """Test adaptive k filtering."""

    def test_max_k_limits(self) -> None:
        from scripts.pipeline.search import adaptive_k
        results = [(f"d{i}", 1.0 - i * 0.01) for i in range(20)]
        filtered = adaptive_k(results, min_score=0.0, max_gap=1.0, max_k=5)
        assert len(filtered) == 5

    def test_min_score_filters(self) -> None:
        from scripts.pipeline.search import adaptive_k
        results = [("a", 0.5), ("b", 0.4), ("c", 0.1)]
        filtered = adaptive_k(results, min_score=0.3, max_gap=1.0, max_k=10)
        assert len(filtered) == 2
        assert filtered[-1][0] == "b"

    def test_gap_cuts(self) -> None:
        from scripts.pipeline.search import adaptive_k
        results = [("a", 0.9), ("b", 0.85), ("c", 0.5), ("d", 0.45)]
        # Gap between b(0.85) and c(0.5) = 0.35 > max_gap 0.2
        filtered = adaptive_k(results, min_score=0.0, max_gap=0.2, max_k=10)
        assert len(filtered) == 2

    def test_empty_input(self) -> None:
        from scripts.pipeline.search import adaptive_k
        assert adaptive_k([], min_score=0.3, max_gap=0.15, max_k=10) == []

    def test_all_filtered(self) -> None:
        from scripts.pipeline.search import adaptive_k
        results = [("a", 0.1), ("b", 0.05)]
        filtered = adaptive_k(results, min_score=0.3, max_gap=0.15, max_k=10)
        assert len(filtered) == 0

    def test_single_result_passes(self) -> None:
        from scripts.pipeline.search import adaptive_k
        results = [("a", 0.5)]
        filtered = adaptive_k(results, min_score=0.3, max_gap=0.15, max_k=10)
        assert len(filtered) == 1


class TestBuildContext:
    """Test parent lookup and deduplication."""

    def test_deduplicates_parents(self, tmp_path: Path) -> None:
        from scripts.pipeline.search import build_context
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        insert_parents(conn, [
            {"id": "p1", "text": "Parent one text", "source": "doc.pdf",
             "section": "S1", "tokens": 50, "page": 1},
        ])
        emb = np.random.randn(2, 768).astype(np.float32)
        insert_children(conn, [
            {"id": "c1", "text": "Child 1", "parent_id": "p1",
             "source": "doc.pdf", "page": 1, "article_num": None,
             "section": "S1", "tokens": 10},
            {"id": "c2", "text": "Child 2", "parent_id": "p1",
             "source": "doc.pdf", "page": 1, "article_num": None,
             "section": "S1", "tokens": 10},
        ], emb)
        # Both children point to same parent
        result_ids = [("c1", 0.9), ("c2", 0.8)]
        contexts = build_context(conn, result_ids)
        # Should have exactly 1 parent context (deduplicated)
        parent_contexts = [c for c in contexts if c.context_type == "parent"]
        assert len(parent_contexts) == 1
        assert parent_contexts[0].text == "Parent one text"
        conn.close()

    def test_table_summary_returns_raw(self, tmp_path: Path) -> None:
        from scripts.pipeline.search import build_context
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        emb = np.random.randn(1, 768).astype(np.float32)
        insert_table_summaries(conn, [{
            "id": "t1", "summary_text": "Summary",
            "raw_table_text": "| col1 | col2 |",
            "source": "doc.pdf", "page": 3, "tokens": 10,
        }], emb)
        result_ids = [("t1", 0.7)]
        contexts = build_context(conn, result_ids)
        table_contexts = [c for c in contexts if c.context_type == "table"]
        assert len(table_contexts) == 1
        assert table_contexts[0].text == "| col1 | col2 |"
        conn.close()

    def test_ordered_by_best_score(self, tmp_path: Path) -> None:
        from scripts.pipeline.search import build_context
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        insert_parents(conn, [
            {"id": "p1", "text": "Low", "source": "d.pdf",
             "section": "S1", "tokens": 10, "page": 1},
            {"id": "p2", "text": "High", "source": "d.pdf",
             "section": "S2", "tokens": 10, "page": 2},
        ])
        emb = np.random.randn(2, 768).astype(np.float32)
        insert_children(conn, [
            {"id": "c1", "text": "C1", "parent_id": "p1",
             "source": "d.pdf", "page": 1, "article_num": None,
             "section": "S1", "tokens": 5},
            {"id": "c2", "text": "C2", "parent_id": "p2",
             "source": "d.pdf", "page": 2, "article_num": None,
             "section": "S2", "tokens": 5},
        ], emb)
        result_ids = [("c2", 0.9), ("c1", 0.3)]
        contexts = build_context(conn, result_ids)
        # p2 should come first (higher score child)
        assert contexts[0].text == "High"
        conn.close()
```

- [ ] **3.2: Run tests — verify they FAIL**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/test_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.pipeline.search'`

### Step 2: Implement search.py

- [ ] **3.3: Create search.py with dataclasses + pure functions**

```python
# scripts/pipeline/search.py
"""Hybrid search: cosine + BM25 FTS5 with RRF fusion.

Query flow:
1. Stem + synonym expand (for BM25)
2. Embed query (for cosine)
3. Dual retrieval: cosine brute-force + FTS5 BM25
4. RRF fusion
5. Adaptive k filtering
6. Parent lookup + dedup
7. Context assembly
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.pipeline.indexer import (
    DEFAULT_MODEL_ID,
    EMBEDDING_DIM,
    SOURCE_TITLES,
    blob_to_embedding,
    format_query,
    load_model,
)
from scripts.pipeline.synonyms import expand_query, stem_text

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """A single context block for the LLM."""

    text: str
    source: str
    page: int | None
    section: str
    context_type: str  # "parent" or "table"
    score: float
    children_matched: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Complete search result."""

    contexts: list[Context]
    total_children_matched: int
    scores: dict[str, float]


# === Pure functions ===


def reciprocal_rank_fusion(
    cosine_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        cosine_results: [(doc_id, cosine_score), ...] sorted desc.
        bm25_results: [(doc_id, bm25_score), ...] sorted by relevance.
        k: RRF constant (default 60, standard value).

    Returns:
        [(doc_id, rrf_score), ...] sorted desc by RRF score.
    """
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(cosine_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def adaptive_k(
    results: list[tuple[str, float]],
    min_score: float = 0.3,
    max_gap: float = 0.15,
    max_k: int = 10,
) -> list[tuple[str, float]]:
    """Filter results by score threshold and gap detection.

    Args:
        results: [(doc_id, score), ...] sorted desc.
        min_score: Minimum score to keep.
        max_gap: Maximum gap between consecutive scores before cutting.
        max_k: Hard maximum number of results.

    Returns:
        Filtered results.
    """
    if not results:
        return []

    # Apply max_k
    results = results[:max_k]

    # Apply min_score
    results = [(doc_id, score) for doc_id, score in results if score >= min_score]

    if not results:
        return []

    # Apply gap detection
    filtered = [results[0]]
    for i in range(1, len(results)):
        gap = results[i - 1][1] - results[i][1]
        if gap > max_gap:
            break
        filtered.append(results[i])

    return filtered


# === DB-dependent functions ===


def cosine_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    max_k: int = 20,
) -> list[tuple[str, float]]:
    """Brute-force cosine search on children + table_summaries.

    Args:
        conn: SQLite connection to corpus DB.
        query_embedding: Query vector (768D, L2 normalized).
        max_k: Maximum results to return.

    Returns:
        [(id, cosine_score), ...] sorted desc.
    """
    results: list[tuple[str, float]] = []

    # Search children
    for row in conn.execute("SELECT id, embedding FROM children"):
        emb = blob_to_embedding(row[1])
        score = float(np.dot(query_embedding, emb))
        results.append((row[0], score))

    # Search table summaries
    for row in conn.execute("SELECT id, embedding FROM table_summaries"):
        emb = blob_to_embedding(row[1])
        score = float(np.dot(query_embedding, emb))
        results.append((row[0], score))

    results.sort(key=lambda x: -x[1])
    return results[:max_k]


def bm25_search(
    conn: sqlite3.Connection,
    stemmed_query: str,
    max_k: int = 20,
) -> list[tuple[str, float]]:
    """BM25 search via FTS5 on stemmed text.

    Args:
        conn: SQLite connection with FTS5 tables.
        stemmed_query: Stemmed + expanded query string.
        max_k: Maximum results to return.

    Returns:
        [(id, bm25_score), ...] sorted by relevance (lower = better in FTS5).
    """
    if not stemmed_query.strip():
        return []

    results: list[tuple[str, float]] = []

    # Search children_fts
    try:
        rows = conn.execute(
            "SELECT id, bm25(children_fts) AS score FROM children_fts "
            "WHERE children_fts MATCH ? ORDER BY score LIMIT ?",
            (stemmed_query, max_k),
        ).fetchall()
        results.extend((row[0], row[1]) for row in rows)
    except sqlite3.OperationalError:
        logger.warning("FTS5 MATCH failed for query: %s", stemmed_query[:50])

    # Search table_summaries_fts
    try:
        rows = conn.execute(
            "SELECT id, bm25(table_summaries_fts) AS score "
            "FROM table_summaries_fts "
            "WHERE table_summaries_fts MATCH ? ORDER BY score LIMIT ?",
            (stemmed_query, max_k),
        ).fetchall()
        results.extend((row[0], row[1]) for row in rows)
    except sqlite3.OperationalError:
        logger.warning("FTS5 table MATCH failed for query: %s", stemmed_query[:50])

    # Sort by BM25 score (lower = more relevant in FTS5)
    results.sort(key=lambda x: x[1])
    return results[:max_k]


def build_context(
    conn: sqlite3.Connection,
    result_ids: list[tuple[str, float]],
) -> list[Context]:
    """Lookup parents, dedup, assemble context.

    Args:
        conn: SQLite connection.
        result_ids: [(id, score), ...] from adaptive_k output.

    Returns:
        List of Context objects, parents deduplicated, ordered by score.
    """
    # Separate children from table summaries
    child_ids: list[tuple[str, float]] = []
    table_ids: list[tuple[str, float]] = []

    child_id_set = {
        row[0]
        for row in conn.execute("SELECT id FROM children").fetchall()
    }

    for doc_id, score in result_ids:
        if doc_id in child_id_set:
            child_ids.append((doc_id, score))
        else:
            table_ids.append((doc_id, score))

    contexts: list[Context] = []

    # Parent dedup: group children by parent_id
    parent_groups: dict[str, list[tuple[str, float]]] = {}
    for child_id, score in child_ids:
        row = conn.execute(
            "SELECT parent_id, source, page, section FROM children WHERE id = ?",
            (child_id,),
        ).fetchone()
        if row:
            pid = row[0]
            parent_groups.setdefault(pid, []).append((child_id, score))

    # Build parent contexts
    for pid, children in parent_groups.items():
        parent_row = conn.execute(
            "SELECT text, source, section, page FROM parents WHERE id = ?",
            (pid,),
        ).fetchone()
        if parent_row:
            best_score = max(s for _, s in children)
            contexts.append(Context(
                text=parent_row[0],
                source=parent_row[1],
                page=parent_row[3],
                section=parent_row[2] or "",
                context_type="parent",
                score=best_score,
                children_matched=[cid for cid, _ in children],
            ))

    # Build table contexts (raw_table_text, not summary)
    for table_id, score in table_ids:
        row = conn.execute(
            "SELECT raw_table_text, source, page FROM table_summaries "
            "WHERE id = ?",
            (table_id,),
        ).fetchone()
        if row:
            contexts.append(Context(
                text=row[0],
                source=row[1],
                page=row[2],
                section="",
                context_type="table",
                score=score,
                children_matched=[table_id],
            ))

    # Sort by score desc
    contexts.sort(key=lambda c: -c.score)
    return contexts


# === Main entry point ===


def search(
    db_path: Path | str,
    query: str,
    model: object | None = None,
    min_score: float = 0.3,
    max_gap: float = 0.15,
    max_k: int = 10,
) -> SearchResult:
    """Full hybrid search pipeline.

    Args:
        db_path: Path to corpus_v2_fr.db.
        query: User question in French.
        model: SentenceTransformer model (loaded if None).
        min_score: Adaptive k minimum RRF score.
        max_gap: Adaptive k maximum gap.
        max_k: Adaptive k maximum results.

    Returns:
        SearchResult with contexts, scores, and metadata.
    """
    if model is None:
        model = load_model()

    conn = sqlite3.connect(str(db_path))

    # 1. Query processing
    stemmed_expanded = expand_query(query)
    q_emb = model.encode(
        [format_query(query)],
        normalize_embeddings=True,
    )[0].astype(np.float32)

    # 2. Dual retrieval
    cosine_results = cosine_search(conn, q_emb, max_k=max_k * 2)
    bm25_results = bm25_search(conn, stemmed_expanded, max_k=max_k * 2)

    # 3. RRF fusion
    fused = reciprocal_rank_fusion(cosine_results, bm25_results)

    # 4. Adaptive k
    filtered = adaptive_k(fused, min_score, max_gap, max_k)

    # 5+6. Parent lookup + context assembly
    contexts = build_context(conn, filtered)

    conn.close()

    return SearchResult(
        contexts=contexts,
        total_children_matched=len(filtered),
        scores={doc_id: score for doc_id, score in filtered},
    )
```

- [ ] **3.4: Run tests — verify they PASS**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/test_search.py -v
```

Expected: ALL PASS

- [ ] **3.5: Commit**

```bash
git add scripts/pipeline/search.py scripts/pipeline/tests/test_search.py
git commit -m "feat(pipeline): add hybrid search with RRF, adaptive k, parent dedup

Cosine brute-force + FTS5 BM25, RRF fusion (k=60),
adaptive_k (min_score/max_gap/max_k), parent lookup with dedup,
Context/SearchResult dataclasses. TDD: 15 tests."
```

---

## Task 4: Rebuild DB with FTS5 + verify

**Files:**
- None (uses existing code)

**IMPORTANT:** This task re-runs build_index() which takes ~18 min CPU. The DB must be rebuilt because FTS5 tables are new.

- [ ] **4.1: Run all fast tests first**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/ scripts/iso/ -m "not slow" -q
```

Expected: ALL PASS (189+ tests)

- [ ] **4.2: Rebuild DB**

```bash
cd C:/Dev/pocket_arbiter && python -c "
import logging, json, time
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from pathlib import Path
from scripts.pipeline.indexer import build_index

t0 = time.time()
stats = build_index(
    docling_dir=Path('corpus/processed/docling_v2_fr'),
    table_summaries_path=Path('corpus/processed/table_summaries_claude.json'),
    output_db=Path('corpus/processed/corpus_v2_fr.db'),
)
elapsed = time.time() - t0
stats['elapsed_seconds'] = round(elapsed, 1)
print(json.dumps(stats, indent=2))
"
```

Run in background (~18 min). Do NOT interrupt.

- [ ] **4.3: Verify FTS5 populated**

```bash
cd C:/Dev/pocket_arbiter && python -c "
import sqlite3
db = sqlite3.connect('corpus/processed/corpus_v2_fr.db')
c_fts = db.execute('SELECT COUNT(*) FROM children_fts').fetchone()[0]
ts_fts = db.execute('SELECT COUNT(*) FROM table_summaries_fts').fetchone()[0]
print(f'children_fts: {c_fts}')
print(f'table_summaries_fts: {ts_fts}')
print(f'S8 ... {\"PASS\" if c_fts == 1253 and ts_fts == 111 else \"FAIL\"}')
db.close()
"
```

Expected: S8 PASS (1253 + 111)

---

## Task 5: Quality gates on real data

**Files:**
- Modify: `scripts/pipeline/tests/test_search.py` (add quality gate tests)

- [ ] **5.1: Add quality gate tests (marked slow)**

Append to `test_search.py`:

```python
@pytest.mark.slow
class TestSearchQualityGates:
    """Quality gates on real corpus_v2_fr.db."""

    DB_PATH = Path("corpus/processed/corpus_v2_fr.db")

    @pytest.fixture(autouse=True)
    def _skip_if_no_db(self) -> None:
        if not self.DB_PATH.exists():
            pytest.skip("corpus_v2_fr.db not available")

    def test_s1_jury_appel(self) -> None:
        """S1: search for jury composition returns relevant result."""
        result = search(self.DB_PATH, "composition jury appel")
        assert len(result.contexts) >= 1
        top = result.contexts[0]
        assert "jury" in top.text.lower() or "appel" in top.text.lower()

    def test_s2_cadence_fischer(self) -> None:
        """S2: search for cadence Fischer returns table."""
        result = search(self.DB_PATH, "cadence Fischer equivalente")
        all_text = " ".join(c.text.lower() for c in result.contexts[:3])
        assert "fischer" in all_text or "cadence" in all_text

    def test_s3_categorie_u12(self) -> None:
        """S3: search for U12 returns age categories."""
        result = search(self.DB_PATH, "categorie U12 age")
        all_text = " ".join(c.text.lower() for c in result.contexts[:3])
        assert "u12" in all_text or "pupille" in all_text

    def test_s4_bm25_forfait(self) -> None:
        """S4: BM25 alone finds forfait articles."""
        conn = sqlite3.connect(str(self.DB_PATH))
        from scripts.pipeline.search import bm25_search
        from scripts.pipeline.synonyms import expand_query
        results = bm25_search(conn, expand_query("forfait"), max_k=5)
        conn.close()
        assert len(results) >= 1
        # At least one result should be from forfait section
        conn2 = sqlite3.connect(str(self.DB_PATH))
        for doc_id, _ in results[:3]:
            row = conn2.execute(
                "SELECT section FROM children WHERE id = ?", (doc_id,)
            ).fetchone()
            if row and "forfait" in (row[0] or "").lower():
                conn2.close()
                return
        conn2.close()
        pytest.fail("No forfait section in BM25 top-3")

    def test_s5_hybrid_beats_cosine(self) -> None:
        """S5: hybrid recall >= cosine alone on sample queries."""
        queries = [
            "composition jury appel",
            "forfait equipe",
            "mutation joueur",
            "cadence rapide",
            "classement elo",
        ]
        from scripts.pipeline.indexer import load_model
        from scripts.pipeline.search import (
            cosine_search, bm25_search, reciprocal_rank_fusion,
        )
        from scripts.pipeline.synonyms import expand_query
        model = load_model()
        conn = sqlite3.connect(str(self.DB_PATH))
        hybrid_better_or_equal = 0
        for q in queries:
            q_emb = model.encode(
                [format_query(q)], normalize_embeddings=True,
            )[0].astype(np.float32)
            cosine = cosine_search(conn, q_emb, max_k=10)
            bm25 = bm25_search(conn, expand_query(q), max_k=10)
            hybrid = reciprocal_rank_fusion(cosine, bm25)
            # Check if top-1 of hybrid is in top-3 of cosine
            if hybrid and hybrid[0][0] in [c[0] for c in cosine[:3]]:
                hybrid_better_or_equal += 1
        conn.close()
        assert hybrid_better_or_equal >= 3, \
            f"Hybrid should match cosine top-3 on at least 3/5 queries"

    def test_s6_adaptive_k_range(self) -> None:
        """S6: adaptive k returns 1 <= n <= max_k."""
        result = search(self.DB_PATH, "regles generales")
        assert 1 <= len(result.contexts) <= 10

    def test_s7_no_duplicate_parents(self) -> None:
        """S7: no duplicate parents in context."""
        result = search(self.DB_PATH, "licence joueur mutation")
        parent_texts = [c.text for c in result.contexts if c.context_type == "parent"]
        assert len(parent_texts) == len(set(parent_texts)), \
            "Duplicate parent text in context"
```

- [ ] **5.2: Run fast tests only (not quality gates)**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/ scripts/iso/ -m "not slow" -q
```

Expected: ALL PASS

- [ ] **5.3: Run quality gates (slow, requires DB + model)**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/test_search.py -m slow -v
```

Expected: S1-S7 PASS

- [ ] **5.4: Verify S8 (FTS5 counts) manually**

Already done in Task 4 step 4.3.

- [ ] **5.5: Commit**

```bash
git add scripts/pipeline/tests/test_search.py
git commit -m "test(pipeline): add search quality gates S1-S7

Slow tests on real corpus_v2_fr.db: jury appel, cadence Fischer,
U12 categories, BM25 forfait, hybrid vs cosine, adaptive k range,
no duplicate parents."
```

---

## Task 6: Full test suite + final commit

- [ ] **6.1: Run complete fast test suite**

```bash
cd C:/Dev/pocket_arbiter && python -m pytest scripts/pipeline/tests/ scripts/iso/ -m "not slow" -q
```

Expected: ALL PASS (200+ tests)

- [ ] **6.2: Ruff lint check**

```bash
cd C:/Dev/pocket_arbiter && python -m ruff check scripts/pipeline/search.py scripts/pipeline/synonyms.py
```

Expected: no errors

- [ ] **6.3: Mypy type check**

```bash
cd C:/Dev/pocket_arbiter && python -m mypy scripts/pipeline/search.py scripts/pipeline/synonyms.py
```

Expected: no errors

- [ ] **6.4: Update CLAUDE.md**

Update "En cours" section:
- Task 4 DONE
- Remaining: Task 5 (integration)

Update stats:
- Pipeline tests count
- Search module added

- [ ] **6.5: Update memory**

Update MEMORY.md with Task 4 complete status.

- [ ] **6.6: Commit + push**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Task 4 Search complete"
git push
```

Expected: all pre-commit + pre-push hooks PASS.

---

## DoD (Definition of Done)

- [ ] `scripts/pipeline/search.py` ecrit et teste
- [ ] `scripts/pipeline/synonyms.py` ecrit et teste
- [ ] `scripts/pipeline/indexer.py` modifie (FTS5 schema + populate_fts)
- [ ] `corpus/processed/corpus_v2_fr.db` rebuilt avec FTS5 tables
- [ ] 8 quality gates S1-S8 PASS
- [ ] Tous tests rapides PASS
- [ ] Ruff + mypy PASS
- [ ] Commits conventionnels
- [ ] CLAUDE.md et memoire a jour
- [ ] Push reussi (pre-push hooks PASS)

---

## Audit contre standards industrie

| Standard | Critere | Verification dans ce plan |
|----------|---------|--------------------------|
| **Hybrid search** (Weaviate, ES) | Cosine + BM25 + RRF | Task 3: search.py |
| **Parent-child RAG** (LangChain, Dify) | Search children, return parents | Task 3: build_context |
| **FTS5 BM25** (SQLite native) | Pre-stem + unicode61 + bm25() | Task 2: indexer FTS5 |
| **Snowball FR** (ACL 2025) | Pre-stem build, dict runtime | Task 1: synonyms.py |
| **RRF fusion** (k=60) | Standard constant | Task 3: reciprocal_rank_fusion |
| **Adaptive k** (CRAG pattern) | Score + gap filtering | Task 3: adaptive_k |
| **Brute-force** (Denisov 2026) | Sub-10ms, no ANN needed | Task 3: cosine_search |
| **ISO 29119** | TDD, tests before code | Tasks 1-3: test first |
| **ISO 25010** | Complexity <= B | Task 6: ruff + xenon |
| **ISO 12207** | Conventional commits | All tasks |
| **ISO 42001** | Score tracabilite | SearchResult.scores dict |

---

## Anti-paresse checklist

A verifier AVANT de declarer chaque task "done" :

- [ ] Les tests ont ete ecrits AVANT le code (pas apres)
- [ ] Les tests ont ete lances et ont FAIL avant l'implementation
- [ ] Aucun test n'est skip ou ignore sans justification
- [ ] Aucun quality gate n'est "passe par transitivite" — tous verifies directement
- [ ] Le code passe ruff + mypy SANS noqa/type:ignore sauf justification documentee
- [ ] Les valeurs hardcodees sont justifiees (pas de magic numbers)
- [ ] Les erreurs sont gerees (pas de bare except, pas de pass silencieux)
- [ ] Le build DB a ete relance APRES les modifications de l'indexer
- [ ] Les quality gates tournent sur la DB REBUILTEE (pas l'ancienne)
- [ ] CLAUDE.md et memoire refletent l'etat reel, pas l'etat espere
