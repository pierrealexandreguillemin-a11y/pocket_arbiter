"""Tests for search (cosine retrieval + parent lookup + adaptive k)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import (
    create_db, insert_children, insert_parents, insert_table_summaries,
)
from scripts.pipeline.search import search, load_index


@pytest.fixture
def populated_db(tmp_path):
    """Create a small DB with known embeddings for testing."""
    db_path = tmp_path / "test.db"
    create_db(db_path)

    parents = [
        {"id": "p0", "text": "Parent about licences and qualifications.",
         "source": "test.pdf", "section": "Licences", "tokens": 20, "page": 1},
        {"id": "p1", "text": "Parent about forfaits and sanctions.",
         "source": "test.pdf", "section": "Forfaits", "tokens": 20, "page": 3},
    ]
    insert_parents(db_path, parents)

    # Distinct embeddings: licence points dim 0, forfait points dim 1
    emb_licence = np.zeros(768, dtype=np.float32)
    emb_licence[0] = 1.0

    emb_forfait = np.zeros(768, dtype=np.float32)
    emb_forfait[1] = 1.0

    children = [
        {"id": "c0", "text": "Licence A obligatoire", "parent_id": "p0",
         "source": "test.pdf", "article_num": "1", "section": "1. Licences",
         "tokens": 10, "page": 1},
        {"id": "c1", "text": "Forfait sportif definition", "parent_id": "p1",
         "source": "test.pdf", "article_num": "3", "section": "3. Forfaits",
         "tokens": 10, "page": 3},
    ]
    embeddings = {"c0": emb_licence, "c1": emb_forfait}
    insert_children(db_path, children, embeddings)

    # Table summary
    emb_table = np.zeros(768, dtype=np.float32)
    emb_table[2] = 1.0
    summaries = [
        {"id": "t0", "summary_text": "Categories d'age", "raw_table_text": "| U8 | <8 |",
         "source": "test.pdf", "page": 5, "tokens": 10},
    ]
    insert_table_summaries(db_path, summaries, {"t0": emb_table})

    return db_path


class TestSearch:
    def test_returns_results(self, populated_db):
        index = load_index(populated_db)
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 1.0
        results = search(index, query_emb, k=2)
        assert len(results) > 0

    def test_nearest_is_correct(self, populated_db):
        index = load_index(populated_db)
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 1.0
        results = search(index, query_emb, k=1)
        assert results[0]["child_id"] == "c0"

    def test_returns_parent_text(self, populated_db):
        index = load_index(populated_db)
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 1.0
        results = search(index, query_emb, k=1)
        assert "licences" in results[0]["parent_text"].lower()

    def test_deduplicates_parents(self, populated_db):
        index = load_index(populated_db)
        query_emb = np.ones(768, dtype=np.float32)
        results = search(index, query_emb, k=3)
        parent_ids = [r["parent_id"] for r in results if r["parent_id"]]
        assert len(parent_ids) == len(set(parent_ids))

    def test_includes_table_summaries(self, populated_db):
        index = load_index(populated_db)
        # Query close to table embedding (dim 2)
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[2] = 1.0
        results = search(index, query_emb, k=3)
        types = [r["result_type"] for r in results]
        assert "table_summary" in types

    def test_table_summary_returns_raw_text(self, populated_db):
        index = load_index(populated_db)
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[2] = 1.0
        results = search(index, query_emb, k=1)
        assert "| U8 |" in results[0]["parent_text"]


class TestAdaptiveK:
    def test_score_threshold_filters(self, populated_db):
        index = load_index(populated_db)
        # Query close to licence only — forfait and table should be filtered
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 1.0
        results = search(index, query_emb, k=10, score_threshold=0.5)
        # Only c0 should have score > 0.5
        assert len(results) == 1
        assert results[0]["child_id"] == "c0"

    def test_gap_threshold_filters(self, populated_db):
        index = load_index(populated_db)
        # Query that matches c0 strongly and c1 weakly
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 0.99
        query_emb[1] = 0.01
        results = search(index, query_emb, k=10, gap_threshold=0.5)
        # Big gap between c0 (score ~1.0) and c1 (score ~0.01) → c1 filtered
        assert len(results) == 1

    def test_no_threshold_returns_all(self, populated_db):
        index = load_index(populated_db)
        query_emb = np.ones(768, dtype=np.float32)
        results = search(index, query_emb, k=10)
        assert len(results) == 3  # 2 children + 1 table summary
