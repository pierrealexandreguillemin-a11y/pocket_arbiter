# scripts/pipeline/tests/test_search.py
"""Unit tests for search module (fast, no real DB needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import create_db, insert_children, insert_parents
from scripts.pipeline.search import (
    adaptive_k,
    bm25_search,
    clear_embedding_cache,
    cosine_search,
    reciprocal_rank_fusion,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear embedding cache between every test."""
    clear_embedding_cache()


class TestCosineSearchCache:
    """Test embedding cache in cosine_search."""

    def test_cache_returns_same_results(self, tmp_path: Path) -> None:
        """Two calls with same db_path return identical results."""
        db_path = tmp_path / "cache.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "parent",
                    "source": "d.pdf",
                    "section": "S1",
                    "tokens": 5,
                    "page": 1,
                },
            ],
        )
        emb = np.random.randn(2, 768).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "Child 1",
                    "parent_id": "p1",
                    "source": "d.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 5,
                },
                {
                    "id": "c2",
                    "text": "Child 2",
                    "parent_id": "p1",
                    "source": "d.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 5,
                },
            ],
            emb,
        )
        q = np.random.randn(768).astype(np.float32)
        q = q / np.linalg.norm(q)

        r1 = cosine_search(conn, q, max_k=5, db_path=str(db_path))
        r2 = cosine_search(conn, q, max_k=5, db_path=str(db_path))
        assert r1 == r2
        conn.close()

    def test_clear_cache_works(self, tmp_path: Path) -> None:
        """After clear, cache is empty."""
        from scripts.pipeline.search import _embedding_cache

        db_path = tmp_path / "cache2.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "p",
                    "source": "d.pdf",
                    "section": "S1",
                    "tokens": 5,
                    "page": 1,
                },
            ],
        )
        emb = np.random.randn(1, 768).astype(np.float32)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "C1",
                    "parent_id": "p1",
                    "source": "d.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 5,
                },
            ],
            emb,
        )
        q = np.random.randn(768).astype(np.float32)
        cosine_search(conn, q, db_path=str(db_path))
        assert len(_embedding_cache) > 0
        clear_embedding_cache()
        assert len(_embedding_cache) == 0
        conn.close()


class TestReciprocalRankFusion:
    """Test RRF score fusion."""

    def test_single_list(self) -> None:
        cosine = [("a", 0.9), ("b", 0.8)]
        bm25: list[tuple[str, float]] = []
        result = reciprocal_rank_fusion(cosine, bm25)
        assert result[0][0] == "a"
        assert result[1][0] == "b"

    def test_overlapping_docs_score_higher(self) -> None:
        cosine = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        bm25 = [("b", -3.0), ("a", -4.0), ("d", -5.0)]
        result = reciprocal_rank_fusion(cosine, bm25)
        ids = [r[0] for r in result]
        assert ids[0] in ("a", "b")
        assert ids[1] in ("a", "b")

    def test_empty_lists(self) -> None:
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_k_parameter_affects_scores(self) -> None:
        cosine = [("a", 0.9)]
        bm25 = [("a", -3.0)]
        r1 = reciprocal_rank_fusion(cosine, bm25, k=60)
        r2 = reciprocal_rank_fusion(cosine, bm25, k=1)
        assert r1[0][1] < r2[0][1]


class TestAdaptiveK:
    """Test adaptive k filtering (EMNLP 2025 largest-gap)."""

    def test_max_k_limits(self) -> None:
        results = [(f"d{i}", 1.0 - i * 0.01) for i in range(20)]
        filtered = adaptive_k(results, min_score=0.0, max_k=5)
        assert len(filtered) <= 5

    def test_min_score_filters(self) -> None:
        """min_score removes low-score results before gap detection."""
        # c(0.1) filtered by min_score, then a(0.5) and b(0.4) remain
        # Single gap 0.1 → largest gap cuts after a → 1 result
        results = [("a", 0.5), ("b", 0.4), ("c", 0.1)]
        filtered = adaptive_k(results, min_score=0.3, max_k=10)
        assert all(s >= 0.3 for _, s in filtered)
        assert "c" not in [did for did, _ in filtered]

    def test_min_score_removes_all(self) -> None:
        results = [("a", 0.1), ("b", 0.05)]
        filtered = adaptive_k(results, min_score=0.3, max_k=10)
        assert len(filtered) == 0

    def test_largest_gap_cuts(self) -> None:
        """Largest gap between b(0.85) and c(0.5) = 0.35 → cut after b.
        But min_k=3 keeps at least 3, so c is kept."""
        results = [("a", 0.9), ("b", 0.85), ("c", 0.5), ("d", 0.45)]
        filtered = adaptive_k(results, min_score=0.0, max_k=10)
        assert len(filtered) == 3  # min_k=3 floor

    def test_largest_gap_cuts_with_min_k_1(self) -> None:
        """With min_k=1, largest gap cuts after b."""
        results = [("a", 0.9), ("b", 0.85), ("c", 0.5), ("d", 0.45)]
        filtered = adaptive_k(results, min_score=0.0, min_k=1, max_k=10)
        assert len(filtered) == 2
        assert filtered[-1][0] == "b"

    def test_largest_gap_at_end(self) -> None:
        """Largest gap at end keeps almost everything."""
        results = [("a", 0.9), ("b", 0.85), ("c", 0.8), ("d", 0.1)]
        filtered = adaptive_k(results, min_score=0.0, max_k=10)
        assert len(filtered) == 3
        assert filtered[-1][0] == "c"

    def test_uniform_scores_keeps_all(self) -> None:
        """Equal gaps → largest gap at idx 0 → keeps 1, but all gaps equal."""
        results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        filtered = adaptive_k(results, min_score=0.0, max_k=10)
        # All gaps equal (0.1), largest gap idx=0, cut at 1
        assert len(filtered) >= 1

    def test_buffer_extends(self) -> None:
        """Buffer adds extra results after gap."""
        results = [("a", 0.9), ("b", 0.85), ("c", 0.5), ("d", 0.45)]
        filtered = adaptive_k(results, min_score=0.0, max_k=10, buffer=1)
        assert len(filtered) == 3  # 2 before gap + 1 buffer

    def test_empty_input(self) -> None:
        assert adaptive_k([], min_score=0.3, max_k=10) == []

    def test_single_result_passes(self) -> None:
        results = [("a", 0.5)]
        filtered = adaptive_k(results, min_score=0.3, max_k=10)
        assert len(filtered) == 1


class TestBm25FtsOrLogic:
    """Test that BM25 uses OR between terms, not AND."""

    def test_or_logic_matches_any_term(self, tmp_path: Path) -> None:
        """FTS5 MATCH with OR returns docs matching ANY term."""
        db_path = tmp_path / "fts_or.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "parent",
                    "source": "d.pdf",
                    "section": "S1",
                    "tokens": 5,
                    "page": 1,
                },
            ],
        )
        emb = np.random.randn(3, 768).astype(np.float32)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "forfait absence",
                    "parent_id": "p1",
                    "source": "d.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 5,
                },
                {
                    "id": "c2",
                    "text": "cadence rapide",
                    "parent_id": "p1",
                    "source": "d.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 5,
                },
                {
                    "id": "c3",
                    "text": "nothing relevant",
                    "parent_id": "p1",
                    "source": "d.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 5,
                },
            ],
            emb,
        )
        from scripts.pipeline.indexer import populate_fts

        populate_fts(conn)

        # bm25_search expects stemmed input (FTS5 index is stemmed)
        results = bm25_search(conn, "forf cadenc", max_k=10)
        ids = [r[0] for r in results]
        assert "c1" in ids, "c1 should match 'forfait'"
        assert "c2" in ids, "c2 should match 'cadence'"
        assert "c3" not in ids, "c3 matches neither term"
        conn.close()

    def test_and_would_miss_partial_matches(self, tmp_path: Path) -> None:
        """Verify OR is needed: a doc with only one term still matches."""
        db_path = tmp_path / "fts_or2.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "parent",
                    "source": "d.pdf",
                    "section": "S1",
                    "tokens": 5,
                    "page": 1,
                },
            ],
        )
        emb = np.random.randn(1, 768).astype(np.float32)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "only forfait here",
                    "parent_id": "p1",
                    "source": "d.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 5,
                },
            ],
            emb,
        )
        from scripts.pipeline.indexer import populate_fts

        populate_fts(conn)

        results = bm25_search(conn, "forf cadenc", max_k=10)
        assert len(results) == 1, "OR logic should match c1 with partial term"
        assert results[0][0] == "c1"
        conn.close()
