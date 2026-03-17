# scripts/pipeline/tests/test_search.py
"""Tests for search module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.pipeline.indexer import (
    create_db,
    insert_children,
    insert_parents,
    insert_table_summaries,
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
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "Parent one text",
                    "source": "doc.pdf",
                    "section": "S1",
                    "tokens": 50,
                    "page": 1,
                },
            ],
        )
        emb = np.random.randn(2, 768).astype(np.float32)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "Child 1",
                    "parent_id": "p1",
                    "source": "doc.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 10,
                },
                {
                    "id": "c2",
                    "text": "Child 2",
                    "parent_id": "p1",
                    "source": "doc.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 10,
                },
            ],
            emb,
        )
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
        insert_table_summaries(
            conn,
            [
                {
                    "id": "t1",
                    "summary_text": "Summary",
                    "raw_table_text": "| col1 | col2 |",
                    "source": "doc.pdf",
                    "page": 3,
                    "tokens": 10,
                }
            ],
            emb,
        )
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
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "Low",
                    "source": "d.pdf",
                    "section": "S1",
                    "tokens": 10,
                    "page": 1,
                },
                {
                    "id": "p2",
                    "text": "High",
                    "source": "d.pdf",
                    "section": "S2",
                    "tokens": 10,
                    "page": 2,
                },
            ],
        )
        emb = np.random.randn(2, 768).astype(np.float32)
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
                {
                    "id": "c2",
                    "text": "C2",
                    "parent_id": "p2",
                    "source": "d.pdf",
                    "page": 2,
                    "article_num": None,
                    "section": "S2",
                    "tokens": 5,
                },
            ],
            emb,
        )
        result_ids = [("c2", 0.9), ("c1", 0.3)]
        contexts = build_context(conn, result_ids)
        # p2 should come first (higher score child)
        assert contexts[0].text == "High"
        conn.close()
