# scripts/pipeline/tests/test_context.py
"""Unit tests for context assembly and search() entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from scripts.pipeline.context import (
    SearchResult,
    _resolve_page,
    build_context,
)
from scripts.pipeline.indexer import (
    create_db,
    insert_children,
    insert_parents,
    insert_table_summaries,
    populate_fts,
)
from scripts.pipeline.search import search


class TestBuildContext:
    """Test parent lookup and deduplication."""

    def test_deduplicates_parents(self, tmp_path: Path) -> None:
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
        result_ids = [("c1", 0.9), ("c2", 0.8)]
        contexts = build_context(conn, result_ids)
        parent_contexts = [c for c in contexts if c.context_type == "parent"]
        assert len(parent_contexts) == 1
        assert parent_contexts[0].text == "Parent one text"
        conn.close()

    def test_table_summary_returns_raw(self, tmp_path: Path) -> None:
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
        assert contexts[0].text == "High"
        conn.close()

    def test_empty_parent_falls_back_to_child(self, tmp_path: Path) -> None:
        """Children with empty-text parents are returned as child contexts."""
        db_path = tmp_path / "empty.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p_empty",
                    "text": "",
                    "source": "d.pdf",
                    "section": "",
                    "tokens": 0,
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
                    "parent_id": "p_empty",
                    "source": "d.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "",
                    "tokens": 5,
                },
            ],
            emb,
        )
        contexts = build_context(conn, [("c1", 0.9)])
        assert len(contexts) == 1
        assert contexts[0].context_type == "child"
        assert contexts[0].text == "C1"
        conn.close()

    def test_parent_page_none_resolved_from_child(self, tmp_path: Path) -> None:
        """When parent.page is None, page is resolved from best child."""
        db_path = tmp_path / "page_fix.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "Parent text",
                    "source": "d.pdf",
                    "section": "S",
                    "tokens": 10,
                    "page": None,  # No page on parent
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
                    "page": 42,
                    "article_num": None,
                    "section": "S",
                    "tokens": 5,
                },
                {
                    "id": "c2",
                    "text": "C2",
                    "parent_id": "p1",
                    "source": "d.pdf",
                    "page": 43,
                    "article_num": None,
                    "section": "S",
                    "tokens": 5,
                },
            ],
            emb,
        )
        # c2 has higher score -> page should be 43
        contexts = build_context(conn, [("c2", 0.9), ("c1", 0.5)])
        assert len(contexts) == 1
        assert contexts[0].page == 43  # resolved from best child
        conn.close()

    def test_resolve_page_prefers_parent(self, tmp_path: Path) -> None:
        """_resolve_page uses parent page when available."""
        db_path = tmp_path / "resolve.db"
        conn = create_db(db_path)
        # No children needed for this unit test
        assert _resolve_page(conn, 10, [("c1", 0.9)]) == 10
        conn.close()


class TestInjectNeighborTables:
    """Test neighbor table injection into context."""

    def _setup_db_with_table(self, tmp_path: Path) -> tuple:
        """Create DB with a child on p.5 and a table on p.6 (adjacent)."""
        db_path = tmp_path / "inject.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "Parent text about forfaits.",
                    "source": "doc.pdf",
                    "section": "S1",
                    "tokens": 10,
                    "page": 5,
                },
            ],
        )
        emb_c = np.random.randn(1, 768).astype(np.float32)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "Child about forfaits.",
                    "parent_id": "p1",
                    "source": "doc.pdf",
                    "page": 5,
                    "article_num": None,
                    "section": "S1",
                    "tokens": 5,
                },
            ],
            emb_c,
        )
        emb_t = np.random.randn(1, 768).astype(np.float32)
        insert_table_summaries(
            conn,
            [
                {
                    "id": "t1",
                    "summary_text": "Table of cadences",
                    "raw_table_text": "| Cadence | Temps |\n|---|---|\n| A | 90+30 |",
                    "source": "doc.pdf",
                    "page": 6,
                    "tokens": 10,
                },
            ],
            emb_t,
        )
        return db_path, conn

    def test_injects_adjacent_table(self, tmp_path: Path) -> None:
        """Table on page+1 is injected when prose child is on page."""
        _, conn = self._setup_db_with_table(tmp_path)
        contexts = build_context(conn, [("c1", 0.9)])
        types = [c.context_type for c in contexts]
        assert "table_injected" in types
        injected = [c for c in contexts if c.context_type == "table_injected"]
        assert len(injected) == 1
        assert injected[0].page == 6
        assert "t1" in injected[0].children_matched
        conn.close()

    def test_dedup_no_double_inject(self, tmp_path: Path) -> None:
        """Already-retrieved table is NOT injected again."""
        _, conn = self._setup_db_with_table(tmp_path)
        # t1 already in results (retrieved directly)
        contexts = build_context(conn, [("c1", 0.9), ("t1", 0.7)])
        injected = [c for c in contexts if c.context_type == "table_injected"]
        assert len(injected) == 0
        conn.close()

    def test_no_inject_distant_table(self, tmp_path: Path) -> None:
        """Table on a distant page is NOT injected."""
        db_path = tmp_path / "distant.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "Parent.",
                    "source": "doc.pdf",
                    "section": "S",
                    "tokens": 5,
                    "page": 5,
                },
            ],
        )
        emb_c = np.random.randn(1, 768).astype(np.float32)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "Child.",
                    "parent_id": "p1",
                    "source": "doc.pdf",
                    "page": 5,
                    "article_num": None,
                    "section": "S",
                    "tokens": 3,
                },
            ],
            emb_c,
        )
        emb_t = np.random.randn(1, 768).astype(np.float32)
        insert_table_summaries(
            conn,
            [
                {
                    "id": "t_far",
                    "summary_text": "Far table",
                    "raw_table_text": "| X |\n|---|\n| Y |",
                    "source": "doc.pdf",
                    "page": 50,
                    "tokens": 5,
                },
            ],
            emb_t,
        )
        contexts = build_context(conn, [("c1", 0.9)])
        injected = [c for c in contexts if c.context_type == "table_injected"]
        assert len(injected) == 0
        conn.close()

    def test_no_inject_without_prose(self, tmp_path: Path) -> None:
        """No injection when only table contexts (no prose child)."""
        db_path = tmp_path / "no_prose.db"
        conn = create_db(db_path)
        emb_t = np.random.randn(1, 768).astype(np.float32)
        insert_table_summaries(
            conn,
            [
                {
                    "id": "t1",
                    "summary_text": "Summary",
                    "raw_table_text": "| A |\n|---|\n| B |",
                    "source": "doc.pdf",
                    "page": 10,
                    "tokens": 5,
                },
            ],
            emb_t,
        )
        contexts = build_context(conn, [("t1", 0.8)])
        injected = [c for c in contexts if c.context_type == "table_injected"]
        assert len(injected) == 0
        conn.close()

    def test_injected_score_lower_than_retrieved(self, tmp_path: Path) -> None:
        """Injected tables have lower score than any retrieved context."""
        _, conn = self._setup_db_with_table(tmp_path)
        contexts = build_context(conn, [("c1", 0.9)])
        retrieved = [c for c in contexts if c.context_type != "table_injected"]
        injected = [c for c in contexts if c.context_type == "table_injected"]
        if injected and retrieved:
            min_retrieved = min(c.score for c in retrieved)
            max_injected = max(c.score for c in injected)
            assert max_injected < min_retrieved
        conn.close()


class TestSearchEntryPoint:
    """Unit test for search() without real corpus (mock model)."""

    def test_returns_search_result(self, tmp_path: Path) -> None:
        """search() returns SearchResult with contexts from a synthetic DB."""
        db_path = tmp_path / "search_unit.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "Le forfait est declare apres 60 minutes.",
                    "source": "test.pdf",
                    "section": "Forfaits",
                    "tokens": 20,
                    "page": 5,
                },
            ],
        )
        # Create a normalized embedding that will produce positive cosine
        emb = np.ones((1, 768), dtype=np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "Le forfait est declare apres 60 minutes.",
                    "parent_id": "p1",
                    "source": "test.pdf",
                    "page": 5,
                    "article_num": None,
                    "section": "Forfaits",
                    "tokens": 20,
                },
            ],
            emb,
        )
        populate_fts(conn)
        conn.close()

        # Mock the embedding model to return the same normalized vector
        mock_model = MagicMock()
        query_emb = np.ones((1, 768), dtype=np.float32)
        query_emb = query_emb / np.linalg.norm(query_emb)
        mock_model.encode.return_value = query_emb.reshape(1, 768)

        result = search(db_path, "forfait", model=mock_model)
        assert isinstance(result, SearchResult)
        assert len(result.contexts) >= 1
        assert result.contexts[0].source == "test.pdf"
        assert result.total_children_matched >= 1

    def test_empty_query_does_not_crash(self, tmp_path: Path) -> None:
        """search() with empty-ish query doesn't raise."""
        db_path = tmp_path / "search_empty.db"
        conn = create_db(db_path)
        insert_parents(
            conn,
            [
                {
                    "id": "p1",
                    "text": "Texte parent.",
                    "source": "t.pdf",
                    "section": "S",
                    "tokens": 5,
                    "page": 1,
                },
            ],
        )
        emb = np.random.randn(1, 768).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        insert_children(
            conn,
            [
                {
                    "id": "c1",
                    "text": "Child text.",
                    "parent_id": "p1",
                    "source": "t.pdf",
                    "page": 1,
                    "article_num": None,
                    "section": "S",
                    "tokens": 5,
                },
            ],
            emb,
        )
        populate_fts(conn)
        conn.close()

        mock_model = MagicMock()
        q_emb = np.random.randn(1, 768).astype(np.float32)
        q_emb = q_emb / np.linalg.norm(q_emb)
        mock_model.encode.return_value = q_emb.reshape(1, 768)

        result = search(db_path, "x", model=mock_model)
        assert isinstance(result, SearchResult)
