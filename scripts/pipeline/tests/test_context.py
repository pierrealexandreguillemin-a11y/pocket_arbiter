# scripts/pipeline/tests/test_context.py
"""Unit tests for context assembly and search() entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from scripts.pipeline.context import SearchResult, build_context
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
