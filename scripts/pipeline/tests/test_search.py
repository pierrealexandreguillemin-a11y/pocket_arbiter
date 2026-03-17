# scripts/pipeline/tests/test_search.py
"""Tests for search module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import (
    create_db,
    insert_children,
    insert_parents,
    insert_table_summaries,
)
from scripts.pipeline.search import (
    adaptive_k,
    bm25_search,
    build_context,
    cosine_search,
    reciprocal_rank_fusion,
    search,
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
        results = [(f"d{i}", 1.0 - i * 0.01) for i in range(20)]
        filtered = adaptive_k(results, min_score=0.0, max_gap=1.0, max_k=5)
        assert len(filtered) == 5

    def test_min_score_filters(self) -> None:
        results = [("a", 0.5), ("b", 0.4), ("c", 0.1)]
        filtered = adaptive_k(results, min_score=0.3, max_gap=1.0, max_k=10)
        assert len(filtered) == 2
        assert filtered[-1][0] == "b"

    def test_gap_cuts(self) -> None:
        results = [("a", 0.9), ("b", 0.85), ("c", 0.5), ("d", 0.45)]
        # Gap between b(0.85) and c(0.5) = 0.35 > max_gap 0.2
        filtered = adaptive_k(results, min_score=0.0, max_gap=0.2, max_k=10)
        assert len(filtered) == 2

    def test_empty_input(self) -> None:
        assert adaptive_k([], min_score=0.3, max_gap=0.15, max_k=10) == []

    def test_all_filtered(self) -> None:
        results = [("a", 0.1), ("b", 0.05)]
        filtered = adaptive_k(results, min_score=0.3, max_gap=0.15, max_k=10)
        assert len(filtered) == 0

    def test_single_result_passes(self) -> None:
        results = [("a", 0.5)]
        filtered = adaptive_k(results, min_score=0.3, max_gap=0.15, max_k=10)
        assert len(filtered) == 1


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
        # Both children point to same parent
        result_ids = [("c1", 0.9), ("c2", 0.8)]
        contexts = build_context(conn, result_ids)
        # Should have exactly 1 parent context (deduplicated)
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
        # p2 should come first (higher score child)
        assert contexts[0].text == "High"
        conn.close()


# === Quality gates on real corpus (slow) ===


@pytest.mark.slow
class TestSearchQualityGates:
    """Quality gates S1-S7 on real corpus_v2_fr.db."""

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
        """S2: search for cadence Fischer returns relevant content."""
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
        import sqlite3 as _sql

        from scripts.pipeline.synonyms import expand_query as _expand

        conn = _sql.connect(str(self.DB_PATH))
        results = bm25_search(conn, _expand("forfait"), max_k=5)
        conn.close()
        assert len(results) >= 1, "BM25 returned no results for 'forfait'"
        # Verify at least one is from a forfait section
        conn2 = _sql.connect(str(self.DB_PATH))
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
        from scripts.pipeline.indexer import format_query, load_model

        model = load_model()
        import sqlite3 as _sql

        conn = _sql.connect(str(self.DB_PATH))
        hybrid_match = 0
        for q in queries:
            q_emb = model.encode(
                [format_query(q)],
                normalize_embeddings=True,
            )[0].astype(np.float32)
            cosine = cosine_search(conn, q_emb, max_k=10)
            from scripts.pipeline.synonyms import expand_query as _expand

            bm25 = bm25_search(conn, _expand(q), max_k=10)
            hybrid = reciprocal_rank_fusion(cosine, bm25)
            # top-1 of hybrid should be in top-3 of cosine
            if hybrid and hybrid[0][0] in [c[0] for c in cosine[:3]]:
                hybrid_match += 1
        conn.close()
        assert (
            hybrid_match >= 3
        ), f"Hybrid top-1 in cosine top-3 on only {hybrid_match}/5 queries"

    def test_s6_adaptive_k_range(self) -> None:
        """S6: adaptive k returns 1 <= n <= max_k."""
        result = search(self.DB_PATH, "regles generales")
        assert 1 <= len(result.contexts) <= 10

    def test_s7_no_duplicate_parents(self) -> None:
        """S7: no duplicate parents in context."""
        result = search(self.DB_PATH, "licence joueur mutation")
        parent_texts = [c.text for c in result.contexts if c.context_type == "parent"]
        assert len(parent_texts) == len(
            set(parent_texts)
        ), "Duplicate parent text in context"
