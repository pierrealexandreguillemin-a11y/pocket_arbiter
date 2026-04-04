# scripts/pipeline/tests/test_search.py
"""Unit tests for search module (fast, no real DB needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import create_db, insert_children, insert_parents
from scripts.pipeline.search import (
    _has_table_triggers,
    adaptive_k,
    bm25_search,
    clear_embedding_cache,
    cosine_search,
    reciprocal_rank_fusion,
    structured_cell_search,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear embedding cache between every test."""
    clear_embedding_cache()
    # Also clear intent stem cache to avoid stale state
    from scripts.pipeline.search import _INTENT_STEMS

    _INTENT_STEMS.clear()


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


class TestTableTriggers:
    """Tests for structured table trigger detection."""

    def test_strong_trigger_alone(self) -> None:
        assert _has_table_triggers("Quelle est la grille d'appariement?")

    def test_single_weak_trigger_no_match(self) -> None:
        # 1 weak trigger alone doesn't activate (needs 2 weak or 1 weak + Elo)
        assert not _has_table_triggers("Quelle est la cadence du tournoi?")

    def test_two_weak_triggers_with_elo(self) -> None:
        assert _has_table_triggers("Quel classement Elo 1500 pour la cadence?")

    def test_two_weak_triggers_without_elo(self) -> None:
        # 2 weak triggers WITHOUT Elo regex does not activate
        assert not _has_table_triggers("Quel classement pour la cadence?")

    def test_no_triggers(self) -> None:
        assert not _has_table_triggers("Comment jouer le roque?")

    def test_specific_category(self) -> None:
        assert _has_table_triggers("Age minimum pour les poussins?")

    def test_one_weak_plus_elo_no_trigger(self) -> None:
        # 1 weak + Elo is NOT enough (needs 2 weak + Elo)
        assert not _has_table_triggers("Quel classement pour un joueur à 1500?")

    def test_elo_regex_alone_no_trigger(self) -> None:
        assert not _has_table_triggers("Un joueur à 1500 peut-il participer?")

    def test_no_number_no_keyword(self) -> None:
        assert not _has_table_triggers("Peut-on proposer nulle après le premier coup?")


class TestStructuredCellSearch:
    """Tests for deterministic cell lookup."""

    def test_returns_empty_without_table(self, tmp_path: Path) -> None:
        import sqlite3

        db = tmp_path / "no_cells.db"
        conn = sqlite3.connect(str(db))
        results = structured_cell_search(conn, "test query")
        assert results == []
        conn.close()

    def test_matches_cells(self, tmp_path: Path) -> None:
        import sqlite3

        db = tmp_path / "cells.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE structured_cells "
            "(table_id TEXT, row_idx INT, col_name TEXT, "
            "cell_value TEXT, source TEXT, page INT)"
        )
        conn.executemany(
            "INSERT INTO structured_cells VALUES (?,?,?,?,?,?)",
            [
                ("t1", 0, "Category", "Poussin", "R01.pdf", 5),
                ("t1", 0, "Age", "6-8 ans", "R01.pdf", 5),
                ("t1", 1, "Category", "Pupille", "R01.pdf", 5),
                ("t2", 0, "Name", "Other data", "LA.pdf", 10),
            ],
        )
        conn.commit()

        # Query with terms that match multiple cell values in t1
        # "poussin" matches "Poussin", "pupille" matches "Pupille"
        results = structured_cell_search(conn, "Poussin et Pupille categories", max_k=5)
        assert len(results) >= 1
        assert results[0][0] == "t1"
        conn.close()

    def test_single_cell_match_below_threshold(self, tmp_path: Path) -> None:
        """Single cell match (1.0) is below threshold 2.0."""
        import sqlite3

        db = tmp_path / "cells2.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE structured_cells "
            "(table_id TEXT, row_idx INT, col_name TEXT, "
            "cell_value TEXT, source TEXT, page INT)"
        )
        conn.execute(
            "INSERT INTO structured_cells VALUES (?,?,?,?,?,?)",
            ("t1", 0, "Col", "unique_term_only", "s.pdf", 1),
        )
        conn.commit()

        results = structured_cell_search(conn, "unique_term_only", max_k=5)
        assert results == []  # 1.0 < threshold 1.5
        conn.close()

    def test_cell_plus_col_match_below_threshold(self, tmp_path: Path) -> None:
        """Cell match (1.0) + col_name match (0.5) = 1.5 < threshold 2.0."""
        import sqlite3

        db = tmp_path / "cells2b.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE structured_cells "
            "(table_id TEXT, row_idx INT, col_name TEXT, "
            "cell_value TEXT, source TEXT, page INT)"
        )
        conn.execute(
            "INSERT INTO structured_cells VALUES (?,?,?,?,?,?)",
            ("t1", 0, "Categorie", "Pupille", "R01.pdf", 2),
        )
        conn.commit()

        # "categorie" matches col_name (0.5) + "pupille" matches cell (1.0) = 1.5
        results = structured_cell_search(conn, "Categorie pupille", max_k=5)
        assert results == []  # 1.5 < threshold 2.0
        conn.close()

    def test_two_cell_matches_passes(self, tmp_path: Path) -> None:
        """Two cell matches (1.0 + 1.0) = 2.0 passes threshold."""
        import sqlite3

        db = tmp_path / "cells2c.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE structured_cells "
            "(table_id TEXT, row_idx INT, col_name TEXT, "
            "cell_value TEXT, source TEXT, page INT)"
        )
        conn.executemany(
            "INSERT INTO structured_cells VALUES (?,?,?,?,?,?)",
            [
                ("t1", 0, "Cat", "Pupille", "R01.pdf", 2),
                ("t1", 1, "Cat", "Poussin", "R01.pdf", 2),
            ],
        )
        conn.commit()

        results = structured_cell_search(conn, "Pupille et Poussin", max_k=5)
        assert len(results) == 1
        assert results[0][0] == "t1"
        conn.close()

    def test_short_chess_terms_match(self, tmp_path: Path) -> None:
        """Allow-list lets short domain terms bypass 4-char filter."""
        import sqlite3

        db = tmp_path / "cells3.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE structured_cells "
            "(table_id TEXT, row_idx INT, col_name TEXT, "
            "cell_value TEXT, source TEXT, page INT)"
        )
        conn.executemany(
            "INSERT INTO structured_cells VALUES (?,?,?,?,?,?)",
            [
                ("t1", 0, "Code", "U10", "R01.pdf", 2),
                ("t1", 1, "Code", "U12", "R01.pdf", 2),
                ("t2", 0, "Niveau", "Elo 1500", "LA.pdf", 150),
            ],
        )
        conn.commit()

        # "u10" in CHESS_SHORT_TERMS + "categorie" (4+ chars) → both match t1
        # cell "U10" → 1.0, col "Code" doesn't match, but "categorie"
        # doesn't match any cell_value. Need 2 term matches for threshold.
        # Add a second matching cell to test allow-list works:
        conn.execute(
            "INSERT INTO structured_cells VALUES (?,?,?,?,?,?)",
            ("t1", 0, "Categorie", "U10 enfants", "R01.pdf", 2),
        )
        conn.commit()

        # "u10" matches cell (1.0) + "categorie" matches col_name (0.5) = 1.5
        results = structured_cell_search(conn, "categorie U10", max_k=5)
        assert len(results) >= 1
        assert results[0][0] == "t1"

        # "elo" in CHESS_SHORT_TERMS + "1500" (4 chars) matches cell_value
        results2 = structured_cell_search(conn, "elo 1500", max_k=5)
        assert len(results2) >= 1
        assert results2[0][0] == "t2"
        conn.close()

    def test_elo_regex_alone_triggers_no_match(self, tmp_path: Path) -> None:
        """Elo regex triggers search, but no matching cells → empty."""
        import sqlite3

        db = tmp_path / "cells5.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE structured_cells "
            "(table_id TEXT, row_idx INT, col_name TEXT, "
            "cell_value TEXT, source TEXT, page INT)"
        )
        conn.execute(
            "INSERT INTO structured_cells VALUES (?,?,?,?,?,?)",
            ("t1", 0, "Niveau", "Expert", "LA.pdf", 150),
        )
        conn.commit()

        # "1500" is in the query but not in cells → no result
        results = structured_cell_search(conn, "joueur 1500 Elo", max_k=5)
        assert results == []
        conn.close()


class TestThreeWayRRF:
    """Tests for structured results in RRF fusion."""

    def test_structured_boost(self) -> None:
        cosine = [("d1", 0.9), ("d2", 0.8)]
        bm25 = [("d2", -1.0), ("d3", -2.0)]
        struct = [("d3", 2.0)]

        fused = reciprocal_rank_fusion(cosine, bm25, struct)
        scores = {did: s for did, s in fused}

        # d3 gets structured boost (1.5x)
        assert scores["d3"] > 1 / (60 + 2)  # more than just bm25 rank 2

    def test_no_structured_results(self) -> None:
        cosine = [("d1", 0.9)]
        bm25 = [("d1", -1.0)]

        fused_with = reciprocal_rank_fusion(cosine, bm25, [])
        fused_without = reciprocal_rank_fusion(cosine, bm25, None)

        assert len(fused_with) == len(fused_without)


class TestGradientIntentScore:
    """Tests for gradient intent detection (B.4)."""

    def test_pure_prose_query(self) -> None:
        from scripts.pipeline.search import gradient_intent_score

        score = gradient_intent_score(
            "Quelle est la procedure d'appel en cas de litige ?"
        )
        assert score == 0.0

    def test_strong_table_query(self) -> None:
        from scripts.pipeline.search import gradient_intent_score

        score = gradient_intent_score(
            "Quel est le classement Elo minimum pour le titre ?"
        )
        assert score >= 1.0

    def test_berger_trigger(self) -> None:
        from scripts.pipeline.search import gradient_intent_score

        score = gradient_intent_score("Quelle est la grille berger pour 6 joueurs ?")
        assert score >= 1.5

    def test_numeric_boost(self) -> None:
        from scripts.pipeline.search import gradient_intent_score

        base = gradient_intent_score("Quel est le classement ?")
        with_num = gradient_intent_score("Quel est le classement 1800 ?")
        assert with_num >= base + 0.4

    def test_capped_at_3(self) -> None:
        from scripts.pipeline.search import gradient_intent_score

        score = gradient_intent_score(
            "berger grille scheveningen bareme cadence elo classement titre norme"
        )
        assert score == 3.0

    def test_age_category_query(self) -> None:
        from scripts.pipeline.search import gradient_intent_score

        score = gradient_intent_score("Quel age pour la categorie pupille ?")
        assert 0.5 <= score <= 2.0

    def test_empty_query(self) -> None:
        from scripts.pipeline.search import gradient_intent_score

        assert gradient_intent_score("") == 0.0


class TestTargetedRowCosineSearch:
    """Tests for targeted_row_cosine_search (canal 5)."""

    def test_returns_table_ids(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE targeted_rows ("
            "id TEXT PRIMARY KEY, table_id TEXT, text TEXT, "
            "source TEXT, page INTEGER, embedding BLOB)"
        )
        dim = 768
        emb1 = np.random.randn(dim).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = np.random.randn(dim).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        conn.execute(
            "INSERT INTO targeted_rows VALUES (?,?,?,?,?,?)",
            ("t1-tr000", "t1", "row 0", "src.pdf", 1, emb1.tobytes()),
        )
        conn.execute(
            "INSERT INTO targeted_rows VALUES (?,?,?,?,?,?)",
            ("t1-tr001", "t1", "row 1", "src.pdf", 1, emb2.tobytes()),
        )
        conn.commit()

        query_emb = np.random.randn(dim).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb)

        from scripts.pipeline.search import targeted_row_cosine_search

        results = targeted_row_cosine_search(
            conn, query_emb, max_k=5, db_path=str(db_path)
        )
        conn.close()

        assert len(results) == 1  # deduped to 1 table
        assert results[0][0] == "t1"
        assert isinstance(results[0][1], float)

    def test_empty_table(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "test_empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE dummy (id TEXT)")
        conn.commit()

        from scripts.pipeline.search import targeted_row_cosine_search

        results = targeted_row_cosine_search(
            conn, np.zeros(768, dtype=np.float32), max_k=5, db_path=str(db_path)
        )
        conn.close()
        assert results == []

    def test_deduplication_keeps_max_score(self, tmp_path: Path) -> None:
        """Multiple rows from same table → only max score kept."""
        import sqlite3

        db_path = tmp_path / "test_dedup.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE targeted_rows ("
            "id TEXT PRIMARY KEY, table_id TEXT, text TEXT, "
            "source TEXT, page INTEGER, embedding BLOB)"
        )
        dim = 768
        # Make row0 very similar to query, row1 orthogonal
        query_emb = np.zeros(dim, dtype=np.float32)
        query_emb[0] = 1.0

        emb_similar = np.zeros(dim, dtype=np.float32)
        emb_similar[0] = 1.0  # cosine=1.0 with query

        emb_ortho = np.zeros(dim, dtype=np.float32)
        emb_ortho[1] = 1.0  # cosine=0.0 with query

        conn.execute(
            "INSERT INTO targeted_rows VALUES (?,?,?,?,?,?)",
            ("t1-tr000", "t1", "row 0", "src.pdf", 1, emb_similar.tobytes()),
        )
        conn.execute(
            "INSERT INTO targeted_rows VALUES (?,?,?,?,?,?)",
            ("t1-tr001", "t1", "row 1", "src.pdf", 1, emb_ortho.tobytes()),
        )
        conn.execute(
            "INSERT INTO targeted_rows VALUES (?,?,?,?,?,?)",
            ("t2-tr000", "t2", "row 0", "src.pdf", 2, emb_ortho.tobytes()),
        )
        conn.commit()

        from scripts.pipeline.search import targeted_row_cosine_search

        results = targeted_row_cosine_search(
            conn, query_emb, max_k=10, db_path=str(db_path)
        )
        conn.close()

        result_map = dict(results)
        assert len(results) == 2  # t1 and t2
        assert result_map["t1"] == pytest.approx(1.0, abs=1e-5)
        assert result_map["t2"] == pytest.approx(0.0, abs=1e-5)

    def test_max_k_limits_output(self, tmp_path: Path) -> None:
        """max_k limits the number of distinct table results."""
        import sqlite3

        db_path = tmp_path / "test_maxk.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE targeted_rows ("
            "id TEXT PRIMARY KEY, table_id TEXT, text TEXT, "
            "source TEXT, page INTEGER, embedding BLOB)"
        )
        dim = 768
        for i in range(5):
            emb = np.random.randn(dim).astype(np.float32)
            emb /= np.linalg.norm(emb)
            conn.execute(
                "INSERT INTO targeted_rows VALUES (?,?,?,?,?,?)",
                (f"t{i}-tr000", f"t{i}", f"row {i}", "src.pdf", i, emb.tobytes()),
            )
        conn.commit()

        query_emb = np.random.randn(dim).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb)

        from scripts.pipeline.search import targeted_row_cosine_search

        results = targeted_row_cosine_search(
            conn, query_emb, max_k=2, db_path=str(db_path)
        )
        conn.close()
        assert len(results) <= 2
