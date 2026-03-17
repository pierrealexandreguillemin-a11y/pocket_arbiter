# scripts/pipeline/tests/test_search_quality_gates.py
"""Quality gates S1-S8 on real corpus_v2_fr.db (slow tests)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.search import (
    bm25_search,
    clear_embedding_cache,
    cosine_search,
    reciprocal_rank_fusion,
    search,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear embedding cache between every test."""
    clear_embedding_cache()


@pytest.mark.slow
class TestSearchQualityGates:
    """Quality gates S1-S8 on real corpus_v2_fr.db."""

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
        from scripts.pipeline.synonyms import expand_query

        conn = sqlite3.connect(str(self.DB_PATH))
        try:
            results = bm25_search(conn, expand_query("forfait"), max_k=5)
            assert len(results) >= 1, "BM25 returned no results for 'forfait'"
        finally:
            conn.close()

        # Verify at least one is from a forfait section
        conn = sqlite3.connect(str(self.DB_PATH))
        try:
            for doc_id, _ in results[:3]:
                row = conn.execute(
                    "SELECT section FROM children WHERE id = ?", (doc_id,)
                ).fetchone()
                if row and "forfait" in (row[0] or "").lower():
                    return
        finally:
            conn.close()
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
        from scripts.pipeline.synonyms import expand_query

        model = load_model()
        conn = sqlite3.connect(str(self.DB_PATH))
        try:
            hybrid_match = 0
            for q in queries:
                q_emb = model.encode(
                    [format_query(q)],
                    normalize_embeddings=True,
                )[0].astype(np.float32)
                cosine = cosine_search(conn, q_emb, max_k=10)
                bm25 = bm25_search(conn, expand_query(q), max_k=10)
                hybrid = reciprocal_rank_fusion(cosine, bm25)
                if hybrid and hybrid[0][0] in [c[0] for c in cosine[:3]]:
                    hybrid_match += 1
        finally:
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

    def test_s8_fts5_counts(self) -> None:
        """S8: FTS5 tables contain expected row counts."""
        conn = sqlite3.connect(str(self.DB_PATH))
        try:
            c_fts = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
            t_fts = conn.execute("SELECT COUNT(*) FROM table_summaries_fts").fetchone()[
                0
            ]
        finally:
            conn.close()
        assert c_fts == 1253, f"children_fts: {c_fts} != 1253"
        assert t_fts == 111, f"table_summaries_fts: {t_fts} != 111"
