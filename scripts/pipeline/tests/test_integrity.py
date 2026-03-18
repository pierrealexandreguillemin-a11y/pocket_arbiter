"""Tests for post-build integrity gates I1-I9."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import (
    create_db,
    insert_children,
    insert_parents,
    insert_table_summaries,
    populate_fts,
)
from scripts.pipeline.integrity import run_integrity_gates


def _build_valid_db(tmp_path: Path) -> Path:
    """Build a minimal valid DB that passes all gates."""
    db_path = tmp_path / "valid.db"
    conn = create_db(db_path)
    # Parent and child tokens must satisfy I7: child >= 90% of parent
    parent_text = "Parent text that is long enough to satisfy coverage gate."
    child_text = parent_text  # Same text = 100% coverage
    insert_parents(
        conn,
        [
            {
                "id": "p1",
                "text": parent_text,
                "source": "t.pdf",
                "section": "S",
                "tokens": 10,
                "page": 1,
            }
        ],
    )
    emb = np.random.randn(1, 768).astype(np.float32)
    insert_children(
        conn,
        [
            {
                "id": "c1",
                "text": child_text,
                "parent_id": "p1",
                "source": "t.pdf",
                "page": 1,
                "article_num": None,
                "section": "S",
                "tokens": 10,
            }
        ],
        emb,
    )
    ts_emb = np.random.randn(1, 768).astype(np.float32)
    insert_table_summaries(
        conn,
        [
            {
                "id": "t1",
                "summary_text": "Summary",
                "raw_table_text": "| A |",
                "source": "t.pdf",
                "page": 1,
                "tokens": 5,
            }
        ],
        ts_emb,
    )
    populate_fts(conn)
    conn.close()
    return db_path


class TestIntegrityGates:
    """Test each gate catches its specific violation."""

    def test_valid_db_passes(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = _build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        run_integrity_gates(conn)
        conn.close()

    def test_i1_invisible_parent(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = _build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO parents (id, text, source, section, tokens, page) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p_orphan", "Invisible", "t.pdf", "S", 20, 1),
        )
        conn.commit()
        with pytest.raises(ValueError, match="I1 FAIL"):
            run_integrity_gates(conn)
        conn.close()

    def test_i3_null_page(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = _build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE children SET page = NULL WHERE id = 'c1'")
        conn.commit()
        with pytest.raises(ValueError, match="I3 FAIL"):
            run_integrity_gates(conn)
        conn.close()

    def test_i6_giant_parent(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = _build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE parents SET tokens = 5000 WHERE id = 'p1'")
        conn.commit()
        with pytest.raises(ValueError, match="I6 FAIL"):
            run_integrity_gates(conn)
        conn.close()

    def test_i8_unresolved_placeholder(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = _build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE children SET text = 'Text <!-- TABLE_0 --> more' WHERE id = 'c1'"
        )
        conn.commit()
        with pytest.raises(ValueError, match="I8 FAIL"):
            run_integrity_gates(conn)
        conn.close()

    def test_i9_null_page_summary(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = _build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE table_summaries SET page = NULL WHERE id = 't1'")
        conn.commit()
        with pytest.raises(ValueError, match="I9 FAIL"):
            run_integrity_gates(conn)
        conn.close()
