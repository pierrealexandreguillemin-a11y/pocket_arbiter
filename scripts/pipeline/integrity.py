"""Post-build integrity gates for corpus DB (I1-I9).

Industry standards: LangChain coverage guarantee, GraphRAG parent cap,
KX multi-vector table linkage, relational integrity.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


def run_integrity_gates(conn: sqlite3.Connection) -> None:
    """Validate relational integrity after build. Raises on failure."""
    _gate_i1_no_invisible_parents(conn)
    _gate_i2_no_orphan_children(conn)
    _gate_i3_no_null_pages(conn)
    _gate_i4_no_null_embeddings(conn)
    _gate_i5_fts5_sync(conn)
    _gate_i6_parent_token_cap(conn)
    _gate_i7_coverage(conn)
    _gate_i8_no_unresolved_placeholders(conn)
    _gate_i9_table_summary_linkage(conn)


def _gate_i1_no_invisible_parents(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT p.id FROM parents p "
        "LEFT JOIN children c ON c.parent_id = p.id "
        "WHERE c.id IS NULL AND p.text != '' AND p.tokens > 0"
    ).fetchall()
    if rows:
        ids = [r[0] for r in rows[:5]]
        msg = f"I1 FAIL: {len(rows)} parents with text but 0 children: {ids}"
        raise ValueError(msg)
    logger.info("I1 PASS: all parents with text have children")


def _gate_i2_no_orphan_children(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT c.id FROM children c "
        "LEFT JOIN parents p ON c.parent_id = p.id "
        "WHERE p.id IS NULL"
    ).fetchall()
    if rows:
        msg = f"I2 FAIL: {len(rows)} children with missing parent"
        raise ValueError(msg)
    logger.info("I2 PASS: all children have valid parent_id")


def _gate_i3_no_null_pages(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM children WHERE page IS NULL").fetchone()[
        0
    ]
    if count:
        msg = f"I3 FAIL: {count} children with NULL page"
        raise ValueError(msg)
    logger.info("I3 PASS: all children have page number")


def _gate_i4_no_null_embeddings(conn: sqlite3.Connection) -> None:
    count = conn.execute(
        "SELECT COUNT(*) FROM children WHERE embedding IS NULL"
    ).fetchone()[0]
    if count:
        msg = f"I4 FAIL: {count} children with NULL embedding"
        raise ValueError(msg)
    logger.info("I4 PASS: all children have embeddings")


def _gate_i5_fts5_sync(conn: sqlite3.Connection) -> None:
    c_count = conn.execute("SELECT COUNT(*) FROM children").fetchone()[0]
    c_fts = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
    if c_count != c_fts:
        msg = f"I5 FAIL: children={c_count} vs children_fts={c_fts}"
        raise ValueError(msg)
    ts_count = conn.execute("SELECT COUNT(*) FROM table_summaries").fetchone()[0]
    ts_fts = conn.execute("SELECT COUNT(*) FROM table_summaries_fts").fetchone()[0]
    if ts_count != ts_fts:
        msg = f"I5 FAIL: table_summaries={ts_count} vs fts={ts_fts}"
        raise ValueError(msg)
    logger.info("I5 PASS: FTS5 counts match (%d + %d)", c_fts, ts_fts)


def _gate_i6_parent_token_cap(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, tokens FROM parents WHERE tokens > 2048").fetchall()
    if rows:
        ids = [f"{r[0]}({r[1]}tok)" for r in rows[:5]]
        msg = f"I6 FAIL: {len(rows)} parents > 2048 tokens: {ids}"
        raise ValueError(msg)
    logger.info("I6 PASS: all parents <= 2048 tokens")


def _gate_i7_coverage(conn: sqlite3.Connection) -> None:
    child_tokens = conn.execute("SELECT SUM(tokens) FROM children").fetchone()[0] or 0
    parent_tokens = (
        conn.execute("SELECT SUM(tokens) FROM parents WHERE text != ''").fetchone()[0]
        or 0
    )
    if parent_tokens == 0:
        logger.info("I7 SKIP: no parent tokens to compare")
        return
    ratio = child_tokens / parent_tokens
    if ratio < 0.9:
        msg = (
            f"I7 FAIL: child tokens ({child_tokens}) < 90% of "
            f"parent tokens ({parent_tokens}), ratio={ratio:.2f}"
        )
        raise ValueError(msg)
    logger.info("I7 PASS: coverage ratio %.2f (>= 0.90)", ratio)


def _gate_i8_no_unresolved_placeholders(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id FROM children WHERE text LIKE '%<!-- TABLE_%'"
    ).fetchall()
    if rows:
        ids = [r[0] for r in rows[:5]]
        msg = f"I8 FAIL: {len(rows)} children with unresolved placeholders: {ids}"
        raise ValueError(msg)
    logger.info("I8 PASS: no unresolved table placeholders")


def _gate_i9_table_summary_linkage(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id FROM table_summaries WHERE source IS NULL OR page IS NULL"
    ).fetchall()
    if rows:
        ids = [r[0] for r in rows[:5]]
        msg = f"I9 FAIL: {len(rows)} table_summaries with NULL source/page: {ids}"
        raise ValueError(msg)
    logger.info("I9 PASS: all table_summaries have valid source+page")
