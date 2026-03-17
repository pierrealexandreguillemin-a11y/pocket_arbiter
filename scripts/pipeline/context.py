# scripts/pipeline/context.py
"""Context assembly: parent lookup, dedup, table context building."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass
class Context:
    """A single context block for the LLM."""

    text: str
    source: str
    page: int | None
    section: str
    context_type: str  # "parent" or "table"
    score: float
    children_matched: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Complete search result."""

    contexts: list[Context]
    total_children_matched: int
    scores: dict[str, float]


def _build_parent_contexts(
    conn: sqlite3.Connection,
    child_ids: list[tuple[str, float]],
) -> list[Context]:
    """Group children by parent, dedup, build parent contexts."""
    parent_groups: dict[str, list[tuple[str, float]]] = {}
    for child_id, score in child_ids:
        row = conn.execute(
            "SELECT parent_id FROM children WHERE id = ?", (child_id,)
        ).fetchone()
        if row:
            parent_groups.setdefault(row[0], []).append((child_id, score))

    contexts: list[Context] = []
    for pid, children in parent_groups.items():
        parent_row = conn.execute(
            "SELECT text, source, section, page FROM parents WHERE id = ?",
            (pid,),
        ).fetchone()
        if parent_row and parent_row[0].strip():
            contexts.append(
                Context(
                    text=parent_row[0],
                    source=parent_row[1],
                    page=parent_row[3],
                    section=parent_row[2] or "",
                    context_type="parent",
                    score=max(s for _, s in children),
                    children_matched=[cid for cid, _ in children],
                )
            )
    return contexts


def _build_table_contexts(
    conn: sqlite3.Connection,
    table_ids: list[tuple[str, float]],
) -> list[Context]:
    """Build table contexts from matched summaries."""
    contexts: list[Context] = []
    for table_id, score in table_ids:
        row = conn.execute(
            "SELECT raw_table_text, source, page FROM table_summaries WHERE id = ?",
            (table_id,),
        ).fetchone()
        if row:
            contexts.append(
                Context(
                    text=row[0],
                    source=row[1],
                    page=row[2],
                    section="",
                    context_type="table",
                    score=score,
                    children_matched=[table_id],
                )
            )
    return contexts


def build_context(
    conn: sqlite3.Connection,
    result_ids: list[tuple[str, float]],
) -> list[Context]:
    """Lookup parents, dedup, assemble context.

    Args:
        conn: SQLite connection.
        result_ids: [(id, score), ...] from adaptive_k output.

    Returns:
        List of Context objects, parents deduplicated, ordered by score.
    """
    # Classify each result as child or table_summary via targeted lookup
    # (avoids full table scan of 1253 children IDs on every call)
    child_ids: list[tuple[str, float]] = []
    table_ids: list[tuple[str, float]] = []
    for did, score in result_ids:
        row = conn.execute("SELECT 1 FROM children WHERE id = ?", (did,)).fetchone()
        if row:
            child_ids.append((did, score))
        else:
            table_ids.append((did, score))

    contexts = _build_parent_contexts(conn, child_ids)
    contexts.extend(_build_table_contexts(conn, table_ids))
    contexts.sort(key=lambda c: -c.score)
    return contexts
