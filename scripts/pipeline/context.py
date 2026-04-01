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
            contexts.append(_parent_context(conn, parent_row, children))
        else:
            contexts.extend(_fallback_child_contexts(conn, children))
    return contexts


def _resolve_page(
    conn: sqlite3.Connection,
    parent_page: int | None,
    children: list[tuple[str, float]],
) -> int | None:
    """Resolve page: use parent page if set, else best-scoring child's page."""
    if parent_page is not None:
        return parent_page
    # Sort by score desc, take first child with a page
    for cid, _ in sorted(children, key=lambda x: -x[1]):
        row = conn.execute("SELECT page FROM children WHERE id = ?", (cid,)).fetchone()
        if row and row[0] is not None:
            return row[0]
    return None


def _parent_context(
    conn: sqlite3.Connection,
    parent_row: tuple,
    children: list[tuple[str, float]],
) -> Context:
    """Build context from a non-empty parent."""
    page = _resolve_page(conn, parent_row[3], children)
    return Context(
        text=parent_row[0],
        source=parent_row[1],
        page=page,
        section=parent_row[2] or "",
        context_type="parent",
        score=max(s for _, s in children),
        children_matched=[cid for cid, _ in children],
    )


def _fallback_child_contexts(
    conn: sqlite3.Connection, children: list[tuple[str, float]]
) -> list[Context]:
    """Fallback for empty parents: return each child as its own context."""
    contexts: list[Context] = []
    for cid, score in children:
        child_row = conn.execute(
            "SELECT text, source, section, page FROM children WHERE id = ?",
            (cid,),
        ).fetchone()
        if child_row:
            contexts.append(
                Context(
                    text=child_row[0],
                    source=child_row[1],
                    page=child_row[3],
                    section=child_row[2] or "",
                    context_type="child",
                    score=score,
                    children_matched=[cid],
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


def _build_table_row_contexts(
    conn: sqlite3.Connection,
    row_ids: list[tuple[str, float]],
) -> list[Context]:
    """Build contexts from matched table rows (multi-vector pattern).

    The row embedding is used for retrieval, but the full raw_table_text
    from the parent table_summary is returned as context for the LLM.
    Deduplicates by parent table_id.
    """
    seen_tables: set[str] = set()
    contexts: list[Context] = []
    for row_id, score in row_ids:
        row = conn.execute(
            "SELECT table_id FROM table_rows WHERE id = ?", (row_id,)
        ).fetchone()
        if not row:
            continue
        table_id = row[0]
        if table_id in seen_tables:
            continue
        seen_tables.add(table_id)
        table = conn.execute(
            "SELECT raw_table_text, source, page FROM table_summaries WHERE id = ?",
            (table_id,),
        ).fetchone()
        if table:
            contexts.append(
                Context(
                    text=table[0],
                    source=table[1],
                    page=table[2],
                    section="",
                    context_type="table",
                    score=score,
                    children_matched=[row_id],
                )
            )
    return contexts


_MAX_INJECTED_WORDS = 500  # ~650 tokens — cap for neighbor table injection


def _collect_neighbor_pages(
    contexts: list[Context],
) -> set[tuple[str, int]]:
    """Collect (source, page) neighborhoods from prose contexts."""
    pages: set[tuple[str, int]] = set()
    for ctx in contexts:
        if ctx.context_type in ("parent", "child") and ctx.page is not None:
            for offset in (-1, 0, 1):
                p = ctx.page + offset
                if p >= 1:
                    pages.add((ctx.source, p))
    return pages


def _collect_existing_tables(
    contexts: list[Context],
) -> tuple[set[str], set[tuple[str, int | None]]]:
    """Collect already-present table IDs and (source, page) pairs for dedup."""
    ids: set[str] = set()
    source_pages: set[tuple[str, int | None]] = set()
    for ctx in contexts:
        if ctx.context_type in ("table", "table_injected"):
            ids.update(ctx.children_matched)
            source_pages.add((ctx.source, ctx.page))
    return ids, source_pages


def _inject_neighbor_tables(
    conn: sqlite3.Connection,
    contexts: list[Context],
) -> list[Context]:
    """Inject table_summaries from pages adjacent to prose contexts.

    When a prose child on page X is retrieved, tables on pages X-1, X, X+1
    from the same source may contain supporting data (cadences, Elo tables,
    grilles Berger) that the LLM needs for a complete answer.

    Only injects tables not already present. Respects a word budget cap.
    """
    neighbor_pages = _collect_neighbor_pages(contexts)
    if not neighbor_pages:
        return contexts

    existing_ids, existing_sp = _collect_existing_tables(contexts)
    min_score = min((c.score for c in contexts), default=0.0) * 0.5
    injected: list[Context] = []
    injected_words = 0

    for source, page in sorted(neighbor_pages):
        if injected_words >= _MAX_INJECTED_WORDS:
            break
        rows = conn.execute(
            "SELECT id, raw_table_text, source, page FROM table_summaries "
            "WHERE source = ? AND page = ?",
            (source, page),
        ).fetchall()
        for tid, raw_text, src, pg in rows:
            if tid in existing_ids or (src, pg) in existing_sp:
                continue
            words = len(raw_text.split())
            if injected_words + words > _MAX_INJECTED_WORDS:
                continue
            injected.append(
                Context(
                    text=raw_text,
                    source=src,
                    page=pg,
                    section="",
                    context_type="table_injected",
                    score=min_score,
                    children_matched=[tid],
                )
            )
            existing_ids.add(tid)
            existing_sp.add((src, pg))
            injected_words += words

    contexts.extend(injected)
    return contexts


def build_context(
    conn: sqlite3.Connection,
    result_ids: list[tuple[str, float]],
) -> list[Context]:
    """Lookup parents, dedup, assemble context, inject neighbor tables.

    Args:
        conn: SQLite connection.
        result_ids: [(id, score), ...] from adaptive_k output.

    Returns:
        List of Context objects, parents deduplicated, ordered by score.
    """
    child_ids: list[tuple[str, float]] = []
    table_ids: list[tuple[str, float]] = []
    table_row_ids: list[tuple[str, float]] = []
    for did, score in result_ids:
        if conn.execute("SELECT 1 FROM children WHERE id = ?", (did,)).fetchone():
            child_ids.append((did, score))
        elif conn.execute("SELECT 1 FROM table_rows WHERE id = ?", (did,)).fetchone():
            table_row_ids.append((did, score))
        else:
            table_ids.append((did, score))

    contexts = _build_parent_contexts(conn, child_ids)
    contexts.extend(_build_table_contexts(conn, table_ids))
    contexts.extend(_build_table_row_contexts(conn, table_row_ids))
    contexts = _inject_neighbor_tables(conn, contexts)
    contexts.sort(key=lambda c: -c.score)
    return contexts
