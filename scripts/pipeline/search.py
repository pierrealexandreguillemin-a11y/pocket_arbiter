# scripts/pipeline/search.py
"""Hybrid search: cosine + BM25 FTS5 with RRF fusion.

Query flow:
1. Stem + synonym expand (for BM25)
2. Embed query (for cosine)
3. Dual retrieval: cosine brute-force + FTS5 BM25
4. RRF fusion
5. Adaptive k filtering
6. Parent lookup + dedup
7. Context assembly
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from scripts.pipeline.indexer import (
    blob_to_embedding,
    format_query,
    load_model,
)
from scripts.pipeline.synonyms import expand_query

logger = logging.getLogger(__name__)


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


# === Pure functions ===


def reciprocal_rank_fusion(
    cosine_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        cosine_results: [(doc_id, cosine_score), ...] sorted desc.
        bm25_results: [(doc_id, bm25_score), ...] sorted by relevance.
        k: RRF constant (default 60, standard value).

    Returns:
        [(doc_id, rrf_score), ...] sorted desc by RRF score.
    """
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(cosine_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def adaptive_k(
    results: list[tuple[str, float]],
    min_score: float = 0.005,
    max_gap: float = 0.01,
    max_k: int = 10,
) -> list[tuple[str, float]]:
    """Filter results by score threshold and gap detection.

    Args:
        results: [(doc_id, score), ...] sorted desc.
        min_score: Minimum score to keep.
        max_gap: Maximum gap between consecutive scores before cutting.
        max_k: Hard maximum number of results.

    Returns:
        Filtered results.
    """
    if not results:
        return []

    # Apply max_k
    results = results[:max_k]

    # Apply min_score
    results = [(doc_id, score) for doc_id, score in results if score >= min_score]

    if not results:
        return []

    # Apply gap detection
    filtered = [results[0]]
    for i in range(1, len(results)):
        gap = results[i - 1][1] - results[i][1]
        if gap > max_gap:
            break
        filtered.append(results[i])

    return filtered


# === DB-dependent functions ===


def cosine_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    max_k: int = 20,
) -> list[tuple[str, float]]:
    """Brute-force cosine search on children + table_summaries.

    Args:
        conn: SQLite connection to corpus DB.
        query_embedding: Query vector (768D, L2 normalized).
        max_k: Maximum results to return.

    Returns:
        [(id, cosine_score), ...] sorted desc.
    """
    results: list[tuple[str, float]] = []

    # Search children
    for row in conn.execute("SELECT id, embedding FROM children"):
        emb = blob_to_embedding(row[1])
        score = float(np.dot(query_embedding, emb))
        results.append((row[0], score))

    # Search table summaries
    for row in conn.execute("SELECT id, embedding FROM table_summaries"):
        emb = blob_to_embedding(row[1])
        score = float(np.dot(query_embedding, emb))
        results.append((row[0], score))

    results.sort(key=lambda x: -x[1])
    return results[:max_k]


def bm25_search(
    conn: sqlite3.Connection,
    stemmed_query: str,
    max_k: int = 20,
) -> list[tuple[str, float]]:
    """BM25 search via FTS5 on stemmed text.

    Args:
        conn: SQLite connection with FTS5 tables.
        stemmed_query: Stemmed + expanded query string.
        max_k: Maximum results to return.

    Returns:
        [(id, bm25_score), ...] sorted by relevance (lower = better in FTS5).
    """
    if not stemmed_query.strip():
        return []

    # FTS5 default is AND between terms. Use OR so synonym expansion
    # matches documents containing ANY expanded term, not ALL.
    fts_query = " OR ".join(stemmed_query.split())

    results: list[tuple[str, float]] = []

    # Search children_fts
    try:
        rows = conn.execute(
            "SELECT id, bm25(children_fts) AS score FROM children_fts "
            "WHERE children_fts MATCH ? ORDER BY score LIMIT ?",
            (fts_query, max_k),
        ).fetchall()
        results.extend((row[0], row[1]) for row in rows)
    except sqlite3.OperationalError:
        logger.warning("FTS5 MATCH failed for query: %s", stemmed_query[:50])

    # Search table_summaries_fts
    try:
        rows = conn.execute(
            "SELECT id, bm25(table_summaries_fts) AS score "
            "FROM table_summaries_fts "
            "WHERE table_summaries_fts MATCH ? ORDER BY score LIMIT ?",
            (fts_query, max_k),
        ).fetchall()
        results.extend((row[0], row[1]) for row in rows)
    except sqlite3.OperationalError:
        logger.warning("FTS5 table MATCH failed for query: %s", stemmed_query[:50])

    # Sort by BM25 score (lower = more relevant in FTS5)
    results.sort(key=lambda x: x[1])
    return results[:max_k]


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
    child_id_set = {
        row[0] for row in conn.execute("SELECT id FROM children").fetchall()
    }

    child_ids = [(did, s) for did, s in result_ids if did in child_id_set]
    table_ids = [(did, s) for did, s in result_ids if did not in child_id_set]

    contexts = _build_parent_contexts(conn, child_ids)
    contexts.extend(_build_table_contexts(conn, table_ids))
    contexts.sort(key=lambda c: -c.score)
    return contexts


# === Main entry point ===


def search(
    db_path: Path | str,
    query: str,
    model: SentenceTransformer | None = None,
    min_score: float = 0.005,
    max_gap: float = 0.01,
    max_k: int = 10,
) -> SearchResult:
    """Full hybrid search pipeline.

    Args:
        db_path: Path to corpus_v2_fr.db.
        query: User question in French.
        model: SentenceTransformer model (loaded if None).
        min_score: Adaptive k minimum RRF score.
        max_gap: Adaptive k maximum gap.
        max_k: Adaptive k maximum results.

    Returns:
        SearchResult with contexts, scores, and metadata.
    """
    if model is None:
        model = load_model()

    conn = sqlite3.connect(str(db_path))

    # 1. Query processing
    stemmed_expanded = expand_query(query)
    q_emb = model.encode(
        [format_query(query)],
        normalize_embeddings=True,
    )[0].astype(np.float32)

    # 2. Dual retrieval
    cosine_results = cosine_search(conn, q_emb, max_k=max_k * 2)
    bm25_results = bm25_search(conn, stemmed_expanded, max_k=max_k * 2)

    # 3. RRF fusion
    fused = reciprocal_rank_fusion(cosine_results, bm25_results)

    # 4. Adaptive k
    filtered = adaptive_k(fused, min_score, max_gap, max_k)

    # 5+6. Parent lookup + context assembly
    contexts = build_context(conn, filtered)

    conn.close()

    return SearchResult(
        contexts=contexts,
        total_children_matched=len(filtered),
        scores={doc_id: score for doc_id, score in filtered},
    )
