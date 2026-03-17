# scripts/pipeline/search.py
"""Hybrid search: cosine + BM25 FTS5 with RRF fusion, adaptive k filtering."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from scripts.pipeline.context import SearchResult, build_context
from scripts.pipeline.indexer import (
    blob_to_embedding,
    format_query,
    load_model,
)
from scripts.pipeline.synonyms import expand_query

logger = logging.getLogger(__name__)

# === Embedding cache ===

_embedding_cache: dict[str, tuple[list[str], np.ndarray]] = {}


_EMBEDDING_SQL = {
    "children": "SELECT id, embedding FROM children",
    "table_summaries": "SELECT id, embedding FROM table_summaries",
}


def _load_embeddings(
    conn: sqlite3.Connection,
    table: str,
    cache_key: str,
) -> tuple[list[str], np.ndarray]:
    """Load and cache all embeddings from a table.

    Args:
        conn: SQLite connection.
        table: "children" or "table_summaries" (whitelist enforced).
        cache_key: Cache key (db_path + table).

    Returns:
        (ids, embeddings_matrix) where matrix is (N, 768) float32.
    """
    if table not in _EMBEDDING_SQL:
        msg = f"Invalid table: {table}. Must be one of {set(_EMBEDDING_SQL)}"
        raise ValueError(msg)

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    rows = conn.execute(_EMBEDDING_SQL[table]).fetchall()
    if not rows:
        _embedding_cache[cache_key] = ([], np.empty((0, 768), dtype=np.float32))
        return _embedding_cache[cache_key]

    ids = [row[0] for row in rows]
    matrix = np.stack([blob_to_embedding(row[1]) for row in rows])
    _embedding_cache[cache_key] = (ids, matrix)
    return ids, matrix


def clear_embedding_cache() -> None:
    """Clear the embedding cache. Call when switching DBs."""
    _embedding_cache.clear()


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
    db_path: str = "",
) -> list[tuple[str, float]]:
    """Brute-force cosine search on children + table_summaries.

    Embeddings are loaded once and cached in memory. Subsequent calls
    with the same db_path skip all BLOB deserialization.

    Args:
        conn: SQLite connection to corpus DB.
        query_embedding: Query vector (768D, L2 normalized).
        max_k: Maximum results to return.
        db_path: DB path for cache key (empty string = no cache reuse).

    Returns:
        [(id, cosine_score), ...] sorted desc.
    """
    child_ids, child_matrix = _load_embeddings(conn, "children", f"{db_path}:children")
    table_ids, table_matrix = _load_embeddings(
        conn, "table_summaries", f"{db_path}:table_summaries"
    )

    results: list[tuple[str, float]] = []

    if child_ids:
        scores = child_matrix @ query_embedding
        results.extend(zip(child_ids, scores.tolist(), strict=True))

    if table_ids:
        scores = table_matrix @ query_embedding
        results.extend(zip(table_ids, scores.tolist(), strict=True))

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
    try:
        # 1. Query processing
        stemmed_expanded = expand_query(query)
        q_emb = model.encode(
            [format_query(query)],
            normalize_embeddings=True,
        )[0].astype(np.float32)

        # 2. Dual retrieval
        cosine_results = cosine_search(
            conn, q_emb, max_k=max_k * 2, db_path=str(db_path)
        )
        bm25_results = bm25_search(conn, stemmed_expanded, max_k=max_k * 2)

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(cosine_results, bm25_results)

        # 4. Adaptive k
        filtered = adaptive_k(fused, min_score, max_gap, max_k)

        # 5+6. Parent lookup + context assembly
        contexts = build_context(conn, filtered)
    finally:
        conn.close()

    return SearchResult(
        contexts=contexts,
        total_children_matched=len(filtered),
        scores={doc_id: score for doc_id, score in filtered},
    )
