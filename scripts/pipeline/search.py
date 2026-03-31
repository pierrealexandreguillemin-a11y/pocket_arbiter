"""Hybrid search: cosine + BM25 FTS5 with RRF fusion, adaptive k filtering."""

from __future__ import annotations

import logging
import re
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
    # table_rows kept in a SEPARATE channel (not mixed in general cosine/BM25).
    # Mixed: +12.3pp tab, -7.3pp prose. Separate channel avoids prose pollution.
}
_TABLE_ROW_SQL = "SELECT id, embedding FROM table_rows"


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
    _VALID_TABLES = {**_EMBEDDING_SQL, "table_rows": _TABLE_ROW_SQL}
    if table not in _VALID_TABLES:
        msg = f"Invalid table: {table}. Must be one of {set(_VALID_TABLES)}"
        raise ValueError(msg)

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    rows = conn.execute(_VALID_TABLES[table]).fetchall()
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
    structured_results: list[tuple[str, float]] | None = None,
    table_row_results: list[tuple[str, float]] | None = None,
    k: int = 60,
    structured_weight: float = 1.5,
    table_row_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """Fuse ranked lists using Reciprocal Rank Fusion.

    Args:
        cosine_results: [(doc_id, cosine_score), ...] sorted desc.
        bm25_results: [(doc_id, bm25_score), ...] sorted by relevance.
        structured_results: Optional structured cell matches (boosted).
        table_row_results: Optional narrative table row cosine matches.
        k: RRF constant (default 60, standard value).
        structured_weight: Boost for structured matches (default 1.5).
        table_row_weight: Weight for table row channel (default 0.5,
            lower than 1.0 to avoid prose pollution).

    Returns:
        [(doc_id, rrf_score), ...] sorted desc by RRF score.
    """
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(cosine_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    if structured_results:
        for rank, (doc_id, _) in enumerate(structured_results):
            scores[doc_id] = scores.get(doc_id, 0) + structured_weight / (k + rank + 1)
    if table_row_results:
        for rank, (doc_id, _) in enumerate(table_row_results):
            scores[doc_id] = scores.get(doc_id, 0) + table_row_weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def adaptive_k(
    results: list[tuple[str, float]],
    min_score: float = 0.005,
    min_k: int = 3,
    max_k: int = 10,
    buffer: int = 0,
) -> list[tuple[str, float]]:
    """Filter results: min_score threshold, then largest-gap cut (EMNLP 2025).

    Args:
        results: [(doc_id, score), ...] sorted desc.
        min_score: Minimum score to keep.
        min_k: Minimum results (floor for gap detection).
        max_k: Hard maximum.
        buffer: Extra results after gap.
    """
    if not results:
        return []

    # Apply max_k
    results = results[:max_k]

    # Apply min_score
    results = [(doc_id, score) for doc_id, score in results if score >= min_score]

    if len(results) <= min_k:
        return results

    # Find largest gap (EMNLP Adaptive-k)
    gaps = [results[i][1] - results[i + 1][1] for i in range(len(results) - 1)]
    largest_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])

    # Keep everything before the largest gap + buffer, but at least min_k
    cut = max(largest_gap_idx + 1 + buffer, min_k)
    cut = min(cut, len(results))
    return results[:cut]


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
    results: list[tuple[str, float]] = []

    for table_name in _EMBEDDING_SQL:
        cache_key = f"{db_path}:{table_name}"
        ids, matrix = _load_embeddings(conn, table_name, cache_key)
        if ids:
            scores = matrix @ query_embedding
            results.extend(zip(ids, scores.tolist(), strict=True))

    results.sort(key=lambda x: -x[1])
    return results[:max_k]


def table_row_cosine_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    max_k: int = 10,
    db_path: str = "",
) -> list[tuple[str, float]]:
    """Cosine search on narrative table_rows only (separate channel).

    Kept separate from cosine_search to avoid polluting prose results.
    Merged via RRF 4th channel with controlled weight.
    """
    cache_key = f"{db_path}:table_rows"
    ids, matrix = _load_embeddings(conn, "table_rows", cache_key)
    if not ids:
        return []
    scores = matrix @ query_embedding
    results = sorted(zip(ids, scores.tolist(), strict=True), key=lambda x: -x[1])
    return results[:max_k]


def _sanitize_fts_query(stemmed_query: str) -> str:
    """Clean stemmed query for FTS5 MATCH: strip punctuation, dedup, OR join."""
    terms = [re.sub(r"[^\w]", "", t) for t in stemmed_query.split()]
    unique = list(dict.fromkeys(t for t in terms if t))
    return " OR ".join(unique) if unique else ""


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

    fts_query = _sanitize_fts_query(stemmed_query)
    if not fts_query:
        return []

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

    # table_rows_fts excluded from general BM25 (separate channel in RRF).
    # Sort by BM25 score (lower = more relevant in FTS5)
    results.sort(key=lambda x: x[1])
    return results[:max_k]


# === Structured table lookup (Level 3) ===

# Allow-list of short chess terms (<=3 chars) that are domain-specific
# enough to use in structured cell search without noise.
# Sources: synonyms.py CHESS_SYNONYMS, enrichment.py ABBREVIATIONS,
# FFE category codes, FIDE title codes, Elo-related terms.
CHESS_SHORT_TERMS: set[str] = {
    # Elo / scoring
    "elo",
    "dp",
    # Age categories (FFE codes)
    "u8",
    "u10",
    "u12",
    "u14",
    "u16",
    "u18",
    "u20",
    # FIDE titles
    "gm",
    "mi",
    "fm",
    "cm",
    "mf",
    "wgm",
    "wmi",
    "wfm",
    # Abbreviations
    "ai",
    "uv",
    # Game terms
    "mat",
    "nul",
    "pat",
    # Cadence letters (FFE)
    # Not included: single letters "a", "b", "c", "d" are too noisy
}

# Strong triggers: specific to table data, low false-positive rate
_TABLE_TRIGGERS_STRONG = {
    "berger",
    "grille",
    "scheveningen",
    "appariement",
    "poussin",
    "pupille",
    "benjamin",
    "minime",
    "cadet",
    "junior",
    "bareme",
    "departage",
    "coefficient",
    "glossaire",
    "definition",
}

# Weak triggers: common terms that benefit from structured lookup
_TABLE_TRIGGERS_WEAK = {
    "cadence",
    "age",
    "elo",
    "classement",
    "titre",
    "norme",
    "echiquier",
    "frais",
    "deplacement",
    "distance",
    "categorie",
}

_ELO_RE = re.compile(r"\b\d{3,4}\b")


def _has_table_triggers(query: str) -> bool:
    """Check if query needs structured table lookup.

    Activates on: 1+ strong trigger, OR 2+ weak triggers + Elo regex.
    Conservative to avoid flooding RRF with false-positive tables.
    """
    q_lower = query.lower()
    strong = sum(1 for t in _TABLE_TRIGGERS_STRONG if t in q_lower)
    if strong >= 1:
        return True
    weak = sum(1 for t in _TABLE_TRIGGERS_WEAK if t in q_lower)
    return weak >= 2 and bool(_ELO_RE.search(query))


def structured_cell_search(
    conn: sqlite3.Connection,
    query: str,
    max_k: int = 5,
) -> list[tuple[str, float]]:
    """Deterministic lookup on structured_cells table.

    Returns table_summary IDs ranked by match quality.

    Args:
        conn: SQLite connection.
        query: User query.
        max_k: Max results.

    Returns:
        [(table_summary_id, score), ...] sorted desc.
    """
    # Check if structured_cells table exists
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='structured_cells'"
    ).fetchone()
    if not has_table:
        return []

    q_lower = query.lower()
    # Accept terms with 4+ chars OR short domain terms from allow-list.
    # Phase 1 chantier 5: allow-list replaces blanket 4-char filter.
    terms = [
        w for w in re.split(r"\W+", q_lower) if len(w) >= 4 or w in CHESS_SHORT_TERMS
    ]
    if not terms:
        return []

    # Search cell values matching query terms
    scores: dict[str, float] = {}
    for term in terms[:10]:  # cap terms to avoid slow queries
        rows = conn.execute(
            "SELECT table_id FROM structured_cells WHERE cell_value LIKE ?",
            (f"%{term}%",),
        ).fetchall()
        for (table_id,) in rows:
            scores[table_id] = scores.get(table_id, 0) + 1.0

    # Also search column names (col_name often contains category info)
    for term in terms[:10]:
        rows = conn.execute(
            "SELECT DISTINCT table_id FROM structured_cells " "WHERE col_name LIKE ?",
            (f"%{term}%",),
        ).fetchall()
        for (table_id,) in rows:
            scores[table_id] = scores.get(table_id, 0) + 0.5

    # Threshold 2.0: requires 2+ cell matches or cell + col_name confirmation.
    ranked = [(tid, s) for tid, s in scores.items() if s >= 2.0]
    ranked.sort(key=lambda x: -x[1])
    return ranked[:max_k]


# === Main entry point ===


def search(
    db_path: Path | str,
    query: str,
    model: SentenceTransformer | None = None,
    min_score: float = 0.005,
    max_k: int = 10,
) -> SearchResult:
    """Full hybrid search pipeline.

    Args:
        db_path: Path to corpus_v2_fr.db.
        query: User question in French.
        model: SentenceTransformer model (loaded if None).
        min_score: Adaptive k minimum RRF score.
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

        # 3. Structured table lookup (if triggers detected)
        struct_results = (
            structured_cell_search(conn, query, max_k=5)
            if _has_table_triggers(query)
            else []
        )

        # 3b. Table row cosine (separate 4th channel, weight 0.5)
        trow_results = table_row_cosine_search(
            conn, q_emb, max_k=10, db_path=str(db_path)
        )

        # 4. RRF fusion (4-way: cosine + BM25 + structured + table_rows)
        fused = reciprocal_rank_fusion(
            cosine_results, bm25_results, struct_results, trow_results
        )

        # 5. Adaptive k (EMNLP 2025 largest-gap)
        filtered = adaptive_k(fused, min_score, max_k)

        # 6. Parent lookup + context assembly
        contexts = build_context(conn, filtered)
    finally:
        conn.close()

    return SearchResult(
        contexts=contexts,
        total_children_matched=len(filtered),
        scores={doc_id: score for doc_id, score in filtered},
    )
