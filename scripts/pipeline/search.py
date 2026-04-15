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
_SYNTHETIC_QUERY_SQL = "SELECT id, embedding FROM synthetic_queries"
_TARGETED_ROW_SQL = "SELECT id, embedding FROM targeted_rows"


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
    _VALID_TABLES = {
        **_EMBEDDING_SQL,
        "table_rows": _TABLE_ROW_SQL,
        "synthetic_queries": _SYNTHETIC_QUERY_SQL,
        "targeted_rows": _TARGETED_ROW_SQL,
    }
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


def _accumulate_rrf(
    scores: dict[str, float],
    results: list[tuple[str, float]] | None,
    k: int,
    weight: float = 1.0,
) -> None:
    """Add RRF contributions from one channel into *scores* dict."""
    if not results:
        return
    for rank, (doc_id, _) in enumerate(results):
        scores[doc_id] = scores.get(doc_id, 0) + weight / (k + rank + 1)


def reciprocal_rank_fusion(
    cosine_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    structured_results: list[tuple[str, float]] | None = None,
    table_row_results: list[tuple[str, float]] | None = None,
    synthetic_results: list[tuple[str, float]] | None = None,
    targeted_results: list[tuple[str, float]] | None = None,
    k: int = 60,
    structured_weight: float = 1.5,
    table_row_weight: float = 0.5,
    synthetic_weight: float = 0.5,
    targeted_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Fuse ranked lists using Reciprocal Rank Fusion (6-way).

    Args:
        cosine_results: [(doc_id, cosine_score), ...] sorted desc.
        bm25_results: [(doc_id, bm25_score), ...] sorted by relevance.
        structured_results: Optional structured cell matches (boosted).
        table_row_results: Optional narrative table row cosine matches.
        synthetic_results: Optional Doc2Query synthetic query matches.
        targeted_results: Optional targeted row-chunk cosine matches.
        k: RRF constant (default 60, standard value).
        structured_weight: Boost for structured matches (default 1.5).
        table_row_weight: Weight for table row channel (default 0.5).
        synthetic_weight: Weight for synthetic query channel (default 0.5).
        targeted_weight: Weight for targeted row channel (default 1.0).

    Returns:
        [(doc_id, rrf_score), ...] sorted desc by RRF score.
    """
    scores: dict[str, float] = {}
    _accumulate_rrf(scores, cosine_results, k)
    _accumulate_rrf(scores, bm25_results, k)
    _accumulate_rrf(scores, structured_results, k, structured_weight)
    _accumulate_rrf(scores, table_row_results, k, table_row_weight)
    _accumulate_rrf(scores, synthetic_results, k, synthetic_weight)
    _accumulate_rrf(scores, targeted_results, k, targeted_weight)
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


def synthetic_query_cosine_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    max_k: int = 10,
    db_path: str = "",
) -> list[tuple[str, float]]:
    """Cosine search on synthetic queries (Doc2Query, 5th channel).

    Returns child_ids (not query ids) — the goal is to find the chunk
    that the synthetic question was generated from. When multiple queries
    match the same child, the best score is kept.
    """
    cache_key = f"{db_path}:synthetic_queries"
    ids, matrix = _load_embeddings(conn, "synthetic_queries", cache_key)
    if not ids:
        return []
    scores = matrix @ query_embedding
    # Map query_id -> child_id, keep best score per child
    child_scores: dict[str, float] = {}
    for qid, score in zip(ids, scores.tolist(), strict=True):
        row = conn.execute(
            "SELECT child_id FROM synthetic_queries WHERE id = ?", (qid,)
        ).fetchone()
        if row:
            cid = row[0]
            if cid not in child_scores or score > child_scores[cid]:
                child_scores[cid] = score
    results = sorted(child_scores.items(), key=lambda x: -x[1])
    return results[:max_k]


def targeted_row_cosine_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    max_k: int = 10,
    db_path: str = "",
) -> list[tuple[str, float]]:
    """Cosine search on targeted_rows (canal 5). Returns table_summary_ids.

    Deduplicates by table_id, keeping max score per table.

    Args:
        conn: SQLite connection to corpus DB.
        query_embedding: Query vector (768D, L2 normalized).
        max_k: Maximum distinct table results to return.
        db_path: DB path for cache key (empty string = no cache reuse).

    Returns:
        [(table_summary_id, cosine_score), ...] sorted desc, deduped by table.
    """
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='targeted_rows'"
    ).fetchone()
    if not has_table:
        return []

    cache_key = f"{db_path}:targeted_rows"
    ids, matrix = _load_embeddings(conn, "targeted_rows", cache_key)
    if not ids:
        return []

    scores_arr = matrix @ query_embedding
    row_scores = list(zip(ids, scores_arr.tolist(), strict=True))

    # Map row IDs to table_ids, keep max score per table
    best: dict[str, float] = {}
    for row_id, score in row_scores:
        table_id = conn.execute(
            "SELECT table_id FROM targeted_rows WHERE id = ?", (row_id,)
        ).fetchone()
        if table_id:
            tid = table_id[0]
            if tid not in best or score > best[tid]:
                best[tid] = score

    return sorted(best.items(), key=lambda x: -x[1])[:max_k]


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

# === Gradient intent detection (B.4) ===
# Replaces binary _has_table_triggers() with a continuous 0.0-3.0 score.
# Each trigger term has a weight reflecting table-relevance strength.

INTENT_WEIGHTS: dict[str, float] = {
    "berger": 1.0,
    "grille": 0.9,
    "scheveningen": 1.0,
    "bareme": 0.8,
    "barème": 0.8,
    "categorie": 0.7,
    "catégorie": 0.7,
    "cadence": 0.7,
    "elo": 0.8,
    "classement": 0.6,
    "titre": 0.6,
    "norme": 0.7,
    "departage": 0.7,
    "départage": 0.7,
    "frais": 0.6,
    "deplacement": 0.5,
    "coefficient": 0.7,
    "age": 0.4,
    "âge": 0.4,
    "poussin": 0.5,
    "pupille": 0.5,
    "benjamin": 0.5,
    "minime": 0.5,
    "cadet": 0.5,
    "junior": 0.5,
    "glossaire": 0.3,
    "definition": 0.3,
    # Team-related (from old _TABLE_TRIGGERS_WEAK + checkpoint 5b gap)
    "equipe": 0.6,
    "echiquier": 0.5,
    "joueur": 0.3,
    "nationale": 0.4,
}

_INTENT_STEMS: dict[str, float] = {}


def _get_intent_stems() -> dict[str, float]:
    """Build stemmed trigger -> weight map (lazy, cached)."""
    if not _INTENT_STEMS:
        from scripts.pipeline.synonyms import _stemmer

        for trigger, weight in INTENT_WEIGHTS.items():
            stemmed = _stemmer.stemWord(trigger.lower())
            if stemmed not in _INTENT_STEMS or weight > _INTENT_STEMS[stemmed]:
                _INTENT_STEMS[stemmed] = weight
    return _INTENT_STEMS


def gradient_intent_score(query: str) -> float:
    """Compute table intent score from query terms (B.4).

    Sums weights for each matched trigger stem in the query.
    Adds 0.5 bonus for Elo-like numeric patterns (3-4 digits).

    Args:
        query: User query in French.

    Returns:
        Score in [0.0, 3.0]. Higher = stronger table intent.
    """
    from scripts.pipeline.synonyms import _stemmer

    stems = _get_intent_stems()
    query_stems = {_stemmer.stemWord(w) for w in re.split(r"\W+", query.lower()) if w}
    score = sum(weight for stem, weight in stems.items() if stem in query_stems)
    if _ELO_RE.search(query):
        score += 0.5
    return min(score, 3.0)


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


# Priority tables: high-value tables that deserve a score boost.
# These are the most-consulted tables in the FFE/FIDE corpus.
# Data-driven: top tables by GS question frequency (audit 2026-04-01).
_PRIORITY_TABLES: set[str] = {
    "LA-octobre2025-table2",  # Resultats fin de partie (21 Q)
    "R01_2025_26_Regles_generales-table0",  # Categories d'age (15 Q)
    "R01_2025_26_Regles_generales-table1",  # Equivalence cadences (12 Q)
    "LA-octobre2025-table75",  # Bareme titres FIDE (8 Q)
    "LA-octobre2025-table73",  # Conversion Elo variante (7 Q)
    "LA-octobre2025-table74",  # Conversion Elo etendue (7 Q)
    "LA-octobre2025-table70",  # Qualification arbitres (5 Q)
    "LA-octobre2025-table69",  # Bareme frais deplacement (5 Q)
    "LA-octobre2025-table82",  # Conditions normes titres FIDE
    "LA-octobre2025-table83",  # Exigences normes titres FIDE
}
_PRIORITY_BOOST = 1.5  # Additive bonus for priority tables


def _extract_search_terms(query: str) -> list[str]:
    """Extract meaningful terms from query for structured cell lookup."""
    q_lower = query.lower()
    return [
        w for w in re.split(r"\W+", q_lower) if len(w) >= 4 or w in CHESS_SHORT_TERMS
    ]


def structured_cell_search(
    conn: sqlite3.Connection,
    query: str,
    max_k: int = 5,
) -> list[tuple[str, float]]:
    """Lookup on structured_cells via FTS5 + LIKE fallback.

    Returns table_summary IDs ranked by match quality.

    Args:
        conn: SQLite connection.
        query: User query.
        max_k: Max results.

    Returns:
        [(table_summary_id, score), ...] sorted desc.
    """
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='structured_cells'"
    ).fetchone()
    if not has_table:
        return []

    terms = _extract_search_terms(query)
    if not terms:
        return []

    scores: dict[str, float] = {}
    has_fts = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='structured_cells_fts'"
    ).fetchone()

    if has_fts:
        _search_cells_fts(conn, terms, scores)
    else:
        _search_cells_like(conn, terms, scores)

    for tid in _PRIORITY_TABLES:
        if tid in scores:
            scores[tid] += _PRIORITY_BOOST

    ranked = [(tid, s) for tid, s in scores.items() if s >= 2.0]
    ranked.sort(key=lambda x: -x[1])
    return ranked[:max_k]


def _search_cells_fts(
    conn: sqlite3.Connection,
    terms: list[str],
    scores: dict[str, float],
) -> None:
    """FTS5 search on structured_cells_fts (col_name + cell_value).

    Uses prefix queries (term*) to handle plural/conjugation without stemming.
    """
    fts_query = " OR ".join(f"{t}*" for t in terms[:10] if t)
    if not fts_query:
        return
    try:
        rows = conn.execute(
            "SELECT table_id FROM structured_cells_fts WHERE cell_value MATCH ?",
            (fts_query,),
        ).fetchall()
        for (table_id,) in rows:
            scores[table_id] = scores.get(table_id, 0) + 1.0
        rows = conn.execute(
            "SELECT table_id FROM structured_cells_fts WHERE col_name MATCH ?",
            (fts_query,),
        ).fetchall()
        for (table_id,) in rows:
            scores[table_id] = scores.get(table_id, 0) + 0.5
    except sqlite3.OperationalError:
        _search_cells_like(conn, terms, scores)


def _search_cells_like(
    conn: sqlite3.Connection,
    terms: list[str],
    scores: dict[str, float],
) -> None:
    """LIKE fallback for structured_cells search."""
    for term in terms[:10]:
        rows = conn.execute(
            "SELECT table_id FROM structured_cells WHERE cell_value LIKE ?",
            (f"%{term}%",),
        ).fetchall()
        for (table_id,) in rows:
            scores[table_id] = scores.get(table_id, 0) + 1.0
    for term in terms[:10]:
        rows = conn.execute(
            "SELECT DISTINCT table_id FROM structured_cells WHERE col_name LIKE ?",
            (f"%{term}%",),
        ).fetchall()
        for (table_id,) in rows:
            scores[table_id] = scores.get(table_id, 0) + 0.5


# === Intro page filter (OPT-5) ===

# Pages with structural content only (TOC, cover, section titles).
# These pollute top-k without providing useful RAG context.
# Verified against PDF source — no substantive content on these pages.
_INTRO_PAGES: set[tuple[str, int]] = {
    ("LA-octobre2025.pdf", 1),  # Cover page (158 chars, title only)
    ("LA-octobre2025.pdf", 100),  # Section title "Systemes d'appariements" (434 chars)
}


_INTRO_LOOKUP_SQL = {
    "children": "SELECT source, page FROM children WHERE id = ?",
    "table_summaries": "SELECT source, page FROM table_summaries WHERE id = ?",
    "table_rows": "SELECT source, page FROM table_rows WHERE id = ?",
}


def _filter_intro_pages(
    conn: sqlite3.Connection,
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Remove results from intro/structural pages (OPT-5)."""
    if not _INTRO_PAGES:
        return results
    filtered: list[tuple[str, float]] = []
    for doc_id, score in results:
        for sql in _INTRO_LOOKUP_SQL.values():
            row = conn.execute(sql, (doc_id,)).fetchone()
            if row:
                if (row[0], row[1]) not in _INTRO_PAGES:
                    filtered.append((doc_id, score))
                break
        else:
            filtered.append((doc_id, score))
    return filtered


# === Main entry point ===


def search(
    db_path: Path | str,
    query: str,
    model: SentenceTransformer | None = None,
    min_score: float = 0.005,
    max_k: int = 10,
    table_row_weight: float = 0.5,
) -> SearchResult:
    """Full hybrid search pipeline.

    Args:
        db_path: Path to corpus_v2_fr.db.
        query: User question in French.
        model: SentenceTransformer model (loaded if None).
        min_score: Adaptive k minimum RRF score.
        max_k: Adaptive k maximum results.
        table_row_weight: Weight for Canal 4 (narrative rows) in RRF.

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

        # 3. Gradient intent score (B.4)
        intent = gradient_intent_score(query)
        # Clipping: cap at old-system levels to prevent table shadowing
        # of prose (Point 4). Isolation test (2026-04-04):
        #   uncapped intent*1.5 → 26/34 human (-6 regressions)
        #   capped min(,1.5)   → 30/34 human (-2: 1 shadowing + 1 rebuild jitter)
        #   all channels OFF   → 31/34 human (-1: rebuild jitter only)
        # Floor 0.3: below this, intent is too weak to justify table boost.
        if intent < 0.3:
            structured_weight = 0.0
            targeted_weight = 0.0
        else:
            structured_weight = min(intent * 1.5, 1.5)
            targeted_weight = min(intent * 1.0, 1.0)

        # 3a. Structured table lookup (scaled by intent)
        struct_results = (
            structured_cell_search(conn, query, max_k=5) if intent > 0 else []
        )

        # 3b. Table row cosine (separate canal 4, fixed weight)
        trow_results = table_row_cosine_search(
            conn, q_emb, max_k=10, db_path=str(db_path)
        )

        # 3c. Targeted row-chunks cosine (canal 5)
        targeted_results = (
            targeted_row_cosine_search(conn, q_emb, max_k=10, db_path=str(db_path))
            if intent > 0
            else []
        )

        # 3d. Synthetic query cosine — DISABLED (w=0.0)
        # Sweep result: w=0.0 best (59.4%), all w>0 degrade recall.
        # Data stays in DB for future experiments.

        # 4. RRF fusion (6-way: cosine + BM25 + structured + table_rows
        #    + targeted + synthetic)
        fused = reciprocal_rank_fusion(
            cosine_results,
            bm25_results,
            struct_results,
            trow_results,
            targeted_results=targeted_results,
            structured_weight=structured_weight,
            table_row_weight=table_row_weight,
            targeted_weight=targeted_weight,
        )

        # 4b. Filter intro pages (OPT-5)
        fused = _filter_intro_pages(conn, fused)

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
