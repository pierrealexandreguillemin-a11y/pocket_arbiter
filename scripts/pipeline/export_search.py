"""
Export Search - Pocket Arbiter

Fonctions de recherche vectorielle et hybride.

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.pipeline.export_serialization import blob_to_embedding
from scripts.pipeline.retrieval_logger import log_retrieval

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Hybrid search parameters (2025 enterprise standard - Funnel Mode)
# Research: sqlite-rag, LlamaIndex Alpha=0.4, VectorHub/Superlinked, Databricks
# - Pool initial large (100) pour ne pas perdre de candidats
# - Weights: BM25=0.6 (keywords, art. numbers), Vector=0.4 (semantic)
# - Rerank sur pool large pour +48% recall (Databricks research)
DEFAULT_VECTOR_WEIGHT = 0.4
DEFAULT_BM25_WEIGHT = 0.6
DEFAULT_INITIAL_K = 100  # Funnel Mode: large initial pool
RRF_K = 60  # Reciprocal Rank Fusion constant (standard)

# =============================================================================
# Glossary Boost Configuration (DNA 2025 - Source canonique definitions)
# =============================================================================
# Research: Glossaires = 30-50% des fails sur corpus reglementaires (definitions)
# Strategy: Boost x3.5 pour chunks glossaire (pages 67-70 LA-octobre + table summaries)
# ISO Reference: ISO 42001 - Traceability, ISO 25010 - Functional suitability
DEFAULT_GLOSSARY_BOOST = 3.5  # Multiplicateur pour chunks glossaire
GLOSSARY_PAGE_RANGE = (67, 70)  # Pages glossaire dans LA-octobre2025

# Patterns pour detecter les chunks glossaire (par ID ou metadata)
GLOSSARY_CHUNK_PATTERNS = [
    "LA-octobre2025-table3",  # Glossaire A-C
    "LA-octobre2025-table4",  # Glossaire C-F
    "LA-octobre2025-table5",  # Glossaire F-O
    "LA-octobre2025-table6",  # Glossaire O-R
    "LA-octobre2025-table7",  # Glossaire R-Z
]


def _is_glossary_chunk(chunk_id: str, source: str, page: int) -> bool:
    """
    Detecte si un chunk appartient au glossaire officiel DNA 2025.

    Criteres:
    1. ID contient un pattern de table glossaire (table3-7 LA-octobre)
    2. Source LA-octobre + page dans range glossaire (67-70)

    Args:
        chunk_id: Identifiant du chunk.
        source: Source du document (e.g. "LA-octobre2025.pdf").
        page: Numero de page.

    Returns:
        True si chunk glossaire, False sinon.
    """
    # Check table summary patterns
    for pattern in GLOSSARY_CHUNK_PATTERNS:
        if pattern in chunk_id:
            return True

    # Check page range for LA-octobre
    if "LA-octobre" in source:
        min_page, max_page = GLOSSARY_PAGE_RANGE
        if min_page <= page <= max_page:
            return True

    return False


def retrieve_similar(
    db_path: Path,
    query_embedding: np.ndarray,
    top_k: int = 5,
    source_filter: str | None = None,
    glossary_boost: float | None = None,
) -> list[dict]:
    """
    Recupere les chunks les plus similaires a une query.

    Utilise la similarite cosinus entre embeddings.
    Les embeddings doivent etre normalises pour des resultats corrects.

    Args:
        db_path: Chemin de la base SQLite.
        query_embedding: Vecteur query normalise.
        top_k: Nombre de resultats a retourner.
        source_filter: Filtre optionnel sur le champ source (e.g. "Statuts", "LA-octobre").
            Utilise LIKE %filter% pour matching partiel.
        glossary_boost: Multiplicateur de score pour chunks glossaire (defaut None = pas de boost).
            Valeur recommandee: 3.5 pour questions definitions.
            Les chunks glossaire sont: pages 67-70 LA-octobre + table summaries.

    Returns:
        Liste de dicts avec id, text, source, page, score, is_glossary (si boost actif).

    Raises:
        FileNotFoundError: Si la base n'existe pas.
        ValueError: Si query_embedding a une mauvaise dimension.

    Example:
        >>> # Recherche globale (tous documents)
        >>> results = retrieve_similar(db_path, query_emb, top_k=5)
        >>> # Recherche filtree sur Statuts uniquement
        >>> results = retrieve_similar(db_path, query_emb, source_filter="Statuts")
        >>> # Recherche avec boost glossaire pour definitions
        >>> results = retrieve_similar(db_path, query_emb, glossary_boost=3.5)
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Normalize query if not already
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        query_embedding = query_embedding / query_norm

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Get embedding dimension from metadata
        cursor.execute("SELECT value FROM metadata WHERE key = 'embedding_dim'")
        row = cursor.fetchone()
        if row is None:
            raise ValueError("Database missing embedding_dim metadata")
        embedding_dim = int(row[0])

        if query_embedding.shape[0] != embedding_dim:
            raise ValueError(
                f"Query dim ({query_embedding.shape[0]}) != db dim ({embedding_dim})"
            )

        # Fetch embeddings (with optional source filter)
        if source_filter:
            cursor.execute(
                """SELECT id, text, source, page, tokens, metadata, embedding
                   FROM chunks WHERE source LIKE ?""",
                (f"%{source_filter}%",),
            )
        else:
            cursor.execute(
                "SELECT id, text, source, page, tokens, metadata, embedding FROM chunks"
            )
        rows = cursor.fetchall()

        results = []
        for row in rows:
            chunk_id, text, source, page, tokens, metadata_json, embedding_blob = row
            embedding = blob_to_embedding(embedding_blob, embedding_dim)

            # Cosine similarity (embeddings should be normalized)
            base_score = float(np.dot(query_embedding, embedding))

            # Apply glossary boost if enabled
            is_glossary = False
            if glossary_boost is not None:
                is_glossary = _is_glossary_chunk(chunk_id, source, page)
                if is_glossary:
                    score = base_score * glossary_boost
                else:
                    score = base_score
            else:
                score = base_score

            result = {
                "id": chunk_id,
                "text": text,
                "source": source,
                "page": page,
                "tokens": tokens,
                "metadata": json.loads(metadata_json) if metadata_json else {},
                "score": round(score, 4),
            }

            # Add glossary flag if boost is active
            if glossary_boost is not None:
                result["is_glossary"] = is_glossary
                if is_glossary:
                    result["base_score"] = round(base_score, 4)

            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    finally:
        conn.close()


def _normalize_accents(text: str) -> str:
    """
    Normalise les accents pour matching FTS5.

    FTS5 indexe avec accents, donc queries doivent aussi avoir accents.
    Cette fonction preserve les accents mais normalise la casse.

    Note: Si corpus indexe sans accents, utiliser unicodedata.normalize('NFD')
    et filtrer les Mn (combining marks).
    """
    return text.lower()


def _prepare_fts_query(query: str, use_stemming: bool = False) -> str:
    """
    Prepare une requete pour FTS5.

    Convertit "mot1 mot2 mot3" en "mot1 OR mot2 OR mot3" pour meilleur recall.
    Retire les caracteres speciaux FTS5 qui pourraient causer des erreurs.
    Normalise la casse (lowercase) pour matching case-insensitive.

    Note: use_stemming=False par defaut car le dictionnaire de synonymes
    (query_expansion.py) est plus precis que le stemming aveugle.

    Args:
        query: Texte de la requete.
        use_stemming: Ajouter stems FR (deconseille, utiliser synonymes).

    Returns:
        Requete FTS5 preparee.
    """
    special_chars = [
        '"',
        "'",
        "(",
        ")",
        "*",
        "-",
        "+",
        ":",
        "^",
        "~",
        "@",
        "?",
        "!",
        ",",
        ".",
        ";",
    ]
    clean_query = query
    for char in special_chars:
        clean_query = clean_query.replace(char, " ")

    # Normalize: lowercase pour matching case-insensitive
    clean_query = _normalize_accents(clean_query)

    words = [w.strip() for w in clean_query.split() if w.strip()]

    if not words:
        return ""

    # Stemming optionnel (deconseille - dictionnaire synonymes plus precis)
    if use_stemming:
        from scripts.pipeline.query_expansion import stem_word

        all_terms = set()
        for word in words:
            all_terms.add(word)
            stem = stem_word(word)
            if stem and len(stem) >= 3 and stem != word:
                all_terms.add(f"{stem}*")
        return " OR ".join(all_terms)

    return " OR ".join(words)


def search_bm25(
    db_path: Path,
    query: str,
    top_k: int = 20,
) -> list[dict]:
    """
    Recherche BM25 full-text avec FTS5.

    Args:
        db_path: Chemin de la base SQLite.
        query: Texte de la requete (mots-cles).
        top_k: Nombre de resultats a retourner.

    Returns:
        Liste de dicts avec id, text, source, page, bm25_score.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    fts_query = _prepare_fts_query(query)
    if not fts_query:
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        if cursor.fetchone() is None:
            raise ValueError("Database missing FTS5 index (chunks_fts)")

        cursor.execute(
            """
            SELECT
                c.id, c.text, c.source, c.page, c.tokens, c.metadata,
                bm25(chunks_fts) as bm25_score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (fts_query, top_k),
        )
        rows = cursor.fetchall()

        results = []
        for row in rows:
            chunk_id, text, source, page, tokens, metadata_json, bm25_score = row
            results.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": source,
                    "page": page,
                    "tokens": tokens,
                    "metadata": json.loads(metadata_json) if metadata_json else {},
                    "bm25_score": round(bm25_score, 4),
                }
            )

        return results

    except sqlite3.OperationalError as e:
        if "fts5" in str(e).lower() or "syntax" in str(e).lower():
            logger.warning(f"BM25 query error: {e}. Query: {query}")
            return []
        raise

    finally:
        conn.close()


def retrieve_hybrid(
    db_path: Path,
    query_embedding: np.ndarray,
    query_text: str,
    top_k: int = 5,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    initial_k: int = DEFAULT_INITIAL_K,
) -> list[dict]:
    """
    Recherche hybride combinant similarite vectorielle et BM25.

    Utilise Reciprocal Rank Fusion (RRF) pour fusionner les resultats.

    Args:
        db_path: Chemin de la base SQLite.
        query_embedding: Vecteur query normalise.
        query_text: Texte de la query pour BM25.
        top_k: Nombre de resultats finaux.
        vector_weight: Poids de la recherche vectorielle (defaut 0.7).
        bm25_weight: Poids de la recherche BM25 (defaut 0.3).
        initial_k: Nombre de resultats initiaux par methode.

    Returns:
        Liste de dicts avec id, text, source, page, hybrid_score.
    """
    vector_results = retrieve_similar(db_path, query_embedding, top_k=initial_k)

    try:
        bm25_results = search_bm25(db_path, query_text, top_k=initial_k)
    except (ValueError, sqlite3.OperationalError):
        logger.warning("BM25 search failed, using vector-only")
        bm25_results = []

    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, result in enumerate(vector_results):
        chunk_id = result["id"]
        rrf_score = vector_weight * (1.0 / (RRF_K + rank + 1))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        chunk_data[chunk_id] = result

    for rank, result in enumerate(bm25_results):
        chunk_id = result["id"]
        rrf_score = bm25_weight * (1.0 / (RRF_K + rank + 1))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = result

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids[:top_k]:
        chunk = chunk_data[chunk_id].copy()
        chunk["hybrid_score"] = round(rrf_scores[chunk_id], 6)
        if "score" in chunk_data[chunk_id]:
            chunk["vector_score"] = chunk_data[chunk_id]["score"]
        if "bm25_score" in chunk_data[chunk_id]:
            chunk["bm25_score"] = chunk_data[chunk_id]["bm25_score"]
        results.append(chunk)

    return results


def retrieve_hybrid_rerank(
    db_path: Path,
    query_embedding: np.ndarray,
    query_text: str,
    reranker: "CrossEncoder",
    top_k_retrieve: int = DEFAULT_INITIAL_K,
    top_k_final: int = 5,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    use_query_expansion: bool = True,
    expanded_query_embedding: "np.ndarray | None" = None,
) -> list[dict]:
    """
    Recherche hybride avec reranking cross-encoder (Funnel Mode).

    Pipeline 2025 enterprise standard pour recall optimal:
    1. Query expansion (synonymes chess FR)
    2. Retrieve top_k_retrieve (100) avec hybrid search (BM25 + vector + RRF)
    3. Rerank pool complet avec cross-encoder
    4. Return top_k_final (5)

    Research: Databricks +48% recall (74%->89%), VectorHub/Superlinked Funnel Mode.

    ISO Reference:
        - ISO/IEC 25010 - Performance efficiency (Recall >= 90%)

    Args:
        db_path: Chemin de la base SQLite.
        query_embedding: Vecteur query normalise.
        query_text: Texte de la query pour BM25 et reranking.
        reranker: CrossEncoder charge pour reranking.
        top_k_retrieve: Pool initial avant rerank (defaut 100 - Funnel Mode).
        top_k_final: Nombre de resultats finaux apres reranking (defaut 5).
        vector_weight: Poids de la recherche vectorielle (defaut 0.4).
        bm25_weight: Poids de la recherche BM25 (defaut 0.6).
        use_query_expansion: Utiliser l'expansion de query (defaut True).
        expanded_query_embedding: Embedding de la query expandue (optionnel).
            Si fourni, utilise pour la recherche vectorielle.

    Returns:
        Liste de dicts avec id, text, source, page, hybrid_score, rerank_score.

    Example:
        >>> from scripts.pipeline.reranker import load_reranker
        >>> reranker = load_reranker()
        >>> results = retrieve_hybrid_rerank(
        ...     db_path, query_emb, "toucher-jouer", reranker
        ... )
    """
    from scripts.pipeline.reranker import rerank

    # Step 0: Query expansion for BM25 (adds synonyms)
    bm25_query = query_text
    if use_query_expansion:
        from scripts.pipeline.query_expansion import expand_query_bm25

        expanded = expand_query_bm25(query_text)
        if expanded:
            bm25_query = f"{query_text} {expanded}"

    # Use expanded embedding if provided, otherwise use original
    vector_embedding = (
        expanded_query_embedding
        if expanded_query_embedding is not None
        else query_embedding
    )

    # Step 1: Hybrid search (BM25 + vector with RRF)
    hybrid_results = retrieve_hybrid(
        db_path,
        vector_embedding,  # Use expanded embedding if available
        bm25_query,  # Use expanded query for BM25
        top_k=top_k_retrieve,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    if not hybrid_results:
        return []

    # Step 2: Rerank with cross-encoder (use original query)
    reranked = rerank(
        query=query_text,  # Use original query for reranking
        chunks=hybrid_results,
        model=reranker,
        top_k=top_k_final,
        content_key="text",
    )

    return reranked


# =============================================================================
# Smart Retrieve (auto source_filter)
# =============================================================================

# Mapping keywords -> source_filter (patterns SPÉCIFIQUES - éviter faux positifs)
# IMPORTANT: Patterns génériques comme "rapide", "cadence" causent des régressions
# car ils matchent des questions sur les championnats FFE (A03, F02, E02)
SOURCE_FILTER_PATTERNS: dict[str, str] = {
    # Statuts FFE (patterns explicites uniquement)
    "selon les statuts": "Statuts",
    "but de la ffe": "Statuts",
    "objectifs de la ffe": "Statuts",
    # LA (Livre de l'Arbitre) - patterns spécifiques règles du jeu
    "cadences officielles": "LA-octobre",
    "règles du jeu": "LA-octobre",
    "annexe a": "LA-octobre",
    "annexe b": "LA-octobre",
    "toucher-jouer": "LA-octobre",
    "pièce touchée": "LA-octobre",
    "coup illégal": "LA-octobre",
    "position illégale": "LA-octobre",
    # Code éthique
    "éthique": "Code_ethique",
    "déontologie": "Code_ethique",
}


def smart_retrieve(
    db_path: Path,
    query_embedding: np.ndarray,
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Recherche intelligente avec auto-detection du source_filter.

    Analyse le texte de la query pour detecter des patterns connus
    et applique automatiquement le filtre source approprie.
    Fallback sur recherche globale si aucun pattern detecte.

    Args:
        db_path: Chemin de la base SQLite.
        query_embedding: Vecteur query normalise.
        query_text: Texte de la query pour detection patterns.
        top_k: Nombre de resultats a retourner.

    Returns:
        Liste de dicts avec id, text, source, page, score.

    Example:
        >>> # Auto-detecte "Statuts" pour questions FFE
        >>> results = smart_retrieve(db, emb, "Quels sont les objectifs de la FFE?")
        >>> # Auto-detecte "LA-octobre" pour cadences
        >>> results = smart_retrieve(db, emb, "Cadences parties rapides?")
    """
    query_lower = query_text.lower()

    # Detect source filter from patterns
    selected_filter = None
    for keyword, source in SOURCE_FILTER_PATTERNS.items():
        if keyword in query_lower:
            selected_filter = source
            logger.debug(f"smart_retrieve: pattern '{keyword}' -> filter '{source}'")
            break

    return retrieve_similar(
        db_path,
        query_embedding,
        top_k=top_k,
        source_filter=selected_filter,
    )


# =============================================================================
# Definition Question Detection (for auto glossary boost)
# =============================================================================

# Patterns indiquant une question de definition
DEFINITION_QUERY_PATTERNS = [
    # Questions directes
    "qu'est-ce que",
    "qu'est ce que",
    "c'est quoi",
    "c est quoi",
    "définition de",
    "definition de",
    "que signifie",
    "que veut dire",
    "comment définir",
    "comment definir",
    # Termes techniques souvent cherchés dans le glossaire
    "selon les règles",
    "selon le règlement",
    "terme officiel",
    "vocabulaire",
    # Patterns anglais (INTL)
    "what is",
    "definition of",
    "meaning of",
]


def _is_definition_query(query_text: str) -> bool:
    """
    Detecte si une query est une question de definition.

    Les questions de definition beneficient du glossary boost car
    le glossaire DNA 2025 est la source canonique pour les termes officiels.

    Args:
        query_text: Texte de la query.

    Returns:
        True si question de definition, False sinon.
    """
    query_lower = query_text.lower()
    return any(pattern in query_lower for pattern in DEFINITION_QUERY_PATTERNS)


def retrieve_with_glossary_boost(
    db_path: Path,
    query_embedding: np.ndarray,
    query_text: str,
    top_k: int = 5,
    boost_factor: float = DEFAULT_GLOSSARY_BOOST,
    auto_detect: bool = True,
) -> list[dict]:
    """
    Recherche avec boost automatique du glossaire pour questions de definition.

    Combine source_filter (smart_retrieve) + glossary_boost pour maximiser
    la precision sur les questions de terminologie officielle.

    Features:
        - Auto-detection des questions de definition (patterns FR/EN)
        - Boost x3.5 pour chunks glossaire (pages 67-70 LA-octobre + table summaries)
        - Fallback intelligent: si boost actif mais 0 chunk glossaire -> global search
        - Logging structure pour analyse post-mortem (logs/retrieval_log.txt)

    Args:
        db_path: Chemin de la base SQLite.
        query_embedding: Vecteur query normalise.
        query_text: Texte de la query pour detection patterns.
        top_k: Nombre de resultats a retourner.
        boost_factor: Multiplicateur glossaire (defaut 3.5).
        auto_detect: Si True, applique le boost uniquement si question definition.
            Si False, applique toujours le boost.

    Returns:
        Liste de dicts avec id, text, source, page, score, is_glossary, boost_applied,
        fallback_used (si applicable).

    Example:
        >>> # Auto-detection: boost applique si question definition
        >>> results = retrieve_with_glossary_boost(db, emb, "Qu'est-ce que le roque?")
        >>> # Boost force (toutes queries)
        >>> results = retrieve_with_glossary_boost(db, emb, "roque", auto_detect=False)
    """
    # Detect source filter from smart_retrieve patterns
    query_lower = query_text.lower()
    selected_filter = None
    for keyword, source in SOURCE_FILTER_PATTERNS.items():
        if keyword in query_lower:
            selected_filter = source
            break

    # Determine if glossary boost should be applied
    is_definition = _is_definition_query(query_text)
    apply_boost = not auto_detect or is_definition

    # Detect matched pattern for logging
    matched_pattern = None
    if is_definition:
        for pattern in DEFINITION_QUERY_PATTERNS:
            if pattern in query_lower:
                matched_pattern = pattern
                break

    # Retrieve with optional glossary boost (fetch larger pool for potential fallback)
    fetch_k = top_k * 2 if apply_boost else top_k
    results = retrieve_similar(
        db_path,
        query_embedding,
        top_k=fetch_k,
        source_filter=selected_filter,
        glossary_boost=boost_factor if apply_boost else None,
    )

    # Count glossary hits in results
    glossary_hits = sum(1 for r in results if r.get("is_glossary", False))
    fallback_used = False

    # Fallback: if boost active but no glossary chunks returned -> retry without boost
    if apply_boost and glossary_hits == 0 and len(results) > 0:
        fallback_used = True
        results = retrieve_similar(
            db_path,
            query_embedding,
            top_k=top_k,
            source_filter=selected_filter,
            glossary_boost=None,
        )

    # Truncate to top_k
    results = results[:top_k]

    # Add metadata flags to results
    for result in results:
        result["boost_applied"] = apply_boost and not fallback_used
        result["fallback_used"] = fallback_used

    # Structured logging (JSONL file for analytics)
    log_retrieval(
        query=query_text,
        is_definition=is_definition,
        matched_pattern=matched_pattern,
        boost_applied=apply_boost,
        boost_factor=boost_factor,
        source_filter=selected_filter,
        results_count=len(results),
        glossary_hits=glossary_hits,
        fallback_used=fallback_used,
        top_scores=[round(r.get("score", 0), 4) for r in results[:3]],
        top_sources=[r.get("source", "?")[:20] for r in results[:3]],
    )

    return results


# Type hint for CrossEncoder (imported only for type checking)
if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder
