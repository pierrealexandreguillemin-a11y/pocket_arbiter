"""
Export SqliteVectorStore - Pocket Arbiter

Ce module exporte les chunks et embeddings au format SQLite compatible
avec le Google AI Edge RAG SDK (SqliteVectorStore).

ISO Reference:
    - ISO/IEC 12207 S7.3.3 - Implementation
    - ISO/IEC 25010 S4.2 - Performance efficiency
    - ISO/IEC 42001 - AI model traceability

Dependencies:
    - numpy >= 1.24.0

Usage:
    python export_sdk.py --chunks chunks_fr.json --embeddings embeddings_fr.npy --output corpus_fr.db
    python export_sdk.py --chunks chunks_intl.json --embeddings embeddings_intl.npy --output corpus_intl.db

Example:
    >>> from scripts.pipeline.export_sdk import create_vector_db, retrieve_similar
    >>> create_vector_db(db_path, chunks, embeddings)
    >>> results = retrieve_similar(db_path, query_embedding, top_k=5)
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Constants ---

SCHEMA_VERSION = "2.0"
EMBEDDING_DTYPE = np.float32

# Hybrid search parameters (from research - CHUNKING_STRATEGY.md)
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_BM25_WEIGHT = 0.3
RRF_K = 60  # Reciprocal Rank Fusion constant


# --- Database Schema ---


def _get_schema_sql() -> str:
    """
    Retourne le schema SQL pour SqliteVectorStore.

    Le schema est compatible avec le Google AI Edge RAG SDK.
    Les embeddings sont stockes en BLOB (float32 array).

    Returns:
        Schema SQL a executer.
    """
    return """
    -- Metadata table for version tracking
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    -- Main chunks table with embeddings
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        page INTEGER NOT NULL,
        tokens INTEGER NOT NULL,
        metadata TEXT,
        embedding BLOB NOT NULL
    );

    -- Index for faster retrieval
    CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
    CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page);

    -- FTS5 virtual table for BM25 full-text search
    -- content=chunks means it references the chunks table (no duplicate storage)
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text,
        content='chunks',
        content_rowid='rowid'
    );

    -- Triggers to keep FTS index synchronized with chunks table
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
    END;
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    END;
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
    END;
    """


# --- Embedding Serialization ---


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """
    Convertit un embedding numpy en BLOB SQLite.

    Args:
        embedding: Vecteur numpy float32 de dimension D.

    Returns:
        Representation binaire (bytes) du vecteur.

    Raises:
        ValueError: Si l'embedding n'est pas 1D.

    Example:
        >>> emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        >>> blob = embedding_to_blob(emb)
        >>> len(blob)
        12
    """
    if embedding.ndim != 1:
        raise ValueError(f"Embedding must be 1D, got shape {embedding.shape}")

    return embedding.astype(EMBEDDING_DTYPE).tobytes()


def blob_to_embedding(blob: bytes, dim: int) -> np.ndarray:
    """
    Convertit un BLOB SQLite en embedding numpy.

    Args:
        blob: Bytes du BLOB SQLite.
        dim: Dimension attendue de l'embedding.

    Returns:
        Vecteur numpy float32.

    Raises:
        ValueError: Si la taille du blob ne correspond pas a dim.

    Example:
        >>> blob = struct.pack('3f', 0.1, 0.2, 0.3)
        >>> emb = blob_to_embedding(blob, 3)
        >>> emb.shape
        (3,)
    """
    expected_size = dim * 4  # float32 = 4 bytes
    if len(blob) != expected_size:
        raise ValueError(f"Blob size {len(blob)} != expected {expected_size}")

    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)


# --- Database Operations ---


def create_vector_db(
    db_path: Path,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_dim: int | None = None,
) -> dict:
    """
    Cree une base SQLite avec chunks et embeddings.

    Args:
        db_path: Chemin de la base SQLite a creer.
        chunks: Liste de chunks conformes au CHUNK_SCHEMA.md.
        embeddings: Array numpy (N, dim) des embeddings.
        embedding_dim: Dimension des embeddings (auto-detectee si None).

    Returns:
        Rapport de creation avec total_chunks, db_size_mb, etc.

    Raises:
        ValueError: Si le nombre de chunks != nombre d'embeddings.
        ValueError: Si chunks contient des champs invalides.

    Example:
        >>> report = create_vector_db(Path("corpus.db"), chunks, embeddings)
        >>> report["total_chunks"]
        100
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})"
        )

    if len(chunks) == 0:
        raise ValueError("Cannot create empty database")

    # Auto-detect embedding dimension
    if embedding_dim is None:
        embedding_dim = embeddings.shape[1]

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    logger.info(f"Creating vector database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Create schema
        cursor.executescript(_get_schema_sql())

        # Insert metadata
        cursor.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )
        cursor.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("embedding_dim", str(embedding_dim)),
        )
        cursor.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("total_chunks", str(len(chunks))),
        )

        # Insert chunks with embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            _validate_chunk(chunk)

            metadata_json = json.dumps(chunk.get("metadata", {}))
            embedding_blob = embedding_to_blob(embedding)

            cursor.execute(
                """
                INSERT INTO chunks (id, text, source, page, tokens, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["id"],
                    chunk["text"],
                    chunk["source"],
                    chunk["page"],
                    chunk["tokens"],
                    metadata_json,
                    embedding_blob,
                ),
            )

        conn.commit()
        logger.info(f"Inserted {len(chunks)} chunks into database")

    finally:
        conn.close()

    # Get database size
    db_size_bytes = db_path.stat().st_size
    db_size_mb = db_size_bytes / (1024 * 1024)

    return {
        "db_path": str(db_path),
        "total_chunks": len(chunks),
        "embedding_dim": embedding_dim,
        "db_size_bytes": db_size_bytes,
        "db_size_mb": round(db_size_mb, 2),
        "schema_version": SCHEMA_VERSION,
    }


def _validate_chunk(chunk: dict) -> None:
    """Valide qu'un chunk contient les champs requis."""
    required_fields = ["id", "text", "source", "page", "tokens"]
    missing = [f for f in required_fields if f not in chunk]
    if missing:
        raise ValueError(f"Chunk missing required fields: {missing}")


def retrieve_similar(
    db_path: Path,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """
    Recupere les chunks les plus similaires a une query.

    Utilise la similarite cosinus entre embeddings.
    Les embeddings doivent etre normalises pour des resultats corrects.

    Args:
        db_path: Chemin de la base SQLite.
        query_embedding: Vecteur query normalise.
        top_k: Nombre de resultats a retourner.

    Returns:
        Liste de dicts avec id, text, source, page, score.

    Raises:
        FileNotFoundError: Si la base n'existe pas.
        ValueError: Si query_embedding a une mauvaise dimension.

    Example:
        >>> results = retrieve_similar(db_path, query_emb, top_k=5)
        >>> results[0]["score"]
        0.95
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

        # Fetch all embeddings and compute similarities
        cursor.execute(
            "SELECT id, text, source, page, tokens, metadata, embedding FROM chunks"
        )
        rows = cursor.fetchall()

        results = []
        for row in rows:
            chunk_id, text, source, page, tokens, metadata_json, embedding_blob = row
            embedding = blob_to_embedding(embedding_blob, embedding_dim)

            # Cosine similarity (embeddings should be normalized)
            score = float(np.dot(query_embedding, embedding))

            results.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": source,
                    "page": page,
                    "tokens": tokens,
                    "metadata": json.loads(metadata_json) if metadata_json else {},
                    "score": round(score, 4),
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    finally:
        conn.close()


def _prepare_fts_query(query: str) -> str:
    """
    Prepare une requete pour FTS5.

    Convertit "mot1 mot2 mot3" en "mot1 OR mot2 OR mot3" pour meilleur recall.
    Retire les caracteres speciaux FTS5 qui pourraient causer des erreurs.

    Args:
        query: Requete texte brute.

    Returns:
        Requete formatee pour FTS5 MATCH.
    """
    # Remove FTS5 special characters and punctuation
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

    # Split into words and filter empty
    words = [w.strip() for w in clean_query.split() if w.strip()]

    if not words:
        return ""

    # Join with OR for better recall (any word matches)
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

    Example:
        >>> results = search_bm25(db_path, "toucher jouer piece", top_k=10)
        >>> results[0]["bm25_score"]
        -5.2
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Prepare FTS5 query
    fts_query = _prepare_fts_query(query)
    if not fts_query:
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check if FTS table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        if cursor.fetchone() is None:
            raise ValueError("Database missing FTS5 index (chunks_fts)")

        # BM25 search - FTS5 returns negative scores (more negative = better match)
        # We use MATCH for FTS5 query syntax
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
        # Handle FTS query syntax errors gracefully
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
    initial_k: int = 20,
) -> list[dict]:
    """
    Recherche hybride combinant similarite vectorielle et BM25.

    Utilise Reciprocal Rank Fusion (RRF) pour fusionner les resultats.
    Score RRF = sum(1 / (k + rank)) pour chaque systeme de ranking.

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

    Example:
        >>> results = retrieve_hybrid(db_path, query_emb, "toucher jouer", top_k=5)
        >>> results[0]["hybrid_score"]
        0.0234
    """
    # Get vector search results
    vector_results = retrieve_similar(db_path, query_embedding, top_k=initial_k)

    # Get BM25 results
    try:
        bm25_results = search_bm25(db_path, query_text, top_k=initial_k)
    except (ValueError, sqlite3.OperationalError):
        # If BM25 fails, fall back to vector-only
        logger.warning("BM25 search failed, using vector-only")
        bm25_results = []

    # Build RRF scores
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    # Add vector results with RRF score
    for rank, result in enumerate(vector_results):
        chunk_id = result["id"]
        rrf_score = vector_weight * (1.0 / (RRF_K + rank + 1))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        chunk_data[chunk_id] = result

    # Add BM25 results with RRF score
    for rank, result in enumerate(bm25_results):
        chunk_id = result["id"]
        rrf_score = bm25_weight * (1.0 / (RRF_K + rank + 1))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = result

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Build final results
    results = []
    for chunk_id in sorted_ids[:top_k]:
        chunk = chunk_data[chunk_id].copy()
        chunk["hybrid_score"] = round(rrf_scores[chunk_id], 6)
        # Preserve original scores if available
        if "score" in chunk_data[chunk_id]:
            chunk["vector_score"] = chunk_data[chunk_id]["score"]
        if "bm25_score" in chunk_data[chunk_id]:
            chunk["bm25_score"] = chunk_data[chunk_id]["bm25_score"]
        results.append(chunk)

    return results


def rebuild_fts_index(db_path: Path) -> int:
    """
    Reconstruit l'index FTS5 pour une base existante.

    Utile pour les bases creees avant l'ajout du support BM25.

    Args:
        db_path: Chemin de la base SQLite.

    Returns:
        Nombre de chunks indexes.

    Raises:
        FileNotFoundError: Si la base n'existe pas.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check if FTS table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        fts_exists = cursor.fetchone() is not None

        if not fts_exists:
            # Create FTS5 table
            logger.info("Creating FTS5 index...")
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='rowid'
                )
                """
            )

            # Create triggers
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                END
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END
                """
            )

        # Rebuild the FTS index from existing data
        logger.info("Populating FTS5 index from chunks...")
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

        # Count indexed documents
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]

        conn.commit()
        logger.info(f"FTS5 index rebuilt: {count} chunks indexed")

        return count

    finally:
        conn.close()


def _check_table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    )
    return cursor.fetchone() is not None


def _validate_metadata(metadata: dict, errors: list[str]) -> None:
    """Validate required metadata fields."""
    if "embedding_dim" not in metadata:
        errors.append("Missing embedding_dim in metadata")
    if "total_chunks" not in metadata:
        errors.append("Missing total_chunks in metadata")


def _validate_chunk_counts(
    metadata: dict, actual_chunks: int, expected_chunks: int | None, errors: list[str]
) -> None:
    """Validate chunk count matches expected values."""
    if "total_chunks" in metadata:
        expected_from_meta = int(metadata["total_chunks"])
        if actual_chunks != expected_from_meta:
            errors.append(
                f"Chunk count mismatch: {actual_chunks} != metadata({expected_from_meta})"
            )
    if expected_chunks is not None and actual_chunks != expected_chunks:
        errors.append(
            f"Chunk count mismatch: {actual_chunks} != expected({expected_chunks})"
        )


def _validate_embeddings(
    cursor: sqlite3.Cursor, metadata: dict, errors: list[str]
) -> None:
    """Validate embedding sizes match expected dimension."""
    if "embedding_dim" not in metadata:
        return
    embedding_dim = int(metadata["embedding_dim"])
    expected_blob_size = embedding_dim * 4  # float32
    cursor.execute("SELECT id, LENGTH(embedding) FROM chunks LIMIT 100")
    for chunk_id, blob_size in cursor.fetchall():
        if blob_size != expected_blob_size:
            errors.append(
                f"Invalid embedding size for {chunk_id}: {blob_size} != {expected_blob_size}"
            )


def _validate_fts_index(
    cursor: sqlite3.Cursor, actual_chunks: int, errors: list[str]
) -> None:
    """Validate FTS5 index exists and has correct count."""
    if not _check_table_exists(cursor, "chunks_fts"):
        logger.warning("FTS5 index missing - BM25 search unavailable")
        return
    cursor.execute("SELECT COUNT(*) FROM chunks_fts")
    fts_count = cursor.fetchone()[0]
    if fts_count != actual_chunks:
        errors.append(
            f"FTS index count mismatch: {fts_count} != chunks({actual_chunks})"
        )


def validate_export(db_path: Path, expected_chunks: int | None = None) -> list[str]:
    """
    Valide l'integrite d'une base SQLite exportee.

    Verifie:
    - Schema correct (tables metadata, chunks, chunks_fts)
    - Metadata coherente
    - Nombre de chunks attendu
    - Embeddings valides (taille correcte)
    - FTS5 index present

    Args:
        db_path: Chemin de la base SQLite.
        expected_chunks: Nombre de chunks attendu (optionnel).

    Returns:
        Liste d'erreurs (vide si valide).

    Example:
        >>> errors = validate_export(db_path, expected_chunks=100)
        >>> if errors:
        ...     print("Validation failed:", errors)
    """
    errors: list[str] = []

    if not db_path.exists():
        return [f"Database not found: {db_path}"]

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        if not _check_table_exists(cursor, "metadata"):
            return ["Missing metadata table"]
        if not _check_table_exists(cursor, "chunks"):
            return ["Missing chunks table"]

        cursor.execute("SELECT key, value FROM metadata")
        metadata = dict(cursor.fetchall())
        _validate_metadata(metadata, errors)

        cursor.execute("SELECT COUNT(*) FROM chunks")
        actual_chunks = cursor.fetchone()[0]
        _validate_chunk_counts(metadata, actual_chunks, expected_chunks, errors)
        _validate_embeddings(cursor, metadata, errors)
        _validate_fts_index(cursor, actual_chunks, errors)

    except sqlite3.Error as e:
        errors.append(f"SQLite error: {e}")
    finally:
        conn.close()

    return errors


def export_corpus(
    chunks_file: Path,
    embeddings_file: Path,
    output_db: Path,
) -> dict:
    """
    Pipeline complet: charge chunks et embeddings, exporte DB.

    Args:
        chunks_file: Fichier JSON des chunks.
        embeddings_file: Fichier numpy des embeddings (.npy).
        output_db: Fichier SQLite de sortie.

    Returns:
        Rapport d'export avec metriques.

    Raises:
        FileNotFoundError: Si fichiers sources n'existent pas.
        ValueError: Si nombre de chunks != nombre d'embeddings.

    Example:
        >>> report = export_corpus(
        ...     Path("chunks_fr.json"),
        ...     Path("embeddings_fr.npy"),
        ...     Path("corpus_fr.db")
        ... )
        >>> report["total_chunks"]
        2047
    """
    from scripts.pipeline.utils import get_timestamp, load_json, save_json

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    logger.info(f"Loading chunks from: {chunks_file}")
    chunks_data = load_json(chunks_file)
    chunks = chunks_data.get("chunks", [])

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_file}")

    logger.info(f"Loading embeddings from: {embeddings_file}")
    embeddings = np.load(embeddings_file)

    logger.info(f"Chunks: {len(chunks)}, Embeddings: {embeddings.shape}")

    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunks ({len(chunks)}) != embeddings ({len(embeddings)})")

    # Create database
    report = create_vector_db(output_db, chunks, embeddings)

    # Validate export
    errors = validate_export(output_db, expected_chunks=len(chunks))
    if errors:
        logger.warning(f"Validation errors: {errors}")
        report["validation_errors"] = errors
    else:
        logger.info("Export validation passed")
        report["validation_errors"] = []

    # Add metadata
    report["chunks_file"] = str(chunks_file)
    report["embeddings_file"] = str(embeddings_file)
    report["timestamp"] = get_timestamp()

    # Save report
    report_file = output_db.with_suffix(".report.json")
    save_json(report, report_file)
    logger.info(f"Report saved: {report_file}")

    return report


# --- CLI ---


def main() -> None:
    """Point d'entree CLI pour l'export SqliteVectorStore."""
    parser = argparse.ArgumentParser(
        description="Export SqliteVectorStore pour Pocket Arbiter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_sdk.py --chunks corpus/processed/chunks_fr.json --embeddings corpus/processed/embeddings_fr.npy --output corpus/processed/corpus_fr.db
    python export_sdk.py --chunks corpus/processed/chunks_intl.json --embeddings corpus/processed/embeddings_intl.npy --output corpus/processed/corpus_intl.db
        """,
    )

    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        required=True,
        help="Fichier JSON des chunks (ex: chunks_fr.json)",
    )

    parser.add_argument(
        "--embeddings",
        "-e",
        type=Path,
        required=True,
        help="Fichier numpy des embeddings (ex: embeddings_fr.npy)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Fichier SQLite de sortie (ex: corpus_fr.db)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher logs detailles",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Chunks: {args.chunks}")
    logger.info(f"Embeddings: {args.embeddings}")
    logger.info(f"Output: {args.output}")

    report = export_corpus(args.chunks, args.embeddings, args.output)

    logger.info("=" * 50)
    logger.info(f"Total chunks: {report['total_chunks']}")
    logger.info(f"Embedding dim: {report['embedding_dim']}")
    logger.info(f"DB size: {report['db_size_mb']} MB")
    logger.info(f"Output: {report['db_path']}")

    if report.get("validation_errors"):
        logger.error(f"Validation errors: {report['validation_errors']}")
    else:
        logger.info("Validation: PASSED")


if __name__ == "__main__":
    main()
