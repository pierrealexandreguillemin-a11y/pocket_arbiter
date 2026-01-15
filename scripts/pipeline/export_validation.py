"""
Export Validation - Pocket Arbiter

Fonctions de validation pour bases SQLite exportees.

ISO Reference:
    - ISO/IEC 25010 S4.2 - Data quality
"""

import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
