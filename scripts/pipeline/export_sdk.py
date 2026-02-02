"""
Export SqliteVectorStore - Pocket Arbiter

Ce module exporte les chunks et embeddings au format SQLite compatible
avec le Google AI Edge RAG SDK (SqliteVectorStore).

ISO Reference:
    - ISO/IEC 12207 S7.3.3 - Implementation
    - ISO/IEC 25010 S4.2 - Performance efficiency
    - ISO/IEC 42001 - AI model traceability

Usage:
    python export_sdk.py --chunks chunks_fr.json --embeddings embeddings_fr.npy --output corpus_fr.db
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

from scripts.pipeline.export_search import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_VECTOR_WEIGHT,
    RRF_K,
    retrieve_hybrid,
    retrieve_similar,
    search_bm25,
)
from scripts.pipeline.export_serialization import (
    EMBEDDING_DTYPE,
    blob_to_embedding,
    embedding_to_blob,
)
from scripts.pipeline.export_validation import validate_export

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "create_vector_db",
    "retrieve_similar",
    "retrieve_hybrid",
    "search_bm25",
    "validate_export",
    "export_corpus",
    "embedding_to_blob",
    "blob_to_embedding",
    "rebuild_fts_index",
    "EMBEDDING_DTYPE",
    "DEFAULT_VECTOR_WEIGHT",
    "DEFAULT_BM25_WEIGHT",
    "RRF_K",
]

# --- Constants ---

SCHEMA_VERSION = "2.0"


# --- Database Schema ---


def _get_schema_sql() -> str:
    """Retourne le schema SQL pour SqliteVectorStore."""
    return """
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        page INTEGER NOT NULL,
        tokens INTEGER NOT NULL,
        metadata TEXT,
        embedding BLOB NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
    CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page);

    -- FTS5 avec tokenizer FR: unicode61 remove_diacritics 2
    -- Permet recherche insensible aux accents (cafe = café)
    -- Ref: https://sqlite.org/fts5.html#tokenizers
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text,
        content='chunks',
        content_rowid='rowid',
        tokenize='unicode61 remove_diacritics 2'
    );

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


def _validate_chunk(chunk: dict) -> None:
    """Valide qu'un chunk contient les champs requis."""
    required_fields = ["id", "text", "source", "page", "tokens"]
    missing = [f for f in required_fields if f not in chunk]
    if missing:
        raise ValueError(f"Chunk missing required fields: {missing}")


def _insert_metadata(
    cursor: sqlite3.Cursor,
    embedding_dim: int,
    total_chunks: int,
    model_id: str | None,
) -> None:
    """Insert metadata rows into the database."""
    for key, value in [
        ("schema_version", SCHEMA_VERSION),
        ("embedding_dim", str(embedding_dim)),
        ("total_chunks", str(total_chunks)),
    ]:
        cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", (key, value))
    if model_id:
        cursor.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("model_id", model_id),
        )


def _insert_chunks(
    cursor: sqlite3.Cursor,
    chunks: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Insert chunk rows with embeddings into the database."""
    core_fields = {"id", "text", "source", "page", "tokens"}
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        _validate_chunk(chunk)
        metadata = {
            k: v for k, v in chunk.items() if k not in core_fields and v is not None
        }
        if "metadata" in chunk and isinstance(chunk["metadata"], dict):
            metadata.update(chunk["metadata"])
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
                json.dumps(metadata),
                embedding_to_blob(embedding),
            ),
        )


def create_vector_db(
    db_path: Path,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_dim: int | None = None,
    model_id: str | None = None,
) -> dict:
    """
    Cree une base SQLite avec chunks et embeddings.

    Args:
        db_path: Chemin de la base SQLite a creer.
        chunks: Liste de chunks conformes au CHUNK_SCHEMA.md.
        embeddings: Array numpy (N, dim) des embeddings.
        embedding_dim: Dimension des embeddings (auto-detectee si None).
        model_id: Identifiant du modele d'embedding (ISO 42001 tracabilite).

    Returns:
        Rapport de creation avec total_chunks, db_size_mb, etc.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})"
        )

    if len(chunks) == 0:
        raise ValueError("Cannot create empty database")

    if embedding_dim is None:
        embedding_dim = embeddings.shape[1]

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    logger.info(f"Creating vector database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.executescript(_get_schema_sql())
        _insert_metadata(cursor, embedding_dim, len(chunks), model_id)
        _insert_chunks(cursor, chunks, embeddings)
        conn.commit()
        logger.info(f"Inserted {len(chunks)} chunks into database")
    finally:
        conn.close()

    db_size_bytes = db_path.stat().st_size
    db_size_mb = db_size_bytes / (1024 * 1024)

    report = {
        "db_path": str(db_path),
        "total_chunks": len(chunks),
        "embedding_dim": embedding_dim,
        "db_size_bytes": db_size_bytes,
        "db_size_mb": round(db_size_mb, 2),
        "schema_version": SCHEMA_VERSION,
    }
    if model_id:
        report["model_id"] = model_id
    return report


def rebuild_fts_index(db_path: Path) -> int:
    """
    Reconstruit l'index FTS5 pour une base existante.

    Args:
        db_path: Chemin de la base SQLite.

    Returns:
        Nombre de chunks indexes.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        fts_exists = cursor.fetchone() is not None

        if not fts_exists:
            logger.info("Creating FTS5 index with FR tokenizer...")
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='rowid',
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
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

        logger.info("Populating FTS5 index from chunks...")
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]

        conn.commit()
        logger.info(f"FTS5 index rebuilt: {count} chunks indexed")

        return count

    finally:
        conn.close()


def export_corpus(
    chunks_file: Path,
    embeddings_file: Path,
    output_db: Path,
    model_id: str | None = None,
) -> dict:
    """
    Pipeline complet: charge chunks et embeddings, exporte DB.

    Args:
        chunks_file: Fichier JSON des chunks.
        embeddings_file: Fichier numpy des embeddings (.npy).
        output_db: Fichier SQLite de sortie.
        model_id: Identifiant du modele d'embedding (ISO 42001 tracabilite).

    Returns:
        Rapport d'export avec metriques.
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

    report = create_vector_db(output_db, chunks, embeddings, model_id=model_id)

    errors = validate_export(output_db, expected_chunks=len(chunks))
    if errors:
        logger.warning(f"Validation errors: {errors}")
        report["validation_errors"] = errors
    else:
        logger.info("Export validation passed")
        report["validation_errors"] = []

    report["chunks_file"] = str(chunks_file)
    report["embeddings_file"] = str(embeddings_file)
    report["timestamp"] = get_timestamp()

    report_file = output_db.with_suffix(".report.json")
    save_json(report, report_file)
    logger.info(f"Report saved: {report_file}")

    return report


# --- CLI ---


def main() -> None:
    """Point d'entree CLI pour l'export SqliteVectorStore."""
    parser = argparse.ArgumentParser(
        description="Export SqliteVectorStore pour Pocket Arbiter",
    )

    parser.add_argument(
        "--chunks", "-c", type=Path, required=True, help="Fichier JSON des chunks"
    )
    parser.add_argument(
        "--embeddings", "-e", type=Path, required=True, help="Fichier numpy embeddings"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Fichier SQLite de sortie"
    )
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        default="google/embeddinggemma-300m-qat-q4_0-unquantized",
        help="Model ID for traceability (default: QAT model)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs detailles")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    report = export_corpus(
        args.chunks, args.embeddings, args.output, model_id=args.model_id
    )

    logger.info("=" * 50)
    logger.info(f"Total chunks: {report['total_chunks']}")
    logger.info(f"Embedding dim: {report['embedding_dim']}")
    logger.info(f"Model ID: {report.get('model_id', 'not specified')}")
    logger.info(f"DB size: {report['db_size_mb']} MB")

    if report.get("validation_errors"):
        logger.error(f"Validation errors: {report['validation_errors']}")
    else:
        logger.info("Validation: PASSED")


if __name__ == "__main__":
    main()
