#!/usr/bin/env python3
"""
Update existing SQLite databases with model_id metadata.

This script patches existing corpus databases to add model traceability
required for the QLoRA fallback strategy.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Model traceability
    - ISO/IEC 12207 - Configuration management

Usage:
    python -m scripts.pipeline.update_db_model_id \
        --db corpus/processed/corpus_mode_b_fr.db \
        --model-id google/embeddinggemma-300m
"""

import argparse
import sqlite3
from pathlib import Path

# Default model for existing DBs - QAT (coherence QLoRA pipeline)
DEFAULT_MODEL_ID = "google/embeddinggemma-300m-qat-q4_0-unquantized"


def update_model_id(db_path: Path, model_id: str) -> dict:
    """
    Add or update model_id in database metadata.

    Args:
        db_path: Path to SQLite database.
        model_id: Model identifier to store.

    Returns:
        Status dict with old and new values.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check current value
        cursor.execute("SELECT value FROM metadata WHERE key = 'model_id'")
        row = cursor.fetchone()
        old_value = row[0] if row else None

        if old_value:
            # Update existing
            cursor.execute(
                "UPDATE metadata SET value = ? WHERE key = 'model_id'",
                (model_id,),
            )
        else:
            # Insert new
            cursor.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("model_id", model_id),
            )

        conn.commit()

        return {
            "db_path": str(db_path),
            "old_model_id": old_value,
            "new_model_id": model_id,
            "action": "updated" if old_value else "inserted",
        }

    finally:
        conn.close()


def get_db_metadata(db_path: Path) -> dict:
    """Get all metadata from database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT key, value FROM metadata")
        return dict(cursor.fetchall())
    finally:
        conn.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Update database model_id metadata for traceability"
    )
    parser.add_argument(
        "--db",
        "-d",
        type=Path,
        required=True,
        help="SQLite database to update",
    )
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Model ID to set (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show current metadata without modifying",
    )

    args = parser.parse_args()

    if args.show:
        metadata = get_db_metadata(args.db)
        print(f"Metadata for {args.db}:")
        for key, value in sorted(metadata.items()):
            print(f"  {key}: {value}")
        return

    result = update_model_id(args.db, args.model_id)
    print(f"Database: {result['db_path']}")
    print(f"Action: {result['action']}")
    print(f"Old model_id: {result['old_model_id']}")
    print(f"New model_id: {result['new_model_id']}")


if __name__ == "__main__":
    main()
