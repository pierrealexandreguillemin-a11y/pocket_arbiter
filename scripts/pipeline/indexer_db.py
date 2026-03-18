"""SQLite DB operations: schema, create, insert, FTS5 population."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from scripts.pipeline.indexer_embed import embedding_to_blob
from scripts.pipeline.synonyms import stem_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS parents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    section TEXT,
    tokens INTEGER,
    page INTEGER
);

CREATE TABLE IF NOT EXISTS children (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    parent_id TEXT NOT NULL REFERENCES parents(id),
    source TEXT NOT NULL,
    page INTEGER,
    article_num TEXT,
    section TEXT,
    tokens INTEGER
);

CREATE TABLE IF NOT EXISTS table_summaries (
    id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,
    raw_table_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    source TEXT NOT NULL,
    page INTEGER,
    tokens INTEGER
);

CREATE VIRTUAL TABLE IF NOT EXISTS children_fts USING fts5(
    id UNINDEXED,
    text_stemmed,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS table_summaries_fts USING fts5(
    id UNINDEXED,
    text_stemmed,
    tokenize='unicode61 remove_diacritics 2'
);
"""


def create_db(path: Path) -> sqlite3.Connection:
    """Create SQLite DB with schema."""
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_parents(conn: sqlite3.Connection, parents: list[dict]) -> None:
    """Insert parent chunks."""
    conn.executemany(
        "INSERT OR REPLACE INTO parents (id, text, source, section, tokens, page) "
        "VALUES (:id, :text, :source, :section, :tokens, :page)",
        parents,
    )
    conn.commit()


def insert_children(
    conn: sqlite3.Connection,
    children: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Insert children with embeddings."""
    conn.executemany(
        "INSERT OR REPLACE INTO children "
        "(id, text, embedding, parent_id, source, page, article_num, section, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                c["id"],
                c["text"],
                embedding_to_blob(embeddings[i]),
                c["parent_id"],
                c["source"],
                c["page"],
                c.get("article_num"),
                c.get("section"),
                c["tokens"],
            )
            for i, c in enumerate(children)
        ],
    )
    conn.commit()


def insert_table_summaries(
    conn: sqlite3.Connection,
    summaries: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Insert table summaries with embeddings."""
    conn.executemany(
        "INSERT OR REPLACE INTO table_summaries "
        "(id, summary_text, raw_table_text, embedding, source, page, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                s["id"],
                s["summary_text"],
                s.get("raw_table_text", ""),
                embedding_to_blob(embeddings[i]),
                s["source"],
                s.get("page"),
                s.get("tokens", 0),
            )
            for i, s in enumerate(summaries)
        ],
    )
    conn.commit()


def populate_fts(conn: sqlite3.Connection) -> None:
    """Populate FTS5 tables with stemmed text."""
    conn.execute("DELETE FROM children_fts")
    conn.execute("DELETE FROM table_summaries_fts")
    rows = conn.execute("SELECT id, text FROM children").fetchall()
    conn.executemany(
        "INSERT INTO children_fts (id, text_stemmed) VALUES (?, ?)",
        [(r[0], stem_text(r[1])) for r in rows],
    )
    rows = conn.execute("SELECT id, summary_text FROM table_summaries").fetchall()
    conn.executemany(
        "INSERT INTO table_summaries_fts (id, text_stemmed) VALUES (?, ?)",
        [(r[0], stem_text(r[1])) for r in rows],
    )
    conn.commit()
