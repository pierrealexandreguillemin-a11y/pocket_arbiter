"""Tests for indexer (embeddings + SQLite)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import (
    create_db,
    insert_children,
    insert_parents,
    insert_table_summaries,
    load_table_summaries,
    contextualize_text,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def sample_children():
    return [
        {"id": "test-c0000", "text": "Article 1 content about licences.",
         "parent_id": "test-p0", "source": "test.pdf", "article_num": "1",
         "section": "1. Licences", "tokens": 15, "page": 5},
        {"id": "test-c0001", "text": "Article 2 content about forfaits.",
         "parent_id": "test-p0", "source": "test.pdf", "article_num": "2",
         "section": "2. Forfaits", "tokens": 15, "page": 7},
    ]


@pytest.fixture
def sample_parents():
    return [
        {"id": "test-p0", "text": "Full parent text with licences and forfaits.",
         "source": "test.pdf", "section": "Regles generales", "tokens": 20,
         "page": 5},
    ]


@pytest.fixture
def sample_summaries():
    return [
        {"id": "test-table0", "summary_text": "Table des categories d'age.",
         "raw_table_text": "| U8 | moins de 8 ans |", "source": "test.pdf",
         "page": 3, "tokens": 10},
    ]


# --- Source title mapping for CCH ---
SOURCE_TITLES = {"test.pdf": "Regles generales FFE 2025-26"}


class TestContextualizeText:
    """Test Contextual Chunk Header generation."""

    def test_basic_contextualization(self):
        result = contextualize_text(
            "Article content.", "test.pdf", "3.2. Forfaits",
            {"test.pdf": "Regles generales"}
        )
        assert "[Document: Regles generales | Section: 3.2. Forfaits]" in result
        assert "Article content." in result

    def test_unknown_source(self):
        result = contextualize_text(
            "Content.", "unknown.pdf", "Section",
            {"test.pdf": "Regles generales"}
        )
        assert "[Document: unknown.pdf | Section: Section]" in result

    def test_none_section(self):
        result = contextualize_text("Content.", "test.pdf", None, {})
        assert "[Document: test.pdf]" in result


class TestCreateDb:
    def test_creates_tables(self, db_path):
        create_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t[0] for t in tables}
        assert "children" in names
        assert "parents" in names
        assert "table_summaries" in names
        conn.close()


class TestInsertData:
    def test_insert_children(self, db_path, sample_children):
        create_db(db_path)
        embeddings = {c["id"]: np.random.randn(768).astype(np.float32)
                      for c in sample_children}
        insert_children(db_path, sample_children, embeddings)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, source, page FROM children").fetchall()
        assert len(rows) == 2
        assert rows[0][2] == 5  # page
        conn.close()

    def test_insert_parents(self, db_path, sample_parents):
        create_db(db_path)
        insert_parents(db_path, sample_parents)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, page FROM parents").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == 5
        conn.close()

    def test_insert_table_summaries(self, db_path, sample_summaries):
        create_db(db_path)
        embeddings = {s["id"]: np.random.randn(768).astype(np.float32)
                      for s in sample_summaries}
        insert_table_summaries(db_path, sample_summaries, embeddings)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, page FROM table_summaries").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == 3
        conn.close()

    def test_embedding_roundtrip(self, db_path, sample_children):
        create_db(db_path)
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embeddings = {c["id"]: emb for c in sample_children}
        insert_children(db_path, sample_children, embeddings)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT embedding FROM children WHERE id=?",
                           (sample_children[0]["id"],)).fetchone()
        recovered = np.frombuffer(row[0], dtype=np.float32)
        np.testing.assert_array_equal(recovered, emb)
        conn.close()


class TestLoadTableSummaries:
    def test_loads_from_json(self, tmp_path):
        data = {
            "summaries": {
                "R01-table0": "Categories d'age",
                "R01-table1": "Cadences equivalentes",
            },
            "metadata": {},
            "total": 2,
        }
        path = tmp_path / "summaries.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = load_table_summaries(path)
        assert len(result) == 2
        assert result[0]["id"] == "R01-table0"
        assert result[0]["summary_text"] == "Categories d'age"
        assert result[0]["source"] == "R01.pdf"
