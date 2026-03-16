"""Tests for indexer module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.pipeline.indexer import (
    SOURCE_TITLES,
    blob_to_embedding,
    create_db,
    embedding_to_blob,
    format_document,
    format_query,
    insert_children,
    insert_parents,
    insert_table_summaries,
    load_table_summaries,
    make_cch_title,
)


class TestMakeCchTitle:
    """Test CCH title generation."""

    def test_known_source(self) -> None:
        title = make_cch_title("R01_2025_26_Regles_generales.pdf", "3.2. Forfait isole")
        assert "Regles Generales" in title
        assert "3.2. Forfait isole" in title
        assert " | " in title

    def test_unknown_source_fallback(self) -> None:
        title = make_cch_title("unknown_doc.pdf", "Section 1")
        assert "unknown doc" in title.lower()
        assert "Section 1" in title

    def test_empty_section(self) -> None:
        title = make_cch_title("LA-octobre2025.pdf", "")
        assert "Lois des Echecs" in title


class TestFormatPrompts:
    """Test Google-recommended prompt formatting."""

    def test_format_document(self) -> None:
        text = format_document("Some chunk text", "Doc Title | Section")
        assert text == "title: Doc Title | Section | text: Some chunk text"

    def test_format_query(self) -> None:
        text = format_query("What is the rule?")
        assert text == "task: search result | query: What is the rule?"


class TestBlobRoundtrip:
    """Test embedding serialization."""

    def test_roundtrip_768d(self) -> None:
        original = np.random.randn(768).astype(np.float32)
        blob = embedding_to_blob(original)
        restored = blob_to_embedding(blob)
        np.testing.assert_array_equal(original, restored)

    def test_blob_size(self) -> None:
        emb = np.zeros(768, dtype=np.float32)
        blob = embedding_to_blob(emb)
        assert len(blob) == 768 * 4  # float32 = 4 bytes

    def test_l2_normalized_roundtrip(self) -> None:
        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        blob = embedding_to_blob(emb)
        restored = blob_to_embedding(blob)
        assert abs(np.linalg.norm(restored) - 1.0) < 0.001


class TestCreateDb:
    """Test SQLite schema creation."""

    def test_creates_three_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "children" in tables
        assert "parents" in tables
        assert "table_summaries" in tables
        conn.close()

    def test_children_schema(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        cursor = conn.execute("PRAGMA table_info(children)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "text",
            "embedding",
            "parent_id",
            "source",
            "page",
            "article_num",
            "section",
            "tokens",
        }
        assert columns == expected
        conn.close()

    def test_parents_schema(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        cursor = conn.execute("PRAGMA table_info(parents)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {"id", "text", "source", "section", "tokens", "page"}
        assert columns == expected
        conn.close()

    def test_table_summaries_schema(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        cursor = conn.execute("PRAGMA table_info(table_summaries)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "summary_text",
            "raw_table_text",
            "embedding",
            "source",
            "page",
            "tokens",
        }
        assert columns == expected
        conn.close()


class TestInsertChildren:
    """Test children insertion with embedding blob roundtrip."""

    def test_insert_and_read(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)

        # Insert a parent first (FK reference)
        conn.execute(
            "INSERT INTO parents (id, text, source, section, tokens, page) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test.pdf-p000", "Parent text", "test.pdf", "Section 1", 50, 1),
        )

        children = [
            {
                "id": "test.pdf-c0000",
                "text": "Child text content",
                "parent_id": "test.pdf-p000",
                "source": "test.pdf",
                "page": 5,
                "article_num": "3.2",
                "section": "3.2. Forfait",
                "tokens": 42,
            }
        ]
        embeddings = np.random.randn(1, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        insert_children(conn, children, embeddings)

        row = conn.execute(
            "SELECT * FROM children WHERE id = ?", ("test.pdf-c0000",)
        ).fetchone()
        assert row is not None
        assert row[1] == "Child text content"  # text
        # Verify embedding roundtrip
        restored = blob_to_embedding(row[2])
        np.testing.assert_allclose(restored, embeddings[0], atol=1e-7)
        conn.close()

    def test_insert_multiple(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        conn.execute(
            "INSERT INTO parents (id, text, source, section, tokens, page) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("src-p000", "", "src.pdf", "root", 0, None),
        )

        children = [
            {
                "id": f"src-c{i:04d}",
                "text": f"Text {i}",
                "parent_id": "src-p000",
                "source": "src.pdf",
                "page": i,
                "article_num": None,
                "section": f"S{i}",
                "tokens": 10,
            }
            for i in range(5)
        ]
        embeddings = np.random.randn(5, 768).astype(np.float32)

        insert_children(conn, children, embeddings)
        count = conn.execute("SELECT COUNT(*) FROM children").fetchone()[0]
        assert count == 5
        conn.close()


class TestInsertParents:
    """Test parents insertion."""

    def test_insert_and_read(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)

        parents = [
            {
                "id": "test.pdf-p000",
                "text": "Full parent text with all sub-sections",
                "source": "test.pdf",
                "section": "3. Forfaits",
                "tokens": 200,
                "page": 5,
            }
        ]

        insert_parents(conn, parents)

        row = conn.execute(
            "SELECT * FROM parents WHERE id = ?", ("test.pdf-p000",)
        ).fetchone()
        assert row is not None
        assert row[1] == "Full parent text with all sub-sections"
        assert row[4] == 200  # tokens
        conn.close()

    def test_parent_page_nullable(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        parents = [
            {
                "id": "root",
                "text": "",
                "source": "test.pdf",
                "section": "root",
                "tokens": 0,
                "page": None,
            }
        ]
        insert_parents(conn, parents)
        row = conn.execute(
            "SELECT page FROM parents WHERE id = ?", ("root",)
        ).fetchone()
        assert row[0] is None
        conn.close()


class TestInsertTableSummaries:
    """Test table summaries insertion with embedding roundtrip."""

    def test_insert_and_read(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)

        summaries = [
            {
                "id": "doc-table0",
                "summary_text": "Summary of the table",
                "raw_table_text": "| col1 | col2 |\n|---|---|\n| a | b |",
                "source": "doc.pdf",
                "page": 3,
                "tokens": 25,
            }
        ]
        embeddings = np.random.randn(1, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        insert_table_summaries(conn, summaries, embeddings)

        row = conn.execute(
            "SELECT * FROM table_summaries WHERE id = ?", ("doc-table0",)
        ).fetchone()
        assert row is not None
        assert row[1] == "Summary of the table"
        assert row[2] == "| col1 | col2 |\n|---|---|\n| a | b |"
        restored = blob_to_embedding(row[3])
        np.testing.assert_allclose(restored, embeddings[0], atol=1e-7)
        conn.close()


class TestLoadTableSummaries:
    """Test loading and cross-referencing table summaries."""

    def test_loads_from_json(self, tmp_path: Path) -> None:
        # Create mock summaries file
        summaries_data = {
            "summaries": {
                "doc-table0": "Summary of table 0",
                "doc-table1": "Summary of table 1",
            },
            "metadata": {},
            "total": 2,
        }
        summaries_path = tmp_path / "summaries.json"
        summaries_path.write_text(json.dumps(summaries_data), encoding="utf-8")

        # Create mock docling extraction with matching tables
        docling_dir = tmp_path / "docling"
        docling_dir.mkdir()
        extraction = {
            "markdown": "# Test",
            "source": "doc.pdf",
            "tables": [
                {
                    "id": "doc-table0",
                    "source": "doc.pdf",
                    "text": "| raw | table0 |",
                    "page": 1,
                },
                {
                    "id": "doc-table1",
                    "source": "doc.pdf",
                    "text": "| raw | table1 |",
                    "page": 2,
                },
            ],
            "heading_pages": {},
        }
        (docling_dir / "doc.json").write_text(json.dumps(extraction), encoding="utf-8")

        result = load_table_summaries(summaries_path, docling_dir)
        assert len(result) == 2
        assert result[0]["id"] == "doc-table0"
        assert result[0]["summary_text"] == "Summary of table 0"
        assert result[0]["raw_table_text"] == "| raw | table0 |"
        assert result[0]["source"] == "doc.pdf"
        assert result[0]["page"] == 1

    def test_missing_raw_table_skipped(self, tmp_path: Path) -> None:
        summaries_data = {
            "summaries": {"doc-table99": "Orphan summary"},
            "metadata": {},
            "total": 1,
        }
        summaries_path = tmp_path / "summaries.json"
        summaries_path.write_text(json.dumps(summaries_data), encoding="utf-8")

        docling_dir = tmp_path / "docling"
        docling_dir.mkdir()
        # No extraction files — no raw tables to match
        result = load_table_summaries(summaries_path, docling_dir)
        assert len(result) == 0  # skipped because no raw table found


class TestSourceTitles:
    """Test SOURCE_TITLES completeness."""

    def test_has_28_entries(self) -> None:
        assert len(SOURCE_TITLES) == 28

    def test_all_pdf_keys(self) -> None:
        for key in SOURCE_TITLES:
            assert key.endswith(".pdf"), f"Key should end with .pdf: {key}"

    def test_known_sources(self) -> None:
        assert "LA-octobre2025.pdf" in SOURCE_TITLES
        assert "R01_2025_26_Regles_generales.pdf" in SOURCE_TITLES
