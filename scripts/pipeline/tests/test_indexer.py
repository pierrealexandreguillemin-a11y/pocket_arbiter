"""Tests for indexer module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import (
    SOURCE_TITLES,
    _table_section_from_summary,
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
    populate_fts,
)
from scripts.pipeline.synonyms import stem_text


class TestMakeCchTitle:
    """Test CCH title generation."""

    def test_known_source(self) -> None:
        title = make_cch_title(
            "R01_2025_26_Regles_generales.pdf",
            "3.2. Forfait isole",
            SOURCE_TITLES,
        )
        assert "Regles Generales" in title
        assert "3.2. Forfait isole" in title
        assert " > " in title

    def test_unknown_source_fallback(self) -> None:
        title = make_cch_title("unknown_doc.pdf", "Section 1", SOURCE_TITLES)
        assert "unknown_doc" in title
        assert "Section 1" in title

    def test_empty_section(self) -> None:
        title = make_cch_title("LA-octobre2025.pdf", "", SOURCE_TITLES)
        assert "Arbitre" in title

    def test_custom_source_titles(self) -> None:
        custom = {"test.pdf": "Custom Title"}
        title = make_cch_title("test.pdf", "S1", source_titles=custom)
        assert title == "Custom Title > S1"

    def test_custom_source_titles_fallback(self) -> None:
        custom = {"other.pdf": "Other"}
        title = make_cch_title("unknown.pdf", "S1", source_titles=custom)
        assert "unknown" in title.lower()

    def test_source_titles_is_required(self) -> None:
        title = make_cch_title(
            "LA-octobre2025.pdf", "Section", source_titles=SOURCE_TITLES
        )
        assert "Arbitre" in title


class TestTableSectionFromSummary:
    """Test _table_section_from_summary helper."""

    def test_colon_truncation(self) -> None:
        summary = "Bareme frais deplacement FFE: 8 tranches distance avec details"
        section = _table_section_from_summary(summary)
        assert section == "Bareme frais deplacement FFE"

    def test_max_words_limit(self) -> None:
        summary = "one two three four five six seven eight nine ten eleven"
        section = _table_section_from_summary(summary, max_words=8)
        assert section == "one two three four five six seven eight"

    def test_short_summary(self) -> None:
        section = _table_section_from_summary("Table simple")
        assert section == "Table simple"

    def test_empty_summary(self) -> None:
        section = _table_section_from_summary("")
        assert section == "Table"

    def test_real_summary(self) -> None:
        summary = (
            "Schema Scheveningen Coupe Loubatiere 2 equipes: "
            "appariements croises A-B sur 3 echiquiers"
        )
        section = _table_section_from_summary(summary)
        assert "Scheveningen" in section
        assert "appariements" not in section


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


# === G8 quality gate: GS text retrouvable in corpus children ===

_GS_PATH = Path("tests/data/gold_standard_annales_fr_v8_adversarial.json")
_DOCLING_DIR = Path("corpus/processed/docling_v2_fr")


def _normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace for fuzzy matching."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Normalize apostrophe variants to straight apostrophe
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    # Remove punctuation except apostrophes (keep l'equipe as l'equipe)
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.lower().split())


def _extract_distinctive_phrases(text: str, min_words: int = 4) -> list[str]:
    """Extract multiple distinctive phrases from answer text for fuzzy matching.

    Splits text into sentences, picks the longest ones, and extracts
    contiguous word spans from different positions.

    Returns:
        List of phrases suitable for substring search (may be empty).
    """
    sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
    if not sentences:
        return []
    # Sort sentences by length (longest first), take up to 3
    ranked = sorted(sentences, key=lambda s: len(s.split()), reverse=True)[:3]
    phrases: list[str] = []
    for sent in ranked:
        words = sent.split()
        if len(words) < min_words:
            phrases.append(" ".join(words))
            continue
        # Take from the middle
        start = max(0, len(words) // 2 - min_words // 2)
        phrases.append(" ".join(words[start : start + min_words]))
        # Also try from the start (after first word to skip boilerplate)
        if len(words) >= min_words + 1:
            phrases.append(" ".join(words[1 : 1 + min_words]))
    return phrases


def _get_answer_text(q: dict) -> str:
    """Extract answer text from a question dict."""
    answer = q.get("content", {}).get("expected_answer", "")
    if not answer:
        answer = q.get("provenance", {}).get("answer_explanation", "")
    return answer


def _check_answers_in_corpus(
    questions: list[dict], corpus_text: str
) -> tuple[list[str], int]:
    """Check which answers can be found in corpus. Returns (not_found, checked)."""
    not_found: list[str] = []
    checked = 0
    for q in questions:
        answer = _get_answer_text(q)
        if not answer:
            continue
        phrases = _extract_distinctive_phrases(answer, min_words=4)
        if not phrases:
            continue
        checked += 1
        found = any(_normalize(phrase) in corpus_text for phrase in phrases)
        if not found:
            not_found.append(f"{q['id']}: phrases={[p[:40] for p in phrases[:2]]}")
    return not_found, checked


@pytest.mark.slow
class TestG8GsTextRetrouvable:
    """G8 quality gate: verify GS answer content exists in v2 corpus children.

    For each answerable GS question, check that a distinctive phrase from
    the expected answer can be found in at least one child chunk text.
    """

    @pytest.fixture(scope="class")
    def corpus_children_text(self) -> str:
        """Load all v2 children texts into a single normalized string."""
        from scripts.pipeline.chunker import chunk_document

        all_text_parts: list[str] = []
        for json_path in sorted(_DOCLING_DIR.glob("*.json")):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            result = chunk_document(
                data["markdown"],
                data["source"],
                data.get("heading_pages"),
            )
            for child in result["children"]:
                all_text_parts.append(child["text"])
        return _normalize(" ".join(all_text_parts))

    @pytest.fixture(scope="class")
    def answerable_questions(self) -> list[dict]:
        """Load answerable questions from the GS."""
        with open(_GS_PATH, encoding="utf-8") as f:
            gs = json.load(f)
        return [
            q
            for q in gs["questions"]
            if not q.get("content", {}).get("is_impossible", False)
        ]

    def test_gs_answerable_count(self, answerable_questions: list[dict]) -> None:
        """Verify we have the expected number of answerable questions."""
        assert len(answerable_questions) >= 290, (
            f"Expected >= 290 answerable questions, got {len(answerable_questions)}"
        )

    def test_gs_text_in_children(
        self,
        answerable_questions: list[dict],
        corpus_children_text: str,
    ) -> None:
        """For each answerable Q, a distinctive phrase must appear in children."""
        not_found, checked = _check_answers_in_corpus(
            answerable_questions, corpus_children_text
        )
        miss_rate = len(not_found) / checked if checked else 0
        assert miss_rate < 0.02, (
            f"G8 FAIL: {len(not_found)}/{checked} answers not found "
            f"in children ({miss_rate:.1%}).\n"
            f"First 5 misses: {not_found[:5]}"
        )
        assert checked >= 250, f"Too few questions checked: {checked} (expected >= 250)"


class TestFts5Schema:
    """Test FTS5 virtual tables creation."""

    def test_creates_fts5_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        # Check FTS5 tables exist
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "children_fts" in tables
        assert "table_summaries_fts" in tables
        conn.close()

    def test_fts5_search_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        conn.execute(
            "INSERT INTO children_fts (id, text_stemmed) VALUES (?, ?)",
            ("child-1", stem_text("arbitrage des competitions")),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT id, bm25(children_fts) FROM children_fts "
            "WHERE children_fts MATCH ?",
            (stem_text("arbitrage"),),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "child-1"
        conn.close()


class TestPopulateFts:
    """Test FTS5 population from children/summaries."""

    def test_populate_children_fts(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        # Insert a parent + child first
        conn.execute(
            "INSERT INTO parents (id, text, source, section, tokens, page) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p1", "Parent", "test.pdf", "S1", 10, 1),
        )
        children = [
            {
                "id": "c1",
                "text": "Les joueurs doivent etre licencies",
                "parent_id": "p1",
                "source": "test.pdf",
                "page": 1,
                "article_num": None,
                "section": "S1",
                "tokens": 10,
            },
        ]
        emb = np.random.randn(1, 768).astype(np.float32)
        insert_children(conn, children, emb)
        populate_fts(conn)
        # Verify FTS5 has the stemmed text
        count = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
        assert count == 1
        # Verify search works
        rows = conn.execute(
            "SELECT id FROM children_fts WHERE children_fts MATCH ?",
            (stem_text("licencies"),),
        ).fetchall()
        assert len(rows) == 1
        conn.close()

    def test_populate_table_summaries_fts(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        # Insert a table summary
        summaries = [
            {
                "id": "doc-table0",
                "summary_text": "Bareme frais deplacement federation",
                "raw_table_text": "| col1 | col2 |",
                "source": "doc.pdf",
                "page": 3,
                "tokens": 25,
            }
        ]
        emb = np.random.randn(1, 768).astype(np.float32)
        insert_table_summaries(conn, summaries, emb)
        populate_fts(conn)
        # Verify FTS5 has the stemmed text
        count = conn.execute("SELECT COUNT(*) FROM table_summaries_fts").fetchone()[0]
        assert count == 1
        # Verify search works
        rows = conn.execute(
            "SELECT id FROM table_summaries_fts WHERE table_summaries_fts MATCH ?",
            (stem_text("deplacement"),),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "doc-table0"
        conn.close()

    def test_populate_idempotent(self, tmp_path: Path) -> None:
        """Calling populate_fts twice should not duplicate rows."""
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        conn.execute(
            "INSERT INTO parents (id, text, source, section, tokens, page) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p1", "Parent", "test.pdf", "S1", 10, 1),
        )
        children = [
            {
                "id": "c1",
                "text": "Texte de test",
                "parent_id": "p1",
                "source": "test.pdf",
                "page": 1,
                "article_num": None,
                "section": "S1",
                "tokens": 5,
            },
        ]
        emb = np.random.randn(1, 768).astype(np.float32)
        insert_children(conn, children, emb)
        populate_fts(conn)
        populate_fts(conn)  # Second call should clear and rebuild
        count = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
        assert count == 1
        conn.close()

    def test_populate_empty_tables(self, tmp_path: Path) -> None:
        """populate_fts on empty tables should not error."""
        db_path = tmp_path / "test.db"
        conn = create_db(db_path)
        populate_fts(conn)
        count_c = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
        count_t = conn.execute("SELECT COUNT(*) FROM table_summaries_fts").fetchone()[0]
        assert count_c == 0
        assert count_t == 0
        conn.close()
