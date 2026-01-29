"""
Tests pour add_expected_chunk_id.py

ISO Reference: ISO/IEC 29119 - Test coverage >= 80%
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from scripts.training.add_expected_chunk_id import (
    find_best_chunk_for_question,
    get_chunks_for_pages,
    normalize_source_name,
)


class TestNormalizeSourceName:
    """Tests pour normalize_source_name()."""

    def test_removes_accents(self):
        """Supprime les accents francais."""
        assert normalize_source_name("règlement") == "reglement"
        assert normalize_source_name("parité") == "parite"
        assert normalize_source_name("éèêë") == "eeee"

    def test_lowercase(self):
        """Convertit en minuscules."""
        assert normalize_source_name("ABC") == "abc"
        assert normalize_source_name("Règlement") == "reglement"

    def test_preserves_non_accented(self):
        """Preserve caracteres sans accent."""
        assert normalize_source_name("test123") == "test123"
        assert normalize_source_name("LA-octobre2025.pdf") == "la-octobre2025.pdf"


class TestGetChunksForPages:
    """Tests pour get_chunks_for_pages()."""

    @pytest.fixture
    def temp_db(self):
        """Cree une base temporaire avec chunks de test."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                metadata TEXT,
                embedding BLOB NOT NULL
            )
        """)

        # Insert test chunks
        test_chunks = [
            ("test.pdf-p001-c0", "Contenu page 1", "test.pdf", 1, 10, None, b""),
            ("test.pdf-p002-c0", "Contenu page 2", "test.pdf", 2, 10, None, b""),
            ("test.pdf-p002-c1", "Autre contenu page 2", "test.pdf", 2, 10, None, b""),
            ("autre.pdf-p001-c0", "Autre doc", "autre.pdf", 1, 10, None, b""),
        ]
        cursor.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            test_chunks,
        )
        conn.commit()
        conn.close()

        yield db_path
        db_path.unlink()

    def test_returns_chunks_for_pages(self, temp_db):
        """Retourne chunks pour les pages demandees."""
        chunks = get_chunks_for_pages(temp_db, "test.pdf", [1, 2])
        assert len(chunks) == 3
        assert all(c["source"] == "test.pdf" for c in chunks)

    def test_filters_by_source(self, temp_db):
        """Filtre par source."""
        chunks = get_chunks_for_pages(temp_db, "autre.pdf", [1])
        assert len(chunks) == 1
        assert chunks[0]["source"] == "autre.pdf"

    def test_returns_empty_for_no_match(self, temp_db):
        """Retourne liste vide si aucun match."""
        chunks = get_chunks_for_pages(temp_db, "inexistant.pdf", [1])
        assert chunks == []


class TestFindBestChunkForQuestion:
    """Tests pour find_best_chunk_for_question()."""

    def test_matches_keywords(self):
        """Trouve chunk avec keywords."""
        question = {
            "id": "Q1",
            "question": "Quelle est la règle?",
            "keywords": ["règle", "article"],
        }
        chunks = [
            {"id": "c1", "text": "Texte sans rapport"},
            {"id": "c2", "text": "Article sur la règle importante"},
        ]
        result = find_best_chunk_for_question(question, chunks)
        assert result == "c2"

    def test_boosts_article_numbers(self):
        """Boost pour numeros d'article."""
        question = {
            "id": "Q1",
            "question": "Article 4.3?",
            "keywords": ["article"],
            "metadata": {"article_num": "4.3"},
        }
        chunks = [
            {"id": "c1", "text": "Article general"},
            {"id": "c2", "text": "Article 4.3 specifique"},
        ]
        result = find_best_chunk_for_question(question, chunks)
        assert result == "c2"

    def test_returns_none_for_no_match(self):
        """Retourne None si aucun keyword trouve."""
        question = {
            "id": "Q1",
            "question": "Question specifique",
            "keywords": ["xyz123"],
        }
        chunks = [
            {"id": "c1", "text": "Texte sans rapport"},
        ]
        result = find_best_chunk_for_question(question, chunks)
        assert result is None

    def test_handles_empty_chunks(self):
        """Gere liste de chunks vide."""
        question = {"id": "Q1", "question": "Test", "keywords": []}
        result = find_best_chunk_for_question(question, [])
        assert result is None
