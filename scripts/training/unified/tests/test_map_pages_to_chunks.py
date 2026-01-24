"""Tests for map_pages_to_chunks.py (Step 1)."""

import pytest

from scripts.training.unified.map_pages_to_chunks import (
    find_chunks_for_pages,
    map_question_to_chunk,
    normalize_source_name,
    score_chunk_for_question,
)


class TestNormalizeSourceName:
    """Tests for normalize_source_name."""

    def test_lowercase(self) -> None:
        assert normalize_source_name("TEST") == "test"

    def test_accents_removed(self) -> None:
        assert normalize_source_name("réglement") == "reglement"
        assert normalize_source_name("éèêë") == "eeee"

    def test_mixed(self) -> None:
        assert normalize_source_name("LA-Règlement.pdf") == "la-reglement.pdf"


class TestFindChunksForPages:
    """Tests for find_chunks_for_pages."""

    @pytest.fixture
    def sample_chunks(self) -> list[dict]:
        return [
            {"id": "c1", "source": "LA-octobre2025.pdf", "pages": [1, 2], "text": "Chunk 1"},
            {"id": "c2", "source": "LA-octobre2025.pdf", "pages": [3, 4], "text": "Chunk 2"},
            {"id": "c3", "source": "other.pdf", "pages": [1], "text": "Chunk 3"},
        ]

    def test_finds_matching_chunks(self, sample_chunks: list[dict]) -> None:
        result = find_chunks_for_pages(sample_chunks, "LA-octobre2025", [1, 2])
        assert len(result) == 1
        assert result[0]["id"] == "c1"

    def test_finds_multiple_chunks(self, sample_chunks: list[dict]) -> None:
        result = find_chunks_for_pages(sample_chunks, "LA-octobre2025", [1, 3])
        assert len(result) == 2

    def test_no_match(self, sample_chunks: list[dict]) -> None:
        result = find_chunks_for_pages(sample_chunks, "nonexistent", [1])
        assert len(result) == 0

    def test_partial_source_match(self, sample_chunks: list[dict]) -> None:
        result = find_chunks_for_pages(sample_chunks, "octobre2025", [1])
        assert len(result) == 1


class TestScoreChunkForQuestion:
    """Tests for score_chunk_for_question."""

    def test_keyword_matching(self) -> None:
        chunk = {"text": "Le joueur doit jouer son coup"}
        question = {"keywords": ["joueur", "coup"]}
        score = score_chunk_for_question(chunk, question)
        assert score >= 2.0  # At least 2 keyword matches

    def test_article_reference_boost(self) -> None:
        chunk = {"text": "Article 4.1 stipule que...", "section": "Article 4.1"}
        question = {"keywords": [], "article_reference": "Article 4.1"}
        score = score_chunk_for_question(chunk, question)
        assert score >= 10.0  # Article match is strong

    def test_no_match_zero_score(self) -> None:
        chunk = {"text": "Texte sans rapport"}
        question = {"keywords": ["absent"]}
        score = score_chunk_for_question(chunk, question)
        assert score == 0.0


class TestMapQuestionToChunk:
    """Tests for map_question_to_chunk."""

    @pytest.fixture
    def sample_chunks(self) -> list[dict]:
        return [
            {
                "id": "c1",
                "source": "LA-octobre2025.pdf",
                "pages": [38],
                "text": "Article 4.1 - Le joueur doit appuyer sur la pendule",
                "section": "Article 4.1",
            },
            {
                "id": "c2",
                "source": "LA-octobre2025.pdf",
                "pages": [39],
                "text": "Article 4.2 - Les pieces doivent etre placees",
                "section": "Article 4.2",
            },
        ]

    def test_maps_with_article_match(self, sample_chunks: list[dict]) -> None:
        question = {
            "id": "Q1",
            "expected_docs": ["LA-octobre2025.pdf"],
            "expected_pages": [38],
            "keywords": ["pendule"],
            "article_reference": "Article 4.1",
        }
        result = map_question_to_chunk(question, sample_chunks)
        assert result["expected_chunk_id"] == "c1"
        assert result["mapping_confidence"] >= 0.9

    def test_adversarial_no_pages(self, sample_chunks: list[dict]) -> None:
        question = {
            "id": "Q2",
            "expected_docs": [],
            "expected_pages": [],
        }
        result = map_question_to_chunk(question, sample_chunks)
        assert result["expected_chunk_id"] is None
        assert result["mapping_method"] == "adversarial_no_pages"

    def test_no_candidates(self, sample_chunks: list[dict]) -> None:
        question = {
            "id": "Q3",
            "expected_docs": ["other.pdf"],
            "expected_pages": [100],
        }
        result = map_question_to_chunk(question, sample_chunks)
        assert result["expected_chunk_id"] is None
        assert result["mapping_method"] == "no_candidates"
