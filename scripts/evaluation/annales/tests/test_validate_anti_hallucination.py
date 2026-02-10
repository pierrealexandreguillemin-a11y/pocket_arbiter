"""
Tests for validate_anti_hallucination module (Phase 3: Anti-hallucination).

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO 42001 A.6.2.2 - Provenance verification
    - ISO 25010 - Accuracy metrics
"""

from unittest.mock import patch

import numpy as np
import pytest

from scripts.evaluation.annales.validate_anti_hallucination import (
    cosine_similarity,
    extract_keywords,
    normalize_text_for_matching,
    validate_keyword_coverage,
    validate_question,
    validate_verbatim,
)


class TestNormalizeTextForMatching:
    """Tests for text normalization."""

    def test_lowercase(self) -> None:
        """Should convert to lowercase."""
        result = normalize_text_for_matching("HELLO World")
        assert result == "hello world"

    def test_remove_accents(self) -> None:
        """Should remove French accents."""
        result = normalize_text_for_matching("éàùç")
        assert result == "eauc"

    def test_normalize_whitespace(self) -> None:
        """Should normalize multiple spaces."""
        result = normalize_text_for_matching("hello   world\n\ttest")
        assert result == "hello world test"


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_basic_extraction(self) -> None:
        """Should extract meaningful keywords."""
        keywords = extract_keywords("L'arbitre verifie la pendule du joueur")
        assert "arbitre" in keywords
        assert "verifie" in keywords
        assert "pendule" in keywords
        assert "joueur" in keywords

    def test_removes_stopwords(self) -> None:
        """Should remove French stopwords."""
        keywords = extract_keywords("pour dans avec cette")
        assert len(keywords) == 0

    def test_min_length(self) -> None:
        """Should filter short words."""
        keywords = extract_keywords("un jeu de test", min_length=4)
        assert "jeu" not in keywords  # 3 chars
        assert "test" in keywords  # 4 chars


class TestValidateVerbatim:
    """Tests for verbatim matching."""

    def test_exact_match(self) -> None:
        """Should find exact substring match."""
        answer = "60 secondes"
        chunk = "Le temps est de 60 secondes par coup."
        assert validate_verbatim(answer, chunk) is True

    def test_case_insensitive(self) -> None:
        """Should match case-insensitively."""
        answer = "L'ARBITRE"
        chunk = "l'arbitre doit verifier"
        assert validate_verbatim(answer, chunk) is True

    def test_accent_insensitive(self) -> None:
        """Should match ignoring accents."""
        answer = "regle"
        chunk = "La règle stipule que"
        assert validate_verbatim(answer, chunk) is True

    def test_no_match(self) -> None:
        """Should return False when no match."""
        answer = "xyz non existant"
        chunk = "Le texte du chunk."
        assert validate_verbatim(answer, chunk) is False


class TestValidateKeywordCoverage:
    """Tests for keyword coverage validation."""

    def test_full_coverage(self) -> None:
        """Should pass with 100% keyword coverage."""
        answer = "arbitre pendule joueur"
        chunk = "L'arbitre verifie la pendule du joueur."
        passed, coverage = validate_keyword_coverage(answer, chunk)
        assert passed is True
        assert coverage == 1.0

    def test_partial_coverage(self) -> None:
        """Should compute partial coverage correctly."""
        answer = "arbitre pendule inexistant"
        chunk = "L'arbitre verifie la pendule."
        passed, coverage = validate_keyword_coverage(answer, chunk, threshold=0.80)
        # 2/3 keywords found = 66.7%
        assert passed is False
        assert coverage == pytest.approx(2 / 3, rel=0.01)

    def test_threshold_respected(self) -> None:
        """Should respect threshold parameter."""
        answer = "arbitre joueur"
        chunk = "L'arbitre observe le joueur."
        passed_high, _ = validate_keyword_coverage(answer, chunk, threshold=0.90)
        passed_low, _ = validate_keyword_coverage(answer, chunk, threshold=0.50)
        # Both should pass since 100% coverage
        assert passed_high is True
        assert passed_low is True


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Should return 1.0 for identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, rel=0.001)

    def test_orthogonal_vectors(self) -> None:
        """Should return 0.0 for orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(0.0, rel=0.001)

    def test_opposite_vectors(self) -> None:
        """Should return -1.0 for opposite vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        sim = cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(-1.0, rel=0.001)

    def test_zero_vector(self) -> None:
        """Should handle zero vectors."""
        vec1 = np.array([0.0, 0.0])
        vec2 = np.array([1.0, 1.0])
        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0


class TestValidateQuestion:
    """Tests for full question validation."""

    def test_unanswerable_skipped(self) -> None:
        """Should skip validation for unanswerable questions."""
        question = {
            "id": "q1",
            "content": {"is_impossible": True, "expected_answer": ""},
        }
        result = validate_question(question, "chunk text", use_semantic=False)
        assert result.passed is True
        assert result.method == "unanswerable_skip"

    def test_verbatim_match_passes(self) -> None:
        """Should pass with verbatim match."""
        question = {
            "id": "q1",
            "content": {"is_impossible": False, "expected_answer": "60 secondes"},
        }
        chunk_text = "Le temps est de 60 secondes par coup."
        result = validate_question(question, chunk_text, use_semantic=False)
        assert result.passed is True
        assert result.method == "verbatim"

    def test_keyword_match_passes(self) -> None:
        """Should pass with keyword coverage."""
        question = {
            "id": "q1",
            "content": {
                "is_impossible": False,
                "expected_answer": "L'arbitre verifie le materiel et la pendule",
            },
        }
        # All keywords present but not verbatim
        chunk_text = "L'arbitre doit verifier le bon fonctionnement du materiel incluant la pendule."
        result = validate_question(question, chunk_text, use_semantic=False)
        assert result.passed is True
        assert result.method == "keyword"

    def test_no_answer_rejected(self) -> None:
        """Should reject questions without expected_answer."""
        question = {
            "id": "q1",
            "content": {"is_impossible": False, "expected_answer": ""},
        }
        result = validate_question(question, "chunk text", use_semantic=False)
        assert result.passed is False
        assert result.method == "REJECTED"

    def test_semantic_fallback(self) -> None:
        """Should fall back to semantic similarity when other methods fail."""
        # Controlled vectors: cosine(vec_answer, vec_chunk) = 0.95
        vec_answer = np.zeros(768)
        vec_answer[0] = 1.0
        vec_chunk = np.zeros(768)
        vec_chunk[0] = 0.95
        vec_chunk[1] = np.sqrt(1 - 0.95**2)

        with patch(
            "scripts.evaluation.annales.validate_anti_hallucination.compute_embedding"
        ) as mock:
            mock.side_effect = [vec_answer, vec_chunk]

            question = {
                "id": "q1",
                "content": {
                    "is_impossible": False,
                    "expected_answer": "Une reponse completement differente",
                },
            }
            chunk_text = "Un texte qui ne match pas du tout."
            result = validate_question(
                question, chunk_text, use_semantic=True, semantic_threshold=0.50
            )
            # Controlled cosine similarity = 0.95 > 0.50 threshold
            assert result.passed is True
            assert result.method == "semantic"
            assert result.semantic_similarity == pytest.approx(0.95, rel=0.01)
            assert mock.call_count == 2

    def test_semantic_fallback_below_threshold(self) -> None:
        """Should reject when semantic similarity is below threshold."""
        # Controlled vectors: cosine(vec_answer, vec_chunk) = 0.0 (orthogonal)
        vec_answer = np.zeros(768)
        vec_answer[0] = 1.0
        vec_chunk = np.zeros(768)
        vec_chunk[1] = 1.0

        with patch(
            "scripts.evaluation.annales.validate_anti_hallucination.compute_embedding"
        ) as mock:
            mock.side_effect = [vec_answer, vec_chunk]

            question = {
                "id": "q1",
                "content": {
                    "is_impossible": False,
                    "expected_answer": "Une reponse completement differente",
                },
            }
            chunk_text = "Un texte qui ne match pas du tout."
            result = validate_question(
                question, chunk_text, use_semantic=True, semantic_threshold=0.50
            )
            assert result.passed is False
            assert result.method == "REJECTED"
            assert result.semantic_similarity == pytest.approx(0.0, abs=0.01)
