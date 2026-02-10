"""
Tests for reformulate_by_design module (Phase 5: reformulation).

14 PURE + 6 MOCK (only model.encode is mocked).

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Semantic preservation
    - ISO/IEC 25010 FA-01 - Quality metrics
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.evaluation.annales.reformulate_by_design import (
    THRESHOLD_REFORMULATION,
    check_answerability,
    cosine_similarity,
    generate_natural_question,
    reformulate_question,
    validate_reformulation,
)

# ---------------------------------------------------------------------------
# TestCosineSimilarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for cosine_similarity (PURE)."""

    def test_identical(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector(self) -> None:
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_opposite(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestCheckAnswerability
# ---------------------------------------------------------------------------


class TestCheckAnswerability:
    """Tests for check_answerability (PURE)."""

    def test_direct_match(self) -> None:
        result = check_answerability(
            "l'arbitre doit verifier",
            "L'arbitre doit verifier les pendules avant la partie.",
        )
        assert result["answer_in_chunk"] is True
        assert result["passed"] is True

    def test_high_word_coverage(self) -> None:
        result = check_answerability(
            "l'arbitre verifie les pendules avant",
            "L'arbitre doit verifier les pendules avant chaque ronde du tournoi.",
        )
        assert result["word_coverage"] >= 0.8
        assert result["passed"] is True

    def test_low_coverage(self) -> None:
        result = check_answerability(
            "le joueur abandonne la partie",
            "L'arbitre surveille le bon deroulement du tournoi de blitz.",
        )
        assert result["passed"] is False

    def test_empty_answer_words(self) -> None:
        # All words <= 3 chars -> empty answer_words
        result = check_answerability("le la un", "Le roi est sur la case.")
        assert "word_coverage" in result

    def test_return_structure(self) -> None:
        result = check_answerability("test", "text with test")
        assert "answer_in_chunk" in result
        assert "word_coverage" in result
        assert "passed" in result


# ---------------------------------------------------------------------------
# TestThresholdReformulation
# ---------------------------------------------------------------------------


class TestThresholdReformulation:
    """Tests for THRESHOLD_REFORMULATION constant."""

    def test_value(self) -> None:
        assert THRESHOLD_REFORMULATION == 0.85


# ---------------------------------------------------------------------------
# TestGenerateNaturalQuestion
# ---------------------------------------------------------------------------


class TestGenerateNaturalQuestion:
    """Tests for generate_natural_question (PURE)."""

    def test_removes_mcq_language(self) -> None:
        original = (
            "Quelle proposition parmi les suivantes est correcte concernant l'arbitre?"
        )
        result = generate_natural_question(original, "reponse", "chunk text")
        assert "parmi les suivantes" not in result.lower()

    def test_adds_question_mark(self) -> None:
        result = generate_natural_question(
            "Quelle est la regle applicable pour le joueur",
            "reponse",
            "chunk text",
        )
        assert result.endswith("?")

    def test_fallback_article_ref(self) -> None:
        # Short question triggers fallback
        result = generate_natural_question(
            "Vrai ou faux",
            "reponse",
            "chunk text",
            article_reference="LA-oct2025 Art. 5.1",
        )
        assert "?" in result
        assert len(result) >= 20

    def test_fallback_answer(self) -> None:
        result = generate_natural_question(
            "Oui",  # Too short, no article_ref
            "L'arbitre doit veiller au bon deroulement de la competition",
            "chunk text",
        )
        assert "?" in result
        assert "arbitre" in result.lower()

    def test_clean_punctuation(self) -> None:
        result = generate_natural_question(
            "Quelle est la regle concernant le roque dans les echecs?",
            "reponse",
            "chunk text",
        )
        assert not result.endswith("??")
        assert result.endswith("?")


# ---------------------------------------------------------------------------
# TestValidateReformulation (MOCK)
# ---------------------------------------------------------------------------


class TestValidateReformulation:
    """Tests for validate_reformulation (mock model.encode)."""

    def _make_mock_model(self, similarity: float) -> MagicMock:
        """Create mock model that produces controlled cosine similarity."""
        model = MagicMock()
        # For similarity s, use vectors (1, 0) and (s, sqrt(1-s^2))
        if similarity >= 1.0:
            v1 = np.array([1.0, 0.0])
            v2 = np.array([1.0, 0.0])
        elif similarity <= -1.0:
            v1 = np.array([1.0, 0.0])
            v2 = np.array([-1.0, 0.0])
        else:
            v1 = np.array([1.0, 0.0])
            v2 = np.array([similarity, np.sqrt(1 - similarity**2)])
        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        call_count = [0]

        def encode_side_effect(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return v1 if idx % 2 == 0 else v2

        model.encode.side_effect = encode_side_effect
        return model

    def test_pass_high_similarity(self) -> None:
        model = self._make_mock_model(0.95)
        result = validate_reformulation("original?", "reformulated?", model)
        assert result["passed"] is True
        assert result["similarity_score"] >= 0.85

    def test_fail_low_similarity(self) -> None:
        model = self._make_mock_model(0.5)
        result = validate_reformulation("original?", "reformulated?", model)
        assert result["passed"] is False
        assert result["similarity_score"] < 0.85

    def test_return_structure(self) -> None:
        model = self._make_mock_model(0.9)
        result = validate_reformulation("q1?", "q2?", model)
        assert "similarity_score" in result
        assert "threshold" in result
        assert "passed" in result


# ---------------------------------------------------------------------------
# TestReformulateQuestion (MOCK)
# ---------------------------------------------------------------------------


class TestReformulateQuestion:
    """Tests for reformulate_question (mock model.encode)."""

    def _make_mock_model(self, similarity: float) -> MagicMock:
        model = MagicMock()
        if similarity >= 1.0:
            v1 = np.array([1.0, 0.0])
            v2 = np.array([1.0, 0.0])
        else:
            v1 = np.array([1.0, 0.0])
            v2 = np.array([similarity, np.sqrt(max(0, 1 - similarity**2))])
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        call_count = [0]

        def encode_side_effect(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return v1 if idx % 2 == 0 else v2

        model.encode.side_effect = encode_side_effect
        return model

    def test_pass_when_both_ok(self) -> None:
        model = self._make_mock_model(0.95)
        question = {
            "id": "test-001",
            "question": "Quelle est la regle pour le roque?",
            "expected_answer": "Le roque est un mouvement special du roi et de la tour.",
            "metadata": {"article_reference": "Art. 5.1"},
        }
        chunk_text = "Le roque est un mouvement special impliquant le roi et une tour."
        result = reformulate_question(question, chunk_text, model)
        assert result["validation"]["overall_passed"] is True

    def test_fail_when_semantic_low(self) -> None:
        model = self._make_mock_model(0.3)
        question = {
            "id": "test-002",
            "question": "Question originale?",
            "expected_answer": "reponse originale",
            "metadata": {},
        }
        chunk_text = "Texte sans rapport avec la question."
        result = reformulate_question(question, chunk_text, model)
        assert result["validation"]["semantic"]["passed"] is False
