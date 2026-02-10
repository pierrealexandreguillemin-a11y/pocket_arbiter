"""
Tests for validate_gs_quality module (Phase 3: anti-hallucination).

16 PURE + 6 MOCK (only embedding model mocked).

ISO Reference:
    - ISO 42001 - Anti-hallucination validation
    - ISO 25010 - Quality metrics
    - ISO 29119 - Test coverage
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.evaluation.annales.validate_gs_quality import (
    ValidationResult,
    compute_keyword_score,
    compute_semantic_similarity,
    extract_keywords,
    load_json,
    normalize_text,
    save_json,
    validate_gold_standard,
    validate_question,
)

# ---------------------------------------------------------------------------
# TestLoadSaveJson
# ---------------------------------------------------------------------------


class TestLoadSaveJson:
    """Tests for load_json and save_json utility functions (PURE)."""

    def test_load_json(self, tmp_path: Path) -> None:
        import json

        p = tmp_path / "test.json"
        p.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        data = load_json(str(p))
        assert data["key"] == "value"

    def test_save_json(self, tmp_path: Path) -> None:
        import json

        p = tmp_path / "out.json"
        save_json({"test": True}, str(p))
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["test"] is True

    def test_roundtrip_utf8(self, tmp_path: Path) -> None:
        p = tmp_path / "utf8.json"
        save_json({"text": "éàü"}, str(p))
        data = load_json(str(p))
        assert data["text"] == "éàü"


# ---------------------------------------------------------------------------
# TestNormalizeText
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """Tests for normalize_text (PURE)."""

    def test_lowercase(self) -> None:
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_french_accents(self) -> None:
        result = normalize_text("éèêë àâä ùûü îï ôö ç")
        assert "e" in result
        assert "a" in result
        assert "u" in result
        assert result == "eeee aaa uuu ii oo c"

    def test_ligatures(self) -> None:
        result = normalize_text("œuvre et cætera")
        assert "oeuvre" in result
        assert "caettera" in result or "caetera" in result


# ---------------------------------------------------------------------------
# TestExtractKeywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    """Tests for extract_keywords (PURE)."""

    def test_words_min_4_chars(self) -> None:
        keywords = extract_keywords("le roi est sur la case du plateau")
        # "roi" has 3 chars -> excluded, "case" has 4 -> included if not stopword
        for kw in keywords:
            assert len(kw) >= 4

    def test_excludes_stopwords(self) -> None:
        keywords = extract_keywords("pour dans avec cette celui faire doit")
        stopwords = {"pour", "dans", "avec", "cette", "celui", "faire", "doit"}
        for kw in keywords:
            assert kw not in stopwords

    def test_min_length_parameter(self) -> None:
        keywords = extract_keywords("arbitre regle competition", min_length=6)
        for kw in keywords:
            assert len(kw) >= 6

    def test_normalizes_before_extraction(self) -> None:
        keywords = extract_keywords("L'Arbitre vérifie les PENDULES")
        # Should be lowercase and accent-stripped
        for kw in keywords:
            assert kw == kw.lower()
            assert "é" not in kw


# ---------------------------------------------------------------------------
# TestComputeKeywordScore
# ---------------------------------------------------------------------------


class TestComputeKeywordScore:
    """Tests for compute_keyword_score (PURE)."""

    def test_perfect_score(self) -> None:
        score = compute_keyword_score(
            "l'arbitre verifie les pendules",
            "L'arbitre doit verifier les pendules avant la competition.",
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_score(self) -> None:
        score = compute_keyword_score(
            "le football americain",
            "L'arbitre surveille les echecs du tournoi de blitz.",
        )
        assert score < 0.5

    def test_partial_score(self) -> None:
        score = compute_keyword_score(
            "l'arbitre joue du basketball",
            "L'arbitre surveille le deroulement du tournoi.",
        )
        assert 0.0 < score < 1.0

    def test_empty_answer(self) -> None:
        score = compute_keyword_score("", "chunk text")
        assert score == 0.0


# ---------------------------------------------------------------------------
# TestValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult dataclass (PURE)."""

    def test_fields(self) -> None:
        vr = ValidationResult(
            question_id="q1",
            chunk_id_valid=True,
            keyword_score=0.8,
            semantic_score=0.9,
            answerable=True,
            issues=[],
        )
        assert vr.question_id == "q1"
        assert vr.answerable is True

    def test_issues_list(self) -> None:
        vr = ValidationResult(
            question_id="q2",
            chunk_id_valid=False,
            keyword_score=0.0,
            semantic_score=None,
            answerable=False,
            issues=["invalid_chunk_id", "low_keyword_score:0.00"],
        )
        assert len(vr.issues) == 2


# ---------------------------------------------------------------------------
# TestComputeSemanticSimilarity (MOCK)
# ---------------------------------------------------------------------------


class TestComputeSemanticSimilarity:
    """Tests for compute_semantic_similarity (mock embedding model)."""

    @patch("scripts.evaluation.annales.validate_gs_quality.get_embedding_model")
    def test_returns_float(self, mock_get_model: MagicMock) -> None:
        model = MagicMock()
        # Return 2 normalized vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.8, 0.6, 0.0])
        model.encode.return_value = np.stack([v1, v2])
        mock_get_model.return_value = model

        score = compute_semantic_similarity("text1", "text2")
        assert isinstance(score, float)

    @patch("scripts.evaluation.annales.validate_gs_quality.get_embedding_model")
    def test_identical_high_score(self, mock_get_model: MagicMock) -> None:
        model = MagicMock()
        v = np.array([1.0, 0.0, 0.0])
        model.encode.return_value = np.stack([v, v])
        mock_get_model.return_value = model

        score = compute_semantic_similarity("same", "same")
        assert score == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# TestValidateQuestion (MIXED)
# ---------------------------------------------------------------------------


class TestValidateQuestion:
    """Tests for validate_question."""

    def test_invalid_chunk_id(self) -> None:
        question = {
            "id": "q1",
            "expected_chunk_id": "nonexistent-chunk",
            "expected_answer": "answer",
            "question": "Q?",
        }
        result = validate_question(question, {}, compute_semantic=False)
        assert result.chunk_id_valid is False
        assert "invalid_chunk_id" in result.issues

    def test_low_keyword_score(self) -> None:
        question = {
            "id": "q2",
            "expected_chunk_id": "chunk1",
            "expected_answer": "totally unrelated basketball answer",
            "question": "Q?",
        }
        chunk_index = {"chunk1": "L'arbitre verifie les pendules du tournoi."}
        result = validate_question(question, chunk_index, compute_semantic=False)
        assert any("low_keyword" in issue for issue in result.issues)

    @patch("scripts.evaluation.annales.validate_gs_quality.get_embedding_model")
    def test_answerable_by_semantic(self, mock_get_model: MagicMock) -> None:
        model = MagicMock()
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.9, np.sqrt(1 - 0.81)])
        model.encode.return_value = np.stack([v1, v2])
        mock_get_model.return_value = model

        question = {
            "id": "q3",
            "expected_chunk_id": "chunk1",
            "expected_answer": "reponse semantique",
            "question": "Question semantique?",
        }
        chunk_index = {"chunk1": "Texte du chunk pour validation semantique."}
        result = validate_question(question, chunk_index, compute_semantic=True)
        assert result.semantic_score is not None

    def test_skip_semantic(self) -> None:
        question = {
            "id": "q4",
            "expected_chunk_id": "chunk1",
            "expected_answer": "L'arbitre verifie les pendules du tournoi",
            "question": "Question?",
        }
        chunk_index = {"chunk1": "L'arbitre verifie les pendules du tournoi de blitz."}
        result = validate_question(question, chunk_index, compute_semantic=False)
        assert result.semantic_score is None
        assert result.keyword_score > 0


# ---------------------------------------------------------------------------
# TestValidateGoldStandard (MOCK)
# ---------------------------------------------------------------------------


class TestValidateGoldStandard:
    """Tests for validate_gold_standard."""

    @patch("scripts.evaluation.annales.validate_gs_quality.get_embedding_model")
    def test_metrics_calculated(self, mock_get_model: MagicMock) -> None:
        model = MagicMock()
        v = np.array([1.0, 0.0])
        model.encode.return_value = np.stack([v, v])
        mock_get_model.return_value = model

        gs = {
            "version": "test",
            "questions": [
                {
                    "id": "q1",
                    "expected_chunk_id": "c1",
                    "expected_answer": "L'arbitre verifie les pendules du tournoi",
                    "question": "Question test?",
                },
            ],
        }
        chunk_index = {"c1": "L'arbitre verifie les pendules du tournoi de blitz."}
        result = validate_gold_standard(gs, chunk_index, compute_semantic=True)
        assert "metrics" in result
        assert result["metrics"]["chunk_id_valid"] == 1

    @patch("scripts.evaluation.annales.validate_gs_quality.get_embedding_model")
    def test_sample_size_respected(self, mock_get_model: MagicMock) -> None:
        model = MagicMock()
        v = np.array([1.0, 0.0])
        model.encode.return_value = np.stack([v, v])
        mock_get_model.return_value = model

        questions = [
            {
                "id": f"q{i}",
                "expected_chunk_id": "c1",
                "expected_answer": "L'arbitre verifie",
                "question": "Q?",
            }
            for i in range(100)
        ]
        gs = {"version": "test", "questions": questions}
        chunk_index = {"c1": "L'arbitre verifie les pendules."}
        result = validate_gold_standard(
            gs, chunk_index, compute_semantic=True, sample_size=10
        )
        assert result["total_validated"] == 10

    @patch("scripts.evaluation.annales.validate_gs_quality.get_embedding_model")
    def test_issues_summary_counted(self, mock_get_model: MagicMock) -> None:
        model = MagicMock()
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        model.encode.return_value = np.stack([v1, v2])
        mock_get_model.return_value = model

        gs = {
            "version": "test",
            "questions": [
                {
                    "id": "q1",
                    "expected_chunk_id": "bad_id",
                    "expected_answer": "answer",
                    "question": "Q?",
                },
            ],
        }
        result = validate_gold_standard(gs, {}, compute_semantic=False)
        assert "invalid_chunk_id" in result["issues_summary"]
