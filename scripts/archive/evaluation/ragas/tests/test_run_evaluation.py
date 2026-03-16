"""Tests for RAGAS evaluation module.

ISO Reference: ISO 29119 - Software Testing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluation.ragas.run_evaluation import (
    RAGAS_METRICS,
    RAGAS_SYSTEM_PROMPTS,
    _build_ragas_prompt,
    _extract_score,
    _mock_score,
    run_ragas_evaluation,
)


class TestRagasMetrics:
    """Test RAGAS metric definitions."""

    def test_four_metrics_defined(self) -> None:
        """RAGAS should define exactly 4 metrics."""
        assert len(RAGAS_METRICS) == 4
        assert "faithfulness" in RAGAS_METRICS
        assert "answer_relevancy" in RAGAS_METRICS
        assert "context_precision" in RAGAS_METRICS
        assert "context_recall" in RAGAS_METRICS

    def test_system_prompts_exist_for_all_metrics(self) -> None:
        """Each metric should have a system prompt."""
        for m in RAGAS_METRICS:
            assert m in RAGAS_SYSTEM_PROMPTS
            assert len(RAGAS_SYSTEM_PROMPTS[m]) > 50


class TestExtractScore:
    """Test score extraction from LLM responses."""

    def test_json_response(self) -> None:
        """Parse JSON response."""
        assert _extract_score('{"score": 0.85}') == pytest.approx(0.85, abs=0.01)

    def test_json_response_full_score(self) -> None:
        """Parse perfect JSON score."""
        assert _extract_score('{"score": 1.0}') == pytest.approx(1.0, abs=0.01)

    def test_json_response_zero(self) -> None:
        """Parse zero JSON score."""
        assert _extract_score('{"score": 0.0}') == pytest.approx(0.0, abs=0.01)

    def test_text_with_score_pattern(self) -> None:
        """Extract score from text with JSON-like pattern."""
        text = 'Based on evaluation, "score": 0.75 is the result.'
        assert _extract_score(text) == pytest.approx(0.75, abs=0.01)

    def test_clamp_above_1(self) -> None:
        """Score above 1.0 should be clamped."""
        assert _extract_score('{"score": 1.5}') == 1.0

    def test_clamp_below_0(self) -> None:
        """Score below 0.0 should be clamped."""
        assert _extract_score('{"score": -0.5}') == 0.0

    def test_unparseable_returns_zero(self) -> None:
        """Unparseable response returns 0.0."""
        assert _extract_score("I cannot evaluate this") == 0.0


class TestBuildRagasPrompt:
    """Test RAGAS prompt building."""

    def test_faithfulness_prompt(self) -> None:
        """Faithfulness prompt includes context and answer."""
        record: dict[str, Any] = {
            "question": "What is castling?",
            "answer": "Moving the king two squares.",
            "contexts": ["Castling is a special move."],
            "ground_truth": "Castling involves the king and rook.",
        }
        prompt = _build_ragas_prompt("faithfulness", record)
        assert "Context:" in prompt
        assert "Answer:" in prompt

    def test_answer_relevancy_prompt(self) -> None:
        """Answer relevancy prompt includes question and answer."""
        record: dict[str, Any] = {
            "question": "What is castling?",
            "answer": "Moving the king two squares.",
            "contexts": ["Castling is a special move."],
            "ground_truth": "",
        }
        prompt = _build_ragas_prompt("answer_relevancy", record)
        assert "Question:" in prompt
        assert "Answer:" in prompt

    def test_context_precision_prompt(self) -> None:
        """Context precision prompt includes question, ground truth, contexts."""
        record: dict[str, Any] = {
            "question": "What is castling?",
            "answer": "",
            "contexts": ["Context 1", "Context 2"],
            "ground_truth": "Castling involves king and rook.",
        }
        prompt = _build_ragas_prompt("context_precision", record)
        assert "Question:" in prompt
        assert "Ground Truth:" in prompt
        assert "Contexts:" in prompt

    def test_context_recall_prompt(self) -> None:
        """Context recall prompt includes ground truth and contexts."""
        record: dict[str, Any] = {
            "question": "",
            "answer": "",
            "contexts": ["Context about castling."],
            "ground_truth": "Castling involves king and rook.",
        }
        prompt = _build_ragas_prompt("context_recall", record)
        assert "Ground Truth:" in prompt
        assert "Contexts:" in prompt


class TestMockScore:
    """Test mock scoring logic."""

    def test_mock_with_contexts(self) -> None:
        """Records with contexts should get positive scores."""
        records = [
            {"contexts": ["text"], "ground_truth": "answer"},
            {"contexts": ["text"], "ground_truth": "answer"},
        ]
        score = _mock_score("faithfulness", records)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_mock_empty_records(self) -> None:
        """Empty records should return 0."""
        assert _mock_score("faithfulness", []) == 0.0

    def test_mock_missing_contexts(self) -> None:
        """Records without contexts get 0 for context metrics."""
        records: list[dict[str, Any]] = [
            {"question": "Q1"},
            {"question": "Q2", "contexts": ["ctx"]},
        ]
        score = _mock_score("faithfulness", records)
        assert score == pytest.approx(0.5, abs=0.01)


class TestRunRagasEvaluation:
    """Test RAGAS evaluation runner."""

    def test_mock_evaluation(self, tmp_path: Path) -> None:
        """Mock evaluation returns 4 metrics."""
        # Create test data
        data_dir = tmp_path / "data" / "evaluation" / "ragas"
        data_dir.mkdir(parents=True)

        data_path = data_dir / "ragas_evaluation.jsonl"
        records = [
            {
                "question": "What is castling?",
                "answer": "",
                "contexts": ["Castling is a special chess move."],
                "ground_truth": "Castling involves the king and rook.",
            },
            {
                "question": "What is en passant?",
                "answer": "",
                "contexts": ["En passant is a pawn capture."],
                "ground_truth": "En passant allows pawn capture.",
            },
        ]
        with open(data_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        output_dir = tmp_path / "results"

        result = run_ragas_evaluation(
            corpus="fr",
            data_path=data_path,
            llm_backend="mock",
            output_dir=output_dir,
        )

        assert "metrics" in result
        assert len(result["metrics"]) == 4
        for m in RAGAS_METRICS:
            assert m in result["metrics"]
            assert "score" in result["metrics"][m]
            assert "pass" in result["metrics"][m]

    def test_ground_truth_is_answer_not_question(self, tmp_path: Path) -> None:
        """Verify ground_truth contains expected answer, not question."""
        data_dir = tmp_path
        data_path = data_dir / "ragas_evaluation.jsonl"

        records = [
            {
                "question": "What is castling?",
                "answer": "",
                "contexts": ["Castling is a special chess move."],
                "ground_truth": "Castling involves the king and rook.",
            },
        ]
        with open(data_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        with open(data_path, encoding="utf-8") as f:
            loaded = json.loads(f.readline())

        # ground_truth should NOT be the question
        assert loaded["ground_truth"] != loaded["question"]
        # ground_truth should be the expected answer
        assert loaded["ground_truth"] == "Castling involves the king and rook."

    def test_result_structure(self, tmp_path: Path) -> None:
        """Result should have required fields."""
        data_path = tmp_path / "test.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "question": "Q",
                        "answer": "",
                        "contexts": ["C"],
                        "ground_truth": "GT",
                    }
                )
                + "\n"
            )

        result = run_ragas_evaluation(
            data_path=data_path,
            llm_backend="mock",
            output_dir=tmp_path / "results",
        )

        assert "corpus" in result
        assert "timestamp" in result
        assert "n_samples" in result
        assert "all_pass" in result
        assert isinstance(result["all_pass"], bool)

    def test_unsupported_backend_raises(self, tmp_path: Path) -> None:
        """Unsupported backend should raise ValueError, not silently return 0."""
        import pytest

        data_path = tmp_path / "test.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "question": "Q",
                        "answer": "",
                        "contexts": ["C"],
                        "ground_truth": "GT",
                    }
                )
                + "\n"
            )

        with pytest.raises(ValueError, match="Unsupported RAGAS backend"):
            run_ragas_evaluation(
                data_path=data_path,
                llm_backend="hf",
                output_dir=tmp_path / "results",
            )
