"""
Tests for verify_regression.py - Snapshot & Compare regression tool.

ISO Reference: ISO/IEC 29119 - Test design
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.annales.verify_regression import (
    ComparisonResult,
    compare_snapshot,
    create_snapshot,
    format_comparison_report,
    load_snapshot,
    save_snapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gs(questions: list[dict]) -> dict:
    """Build minimal GS data wrapper."""
    return {"questions": questions}


def _make_question(
    qid: str = "gs:scratch:answerable:0001:abc",
    is_impossible: bool = False,
    cognitive_level: str = "Understand",
    question_type: str = "procedural",
    difficulty: float = 0.5,
    reasoning_class: str = "reasoning",
    answer_type: str = "extractive",
    chunk_match_score: int = 100,
    priority_boost: float | None = 0.0,
) -> dict:
    """Build a minimal valid Schema v2 question."""
    processing: dict = {
        "chunk_match_score": chunk_match_score,
        "chunk_match_method": "by_design_input",
        "reasoning_class_method": "generation_prompt",
        "triplet_ready": not is_impossible,
        "extraction_flags": ["by_design"],
        "answer_source": "chunk_extraction",
        "quality_score": 0.8,
    }
    if priority_boost is not None:
        processing["priority_boost"] = priority_boost

    return {
        "id": qid,
        "legacy_id": "",
        "content": {
            "question": "Test question?",
            "expected_answer": "Answer" if not is_impossible else "",
            "is_impossible": is_impossible,
        },
        "mcq": {
            "original_question": "Test?",
            "choices": {},
            "mcq_answer": "",
            "correct_answer": "",
            "original_answer": "",
        },
        "provenance": {
            "chunk_id": "test.pdf-p001-parent001-child00",
            "docs": ["test.pdf"],
            "pages": [1],
            "article_reference": "Art. 1",
            "answer_explanation": "",
            "annales_source": None,
        },
        "classification": {
            "category": "arbitrage",
            "keywords": ["test"],
            "difficulty": difficulty,
            "question_type": question_type,
            "cognitive_level": cognitive_level,
            "reasoning_type": "single-hop",
            "reasoning_class": reasoning_class,
            "answer_type": answer_type,
            "hard_type": "ANSWERABLE" if not is_impossible else "OUT_OF_DATABASE",
        },
        "validation": {
            "status": "VALIDATED",
            "method": "by_design_generation",
            "reviewer": "claude_code",
            "answer_current": True,
            "verified_date": "2026-01-01",
            "pages_verified": True,
            "batch": "test",
        },
        "processing": processing,
        "audit": {
            "history": "[BY DESIGN] test",
            "qat_revalidation": None,
            "requires_inference": False,
        },
    }


# ===========================================================================
# TestCreateSnapshot
# ===========================================================================


class TestCreateSnapshot:
    """Tests for create_snapshot()."""

    def test_counts(self) -> None:
        """Snapshot counts answerable and unanswerable correctly."""
        questions = [
            _make_question("q1", is_impossible=False),
            _make_question("q2", is_impossible=False),
            _make_question("q3", is_impossible=True, priority_boost=None),
        ]
        snap = create_snapshot(_make_gs(questions))
        assert snap.total_questions == 3
        assert snap.answerable_count == 2
        assert snap.unanswerable_count == 1

    def test_ids(self) -> None:
        """All question IDs are captured."""
        questions = [_make_question(f"q{i}") for i in range(5)]
        snap = create_snapshot(_make_gs(questions))
        assert snap.question_ids == [f"q{i}" for i in range(5)]

    def test_cognitive_level_distribution(self) -> None:
        """Cognitive level distribution is correct."""
        questions = [
            _make_question("q1", cognitive_level="Remember"),
            _make_question("q2", cognitive_level="Understand"),
            _make_question("q3", cognitive_level="Understand"),
        ]
        snap = create_snapshot(_make_gs(questions))
        assert snap.cognitive_level_dist == {"Remember": 1, "Understand": 2}

    def test_difficulty_buckets(self) -> None:
        """Difficulty buckets (easy/medium/hard) are computed correctly."""
        questions = [
            _make_question("q1", difficulty=0.2),  # easy
            _make_question("q2", difficulty=0.5),  # medium
            _make_question("q3", difficulty=0.7),  # hard
            _make_question("q4", difficulty=0.39),  # easy (< 0.4)
            _make_question("q5", difficulty=0.4),  # medium (>= 0.4)
        ]
        snap = create_snapshot(_make_gs(questions))
        assert snap.difficulty_buckets == {"easy": 2, "medium": 2, "hard": 1}

    def test_field_counts_stats(self) -> None:
        """field_counts has min, max, avg."""
        questions = [_make_question("q1"), _make_question("q2")]
        snap = create_snapshot(_make_gs(questions))
        assert "min" in snap.field_counts
        assert "max" in snap.field_counts
        assert "avg" in snap.field_counts
        assert snap.field_counts["min"] > 0
        assert snap.field_counts["avg"] > 0

    def test_chunk_match_scores(self) -> None:
        """Chunk match scores captured for all questions."""
        questions = [_make_question("q1", chunk_match_score=100)]
        snap = create_snapshot(_make_gs(questions))
        assert snap.chunk_match_scores == [100]

    def test_empty_gs(self) -> None:
        """Empty GS produces zero counts."""
        snap = create_snapshot(_make_gs([]))
        assert snap.total_questions == 0
        assert snap.answerable_count == 0
        assert snap.unanswerable_count == 0
        assert snap.question_ids == []
        assert snap.field_counts == {"min": 0.0, "max": 0.0, "avg": 0.0}

    def test_source_file_recorded(self) -> None:
        """Source file is stored in snapshot."""
        snap = create_snapshot(_make_gs([]), source_file="test.json")
        assert snap.source_file == "test.json"

    def test_question_type_distribution(self) -> None:
        """question_type distribution is computed."""
        questions = [
            _make_question("q1", question_type="procedural"),
            _make_question("q2", question_type="factual"),
        ]
        snap = create_snapshot(_make_gs(questions))
        assert snap.question_type_dist == {"procedural": 1, "factual": 1}


# ===========================================================================
# TestSaveLoadSnapshot
# ===========================================================================


class TestSaveLoadSnapshot:
    """Tests for save_snapshot() and load_snapshot()."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Save then load produces identical snapshot."""
        questions = [
            _make_question("q1"),
            _make_question("q2", is_impossible=True, priority_boost=None),
        ]
        original = create_snapshot(_make_gs(questions), source_file="test.json")
        path = tmp_path / "snapshot.json"
        save_snapshot(original, path)

        loaded = load_snapshot(path)
        assert loaded.total_questions == original.total_questions
        assert loaded.question_ids == original.question_ids
        assert loaded.field_counts == original.field_counts
        assert loaded.cognitive_level_dist == original.cognitive_level_dist

    def test_file_created(self, tmp_path: Path) -> None:
        """Save creates the file on disk."""
        snap = create_snapshot(_make_gs([]))
        path = tmp_path / "sub" / "snapshot.json"
        save_snapshot(snap, path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["total_questions"] == 0


# ===========================================================================
# TestCompareSnapshot
# ===========================================================================


class TestCompareSnapshot:
    """Tests for compare_snapshot()."""

    def test_pass_identical(self) -> None:
        """Identical GS passes comparison."""
        questions = [_make_question("q1"), _make_question("q2")]
        gs = _make_gs(questions)
        baseline = create_snapshot(gs)
        result = compare_snapshot(baseline, gs)
        assert result.passed is True
        assert result.ids_lost == []
        assert result.ids_preserved == 2

    def test_pass_with_additions(self) -> None:
        """Adding new questions still passes."""
        orig = [_make_question("q1")]
        baseline = create_snapshot(_make_gs(orig))
        updated = [_make_question("q1"), _make_question("q2")]
        result = compare_snapshot(baseline, _make_gs(updated))
        assert result.passed is True
        assert result.ids_added == ["q2"]
        assert result.ids_preserved == 1

    def test_fail_ids_lost(self) -> None:
        """Losing an ID fails comparison."""
        orig = [_make_question("q1"), _make_question("q2")]
        baseline = create_snapshot(_make_gs(orig))
        reduced = [_make_question("q1")]
        result = compare_snapshot(baseline, _make_gs(reduced))
        assert result.passed is False
        assert result.ids_lost == ["q2"]

    def test_fail_chunk_score(self) -> None:
        """Non-100 chunk_match_score fails comparison."""
        questions = [_make_question("q1", chunk_match_score=100)]
        baseline = create_snapshot(_make_gs(questions))
        bad = [_make_question("q1", chunk_match_score=80)]
        result = compare_snapshot(baseline, _make_gs(bad))
        assert result.passed is False
        assert result.chunk_scores_valid is False

    def test_fail_field_count_degraded(self) -> None:
        """Decreasing field_counts.min fails comparison."""
        # Create baseline with priority_boost (more fields)
        q_full = _make_question("q1", priority_boost=0.1)
        baseline = create_snapshot(_make_gs([q_full]))

        # Current without priority_boost (fewer fields)
        q_reduced = _make_question("q1", priority_boost=None)
        result = compare_snapshot(baseline, _make_gs([q_reduced]))
        assert result.passed is False
        assert result.field_counts_valid is False


# ===========================================================================
# TestFormatReport
# ===========================================================================


class TestFormatReport:
    """Tests for format_comparison_report()."""

    def test_pass_report(self) -> None:
        """PASS report contains PASS keyword."""
        result = ComparisonResult(
            passed=True,
            ids_lost=[],
            ids_added=[],
            ids_preserved=10,
            chunk_scores_valid=True,
            field_counts_valid=True,
            messages=["PASS: All OK"],
        )
        report = format_comparison_report(result)
        assert "PASS" in report
        assert "IDs preserved: 10" in report

    def test_fail_report(self) -> None:
        """FAIL report contains FAIL keyword."""
        result = ComparisonResult(
            passed=False,
            ids_lost=["q1"],
            ids_added=[],
            ids_preserved=9,
            chunk_scores_valid=True,
            field_counts_valid=True,
            messages=["FAIL: 1 IDs lost"],
        )
        report = format_comparison_report(result)
        assert "FAIL" in report
        assert "IDs lost: 1" in report


# ===========================================================================
# TestCLI
# ===========================================================================


class TestCLI:
    """Tests for main() CLI."""

    def test_snapshot_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--snapshot creates a snapshot file."""
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(
            json.dumps({"questions": [_make_question("q1")]}),
            encoding="utf-8",
        )
        out_path = tmp_path / "snapshot.json"
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--snapshot", "--gs", str(gs_path), "--output", str(out_path)],
        )
        from scripts.evaluation.annales.verify_regression import main

        assert main() == 0
        assert out_path.exists()

    def test_compare_mode_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--compare returns 0 on pass."""
        q = _make_question("q1")
        gs_data = {"questions": [q]}
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        # Create baseline
        snap = create_snapshot(gs_data)
        snap_path = tmp_path / "baseline.json"
        save_snapshot(snap, snap_path)

        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--compare", "--gs", str(gs_path), "--baseline", str(snap_path)],
        )
        from scripts.evaluation.annales.verify_regression import main

        assert main() == 0

    def test_compare_mode_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--compare returns 1 on fail (lost ID)."""
        q1 = _make_question("q1")
        q2 = _make_question("q2")
        baseline_data = {"questions": [q1, q2]}
        current_data = {"questions": [q1]}

        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(current_data), encoding="utf-8")

        snap = create_snapshot(baseline_data)
        snap_path = tmp_path / "baseline.json"
        save_snapshot(snap, snap_path)

        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--compare", "--gs", str(gs_path), "--baseline", str(snap_path)],
        )
        from scripts.evaluation.annales.verify_regression import main

        assert main() == 1
