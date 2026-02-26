"""Tests for generate_v2_coverage (Phase B orchestrator).

ISO Reference: ISO/IEC 29119-3 - Test data coverage generation
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

from scripts.evaluation.annales.generate_v2_coverage import (
    STOP_KEYWORD_RATIO,
    BatchReport,
    CumulativeStats,
    check_stop_gates,
    format_progress,
    generate_batch,
    prioritize_chunks,
    run_phase_b_gates,
    update_cumulative,
    validate_question,
)

# ---------------------------------------------------------------------------
# CumulativeStats
# ---------------------------------------------------------------------------


class TestCumulativeStats:
    """Test cumulative stats tracking."""

    def test_fact_single_ratio_empty(self) -> None:
        cs = CumulativeStats()
        assert cs.fact_single_ratio() == 0.0

    def test_fact_single_ratio(self) -> None:
        cs = CumulativeStats()
        cs.reasoning_class = Counter(
            {"fact_single": 60, "summary": 20, "reasoning": 20}
        )
        assert cs.fact_single_ratio() == 0.6

    def test_hard_ratio_empty(self) -> None:
        cs = CumulativeStats()
        assert cs.hard_ratio() == 0.0

    def test_hard_ratio(self) -> None:
        cs = CumulativeStats()
        cs.difficulty_buckets = Counter({"easy": 30, "medium": 55, "hard": 15})
        assert cs.hard_ratio() == 0.15


# ---------------------------------------------------------------------------
# prioritize_chunks
# ---------------------------------------------------------------------------


class TestPrioritizeChunks:
    """Test chunk priority ordering."""

    def test_small_docs_before_la(self) -> None:
        chunks = [
            {"id": "LA-oct-p1", "source": "LA-octobre2025.pdf", "page": 1},
            {"id": "C01-p1", "source": "C01_Coupe.pdf", "page": 1},
            {"id": "J01-p1", "source": "J01_Jeunes.pdf", "page": 1},
        ]
        result = prioritize_chunks(chunks)
        # C01 and J01 should come before LA
        assert result[0]["id"] == "C01-p1"
        assert result[1]["id"] == "J01-p1"
        assert result[2]["id"] == "LA-oct-p1"

    def test_page_order_within_group(self) -> None:
        chunks = [
            {"id": "doc-p3", "source": "doc.pdf", "page": 3},
            {"id": "doc-p1", "source": "doc.pdf", "page": 1},
            {"id": "doc-p2", "source": "doc.pdf", "page": 2},
        ]
        result = prioritize_chunks(chunks)
        assert [c["page"] for c in result] == [1, 2, 3]

    def test_empty_list(self) -> None:
        assert prioritize_chunks([]) == []


# ---------------------------------------------------------------------------
# check_stop_gates
# ---------------------------------------------------------------------------


class TestCheckStopGates:
    """Test stop gate logic."""

    def test_no_stop_normal_batch(self) -> None:
        report = BatchReport(
            batch_num=0,
            chunks_processed=30,
            questions_generated=50,
            empty_chunks=3,
            low_keyword_count=2,
        )
        cumulative = CumulativeStats()
        should_stop, reason = check_stop_gates(report, cumulative)
        assert should_stop is False
        assert reason == ""

    def test_stop_high_low_keyword(self) -> None:
        report = BatchReport(
            batch_num=0,
            chunks_processed=30,
            questions_generated=20,
            empty_chunks=0,
            low_keyword_count=10,  # 50% > STOP_KEYWORD_RATIO
        )
        cumulative = CumulativeStats()
        should_stop, reason = check_stop_gates(report, cumulative)
        assert should_stop is True
        assert "keyword" in reason.lower()

    def test_stop_too_many_empty(self) -> None:
        report = BatchReport(
            batch_num=0,
            chunks_processed=30,
            questions_generated=5,
            empty_chunks=20,  # 67% > 50%
        )
        cumulative = CumulativeStats()
        should_stop, reason = check_stop_gates(report, cumulative)
        assert should_stop is True
        assert "0 questions" in reason

    def test_no_stop_zero_questions(self) -> None:
        """Edge case: batch produced 0 questions (empty_chunks check)."""
        report = BatchReport(
            batch_num=0,
            chunks_processed=30,
            questions_generated=0,
            empty_chunks=30,
        )
        cumulative = CumulativeStats()
        should_stop, _ = check_stop_gates(report, cumulative)
        assert should_stop is True

    def test_keyword_threshold_boundary(self) -> None:
        """Exactly at threshold should not stop."""
        report = BatchReport(
            batch_num=0,
            chunks_processed=30,
            questions_generated=100,
            empty_chunks=0,
            low_keyword_count=int(100 * STOP_KEYWORD_RATIO),  # exactly at threshold
        )
        cumulative = CumulativeStats()
        should_stop, _ = check_stop_gates(report, cumulative)
        assert should_stop is False


# ---------------------------------------------------------------------------
# update_cumulative
# ---------------------------------------------------------------------------


class TestUpdateCumulative:
    """Test cumulative stats update."""

    def test_basic_update(self) -> None:
        cumulative = CumulativeStats()
        questions = [
            {
                "reasoning_class": "fact_single",
                "cognitive_level": "Remember",
                "question_type": "factual",
                "answer_type": "extractive",
                "difficulty": 0.3,
                "chunk_id": "c1",
            },
            {
                "reasoning_class": "reasoning",
                "cognitive_level": "Apply",
                "question_type": "scenario",
                "answer_type": "inferential",
                "difficulty": 0.8,
                "chunk_id": "c2",
            },
        ]
        report = BatchReport(
            batch_num=0,
            chunks_processed=2,
            questions_generated=2,
            empty_chunks=0,
        )
        update_cumulative(cumulative, questions, report)

        assert cumulative.total_chunks == 2
        assert cumulative.total_questions == 2
        assert cumulative.reasoning_class["fact_single"] == 1
        assert cumulative.reasoning_class["reasoning"] == 1
        assert cumulative.cognitive_level["Apply"] == 1
        assert cumulative.difficulty_buckets["easy"] == 1
        assert cumulative.difficulty_buckets["hard"] == 1
        assert len(cumulative.covered_chunks) == 2

    def test_accumulates_across_batches(self) -> None:
        cumulative = CumulativeStats()
        q1 = [
            {
                "reasoning_class": "fact_single",
                "cognitive_level": "Remember",
                "question_type": "factual",
                "answer_type": "extractive",
                "difficulty": 0.3,
                "chunk_id": "c1",
            }
        ]
        q2 = [
            {
                "reasoning_class": "summary",
                "cognitive_level": "Understand",
                "question_type": "factual",
                "answer_type": "extractive",
                "difficulty": 0.5,
                "chunk_id": "c2",
            }
        ]
        r1 = BatchReport(0, 1, 1, 0)
        r2 = BatchReport(1, 1, 1, 0)

        update_cumulative(cumulative, q1, r1)
        update_cumulative(cumulative, q2, r2)

        assert cumulative.total_questions == 2
        assert cumulative.total_chunks == 2


# ---------------------------------------------------------------------------
# format_progress
# ---------------------------------------------------------------------------


class TestFormatProgress:
    """Test progress formatting."""

    def test_basic_format(self) -> None:
        report = BatchReport(0, 30, 50, 3)
        cumulative = CumulativeStats()
        cumulative.total_questions = 50
        cumulative.covered_chunks = {"c1", "c2", "c3"}
        cumulative.reasoning_class = Counter({"fact_single": 25, "summary": 25})
        cumulative.difficulty_buckets = Counter({"easy": 10, "medium": 30, "hard": 10})

        result = format_progress(1, 10, report, cumulative)

        assert "Batch 1/10" in result
        assert "+50Q" in result
        assert "3 empty" in result
        assert "Total: 50Q" in result


# ---------------------------------------------------------------------------
# run_phase_b_gates
# ---------------------------------------------------------------------------


class TestRunPhaseBGates:
    """Test Phase B quality gates."""

    def test_all_pass(self) -> None:
        cumulative = CumulativeStats()
        cumulative.cognitive_level = Counter(
            {"Remember": 30, "Understand": 30, "Apply": 20, "Analyze": 20}
        )
        cumulative.question_type = Counter(
            {"factual": 40, "procedural": 30, "scenario": 20, "comparative": 10}
        )
        cumulative.difficulty_buckets = Counter({"easy": 20, "medium": 50, "hard": 30})
        cumulative.covered_chunks = set(range(1200))

        gates = run_phase_b_gates(
            cumulative, total_corpus_chunks=1857, existing_covered=356
        )
        # 356 + 1200 = 1556 / 1857 = 83.8%
        assert all(g.passed for g in gates)

    def test_coverage_fail(self) -> None:
        cumulative = CumulativeStats()
        cumulative.cognitive_level = Counter({"Remember": 50, "Apply": 50})
        cumulative.question_type = Counter({"factual": 50, "comparative": 50})
        cumulative.difficulty_buckets = Counter({"hard": 100})
        cumulative.covered_chunks = set(range(100))

        gates = run_phase_b_gates(
            cumulative, total_corpus_chunks=1857, existing_covered=356
        )
        coverage_gate = gates[0]
        assert coverage_gate.name == "B-G1: coverage >= 80%"
        assert coverage_gate.passed is False

    def test_apply_fail(self) -> None:
        cumulative = CumulativeStats()
        cumulative.cognitive_level = Counter({"Remember": 90, "Understand": 10})
        cumulative.question_type = Counter({"factual": 100})
        cumulative.difficulty_buckets = Counter({"medium": 100})
        cumulative.covered_chunks = set(range(1200))

        gates = run_phase_b_gates(
            cumulative, total_corpus_chunks=1857, existing_covered=356
        )
        apply_gate = next(g for g in gates if "Apply" in g.name)
        assert apply_gate.passed is False

    def test_hard_fail(self) -> None:
        cumulative = CumulativeStats()
        cumulative.cognitive_level = Counter({"Apply": 50, "Analyze": 50})
        cumulative.question_type = Counter({"comparative": 100})
        cumulative.difficulty_buckets = Counter({"easy": 50, "medium": 50})
        cumulative.covered_chunks = set(range(1200))

        gates = run_phase_b_gates(
            cumulative, total_corpus_chunks=1857, existing_covered=356
        )
        hard_gate = next(g for g in gates if "hard" in g.name)
        assert hard_gate.passed is False

    def test_empty_stats(self) -> None:
        cumulative = CumulativeStats()
        gates = run_phase_b_gates(cumulative, 1857, 0)
        assert len(gates) == 5
        assert not any(g.passed for g in gates)


# ---------------------------------------------------------------------------
# validate_question (Plan §5.5: G1-1, G1-2, G1-4)
# ---------------------------------------------------------------------------


class TestValidateQuestion:
    """Test per-question gate validation."""

    def test_valid_question(self) -> None:
        q = {
            "question": "Que doit faire l'arbitre?",
            "expected_answer": "L'arbitre doit prendre une decision",
        }
        chunk = {"text": "L'arbitre doit prendre une decision dans les 5 minutes."}
        assert validate_question(q, chunk) is True

    def test_rejects_no_question_mark(self) -> None:
        """G1-4: question must end with '?'."""
        q = {
            "question": "Que doit faire l'arbitre",
            "expected_answer": "L'arbitre doit prendre une decision",
        }
        chunk = {"text": "L'arbitre doit prendre une decision."}
        assert validate_question(q, chunk) is False

    def test_rejects_short_answer(self) -> None:
        """G1-2: answer must be >= 6 chars."""
        q = {"question": "Test?", "expected_answer": "Oui"}
        chunk = {"text": "Oui c'est correct."}
        assert validate_question(q, chunk) is False

    def test_rejects_empty_answer(self) -> None:
        q = {"question": "Test?", "expected_answer": ""}
        chunk = {"text": "Some text."}
        assert validate_question(q, chunk) is False

    def test_rejects_low_keyword_overlap(self) -> None:
        """G1-2: answer words must overlap with chunk."""
        q = {
            "question": "Quelle est la regle?",
            "expected_answer": "Le joueur doit respecter les consignes specifiques",
        }
        chunk = {"text": "L'arbitre veille au bon deroulement de la competition."}
        assert validate_question(q, chunk) is False

    def test_accepts_boundary_overlap(self) -> None:
        """At exactly STOP_KEYWORD_THRESHOLD, should pass."""
        # Create answer where exactly threshold fraction of words are in chunk
        q = {
            "question": "Quelle obligation?",
            "expected_answer": "L'arbitre doit veiller au deroulement",
        }
        chunk = {"text": "L'arbitre doit veiller au bon deroulement de la competition."}
        assert validate_question(q, chunk) is True

    def test_empty_chunk_text(self) -> None:
        """Empty chunk text: still valid if answer is long enough (no overlap check)."""
        q = {
            "question": "Quelle regle?",
            "expected_answer": "Regle importante du reglement",
        }
        chunk = {"text": ""}
        assert validate_question(q, chunk) is True


# ---------------------------------------------------------------------------
# generate_batch (with mock)
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    """Test batch generation with mocked question generator."""

    def test_basic_batch(self) -> None:
        chunks = [
            {
                "id": "c1",
                "text": "L'arbitre doit prendre une decision.",
                "source": "doc.pdf",
                "page": 1,
                "pages": [1],
            },
            {
                "id": "c2",
                "text": "Le joueur peut demander une pause.",
                "source": "doc.pdf",
                "page": 2,
                "pages": [2],
            },
        ]

        def mock_gen(chunk: dict, target_count: int = 2) -> list[dict]:
            """Return a valid question whose answer matches the chunk."""
            text = chunk.get("text", "")
            return [
                {
                    "question": "Quelle est la regle?",
                    "expected_answer": text.rstrip("."),
                    "reasoning_class": "fact_single",
                    "cognitive_level": "Remember",
                    "question_type": "factual",
                },
            ]

        with patch(
            "scripts.evaluation.annales.generate_v2_coverage.generate_questions_from_chunk",
            side_effect=mock_gen,
        ):
            questions, report = generate_batch(chunks, batch_num=0)

        assert report.chunks_processed == 2
        assert report.questions_generated == 2  # 1 per chunk, 2 chunks
        assert report.empty_chunks == 0
        assert all(q.get("chunk_id") for q in questions)

    def test_empty_chunk_handling(self) -> None:
        chunks = [{"id": "c1", "text": "Short.", "source": "doc.pdf", "page": 1}]

        with patch(
            "scripts.evaluation.annales.generate_v2_coverage.generate_questions_from_chunk",
            return_value=[],
        ):
            questions, report = generate_batch(chunks, batch_num=0)

        assert report.empty_chunks == 1
        assert report.questions_generated == 0
        assert questions == []

    def test_filters_invalid_questions(self) -> None:
        """Questions failing validate_question are excluded."""
        chunks = [
            {
                "id": "c1",
                "text": "L'arbitre doit prendre une decision.",
                "source": "doc.pdf",
                "page": 1,
            }
        ]
        bad_question = [
            {
                "question": "No question mark",
                "expected_answer": "L'arbitre doit prendre une decision",
                "reasoning_class": "fact_single",
            },
        ]

        with patch(
            "scripts.evaluation.annales.generate_v2_coverage.generate_questions_from_chunk",
            return_value=bad_question,
        ):
            questions, report = generate_batch(chunks, batch_num=0)

        assert report.questions_generated == 0
        assert questions == []
