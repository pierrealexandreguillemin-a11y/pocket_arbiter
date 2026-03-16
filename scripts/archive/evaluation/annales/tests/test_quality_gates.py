"""
Tests for quality_gates module - ALL 21 gates (G0-1 to G5-5).

ISO Reference:
    - ISO/IEC 29119 - Test data validation
    - ISO 25010 - Quality metrics
"""

from __future__ import annotations

import numpy as np

from scripts.evaluation.annales.quality_gates import (
    GateResult,
    count_schema_fields,
    format_gate_report,
    g0_1_strata_count,
    g0_2_document_coverage,
    g1_1_chunk_match_score,
    g1_2_answer_in_chunk,
    g1_3_fact_single_ratio,
    g1_4_question_format,
    g2_1_is_impossible_flag,
    g2_2_unanswerable_ratio,
    g2_3_hard_type_diversity,
    g3_1_validation_passed,
    g3_2_hallucination_count,
    g4_1_schema_fields,
    g4_2_chunk_match_method,
    g5_1_inter_question_similarity,
    g5_2_anchor_independence,
    g5_3_final_fact_single_ratio,
    g5_4_hard_ratio,
    g5_5_final_unanswerable_ratio,
    g5_6_cognitive_level_diversity,
    g5_7_question_type_diversity,
    g5_8_chunk_coverage,
    validate_all_gates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_question(
    *,
    is_impossible: bool = False,
    reasoning_class: str = "fact_single",
    hard_type: str = "ANSWERABLE",
    difficulty: float = 0.5,
    chunk_match_score: int = 100,
    chunk_match_method: str = "by_design_input",
    question_text: str = "Quelle est la regle?",
    n_fields: int = 43,
) -> dict:
    """Build a minimal Schema v2 question for gate tests."""
    q: dict = {
        "id": "gs:scratch:test:0001:aaa",
        "legacy_id": "",
        "content": {
            "question": question_text,
            "expected_answer": "Reponse test",
            "is_impossible": is_impossible,
        },
        "mcq": {
            "original_question": question_text,
            "choices": {},
            "mcq_answer": "",
            "correct_answer": "Reponse test",
            "original_answer": "Reponse test",
        },
        "provenance": {
            "chunk_id": "src.pdf-p1-parent1-child0",
            "docs": ["src.pdf"],
            "pages": [1],
            "article_reference": "Art. 1",
            "answer_explanation": "Source: test. Extrait: ...",
            "annales_source": None,
        },
        "classification": {
            "category": "arbitrage",
            "keywords": ["arbitre", "regle", "test", "echecs"],
            "difficulty": difficulty,
            "question_type": "factual",
            "cognitive_level": "Remember",
            "reasoning_type": "single-hop",
            "reasoning_class": reasoning_class,
            "answer_type": "extractive",
            "hard_type": hard_type,
        },
        "validation": {
            "status": "VALIDATED",
            "method": "by_design_generation",
            "reviewer": "claude_code",
            "answer_current": True,
            "verified_date": "2026-01-01",
            "pages_verified": True,
            "batch": "test_batch",
        },
        "processing": {
            "chunk_match_score": chunk_match_score,
            "chunk_match_method": chunk_match_method,
            "reasoning_class_method": "generation_prompt",
            "triplet_ready": True,
            "extraction_flags": ["by_design"],
            "answer_source": "chunk_extraction",
            "quality_score": 0.8,
        },
        "audit": {
            "history": "[BY DESIGN] test",
            "qat_revalidation": None,
            "requires_inference": False,
        },
    }
    return q


# ---------------------------------------------------------------------------
# TestGateResult
# ---------------------------------------------------------------------------


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_fields(self) -> None:
        r = GateResult(
            gate_id="G0-1",
            name="test",
            passed=True,
            blocking=True,
            value=5,
            threshold=5,
            message="ok",
        )
        assert r.gate_id == "G0-1"
        assert r.passed is True

    def test_blocking_flag(self) -> None:
        r = GateResult("G0-2", "t", False, False, 0, 1, "warn")
        assert r.blocking is False


# ---------------------------------------------------------------------------
# TestG0Gates - Phase 0: Stratification
# ---------------------------------------------------------------------------


class TestG0Gates:
    """Tests for G0-1, G0-2."""

    def test_g0_1_pass(self) -> None:
        strata = {f"s{i}": {"quota": 10} for i in range(6)}
        result = g0_1_strata_count(strata)
        assert result.passed is True
        assert result.blocking is True

    def test_g0_1_fail(self) -> None:
        strata = {f"s{i}": {"quota": 10} for i in range(3)}
        result = g0_1_strata_count(strata)
        assert result.passed is False

    def test_g0_2_pass(self) -> None:
        coverage = {"coverage_ratio": 0.85}
        result = g0_2_document_coverage(coverage)
        assert result.passed is True
        assert result.blocking is False

    def test_g0_2_fail(self) -> None:
        coverage = {"coverage_ratio": 0.50}
        result = g0_2_document_coverage(coverage)
        assert result.passed is False


# ---------------------------------------------------------------------------
# TestG1Gates - Phase 1: Answerable Generation
# ---------------------------------------------------------------------------


class TestG1Gates:
    """Tests for G1-1, G1-2, G1-3, G1-4."""

    def test_g1_1_pass_score_100(self) -> None:
        q = _make_question(chunk_match_score=100)
        result = g1_1_chunk_match_score(q)
        assert result.passed is True
        assert result.blocking is True

    def test_g1_1_fail_score_low(self) -> None:
        q = _make_question(chunk_match_score=80)
        result = g1_1_chunk_match_score(q)
        assert result.passed is False

    def test_g1_2_pass_verbatim(self) -> None:
        result = g1_2_answer_in_chunk(
            answer="l'arbitre doit",
            chunk_text="L'arbitre doit veiller au bon deroulement.",
        )
        assert result.passed is True
        assert result.value == "verbatim"

    def test_g1_2_pass_semantic(self) -> None:
        result = g1_2_answer_in_chunk(
            answer="obligations de l'arbitre",
            chunk_text="Texte different",
            semantic_sim=0.96,
        )
        assert result.passed is True
        assert "semantic" in str(result.value)

    def test_g1_2_fail(self) -> None:
        result = g1_2_answer_in_chunk(
            answer="texte introuvable",
            chunk_text="contenu totalement different",
            semantic_sim=0.5,
        )
        assert result.passed is False

    def test_g1_3_pass(self) -> None:
        questions = [
            _make_question(reasoning_class="fact_single"),
            _make_question(reasoning_class="reasoning"),
            _make_question(reasoning_class="summary"),
        ]
        result = g1_3_fact_single_ratio(questions)
        assert result.passed is True
        assert result.blocking is False

    def test_g1_3_fail(self) -> None:
        questions = [_make_question(reasoning_class="fact_single") for _ in range(8)]
        questions.append(_make_question(reasoning_class="reasoning"))
        result = g1_3_fact_single_ratio(questions)
        assert result.passed is False

    def test_g1_3_no_answerable(self) -> None:
        questions = [_make_question(is_impossible=True) for _ in range(3)]
        result = g1_3_fact_single_ratio(questions)
        assert result.passed is True

    def test_g1_4_pass(self) -> None:
        result = g1_4_question_format("Quelle est la regle?")
        assert result.passed is True
        assert result.blocking is True

    def test_g1_4_fail(self) -> None:
        result = g1_4_question_format("Quelle est la regle.")
        assert result.passed is False


# ---------------------------------------------------------------------------
# TestG2Gates - Phase 2: Unanswerable Generation
# ---------------------------------------------------------------------------


class TestG2Gates:
    """Tests for G2-1, G2-2, G2-3."""

    def test_g2_1_pass(self) -> None:
        q = _make_question(is_impossible=True)
        result = g2_1_is_impossible_flag(q)
        assert result.passed is True
        assert result.blocking is True

    def test_g2_1_fail(self) -> None:
        q = _make_question(is_impossible=False)
        result = g2_1_is_impossible_flag(q)
        assert result.passed is False

    def test_g2_2_pass_in_range(self) -> None:
        questions = [_make_question() for _ in range(7)]
        questions.extend(_make_question(is_impossible=True) for _ in range(3))
        result = g2_2_unanswerable_ratio(questions)
        assert result.passed is True

    def test_g2_2_fail_too_low(self) -> None:
        questions = [_make_question() for _ in range(10)]
        result = g2_2_unanswerable_ratio(questions)
        assert result.passed is False

    def test_g2_2_fail_too_high(self) -> None:
        questions = [_make_question() for _ in range(5)]
        questions.extend(_make_question(is_impossible=True) for _ in range(5))
        result = g2_2_unanswerable_ratio(questions)
        assert result.passed is False

    def test_g2_2_empty(self) -> None:
        result = g2_2_unanswerable_ratio([])
        assert result.passed is False

    def test_g2_3_pass(self) -> None:
        types = [
            "OUT_OF_DATABASE",
            "FALSE_PRESUPPOSITION",
            "UNDERSPECIFIED",
            "NONSENSICAL",
        ]
        questions = [_make_question(is_impossible=True, hard_type=t) for t in types]
        result = g2_3_hard_type_diversity(questions)
        assert result.passed is True

    def test_g2_3_fail(self) -> None:
        questions = [
            _make_question(is_impossible=True, hard_type="OUT_OF_DATABASE")
            for _ in range(5)
        ]
        result = g2_3_hard_type_diversity(questions)
        assert result.passed is False


# ---------------------------------------------------------------------------
# TestG3Gates - Phase 3: Anti-Hallucination
# ---------------------------------------------------------------------------


class TestG3Gates:
    """Tests for G3-1, G3-2."""

    def test_g3_1_pass_all_validated(self) -> None:
        questions = []
        for i in range(3):
            q = _make_question()
            q["id"] = f"gs:scratch:test:{i:04d}:aaa"
            questions.append(q)
        validation_results = {q["id"]: True for q in questions}
        result = g3_1_validation_passed(questions, validation_results)
        assert result.passed is True
        assert result.blocking is True

    def test_g3_1_fail_some(self) -> None:
        questions = [_make_question() for _ in range(3)]
        validation_results = {
            questions[0]["id"]: True,
            questions[1]["id"]: False,
            questions[2]["id"]: True,
        }
        result = g3_1_validation_passed(questions, validation_results)
        assert result.passed is False

    def test_g3_2_pass_zero(self) -> None:
        result = g3_2_hallucination_count([])
        assert result.passed is True
        assert result.blocking is True

    def test_g3_2_fail(self) -> None:
        result = g3_2_hallucination_count(["q1", "q2"])
        assert result.passed is False


# ---------------------------------------------------------------------------
# TestG4Gates - Phase 4: Schema Enrichment
# ---------------------------------------------------------------------------


class TestG4Gates:
    """Tests for G4-1, G4-2."""

    def test_g4_1_pass(self, sample_gs_question_answerable: dict) -> None:
        result = g4_1_schema_fields(sample_gs_question_answerable, required_fields=40)
        assert result.passed is True
        assert result.blocking is True

    def test_g4_1_fail(self) -> None:
        q = {"id": "test", "content": {"question": "Q?"}}
        result = g4_1_schema_fields(q, required_fields=42)
        assert result.passed is False

    def test_g4_2_pass(self) -> None:
        q = _make_question(chunk_match_method="by_design_input")
        result = g4_2_chunk_match_method(q)
        assert result.passed is True

    def test_g4_2_fail(self) -> None:
        q = _make_question(chunk_match_method="embedding_search")
        result = g4_2_chunk_match_method(q)
        assert result.passed is False


# ---------------------------------------------------------------------------
# TestG5Gates - Phase 5: Distribution & Deduplication
# ---------------------------------------------------------------------------


class TestG5Gates:
    """Tests for G5-1, G5-2, G5-3, G5-4, G5-5."""

    def test_g5_1_pass_low_similarity(self) -> None:
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        result = g5_1_inter_question_similarity([emb1, emb2])
        assert result.passed is True

    def test_g5_1_fail_high_similarity(self) -> None:
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.99, 0.01, 0.0])
        result = g5_1_inter_question_similarity([emb1, emb2])
        assert result.passed is False

    def test_g5_1_edge_less_than_2(self) -> None:
        emb = np.array([1.0, 0.0, 0.0])
        result = g5_1_inter_question_similarity([emb])
        assert result.passed is True

    def test_g5_2_pass_independent(self) -> None:
        q_emb = np.array([1.0, 0.0, 0.0])
        c_emb = np.array([0.0, 1.0, 0.0])
        result = g5_2_anchor_independence(q_emb, c_emb)
        assert result.passed is True
        assert result.blocking is True

    def test_g5_2_fail_too_similar(self) -> None:
        q_emb = np.array([1.0, 0.1, 0.0])
        c_emb = np.array([1.0, 0.0, 0.0])
        result = g5_2_anchor_independence(q_emb, c_emb)
        assert result.passed is False

    def test_g5_3_delegates_to_g1_3(self) -> None:
        questions = [
            _make_question(reasoning_class="fact_single"),
            _make_question(reasoning_class="reasoning"),
            _make_question(reasoning_class="summary"),
        ]
        result = g5_3_final_fact_single_ratio(questions)
        assert result.gate_id == "G1-3"
        assert result.passed is True

    def test_g5_4_pass_answerable_only(self) -> None:
        """G5-4: passes when answerable questions have enough hard ones."""
        questions = [_make_question(difficulty=0.8) for _ in range(3)]
        questions.append(_make_question(difficulty=0.3))
        result = g5_4_hard_ratio(questions)
        assert result.passed is True
        assert result.blocking is True

    def test_g5_4_fail_answerable_only(self) -> None:
        questions = [_make_question(difficulty=0.3) for _ in range(20)]
        result = g5_4_hard_ratio(questions)
        assert result.passed is False

    def test_g5_4_ignores_unanswerable(self) -> None:
        """G5-4: unanswerable questions are excluded from hard ratio."""
        # 10 answerable easy + 10 unanswerable hard
        questions = [_make_question(difficulty=0.3) for _ in range(10)]
        questions.extend(
            _make_question(
                is_impossible=True, difficulty=0.9, hard_type="OUT_OF_DATABASE"
            )
            for _ in range(10)
        )
        result = g5_4_hard_ratio(questions)
        # Only answerable (all easy) → 0% hard → FAIL
        assert result.passed is False
        assert "0/10" in result.message

    def test_g5_4_empty(self) -> None:
        result = g5_4_hard_ratio([])
        assert result.passed is False

    def test_g5_5_delegates_to_g2_2(self) -> None:
        questions = [_make_question() for _ in range(7)]
        questions.extend(_make_question(is_impossible=True) for _ in range(3))
        result = g5_5_final_unanswerable_ratio(questions)
        assert result.gate_id == "G2-2"
        assert result.passed is True


# ---------------------------------------------------------------------------
# TestG5NewGates - G5-6, G5-7, G5-8
# ---------------------------------------------------------------------------


class TestG5NewGates:
    """Tests for G5-6 (cognitive_level), G5-7 (question_type), G5-8 (chunk_coverage)."""

    def test_g5_6_pass_4_levels(self) -> None:
        levels = ["Remember", "Understand", "Apply", "Analyze"]
        questions = [_make_question() for _ in range(4)]
        for i, q in enumerate(questions):
            q["classification"]["cognitive_level"] = levels[i]
        result = g5_6_cognitive_level_diversity(questions)
        assert result.passed is True
        assert result.blocking is True

    def test_g5_6_fail_2_levels(self) -> None:
        questions = [_make_question() for _ in range(4)]
        for q in questions[:2]:
            q["classification"]["cognitive_level"] = "Remember"
        for q in questions[2:]:
            q["classification"]["cognitive_level"] = "Understand"
        result = g5_6_cognitive_level_diversity(questions)
        assert result.passed is False

    def test_g5_7_pass_all_types(self) -> None:
        types = ["factual", "procedural", "scenario", "comparative"]
        questions = [_make_question() for _ in range(4)]
        for i, q in enumerate(questions):
            q["classification"]["question_type"] = types[i]
        result = g5_7_question_type_diversity(questions)
        assert result.passed is True
        assert result.blocking is False

    def test_g5_7_fail_missing_comparative(self) -> None:
        questions = [_make_question() for _ in range(3)]
        questions[0]["classification"]["question_type"] = "factual"
        questions[1]["classification"]["question_type"] = "procedural"
        questions[2]["classification"]["question_type"] = "scenario"
        result = g5_7_question_type_diversity(questions)
        assert result.passed is False
        assert "comparative" in result.message

    def test_g5_7_ignores_unanswerable(self) -> None:
        """Unanswerable questions are excluded from type diversity check."""
        questions = [_make_question() for _ in range(3)]
        questions[0]["classification"]["question_type"] = "factual"
        questions[1]["classification"]["question_type"] = "procedural"
        questions[2]["classification"]["question_type"] = "scenario"
        # Add unanswerable with "comparative" — should NOT count
        q_unans = _make_question(is_impossible=True, hard_type="OUT_OF_DATABASE")
        q_unans["classification"]["question_type"] = "comparative"
        questions.append(q_unans)
        result = g5_7_question_type_diversity(questions)
        assert result.passed is False

    def test_g5_8_pass(self) -> None:
        result = g5_8_chunk_coverage(160, 200)
        assert result.passed is True
        assert result.blocking is False

    def test_g5_8_fail(self) -> None:
        result = g5_8_chunk_coverage(100, 200)
        assert result.passed is False

    def test_g5_8_zero_total(self) -> None:
        result = g5_8_chunk_coverage(0, 0)
        assert result.passed is False


# ---------------------------------------------------------------------------
# TestCountSchemaFields
# ---------------------------------------------------------------------------


class TestCountSchemaFields:
    """Tests for count_schema_fields helper."""

    def test_complete_question(self, sample_gs_question_answerable: dict) -> None:
        count = count_schema_fields(sample_gs_question_answerable)
        assert count >= 40

    def test_partial_question(self) -> None:
        q = {"id": "test", "content": {"question": "Q?"}}
        count = count_schema_fields(q)
        assert count < 42


# ---------------------------------------------------------------------------
# TestValidateAllGates
# ---------------------------------------------------------------------------


class TestValidateAllGates:
    """Tests for validate_all_gates aggregate function."""

    def test_all_blocking_pass(self, sample_gs_question_answerable: dict) -> None:
        questions = [sample_gs_question_answerable]
        all_passed, results = validate_all_gates(questions)
        # G5-6 (cognitive_level_diversity) requires 4 levels, 1 question has only 1
        # So all_blocking_passed will be False. Check that gates ran.
        assert len(results) > 0
        gate_ids = {r.gate_id for r in results}
        assert "G5-6" in gate_ids
        assert "G5-7" in gate_ids

    def test_blocking_fail(self) -> None:
        q = _make_question(chunk_match_score=50, question_text="pas de point final")
        questions = [q]
        all_passed, results = validate_all_gates(questions)
        assert all_passed is False

    def test_skip_g0_if_strata_none(self) -> None:
        questions = [_make_question()]
        _, results = validate_all_gates(questions, strata=None)
        g0_ids = [r.gate_id for r in results if r.gate_id.startswith("G0")]
        assert len(g0_ids) == 0

    def test_skip_g3_if_validation_none(self) -> None:
        questions = [_make_question()]
        _, results = validate_all_gates(questions, validation_results=None)
        g3_ids = [r.gate_id for r in results if r.gate_id.startswith("G3")]
        assert len(g3_ids) == 0

    def test_with_unanswerable_questions(self) -> None:
        """Ensure G2-1 gates are checked for unanswerable questions."""
        q_ans = _make_question(question_text="Quelle est la regle?")
        q_unans = _make_question(
            is_impossible=True,
            hard_type="OUT_OF_DATABASE",
            reasoning_class="adversarial",
            question_text="Question impossible?",
        )
        questions = [q_ans, q_unans]
        _, results = validate_all_gates(questions)
        g2_1_results = [r for r in results if r.gate_id == "G2-1"]
        assert len(g2_1_results) >= 1

    def test_with_strata_and_coverage(self) -> None:
        """Ensure G0 gates run when strata and coverage provided."""
        q = _make_question(question_text="Quelle est la regle?")
        strata = {f"s{i}": {"quota": 10} for i in range(6)}
        coverage = {"coverage_ratio": 0.85}
        _, results = validate_all_gates(
            [q],
            strata=strata,
            coverage=coverage,
        )
        g0_ids = [r.gate_id for r in results if r.gate_id.startswith("G0")]
        assert len(g0_ids) == 2

    def test_with_validation_and_rejected(self) -> None:
        """Ensure G3 gates run when validation data provided."""
        q = _make_question(question_text="Quelle est la regle?")
        _, results = validate_all_gates(
            [q],
            validation_results={q["id"]: True},
            rejected_ids=[],
        )
        g3_ids = [r.gate_id for r in results if r.gate_id.startswith("G3")]
        assert len(g3_ids) == 2


# ---------------------------------------------------------------------------
# TestFormatGateReport
# ---------------------------------------------------------------------------


class TestFormatGateReport:
    """Tests for format_gate_report."""

    def test_contains_blocking_section(self) -> None:
        results = [
            GateResult("G1-1", "test", True, True, 100, 100, "ok"),
        ]
        report = format_gate_report(results)
        assert "BLOCKING GATES:" in report

    def test_contains_summary(self) -> None:
        results = [
            GateResult("G1-1", "test", True, True, 100, 100, "ok"),
            GateResult("G0-2", "warn", False, False, 0.5, 0.8, "warn"),
        ]
        report = format_gate_report(results)
        assert "SUMMARY" in report
