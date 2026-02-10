"""
Integration tests for GS BY DESIGN data integrity.

Tests committed JSON files against quality gates - no module under test.
All tests marked with @pytest.mark.integration.

ISO Reference:
    - ISO/IEC 29119 - Integration testing
    - ISO 42001 - Data quality validation
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent


@pytest.mark.integration
class TestGSSchemaCompliance:
    """G4-1, G4-2: Schema v2 compliance for all 614 questions."""

    def test_614_questions(self, gs_scratch_data: dict) -> None:
        """614 = 584 original + 30 new (15 MODALITY_LIMITED + 15 SAFETY_CONCERNED)."""
        assert len(gs_scratch_data["questions"]) == 614

    def test_7_required_groups(self, gs_scratch_data: dict) -> None:
        required_groups = {
            "content",
            "mcq",
            "provenance",
            "classification",
            "validation",
            "processing",
            "audit",
        }
        for q in gs_scratch_data["questions"]:
            present = set(q.keys()) & required_groups
            assert (
                present == required_groups
            ), f"Missing groups in {q['id']}: {required_groups - present}"

    def test_id_non_empty(self, gs_scratch_data: dict) -> None:
        for q in gs_scratch_data["questions"]:
            assert q.get("id"), "Question has empty id"

    def test_g4_1_schema_fields(self, gs_scratch_data: dict) -> None:
        """G4-1: answerable >= 40 fields, unanswerable >= 36 fields."""
        from scripts.evaluation.annales.quality_gates import count_schema_fields

        for q in gs_scratch_data["questions"]:
            count = count_schema_fields(q)
            if q["content"]["is_impossible"]:
                assert count >= 36, f"{q['id']} has only {count} fields (unanswerable)"
            else:
                assert count >= 40, f"{q['id']} has only {count} fields (answerable)"

    def test_g4_2_chunk_match_method(self, gs_scratch_data: dict) -> None:
        """G4-2: All questions use by_design_input method."""
        for q in gs_scratch_data["questions"]:
            method = q["processing"]["chunk_match_method"]
            assert method == "by_design_input", f"{q['id']} has method={method}"


@pytest.mark.integration
class TestChunkLinkage:
    """G1-1, G1-2: All chunk_ids exist in corpus."""

    def test_all_chunk_ids_exist(
        self, gs_scratch_data: dict, chunk_index: dict[str, str]
    ) -> None:
        """G1-1: Every question's chunk_id must exist in chunk corpus."""
        for q in gs_scratch_data["questions"]:
            chunk_id = q["provenance"]["chunk_id"]
            assert chunk_id in chunk_index, f"Orphan chunk_id: {chunk_id} in {q['id']}"

    def test_g1_2_answer_in_chunk_sample(
        self, gs_scratch_data: dict, chunk_index: dict[str, str]
    ) -> None:
        """G1-2: Sample of answerable questions have answer extractable from chunk."""
        answerable = [
            q for q in gs_scratch_data["questions"] if not q["content"]["is_impossible"]
        ]
        random.seed(42)
        sample = random.sample(answerable, min(50, len(answerable)))

        for q in sample:
            chunk_text = chunk_index.get(q["provenance"]["chunk_id"], "")
            answer = q["content"]["expected_answer"].lower()
            chunk_lower = chunk_text.lower()
            # Check keyword overlap (pure Python, no embeddings)
            answer_words = [w for w in answer.split() if len(w) > 3]
            if answer_words:
                found = sum(1 for w in answer_words if w in chunk_lower)
                coverage = found / len(answer_words)
                assert (
                    coverage >= 0.3
                ), f"{q['id']}: low keyword coverage {coverage:.2f} for answer in chunk"

    def test_chunk_match_score_100(self, gs_scratch_data: dict) -> None:
        for q in gs_scratch_data["questions"]:
            score = q["processing"]["chunk_match_score"]
            assert score == 100, f"{q['id']} has score={score}"


@pytest.mark.integration
class TestDistributionTargets:
    """G2-2, G2-3, G1-3, G5-3, G5-4, G5-5: Distribution gates."""

    def test_g2_2_g5_5_unanswerable_25_40(self, gs_scratch_data: dict) -> None:
        """G2-2/G5-5: Unanswerable ratio 25-40% (SQuAD 2.0 train=33.4%, dev=50%)."""
        total = len(gs_scratch_data["questions"])
        unanswerable = sum(
            1 for q in gs_scratch_data["questions"] if q["content"]["is_impossible"]
        )
        ratio = unanswerable / total
        assert 0.25 <= ratio <= 0.40, f"Unanswerable ratio: {ratio:.3f}"

    def test_g2_3_at_least_4_hard_types(self, gs_scratch_data: dict) -> None:
        """G2-3: At least 4 different hard_types (project-adapted categories)."""
        unanswerable = [
            q for q in gs_scratch_data["questions"] if q["content"]["is_impossible"]
        ]
        hard_types = {
            q["classification"]["hard_type"]
            for q in unanswerable
            if q["classification"]["hard_type"] not in ("UNKNOWN", "ANSWERABLE")
        }
        assert len(hard_types) >= 4, f"Only {len(hard_types)} hard_types: {hard_types}"

    def test_6_uaeval4rag_categories(self, gs_scratch_data: dict) -> None:
        """G2-3: All 6 UAEval4RAG categories present (arXiv:2412.12300)."""
        unanswerable = [
            q for q in gs_scratch_data["questions"] if q["content"]["is_impossible"]
        ]
        hard_types = {q["classification"]["hard_type"] for q in unanswerable}
        expected = {
            "OUT_OF_DATABASE",
            "FALSE_PRESUPPOSITION",
            "UNDERSPECIFIED",
            "NONSENSICAL",
            "MODALITY_LIMITED",
            "SAFETY_CONCERNED",
        }
        assert hard_types == expected, f"Got {hard_types}, expected {expected}"

    def test_g1_3_g5_3_fact_single_under_60(self, gs_scratch_data: dict) -> None:
        """G1-3/G5-3: fact_single ratio < 60%."""
        answerable = [
            q for q in gs_scratch_data["questions"] if not q["content"]["is_impossible"]
        ]
        fact_single = sum(
            1
            for q in answerable
            if q["classification"]["reasoning_class"] == "fact_single"
        )
        ratio = fact_single / len(answerable)
        assert ratio < 0.60, f"fact_single ratio: {ratio:.3f}"

    def test_g5_4_hard_at_least_10_percent(self, gs_scratch_data: dict) -> None:
        """G5-4: At least 10% hard questions (difficulty >= 0.7)."""
        total = len(gs_scratch_data["questions"])
        hard = sum(
            1
            for q in gs_scratch_data["questions"]
            if q["classification"]["difficulty"] >= 0.7
        )
        ratio = hard / total
        assert ratio >= 0.10, f"Hard ratio: {ratio:.3f}"

    def test_at_least_3_reasoning_classes(self, gs_scratch_data: dict) -> None:
        classes = {
            q["classification"]["reasoning_class"] for q in gs_scratch_data["questions"]
        }
        assert len(classes) >= 3, f"Only {len(classes)} classes: {classes}"


@pytest.mark.integration
class TestAnswerInChunk:
    """G3-1, G2-1: Answer validation."""

    def test_answerable_keyword_score(
        self, gs_scratch_data: dict, chunk_index: dict[str, str]
    ) -> None:
        """G3-1: Sample answerable questions keyword_score >= 0.3."""
        from scripts.evaluation.annales.validate_gs_quality import compute_keyword_score

        answerable = [
            q for q in gs_scratch_data["questions"] if not q["content"]["is_impossible"]
        ]
        random.seed(42)
        sample = random.sample(answerable, min(50, len(answerable)))

        low_score_count = 0
        for q in sample:
            chunk_text = chunk_index.get(q["provenance"]["chunk_id"], "")
            score = compute_keyword_score(q["content"]["expected_answer"], chunk_text)
            if score < 0.3:
                low_score_count += 1

        # BY DESIGN questions: answer extracted from chunk, <=25% tolerance
        assert (
            low_score_count / len(sample) < 0.25
        ), f"Too many low keyword scores: {low_score_count}/{len(sample)}"

    def test_unanswerable_empty_answer(self, gs_scratch_data: dict) -> None:
        unanswerable = [
            q for q in gs_scratch_data["questions"] if q["content"]["is_impossible"]
        ]
        for q in unanswerable:
            assert (
                q["content"]["expected_answer"] == ""
            ), f"{q['id']} is unanswerable but has expected_answer"

    def test_g2_1_unanswerable_flag(self, gs_scratch_data: dict) -> None:
        """G2-1: All questions with hard_type != ANSWERABLE must be is_impossible."""
        for q in gs_scratch_data["questions"]:
            if q["classification"]["hard_type"] != "ANSWERABLE":
                assert q["content"]["is_impossible"] is True, (
                    f"{q['id']} has hard_type={q['classification']['hard_type']} "
                    f"but is_impossible={q['content']['is_impossible']}"
                )


@pytest.mark.integration
class TestChunksFileIntegrity:
    """Validate chunks corpus file."""

    def test_1857_chunks(self, chunks_data: dict) -> None:
        assert len(chunks_data["chunks"]) == 1857

    def test_required_fields(self, chunks_data: dict) -> None:
        required = {"id", "text", "source", "page"}
        for chunk in chunks_data["chunks"]:
            present = set(chunk.keys()) & required
            assert present == required, f"Missing fields in {chunk.get('id', '?')}"

    def test_no_empty_text(self, chunks_data: dict) -> None:
        for chunk in chunks_data["chunks"]:
            assert chunk["text"].strip(), f"Empty text in chunk {chunk['id']}"


@pytest.mark.integration
class TestValidationReportConsistency:
    """Validate validation_report_iso.json consistency."""

    @pytest.fixture()
    def report_data(self) -> dict:
        path = _PROJECT_ROOT / "data" / "gs_generation" / "validation_report_iso.json"
        if not path.exists():
            pytest.skip(f"Validation report not found: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_status_validated(self, report_data: dict) -> None:
        assert report_data["status"] == "VALIDATED"

    def test_all_gates_true(self, report_data: dict) -> None:
        gates = report_data.get("validation_gates", {})
        for gate_name, gate_value in gates.items():
            assert gate_value is True, f"Gate {gate_name} is not True"

    def test_counts_match_gs(self, report_data: dict, gs_scratch_data: dict) -> None:
        report_total = report_data["coverage"]["total_questions"]
        gs_total = len(gs_scratch_data["questions"])
        assert report_total == gs_total

    def test_unanswerable_ratio_matches_gs(
        self, report_data: dict, gs_scratch_data: dict
    ) -> None:
        """Cross-validate report unanswerable ratio against actual GS data."""
        actual_unanswerable = sum(
            1 for q in gs_scratch_data["questions"] if q["content"]["is_impossible"]
        )
        actual_total = len(gs_scratch_data["questions"])
        actual_ratio = round(actual_unanswerable / actual_total, 2)
        report_ratio = report_data["coverage"]["unanswerable_ratio"]
        assert (
            abs(actual_ratio - report_ratio) < 0.02
        ), f"Report unanswerable_ratio={report_ratio} vs actual={actual_ratio}"


@pytest.mark.integration
class TestCoherenceConstraints:
    """C1-C8: Coherence constraints from GS_SCHEMA_V2.md Section 5."""

    def test_c1_correct_answer_matches_choice(self, gs_scratch_data: dict) -> None:
        """C1: mcq.correct_answer == mcq.choices[mcq.mcq_answer]."""
        for q in gs_scratch_data["questions"]:
            mcq = q["mcq"]
            if mcq["mcq_answer"] and mcq["choices"]:
                expected = mcq["choices"].get(mcq["mcq_answer"], "")
                assert (
                    mcq["correct_answer"] == expected
                ), f"{q['id']}: C1 violation - correct_answer != choices[{mcq['mcq_answer']}]"

    def test_c2_docs_in_chunk_id(self, gs_scratch_data: dict) -> None:
        """C2: provenance.docs[0] appears in provenance.chunk_id."""
        for q in gs_scratch_data["questions"]:
            prov = q["provenance"]
            if prov["docs"]:
                doc = prov["docs"][0]
                # chunk_id format: "source.pdf-pXXX-parentXXX-childXX"
                # docs[0] should be the source part
                assert (
                    doc in prov["chunk_id"]
                ), f"{q['id']}: C2 violation - doc '{doc}' not in chunk_id '{prov['chunk_id']}'"

    def test_c3_pages_in_chunk_id(self, gs_scratch_data: dict) -> None:
        """C3: provenance.pages[0] appears in provenance.chunk_id."""
        for q in gs_scratch_data["questions"]:
            prov = q["provenance"]
            if prov["pages"]:
                page = prov["pages"][0]
                # chunk_id contains page as "pXXX"
                page_pattern = f"p{page:03d}"
                assert page_pattern in prov["chunk_id"], (
                    f"{q['id']}: C3 violation - page {page} (as {page_pattern}) "
                    f"not in chunk_id '{prov['chunk_id']}'"
                )

    def test_c6_question_ends_with_question_mark(self, gs_scratch_data: dict) -> None:
        """C6: content.question finit par '?'."""
        for q in gs_scratch_data["questions"]:
            assert (
                q["content"]["question"].strip().endswith("?")
            ), f"{q['id']}: C6 violation - question does not end with '?'"

    def test_c7_expected_answer_gt_5_chars(self, gs_scratch_data: dict) -> None:
        """C7: content.expected_answer > 5 chars (answerable only)."""
        for q in gs_scratch_data["questions"]:
            if not q["content"]["is_impossible"]:
                answer = q["content"]["expected_answer"]
                assert (
                    len(answer) > 5
                ), f"{q['id']}: C7 violation - expected_answer '{answer}' <= 5 chars"

    def test_c8_difficulty_in_0_1(self, gs_scratch_data: dict) -> None:
        """C8: classification.difficulty in [0, 1]."""
        for q in gs_scratch_data["questions"]:
            diff = q["classification"]["difficulty"]
            assert (
                0 <= diff <= 1
            ), f"{q['id']}: C8 violation - difficulty={diff} not in [0, 1]"


@pytest.mark.integration
class TestFormatCriteria:
    """Format criteria from GS_SCHEMA_V2.md Section 3.3."""

    def test_question_at_least_10_chars(self, gs_scratch_data: dict) -> None:
        """content.question >= 10 chars (GS_SCHEMA_V2 Section 3.3)."""
        for q in gs_scratch_data["questions"]:
            question = q["content"]["question"]
            assert (
                len(question) >= 10
            ), f"{q['id']}: question '{question}' is < 10 chars"

    def test_answerable_answer_gt_5_chars(self, gs_scratch_data: dict) -> None:
        """content.expected_answer > 5 chars for answerable (GS_SCHEMA_V2 Section 3.3)."""
        for q in gs_scratch_data["questions"]:
            if not q["content"]["is_impossible"]:
                answer = q["content"]["expected_answer"]
                assert (
                    len(answer) > 5
                ), f"{q['id']}: expected_answer '{answer}' <= 5 chars"


@pytest.mark.integration
class TestProvenanceQuality:
    """ISO 42001 A.6.2.2: Provenance tracking quality on real data."""

    def test_answerable_has_explanation(self, gs_scratch_data: dict) -> None:
        """Answerable questions must have non-empty answer_explanation."""
        for q in gs_scratch_data["questions"]:
            if not q["content"]["is_impossible"]:
                explanation = q["provenance"]["answer_explanation"]
                assert (
                    explanation.strip()
                ), f"{q['id']}: answerable question has empty answer_explanation"

    def test_explanation_min_length(self, gs_scratch_data: dict) -> None:
        """answer_explanation should be meaningful (> 20 chars)."""
        for q in gs_scratch_data["questions"]:
            if not q["content"]["is_impossible"]:
                explanation = q["provenance"]["answer_explanation"]
                assert (
                    len(explanation) > 20
                ), f"{q['id']}: answer_explanation too short ({len(explanation)} chars)"

    def test_answerable_has_article_reference(self, gs_scratch_data: dict) -> None:
        """Answerable questions should have article_reference."""
        empty_count = 0
        answerable = [
            q for q in gs_scratch_data["questions"] if not q["content"]["is_impossible"]
        ]
        for q in answerable:
            if not q["provenance"]["article_reference"].strip():
                empty_count += 1
        # Allow up to 10% without article_reference
        ratio = empty_count / len(answerable)
        assert ratio < 0.10, (
            f"{empty_count}/{len(answerable)} ({ratio:.1%}) answerable questions "
            "have empty article_reference"
        )


@pytest.mark.integration
class TestQuestionFormatting:
    """G1-4: Question format validation."""

    def test_g1_4_all_end_with_question_mark(self, gs_scratch_data: dict) -> None:
        """G1-4: All questions end with '?'."""
        for q in gs_scratch_data["questions"]:
            question = q["content"]["question"]
            assert question.strip().endswith(
                "?"
            ), f"{q['id']}: question does not end with '?': ...{question[-20:]}"

    def test_id_format(self, gs_scratch_data: dict) -> None:
        for q in gs_scratch_data["questions"]:
            assert q["id"].startswith("gs:scratch:"), f"Bad ID format: {q['id']}"

    def test_no_duplicate_ids(self, gs_scratch_data: dict) -> None:
        ids = [q["id"] for q in gs_scratch_data["questions"]]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"
