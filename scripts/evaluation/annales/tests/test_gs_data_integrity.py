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
    """G4-1, G4-2: Schema v2 compliance for all 584 questions."""

    def test_584_questions(self, gs_scratch_data: dict) -> None:
        assert len(gs_scratch_data["questions"]) == 584

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

    def test_zero_orphans(
        self, gs_scratch_data: dict, chunk_index: dict[str, str]
    ) -> None:
        orphans = [
            q["id"]
            for q in gs_scratch_data["questions"]
            if q["provenance"]["chunk_id"] not in chunk_index
        ]
        assert len(orphans) == 0, f"Found {len(orphans)} orphans"

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
                    coverage >= 0.2
                ), f"{q['id']}: low keyword coverage {coverage:.2f} for answer in chunk"

    def test_chunk_match_score_100(self, gs_scratch_data: dict) -> None:
        for q in gs_scratch_data["questions"]:
            score = q["processing"]["chunk_match_score"]
            assert score == 100, f"{q['id']} has score={score}"


@pytest.mark.integration
class TestDistributionTargets:
    """G2-2, G2-3, G1-3, G5-3, G5-4, G5-5: Distribution gates."""

    def test_g2_2_g5_5_unanswerable_25_33(self, gs_scratch_data: dict) -> None:
        """G2-2/G5-5: Unanswerable ratio 25-33%."""
        total = len(gs_scratch_data["questions"])
        unanswerable = sum(
            1 for q in gs_scratch_data["questions"] if q["content"]["is_impossible"]
        )
        ratio = unanswerable / total
        assert 0.25 <= ratio <= 0.33, f"Unanswerable ratio: {ratio:.3f}"

    def test_g2_3_at_least_4_hard_types(self, gs_scratch_data: dict) -> None:
        """G2-3: At least 4 different UAEval4RAG hard_types."""
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
        unanswerable = [
            q for q in gs_scratch_data["questions"] if q["content"]["is_impossible"]
        ]
        hard_types = {q["classification"]["hard_type"] for q in unanswerable}
        expected = {
            "OUT_OF_SCOPE",
            "INSUFFICIENT_INFO",
            "FALSE_PREMISE",
            "TEMPORAL_MISMATCH",
            "AMBIGUOUS",
            "COUNTERFACTUAL",
        }
        assert hard_types == expected

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

        # Allow some tolerance (not all questions need perfect keyword match)
        assert (
            low_score_count / len(sample) < 0.5
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
