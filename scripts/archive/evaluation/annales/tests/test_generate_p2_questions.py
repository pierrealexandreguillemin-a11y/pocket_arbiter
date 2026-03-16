"""Tests for generate_p2_questions.py (Phase A-P2).

Validates that hand-crafted questions conform to profiles and schema.

ISO Reference: ISO/IEC 29119 - Test design, ISO/IEC 42001 - AI quality
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.evaluation.annales.generate_p2_questions import (  # noqa: E402
    HARD_ANALYZE,
    HARD_APPLY,
    MED_ANALYZE_COMP,
    MED_APPLY_INF,
    _make_question,
)
from scripts.evaluation.annales.regenerate_targeted import PROFILES  # noqa: E402

ALL_ITEMS = {
    "HARD_APPLY": HARD_APPLY,
    "HARD_ANALYZE": HARD_ANALYZE,
    "MED_APPLY_INF": MED_APPLY_INF,
    "MED_ANALYZE_COMP": MED_ANALYZE_COMP,
}


class TestMakeQuestion:
    """Test _make_question builds valid schema v2 dicts."""

    def test_has_required_top_keys(self) -> None:
        q = _make_question(
            question="Test question?",
            expected_answer="A valid answer here.",
            chunk_id="test.pdf-p001-parent001-child00",
            source="test.pdf",
            pages=[1],
            article_ref="Art. 1",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["test"],
        )
        for key in (
            "id",
            "legacy_id",
            "content",
            "provenance",
            "classification",
            "validation",
            "processing",
            "audit",
        ):
            assert key in q, f"missing top key: {key}"

    def test_content_structure(self) -> None:
        q = _make_question(
            question="Test?",
            expected_answer="Answer text here.",
            chunk_id="c1",
            source="s.pdf",
            pages=[1],
            article_ref="Art",
            cognitive_level="Apply",
            difficulty=0.5,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["k"],
        )
        assert q["content"]["question"] == "Test?"
        assert q["content"]["expected_answer"] == "Answer text here."
        assert q["content"]["is_impossible"] is False

    def test_chunk_match_score_is_100(self) -> None:
        q = _make_question(
            question="Test?",
            expected_answer="Answer text here.",
            chunk_id="c1",
            source="s.pdf",
            pages=[1],
            article_ref="Art",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["k"],
        )
        assert q["processing"]["chunk_match_score"] == 100

    def test_batch_is_p2(self) -> None:
        q = _make_question(
            question="Test?",
            expected_answer="Answer text here.",
            chunk_id="c1",
            source="s.pdf",
            pages=[1],
            article_ref="Art",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["k"],
        )
        assert q["validation"]["batch"] == "gs_v1_step1_p2"

    def test_mcq_fields_populated(self) -> None:
        q = _make_question(
            question="Test?",
            expected_answer="My answer text.",
            chunk_id="c1",
            source="s.pdf",
            pages=[1],
            article_ref="Art",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["k"],
        )
        assert q["mcq"]["correct_answer"] == "My answer text."
        assert q["mcq"]["original_answer"] == "My answer text."
        assert q["legacy_id"] == ""


class TestProfileConformance:
    """Each of the 80 questions must conform to its target profile."""

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_count_is_20(self, profile_name: str) -> None:
        assert len(ALL_ITEMS[profile_name]) == 20

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_cognitive_level(self, profile_name: str) -> None:
        spec = PROFILES[profile_name]
        for item in ALL_ITEMS[profile_name]:
            cls = item["q"]["classification"]
            assert cls["cognitive_level"] == spec["cognitive_level"], (
                f"{item['old_id']}: {cls['cognitive_level']} "
                f"!= {spec['cognitive_level']}"
            )

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_question_type(self, profile_name: str) -> None:
        spec = PROFILES[profile_name]
        for item in ALL_ITEMS[profile_name]:
            cls = item["q"]["classification"]
            assert cls["question_type"] == spec["question_type"], (
                f"{item['old_id']}: {cls['question_type']} "
                f"!= {spec['question_type']}"
            )

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_answer_type(self, profile_name: str) -> None:
        spec = PROFILES[profile_name]
        for item in ALL_ITEMS[profile_name]:
            cls = item["q"]["classification"]
            assert cls["answer_type"] == spec["answer_type"], (
                f"{item['old_id']}: {cls['answer_type']} " f"!= {spec['answer_type']}"
            )

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_difficulty_in_range(self, profile_name: str) -> None:
        spec = PROFILES[profile_name]
        for item in ALL_ITEMS[profile_name]:
            diff = item["q"]["classification"]["difficulty"]
            assert spec["difficulty_min"] <= diff <= spec["difficulty_max"], (
                f"{item['old_id']}: difficulty={diff} not in "
                f"[{spec['difficulty_min']}, {spec['difficulty_max']}]"
            )


class TestQuestionQuality:
    """ISO 42001: all questions must meet basic quality gates."""

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_questions_end_with_mark(self, profile_name: str) -> None:
        for item in ALL_ITEMS[profile_name]:
            text = item["q"]["content"]["question"]
            assert text.strip().endswith(
                "?"
            ), f"{item['old_id']}: question doesn't end with '?'"

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_answers_not_empty(self, profile_name: str) -> None:
        for item in ALL_ITEMS[profile_name]:
            answer = item["q"]["content"]["expected_answer"]
            assert len(answer) > 5, f"{item['old_id']}: expected_answer too short"

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_chunk_id_not_empty(self, profile_name: str) -> None:
        for item in ALL_ITEMS[profile_name]:
            cid = item["q"]["provenance"]["chunk_id"]
            assert cid, f"{item['old_id']}: empty chunk_id"

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_chunk_match_score_100(self, profile_name: str) -> None:
        for item in ALL_ITEMS[profile_name]:
            score = item["q"]["processing"]["chunk_match_score"]
            assert score == 100, f"{item['old_id']}: chunk_match_score={score}"

    def test_no_duplicate_old_ids(self) -> None:
        all_old = [item["old_id"] for items in ALL_ITEMS.values() for item in items]
        assert len(all_old) == len(set(all_old))


class TestChunkAlignment:
    """Verify answers are grounded in chunk text (ISO 42001 anti-hallucination)."""

    @pytest.fixture()
    def chunk_index(self) -> dict[str, str]:
        """Load chunk index from candidates file."""
        candidates_path = (
            _project_root / "data" / "gs_generation" / "p2_candidates.json"
        )
        if not candidates_path.exists():
            pytest.skip("p2_candidates.json not found")
        import json

        data = json.loads(candidates_path.read_text(encoding="utf-8"))
        index: dict[str, str] = {}
        for tasks in data.values():
            for task in tasks:
                index[task["chunk_id"]] = task["chunk_text"]
        return index

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip accents and punctuation for fuzzy matching."""
        import re
        import unicodedata

        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = re.sub(r"[^\w\s]", "", text)
        return text.lower()

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_answer_keywords_in_chunk(
        self,
        profile_name: str,
        chunk_index: dict[str, str],
    ) -> None:
        """At least 15% of answer keywords must appear in chunk text."""
        for item in ALL_ITEMS[profile_name]:
            q = item["q"]
            chunk_id = q["provenance"]["chunk_id"]
            if chunk_id not in chunk_index:
                continue
            chunk_text = self._normalize(chunk_index[chunk_id])
            answer = self._normalize(q["content"]["expected_answer"])
            # Extract significant words (>= 4 chars)
            words = [w for w in answer.split() if len(w) >= 4]
            if not words:
                continue
            matches = sum(1 for w in words if w in chunk_text)
            ratio = matches / len(words)
            assert ratio >= 0.15, (
                f"{item['old_id']}: only {matches}/{len(words)} answer "
                f"keywords found in chunk ({ratio:.0%})"
            )
