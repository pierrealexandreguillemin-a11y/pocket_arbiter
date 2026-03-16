"""Tests for verify_gs_metadata.py."""

import json
from pathlib import Path
from typing import Any

from scripts.evaluation.annales.verify_gs_metadata import (
    auto_populate,
    compute_triplet_ready,
    derive_answer_type,
    derive_category_from_id,
    derive_difficulty_from_rate,
    extract_keywords_from_question,
    validate_question,
    verify_gs_metadata,
)

# --- Helpers ---


def _make_question(**overrides: Any) -> dict[str, Any]:
    """Create a valid question dict with all required fields."""
    q: dict[str, Any] = {
        "id": "ffe:annales:rules:001:abc12345",
        "question": "Quelle est la cadence applicable pour un tournoi classique?",
        "expected_answer": "La cadence est de 90 minutes pour les 40 premiers coups.",
        "is_impossible": False,
        "expected_chunk_id": "chunk_001",
        "expected_docs": ["doc.pdf"],
        "expected_pages": [1],
        "category": "regles_jeu",
        "keywords": ["cadence", "tournoi"],
        "metadata": {
            "answer_type": "multiple_choice",
            "question_type": "factual",
            "reasoning_type": "single-hop",
            "cognitive_level": "Remember",
            "reasoning_class": "fact_single",
            "difficulty": 0.75,
            "quality_score": 85.0,
            "chunk_match_score": 90.0,
            "chunk_match_method": "doc_page_semantic",
            "triplet_ready": True,
            "article_reference": "Art. 3.1",
            "choices": {"A": "90 min", "B": "60 min", "C": "120 min"},
            "mcq_answer": "A",
        },
    }
    for key, val in overrides.items():
        if key == "metadata":
            q["metadata"].update(val)
        else:
            q[key] = val
    return q


# --- derive_category_from_id ---


class TestDeriveCategoryFromId:
    def test_standard_id(self) -> None:
        assert derive_category_from_id("ffe:annales:rules:001:abc") == "rules"

    def test_human_id(self) -> None:
        assert derive_category_from_id("ffe:human:open:001:xyz") == "open"

    def test_short_id(self) -> None:
        assert derive_category_from_id("short") is None

    def test_annales_clubs(self) -> None:
        assert derive_category_from_id("ffe:annales:clubs:050:def") == "clubs"


# --- derive_answer_type ---


class TestDeriveAnswerType:
    def test_with_choices(self) -> None:
        assert derive_answer_type({"A": "text", "B": "text"}) == "mcq"

    def test_without_choices(self) -> None:
        assert derive_answer_type(None) == "open"

    def test_empty_choices(self) -> None:
        assert derive_answer_type({}) == "open"


# --- derive_difficulty_from_rate ---


class TestDeriveDifficultyFromRate:
    def test_none(self) -> None:
        assert derive_difficulty_from_rate(None) is None

    def test_high_rate(self) -> None:
        assert derive_difficulty_from_rate(0.95) == 0.95

    def test_low_rate(self) -> None:
        assert derive_difficulty_from_rate(0.15) == 0.15

    def test_rounding(self) -> None:
        assert derive_difficulty_from_rate(0.123456) == 0.12


# --- extract_keywords_from_question ---


class TestExtractKeywords:
    def test_chess_terms(self) -> None:
        kw = extract_keywords_from_question(
            "Le roque est-il valide dans cette cadence?"
        )
        assert "roque" in kw
        assert "cadence" in kw

    def test_with_article_ref(self) -> None:
        kw = extract_keywords_from_question("Question?", "Art. 3.7")
        assert any("Art" in k for k in kw)

    def test_no_matches_gives_default(self) -> None:
        kw = extract_keywords_from_question("Rien de special")
        assert kw == ["arbitrage"]

    def test_multiple_terms(self) -> None:
        kw = extract_keywords_from_question("Le joueur abandonne la partie en tournoi")
        assert "joueur" in kw
        assert "abandon" in kw
        assert "partie" in kw
        assert "tournoi" in kw


# --- compute_triplet_ready ---


class TestComputeTripletReady:
    def test_all_present(self) -> None:
        assert (
            compute_triplet_ready(
                "A long enough question text here",
                "A valid answer text",
                "chunk_001",
            )
            is True
        )

    def test_no_chunk(self) -> None:
        assert (
            compute_triplet_ready(
                "A long enough question text here",
                "A valid answer text",
                None,
            )
            is False
        )

    def test_short_question(self) -> None:
        assert compute_triplet_ready("Short", "A valid answer", "chunk_001") is False

    def test_empty_answer(self) -> None:
        assert compute_triplet_ready("A long enough question", "", "chunk_001") is False

    def test_short_answer(self) -> None:
        assert (
            compute_triplet_ready(
                "A long enough question text here",
                "AB",
                "chunk_001",
            )
            is False
        )


# --- validate_question ---


class TestValidateQuestion:
    def test_valid_question_no_errors(self) -> None:
        q = _make_question()
        errors, warnings = validate_question(q, 0)
        assert len(errors) == 0

    def test_missing_id(self) -> None:
        q = _make_question(id="")
        errors, _ = validate_question(q, 0)
        assert any("id" in e for e in errors)

    def test_short_question(self) -> None:
        q = _make_question(question="Short")
        errors, _ = validate_question(q, 0)
        assert any("question too short" in e for e in errors)

    def test_image_question_is_warning(self) -> None:
        q = _make_question(question="<!-- image -->")
        errors, warnings = validate_question(q, 0)
        assert not any("question too short" in e for e in errors)
        assert any("image-dependent" in w for w in warnings)

    def test_fusion_artifact(self) -> None:
        q = _make_question(
            question="Some text ## with fusion artifact that is long enough"
        )
        errors, _ = validate_question(q, 0)
        assert any("## fusion" in e for e in errors)

    def test_empty_answer(self) -> None:
        q = _make_question(expected_answer="")
        errors, _ = validate_question(q, 0)
        assert any("expected_answer empty" in e for e in errors)

    def test_invalid_category(self) -> None:
        q = _make_question(category="invalid_cat")
        errors, _ = validate_question(q, 0)
        assert any("category" in e for e in errors)

    def test_invalid_answer_type(self) -> None:
        q = _make_question(metadata={"answer_type": "bad_type"})
        errors, _ = validate_question(q, 0)
        assert any("answer_type" in e for e in errors)

    def test_invalid_question_type(self) -> None:
        q = _make_question(metadata={"question_type": "wrong"})
        errors, _ = validate_question(q, 0)
        assert any("question_type" in e for e in errors)

    def test_invalid_reasoning_type(self) -> None:
        q = _make_question(metadata={"reasoning_type": "wrong"})
        errors, _ = validate_question(q, 0)
        assert any("reasoning_type" in e for e in errors)

    def test_invalid_cognitive_level(self) -> None:
        q = _make_question(metadata={"cognitive_level": "wrong"})
        errors, _ = validate_question(q, 0)
        assert any("cognitive_level" in e for e in errors)

    def test_difficulty_out_of_range(self) -> None:
        q = _make_question(metadata={"difficulty": 2.0})
        errors, _ = validate_question(q, 0)
        assert any("difficulty" in e for e in errors)

    def test_difficulty_valid(self) -> None:
        q = _make_question(metadata={"difficulty": 0.85})
        errors, _ = validate_question(q, 0)
        assert not any("difficulty" in e for e in errors)

    def test_quality_score_out_of_range(self) -> None:
        q = _make_question(metadata={"quality_score": 150.0})
        errors, _ = validate_question(q, 0)
        assert any("quality_score" in e for e in errors)

    def test_chunk_match_score_out_of_range(self) -> None:
        q = _make_question(metadata={"chunk_match_score": 120.0})
        errors, _ = validate_question(q, 0)
        assert any("chunk_match_score" in e for e in errors)

    def test_chunk_match_score_110_valid(self) -> None:
        q = _make_question(metadata={"chunk_match_score": 110.0})
        errors, _ = validate_question(q, 0)
        assert not any("chunk_match_score" in e for e in errors)

    def test_invalid_chunk_match_method(self) -> None:
        q = _make_question(metadata={"chunk_match_method": "invalid"})
        errors, _ = validate_question(q, 0)
        assert any("chunk_match_method" in e for e in errors)

    def test_is_impossible_not_bool(self) -> None:
        q = _make_question(is_impossible="yes")
        errors, _ = validate_question(q, 0)
        assert any("is_impossible" in e for e in errors)

    def test_triplet_ready_not_bool(self) -> None:
        q = _make_question(metadata={"triplet_ready": "yes"})
        errors, _ = validate_question(q, 0)
        assert any("triplet_ready" in e for e in errors)

    def test_mcq_answer_not_in_choices(self) -> None:
        q = _make_question(metadata={"choices": {"A": "text"}, "mcq_answer": "D"})
        _, warnings = validate_question(q, 0)
        assert any("mcq_answer" in w for w in warnings)

    def test_mcq_answer_in_choices(self) -> None:
        q = _make_question(
            metadata={"choices": {"A": "text", "B": "other"}, "mcq_answer": "A"}
        )
        _, warnings = validate_question(q, 0)
        assert not any("mcq_answer" in w for w in warnings)

    def test_empty_keywords_warning(self) -> None:
        q = _make_question(keywords=[])
        _, warnings = validate_question(q, 0)
        assert any("keywords" in w for w in warnings)

    def test_empty_docs_warning(self) -> None:
        q = _make_question(expected_docs=[])
        _, warnings = validate_question(q, 0)
        assert any("expected_docs" in w for w in warnings)

    def test_annales_invalid_session(self) -> None:
        q = _make_question(
            metadata={
                "annales_source": {"session": "invalid", "uv": "UVR", "question_num": 1}
            }
        )
        errors, _ = validate_question(q, 0)
        assert any("annales session" in e for e in errors)

    def test_annales_invalid_uv(self) -> None:
        q = _make_question(
            metadata={
                "annales_source": {"session": "dec2024", "uv": "UVX", "question_num": 1}
            }
        )
        errors, _ = validate_question(q, 0)
        assert any("annales uv" in e for e in errors)

    def test_annales_valid(self) -> None:
        q = _make_question(
            metadata={
                "annales_source": {"session": "dec2024", "uv": "UVR", "question_num": 5}
            }
        )
        errors, _ = validate_question(q, 0)
        assert not any("annales" in e for e in errors)


# --- auto_populate ---


class TestAutoPopulate:
    def test_populates_keywords(self) -> None:
        q = _make_question(keywords=[])
        gs = {"questions": [q]}
        _, corrections = auto_populate(gs)
        assert gs["questions"][0]["keywords"]
        assert any(c["field"] == "keywords" for c in corrections)

    def test_populates_difficulty(self) -> None:
        q = _make_question(
            metadata={
                "difficulty": None,
                "annales_source": {
                    "session": "dec2024",
                    "uv": "UVR",
                    "question_num": 1,
                    "success_rate": 0.85,
                },
            }
        )
        gs = {"questions": [q]}
        _, corrections = auto_populate(gs)
        assert gs["questions"][0]["metadata"]["difficulty"] == 0.85
        assert any(c["field"] == "difficulty" for c in corrections)

    def test_corrects_impossible_annales(self) -> None:
        q = _make_question(
            is_impossible=True,
            metadata={
                "annales_source": {"session": "dec2024", "uv": "UVR", "question_num": 1}
            },
        )
        gs = {"questions": [q]}
        _, corrections = auto_populate(gs)
        assert gs["questions"][0]["is_impossible"] is False
        assert any(c["field"] == "is_impossible" for c in corrections)

    def test_recalculates_triplet_ready(self) -> None:
        q = _make_question()
        q["metadata"]["triplet_ready"] = False  # Wrong, should be True
        gs = {"questions": [q]}
        _, corrections = auto_populate(gs)
        assert gs["questions"][0]["metadata"]["triplet_ready"] is True
        assert any(c["field"] == "triplet_ready" for c in corrections)

    def test_no_changes_when_complete(self) -> None:
        q = _make_question()
        gs = {"questions": [q]}
        _, corrections = auto_populate(gs)
        # Only triplet_ready check may trigger, but it should be correct already
        field_corrections = [
            c for c in corrections if c["field"] not in ("triplet_ready",)
        ]
        assert len(field_corrections) == 0


# --- verify_gs_metadata ---


class TestVerifyGsMetadata:
    def test_valid_gs_passes(self, tmp_path: Path) -> None:
        q = _make_question()
        gs = {"questions": [q]}
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        report = verify_gs_metadata(gs_path)
        assert report["total_errors"] == 0
        assert report["gate5_validation"]["overall"] == "PASS"

    def test_invalid_gs_fails(self, tmp_path: Path) -> None:
        q = _make_question(category="invalid", question="short")
        gs = {"questions": [q]}
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        report = verify_gs_metadata(gs_path, auto_fix=False)
        assert report["total_errors"] > 0
        assert report["gate5_validation"]["overall"] == "FAIL"

    def test_saves_output(self, tmp_path: Path) -> None:
        q = _make_question(keywords=[])
        gs = {"questions": [q]}
        gs_path = tmp_path / "gs.json"
        out_path = tmp_path / "out.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        verify_gs_metadata(gs_path, output_path=out_path)
        assert out_path.exists()
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["keywords"]  # populated

    def test_saves_report(self, tmp_path: Path) -> None:
        q = _make_question()
        gs = {"questions": [q]}
        gs_path = tmp_path / "gs.json"
        rpt_path = tmp_path / "report.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        verify_gs_metadata(gs_path, report_path=rpt_path)
        assert rpt_path.exists()
        with open(rpt_path, encoding="utf-8") as f:
            report = json.load(f)
        assert "gate5_validation" in report
