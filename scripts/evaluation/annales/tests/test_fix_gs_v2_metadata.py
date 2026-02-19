"""
Tests for fix_gs_v2_metadata.py - Phase A safe corrections.

ISO Reference: ISO/IEC 29119 - Test design
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.annales.fix_gs_v2_metadata import (
    CorrectionRecord,
    apply_all_corrections,
    format_correction_report,
    normalize_schema,
    safe_cognitive_reclassify,
    update_audit_trail,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_question(
    qid: str = "gs:scratch:answerable:0001:abc",
    is_impossible: bool = False,
    cognitive_level: str = "Understand",
    question_text: str = "Test question?",
    has_priority_boost: bool = True,
    priority_boost_value: float = 0.1,
) -> dict:
    """Build minimal valid Schema v2 question for testing."""
    processing: dict = {
        "chunk_match_score": 100,
        "chunk_match_method": "by_design_input",
        "reasoning_class_method": "generation_prompt",
        "triplet_ready": not is_impossible,
        "extraction_flags": ["by_design"],
        "answer_source": "chunk_extraction",
        "quality_score": 0.8,
    }
    if has_priority_boost:
        processing["priority_boost"] = priority_boost_value

    return {
        "id": qid,
        "legacy_id": "",
        "content": {
            "question": question_text,
            "expected_answer": "Answer" if not is_impossible else "",
            "is_impossible": is_impossible,
        },
        "mcq": {
            "original_question": question_text,
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
            "difficulty": 0.5,
            "question_type": "procedural",
            "cognitive_level": cognitive_level,
            "reasoning_type": "single-hop",
            "reasoning_class": "reasoning",
            "answer_type": "extractive",
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
# TestNormalizeSchema
# ===========================================================================


class TestNormalizeSchema:
    """Tests for normalize_schema() - A1 corrections."""

    def test_adds_priority_boost(self) -> None:
        """Adds priority_boost=0.0 when missing."""
        q = _make_question(has_priority_boost=False, is_impossible=True)
        assert "priority_boost" not in q["processing"]
        records = normalize_schema(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].correction_type == "A1_schema"
        assert records[0].new_value == 0.0
        assert q["processing"]["priority_boost"] == 0.0

    def test_preserves_existing(self) -> None:
        """Does not modify existing priority_boost."""
        q = _make_question(has_priority_boost=True, priority_boost_value=0.5)
        records = normalize_schema(q, "2026-02-19")
        assert len(records) == 0
        assert q["processing"]["priority_boost"] == 0.5

    def test_correction_record_fields(self) -> None:
        """CorrectionRecord has correct fields."""
        q = _make_question(qid="test:001", has_priority_boost=False)
        records = normalize_schema(q, "2026-02-19")
        assert records[0].question_id == "test:001"
        assert records[0].field == "processing.priority_boost"
        assert records[0].old_value is None

    def test_no_double_add(self) -> None:
        """Calling twice does not add priority_boost again."""
        q = _make_question(has_priority_boost=False)
        normalize_schema(q, "2026-02-19")
        assert q["processing"]["priority_boost"] == 0.0
        records = normalize_schema(q, "2026-02-19")
        assert len(records) == 0


# ===========================================================================
# TestSafeCognitiveReclassify
# ===========================================================================


class TestSafeCognitiveReclassify:
    """Tests for safe_cognitive_reclassify() - A2 corrections."""

    def test_que_doit_faire_apply(self) -> None:
        """'que doit faire' reclassifies to Apply."""
        q = _make_question(
            question_text="Que doit faire l'arbitre en cas de litige?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"
        assert q["classification"]["cognitive_level"] == "Apply"

    def test_que_doit_on_faire_apply(self) -> None:
        """'que doit-on faire' reclassifies to Apply."""
        q = _make_question(
            question_text="Que doit-on faire selon l'article?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"

    def test_comment_doit_apply(self) -> None:
        """'comment doit' reclassifies to Apply."""
        q = _make_question(
            question_text="Comment doit proceder l'arbitre?",
            cognitive_level="Remember",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"
        assert records[0].old_value == "Remember"

    def test_que_faire_si_apply(self) -> None:
        """'que faire si' reclassifies to Apply."""
        q = _make_question(
            question_text="Que faire si un joueur conteste?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"

    def test_quelle_difference_analyze(self) -> None:
        """'quelle difference' reclassifies to Analyze."""
        q = _make_question(
            question_text="Quelle difference entre rapide et blitz?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Analyze"

    def test_skip_already_apply(self) -> None:
        """Does not reclassify if already Apply."""
        q = _make_question(
            question_text="Que doit faire l'arbitre?",
            cognitive_level="Apply",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 0

    def test_skip_unanswerable(self) -> None:
        """Does not reclassify unanswerable questions."""
        q = _make_question(
            question_text="Que doit faire l'arbitre?",
            cognitive_level="Understand",
            is_impossible=True,
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 0

    def test_remember_to_apply(self) -> None:
        """Remember can be reclassified to Apply."""
        q = _make_question(
            question_text="Que doit-on faire dans ce cas?",
            cognitive_level="Remember",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].old_value == "Remember"
        assert records[0].new_value == "Apply"

    def test_case_insensitive(self) -> None:
        """Pattern matching is case-insensitive."""
        q = _make_question(
            question_text="QUE DOIT FAIRE l'arbitre?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"

    def test_no_match(self) -> None:
        """No reclassification if no pattern matches."""
        q = _make_question(
            question_text="Quel est le nombre maximum de joueurs?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 0

    def test_pattern_recorded(self) -> None:
        """Matched pattern is recorded in CorrectionRecord."""
        q = _make_question(
            question_text="Que doit faire l'arbitre?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert records[0].pattern_matched == "que doit[ -]faire"


# ===========================================================================
# TestUpdateAuditTrail
# ===========================================================================


class TestUpdateAuditTrail:
    """Tests for update_audit_trail() - A3 corrections."""

    def test_a1_tag(self) -> None:
        """A1 correction adds [PHASE A] schema normalized tag."""
        q = _make_question()
        corrs = [
            CorrectionRecord("q1", "A1_schema", "processing.priority_boost", None, 0.0),
        ]
        update_audit_trail(q, corrs, "2026-02-19")
        assert "[PHASE A] schema normalized on 2026-02-19" in q["audit"]["history"]

    def test_a2_tag(self) -> None:
        """A2 correction adds reclassified tag."""
        q = _make_question()
        corrs = [
            CorrectionRecord(
                "q1",
                "A2_cognitive",
                "classification.cognitive_level",
                "Understand",
                "Apply",
                "que doit[ -]faire",
            ),
        ]
        update_audit_trail(q, corrs, "2026-02-19")
        assert (
            "[PHASE A] reclassified cognitive_level: Understand -> Apply on 2026-02-19"
            in q["audit"]["history"]
        )

    def test_date_in_tag(self) -> None:
        """Date is included in audit tag."""
        q = _make_question()
        corrs = [
            CorrectionRecord("q1", "A1_schema", "f", None, 0.0),
        ]
        update_audit_trail(q, corrs, "2026-03-15")
        assert "2026-03-15" in q["audit"]["history"]

    def test_multiple_corrections(self) -> None:
        """Multiple corrections produce multiple tags separated by |."""
        q = _make_question()
        corrs = [
            CorrectionRecord("q1", "A1_schema", "f1", None, 0.0),
            CorrectionRecord("q1", "A2_cognitive", "f2", "U", "A", "pat"),
        ]
        update_audit_trail(q, corrs, "2026-02-19")
        history = q["audit"]["history"]
        assert "schema normalized" in history
        assert "reclassified" in history
        assert " | " in history

    def test_no_corrections_no_change(self) -> None:
        """No corrections leaves audit unchanged."""
        q = _make_question()
        original_history = q["audit"]["history"]
        update_audit_trail(q, [], "2026-02-19")
        assert q["audit"]["history"] == original_history

    def test_appends_to_existing_history(self) -> None:
        """Tags are appended to existing history with | separator."""
        q = _make_question()
        q["audit"]["history"] = "original entry"
        corrs = [CorrectionRecord("q1", "A1_schema", "f", None, 0.0)]
        update_audit_trail(q, corrs, "2026-02-19")
        assert q["audit"]["history"].startswith("original entry | ")


# ===========================================================================
# TestApplyAllCorrections
# ===========================================================================


class TestApplyAllCorrections:
    """Tests for apply_all_corrections() pipeline."""

    def test_full_pipeline(self) -> None:
        """Pipeline applies A1 + A2 + A3 corrections."""
        questions = [
            _make_question("q1", is_impossible=True, has_priority_boost=False),
            _make_question(
                "q2",
                question_text="Que doit faire l'arbitre?",
                cognitive_level="Understand",
            ),
            _make_question("q3", cognitive_level="Remember"),
        ]
        gs = {"questions": questions}
        result, report = apply_all_corrections(gs)
        assert report["a1_schema_normalized"] == 1
        assert report["a2_cognitive_reclassified"] == 1
        assert report["total_corrections"] == 2

    def test_dry_run_preserves_original(self) -> None:
        """dry_run returns original data unchanged."""
        q = _make_question("q1", is_impossible=True, has_priority_boost=False)
        gs = {"questions": [q]}
        original_keys = set(gs["questions"][0]["processing"].keys())
        result, report = apply_all_corrections(gs, dry_run=True)
        assert report["dry_run"] is True
        # Original unchanged
        assert "priority_boost" not in result["questions"][0]["processing"]
        assert set(result["questions"][0]["processing"].keys()) == original_keys

    def test_report_summary(self) -> None:
        """Report includes cognitive before/after."""
        questions = [
            _make_question(
                "q1",
                cognitive_level="Understand",
                question_text="Que doit faire l'arbitre?",
            ),
            _make_question("q2", cognitive_level="Remember"),
        ]
        gs = {"questions": questions}
        _, report = apply_all_corrections(gs)
        assert "Understand" in report["cognitive_before"]
        assert "Remember" in report["cognitive_before"]
        # After: q1 became Apply, q2 stays Remember
        assert report["cognitive_after"].get("Apply", 0) == 1
        assert report["cognitive_after"].get("Remember", 0) == 1

    def test_before_after_cognitive(self) -> None:
        """Cognitive distribution changes correctly."""
        questions = [
            _make_question(
                "q1", cognitive_level="Understand", question_text="Que doit-on faire?"
            ),
            _make_question(
                "q2",
                cognitive_level="Understand",
                question_text="Que doit faire l'arbitre?",
            ),
            _make_question("q3", cognitive_level="Remember"),
        ]
        gs = {"questions": questions}
        _, report = apply_all_corrections(gs)
        assert report["cognitive_before"] == {"Understand": 2, "Remember": 1}
        assert report["cognitive_after"] == {"Apply": 2, "Remember": 1}


# ===========================================================================
# TestFormatCorrectionReport
# ===========================================================================


class TestFormatCorrectionReport:
    """Tests for format_correction_report()."""

    def test_contains_totals(self) -> None:
        """Report includes correction counts."""
        report = {
            "date": "2026-02-19",
            "total_corrections": 5,
            "a1_schema_normalized": 3,
            "a2_cognitive_reclassified": 2,
            "a2_by_pattern": {"que doit[ -]faire": 2},
            "cognitive_before": {"Understand": 10},
            "cognitive_after": {"Understand": 8, "Apply": 2},
            "dry_run": False,
        }
        text = format_correction_report(report)
        assert "Total corrections: 5" in text
        assert "A1 schema normalized: 3" in text
        assert "A2 cognitive reclassified: 2" in text

    def test_dry_run_label(self) -> None:
        """Dry run report includes (DRY RUN) label."""
        report = {
            "date": "2026-02-19",
            "total_corrections": 0,
            "a1_schema_normalized": 0,
            "a2_cognitive_reclassified": 0,
            "a2_by_pattern": {},
            "cognitive_before": {},
            "cognitive_after": {},
            "dry_run": True,
        }
        text = format_correction_report(report)
        assert "DRY RUN" in text


# ===========================================================================
# TestCLI
# ===========================================================================


class TestCLI:
    """Tests for main() CLI."""

    def test_dry_run_no_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run does not write output file."""
        gs_path = tmp_path / "gs.json"
        gs_data = {
            "questions": [
                _make_question("q1", is_impossible=True, has_priority_boost=False)
            ]
        }
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")
        out_path = tmp_path / "output.json"

        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--input", str(gs_path), "--output", str(out_path), "--dry-run"],
        )
        from scripts.evaluation.annales.fix_gs_v2_metadata import main

        assert main() == 0
        assert not out_path.exists()

    def test_real_run_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real run writes corrected output file."""
        gs_path = tmp_path / "gs.json"
        gs_data = {
            "questions": [
                _make_question("q1", is_impossible=True, has_priority_boost=False)
            ]
        }
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")
        out_path = tmp_path / "output.json"

        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--input", str(gs_path), "--output", str(out_path)],
        )
        from scripts.evaluation.annales.fix_gs_v2_metadata import main

        assert main() == 0
        assert out_path.exists()
        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert result["questions"][0]["processing"]["priority_boost"] == 0.0
