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
    _sync_coverage_header,
    apply_all_corrections,
    format_correction_report,
    normalize_schema,
    safe_cognitive_reclassify,
    update_audit_trail,
)
from scripts.evaluation.annales.tests.conftest import make_gs_question

# ===========================================================================
# TestNormalizeSchema
# ===========================================================================


class TestNormalizeSchema:
    """Tests for normalize_schema() - A1 corrections."""

    def test_adds_priority_boost(self) -> None:
        """Adds priority_boost=0.0 when missing."""
        q = make_gs_question(has_priority_boost=False, is_impossible=True)
        assert "priority_boost" not in q["processing"]
        records = normalize_schema(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].correction_type == "A1_schema"
        assert records[0].new_value == 0.0
        assert q["processing"]["priority_boost"] == 0.0

    def test_preserves_existing(self) -> None:
        """Does not modify existing priority_boost."""
        q = make_gs_question(has_priority_boost=True, priority_boost_value=0.5)
        records = normalize_schema(q, "2026-02-19")
        assert len(records) == 0
        assert q["processing"]["priority_boost"] == 0.5

    def test_correction_record_fields(self) -> None:
        """CorrectionRecord has correct fields."""
        q = make_gs_question(qid="test:001", has_priority_boost=False)
        records = normalize_schema(q, "2026-02-19")
        assert records[0].question_id == "test:001"
        assert records[0].field == "processing.priority_boost"
        assert records[0].old_value is None

    def test_no_double_add(self) -> None:
        """Calling twice does not add priority_boost again."""
        q = make_gs_question(has_priority_boost=False)
        normalize_schema(q, "2026-02-19")
        assert q["processing"]["priority_boost"] == 0.0
        records = normalize_schema(q, "2026-02-19")
        assert len(records) == 0

    def test_answerable_without_priority_boost(self) -> None:
        """Works on answerable questions too (generic, not filtered by is_impossible)."""
        q = make_gs_question(is_impossible=False, has_priority_boost=False)
        records = normalize_schema(q, "2026-02-19")
        assert len(records) == 1
        assert q["processing"]["priority_boost"] == 0.0


# ===========================================================================
# TestSafeCognitiveReclassify
# ===========================================================================


class TestSafeCognitiveReclassify:
    """Tests for safe_cognitive_reclassify() - A2 corrections."""

    def test_que_doit_faire_apply(self) -> None:
        """'que doit faire' reclassifies to Apply."""
        q = make_gs_question(
            question_text="Que doit faire l'arbitre en cas de litige?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"
        assert q["classification"]["cognitive_level"] == "Apply"

    def test_que_doit_on_faire_apply(self) -> None:
        """'que doit-on faire' reclassifies to Apply."""
        q = make_gs_question(
            question_text="Que doit-on faire selon l'article?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"

    def test_comment_doit_apply(self) -> None:
        """'comment doit' reclassifies to Apply."""
        q = make_gs_question(
            question_text="Comment doit proceder l'arbitre?",
            cognitive_level="Remember",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"
        assert records[0].old_value == "Remember"

    def test_que_faire_si_apply(self) -> None:
        """'que faire si' reclassifies to Apply."""
        q = make_gs_question(
            question_text="Que faire si un joueur conteste?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"

    def test_quelle_difference_analyze(self) -> None:
        """'quelle difference' reclassifies to Analyze."""
        q = make_gs_question(
            question_text="Quelle difference entre rapide et blitz?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Analyze"

    def test_comparez_analyze(self) -> None:
        """'comparez' reclassifies to Analyze."""
        q = make_gs_question(
            question_text="Comparez les regles du rapide et du blitz.",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Analyze"
        assert records[0].pattern_matched == "comparez"
        assert q["classification"]["cognitive_level"] == "Analyze"

    def test_pourquoi_plutot_que_analyze(self) -> None:
        """'pourquoi ... plutot que' reclassifies to Analyze."""
        q = make_gs_question(
            question_text="Pourquoi utiliser la pendule Fischer plutot que la classique?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Analyze"
        assert records[0].pattern_matched == "pourquoi .+ plut[oô]t que"
        assert q["classification"]["cognitive_level"] == "Analyze"

    def test_skip_already_apply(self) -> None:
        """Does not reclassify if already Apply."""
        q = make_gs_question(
            question_text="Que doit faire l'arbitre?",
            cognitive_level="Apply",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 0

    def test_skip_unanswerable(self) -> None:
        """Does not reclassify unanswerable questions."""
        q = make_gs_question(
            question_text="Que doit faire l'arbitre?",
            cognitive_level="Understand",
            is_impossible=True,
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 0

    def test_remember_to_apply(self) -> None:
        """Remember can be reclassified to Apply."""
        q = make_gs_question(
            question_text="Que doit-on faire dans ce cas?",
            cognitive_level="Remember",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].old_value == "Remember"
        assert records[0].new_value == "Apply"

    def test_case_insensitive(self) -> None:
        """Pattern matching is case-insensitive."""
        q = make_gs_question(
            question_text="QUE DOIT FAIRE l'arbitre?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].new_value == "Apply"

    def test_no_match(self) -> None:
        """No reclassification if no pattern matches."""
        q = make_gs_question(
            question_text="Quel est le nombre maximum de joueurs?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 0

    def test_pattern_recorded(self) -> None:
        """Matched pattern is recorded in CorrectionRecord."""
        q = make_gs_question(
            question_text="Que doit faire l'arbitre?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert records[0].pattern_matched == "que doit[ -]faire"

    def test_first_match_wins(self) -> None:
        """When multiple Apply patterns could match, first one wins."""
        # "Que doit-on faire si..." matches both "que doit-on faire" and "que faire si"
        # Pattern order: "que doit[ -]faire" first, then "que doit-on faire"
        # "que doit-on faire" does NOT match "que doit[ -]faire" (verified earlier)
        # So "que doit-on faire" pattern matches
        q = make_gs_question(
            question_text="Que doit-on faire si un joueur refuse?",
            cognitive_level="Understand",
        )
        records = safe_cognitive_reclassify(q, "2026-02-19")
        assert len(records) == 1
        assert records[0].pattern_matched == "que doit-on faire"


# ===========================================================================
# TestUpdateAuditTrail
# ===========================================================================


class TestUpdateAuditTrail:
    """Tests for update_audit_trail() - A3 corrections."""

    def test_a1_tag(self) -> None:
        """A1 correction adds [PHASE A] schema normalized tag."""
        q = make_gs_question()
        corrs = [
            CorrectionRecord("q1", "A1_schema", "processing.priority_boost", None, 0.0),
        ]
        update_audit_trail(q, corrs, "2026-02-19")
        assert "[PHASE A] schema normalized on 2026-02-19" in q["audit"]["history"]

    def test_a2_tag(self) -> None:
        """A2 correction adds reclassified tag."""
        q = make_gs_question()
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
        q = make_gs_question()
        corrs = [
            CorrectionRecord("q1", "A1_schema", "f", None, 0.0),
        ]
        update_audit_trail(q, corrs, "2026-03-15")
        assert "2026-03-15" in q["audit"]["history"]

    def test_multiple_corrections(self) -> None:
        """Multiple corrections produce multiple tags separated by |."""
        q = make_gs_question()
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
        q = make_gs_question()
        original_history = q["audit"]["history"]
        update_audit_trail(q, [], "2026-02-19")
        assert q["audit"]["history"] == original_history

    def test_appends_to_existing_history(self) -> None:
        """Tags are appended to existing history with | separator."""
        q = make_gs_question()
        q["audit"]["history"] = "original entry"
        corrs = [CorrectionRecord("q1", "A1_schema", "f", None, 0.0)]
        update_audit_trail(q, corrs, "2026-02-19")
        assert q["audit"]["history"].startswith("original entry | ")


# ===========================================================================
# TestSyncCoverageHeader
# ===========================================================================


class TestSyncCoverageHeader:
    """Tests for _sync_coverage_header()."""

    def test_updates_counts(self) -> None:
        """Recalculates coverage from actual questions."""
        gs = {
            "coverage": {"total_questions": 0, "answerable": 0, "unanswerable": 0},
            "questions": [
                make_gs_question("q1", is_impossible=False),
                make_gs_question("q2", is_impossible=True),
                make_gs_question("q3", is_impossible=False),
            ],
        }
        _sync_coverage_header(gs)
        assert gs["coverage"]["total_questions"] == 3
        assert gs["coverage"]["answerable"] == 2
        assert gs["coverage"]["unanswerable"] == 1

    def test_creates_header_if_missing(self) -> None:
        """Creates coverage header if not present."""
        gs = {"questions": [make_gs_question("q1")]}
        _sync_coverage_header(gs)
        assert gs["coverage"]["total_questions"] == 1


# ===========================================================================
# TestApplyAllCorrections
# ===========================================================================


class TestApplyAllCorrections:
    """Tests for apply_all_corrections() pipeline."""

    def test_full_pipeline(self) -> None:
        """Pipeline applies A1 + A2 + A3 corrections."""
        questions = [
            make_gs_question("q1", is_impossible=True, has_priority_boost=False),
            make_gs_question(
                "q2",
                question_text="Que doit faire l'arbitre?",
                cognitive_level="Understand",
            ),
            make_gs_question("q3", cognitive_level="Remember"),
        ]
        gs = {"questions": questions}
        result, report = apply_all_corrections(gs)
        assert report["a1_schema_normalized"] == 1
        assert report["a2_cognitive_reclassified"] == 1
        assert report["total_corrections"] == 2

    def test_dry_run_preserves_original(self) -> None:
        """dry_run returns original data unchanged."""
        q = make_gs_question("q1", is_impossible=True, has_priority_boost=False)
        gs = {"questions": [q]}
        original_keys = set(gs["questions"][0]["processing"].keys())
        result, report = apply_all_corrections(gs, dry_run=True)
        assert report["dry_run"] is True
        assert "priority_boost" not in result["questions"][0]["processing"]
        assert set(result["questions"][0]["processing"].keys()) == original_keys

    def test_report_summary(self) -> None:
        """Report includes cognitive before/after."""
        questions = [
            make_gs_question(
                "q1",
                cognitive_level="Understand",
                question_text="Que doit faire l'arbitre?",
            ),
            make_gs_question("q2", cognitive_level="Remember"),
        ]
        gs = {"questions": questions}
        _, report = apply_all_corrections(gs)
        assert "Understand" in report["cognitive_before"]
        assert "Remember" in report["cognitive_before"]
        assert report["cognitive_after"].get("Apply", 0) == 1
        assert report["cognitive_after"].get("Remember", 0) == 1

    def test_before_after_cognitive(self) -> None:
        """Cognitive distribution changes correctly."""
        questions = [
            make_gs_question(
                "q1", cognitive_level="Understand", question_text="Que doit-on faire?"
            ),
            make_gs_question(
                "q2",
                cognitive_level="Understand",
                question_text="Que doit faire l'arbitre?",
            ),
            make_gs_question("q3", cognitive_level="Remember"),
        ]
        gs = {"questions": questions}
        _, report = apply_all_corrections(gs)
        assert report["cognitive_before"] == {"Understand": 2, "Remember": 1}
        assert report["cognitive_after"] == {"Apply": 2, "Remember": 1}

    def test_coverage_header_synced(self) -> None:
        """Coverage header is updated after corrections."""
        gs = {
            "coverage": {"total_questions": 0, "answerable": 0, "unanswerable": 0},
            "questions": [
                make_gs_question("q1", is_impossible=False),
                make_gs_question("q2", is_impossible=True, has_priority_boost=False),
            ],
        }
        result, _ = apply_all_corrections(gs)
        assert result["coverage"]["total_questions"] == 2
        assert result["coverage"]["answerable"] == 1
        assert result["coverage"]["unanswerable"] == 1


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

    def test_no_dry_run_label_in_real(self) -> None:
        """Real run report does not contain (DRY RUN)."""
        report = {
            "date": "2026-02-19",
            "total_corrections": 1,
            "a1_schema_normalized": 1,
            "a2_cognitive_reclassified": 0,
            "a2_by_pattern": {},
            "cognitive_before": {"Understand": 1},
            "cognitive_after": {"Understand": 1},
            "dry_run": False,
        }
        text = format_correction_report(report)
        assert "DRY RUN" not in text


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
                make_gs_question("q1", is_impossible=True, has_priority_boost=False)
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
                make_gs_question("q1", is_impossible=True, has_priority_boost=False)
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

    def test_overwrite_input_when_no_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --output, input file is overwritten."""
        gs_path = tmp_path / "gs.json"
        gs_data = {
            "questions": [
                make_gs_question("q1", is_impossible=True, has_priority_boost=False)
            ]
        }
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--input", str(gs_path)],
        )
        from scripts.evaluation.annales.fix_gs_v2_metadata import main

        assert main() == 0
        result = json.loads(gs_path.read_text(encoding="utf-8"))
        assert result["questions"][0]["processing"]["priority_boost"] == 0.0
