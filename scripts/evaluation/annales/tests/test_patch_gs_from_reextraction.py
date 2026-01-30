"""Tests for patch_gs_from_reextraction.py."""

import json
from pathlib import Path
from typing import Any

from scripts.evaluation.annales.patch_gs_from_reextraction import (
    _clean_question_text,
    _is_ref_only,
    _summarize_changes,
    build_expected_answer,
    patch_gold_standard,
    should_update_question,
    validate_patched_gs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gs_question(**overrides: Any) -> dict[str, Any]:
    """Minimal valid GS question with annales_source."""
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
            "answer_type": "mcq",
            "choices": {"A": "90 min", "B": "60 min", "C": "120 min"},
            "mcq_answer": "A",
            "annales_source": {
                "session": "dec2024",
                "uv": "UVR",
                "question_num": 1,
                "success_rate": 0.75,
            },
        },
    }
    for key, val in overrides.items():
        if key == "metadata":
            q["metadata"].update(val)
        else:
            q[key] = val
    return q


def _make_result(**overrides: Any) -> dict[str, Any]:
    """Minimal re-extraction result."""
    r: dict[str, Any] = {
        "id": "ffe:annales:rules:001:abc12345",
        "question_full": "",
        "choices": {},
        "mcq_answer": None,
        "answer_text_from_choice": None,
        "answer_explanation": None,
        "article_reference": None,
        "success_rate": None,
        "extraction_flags": [],
    }
    r.update(overrides)
    return r


def _write_gs(tmp_path: Path, questions: list[dict[str, Any]]) -> Path:
    gs_path = tmp_path / "gs.json"
    gs_path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return gs_path


def _write_results(tmp_path: Path, results: list[dict[str, Any]]) -> Path:
    res_path = tmp_path / "results.json"
    res_path.write_text(json.dumps(results), encoding="utf-8")
    return res_path


# ---------------------------------------------------------------------------
# _is_ref_only
# ---------------------------------------------------------------------------


class TestIsRefOnly:
    def test_empty_string(self) -> None:
        assert _is_ref_only("") is True

    def test_whitespace(self) -> None:
        assert _is_ref_only("   ") is True

    def test_art_reference(self) -> None:
        assert _is_ref_only("Art. 3.7") is True

    def test_art_without_dot(self) -> None:
        assert _is_ref_only("Art 3.7") is True

    def test_art_multiple(self) -> None:
        assert _is_ref_only("Art. 3.7 et 4.2") is True

    def test_la_reference(self) -> None:
        assert _is_ref_only("LA - 2.1") is True

    def test_la_dash_variants(self) -> None:
        assert _is_ref_only("LA – 3.5") is True
        assert _is_ref_only("LA — 1.0") is True

    def test_chapitre(self) -> None:
        assert _is_ref_only("Chapitre 5") is True

    def test_valid_short_answer(self) -> None:
        assert _is_ref_only("Le jeudi.") is False

    def test_valid_number_answer(self) -> None:
        assert _is_ref_only("4.") is False

    def test_valid_long_answer(self) -> None:
        assert _is_ref_only("La cadence est de 90 minutes.") is False

    def test_none_like(self) -> None:
        # Empty string == ref-only
        assert _is_ref_only("") is True


# ---------------------------------------------------------------------------
# build_expected_answer
# ---------------------------------------------------------------------------


class TestBuildExpectedAnswer:
    def test_priority1_choice(self) -> None:
        result = {
            "answer_text_from_choice": "90 minutes pour 40 coups",
            "answer_explanation": "Explication longue",
        }
        answer, source = build_expected_answer(result, "existing")
        assert answer == "90 minutes pour 40 coups"
        assert source == "choice"

    def test_priority2_explanation(self) -> None:
        result = {
            "answer_text_from_choice": None,
            "answer_explanation": "Explication détaillée du corrigé",
        }
        answer, source = build_expected_answer(result, "existing")
        assert answer == "Explication détaillée du corrigé"
        assert source == "explanation"

    def test_priority3_existing(self) -> None:
        result = {
            "answer_text_from_choice": None,
            "answer_explanation": None,
        }
        answer, source = build_expected_answer(result, "Ma réponse existante")
        assert answer == "Ma réponse existante"
        assert source == "existing"

    def test_choice_ref_only_falls_to_explanation(self) -> None:
        result = {
            "answer_text_from_choice": "Art. 3.7",
            "answer_explanation": "La bonne réponse est 90 minutes.",
        }
        answer, source = build_expected_answer(result, "existing")
        assert answer == "La bonne réponse est 90 minutes."
        assert source == "explanation"

    def test_both_ref_only_keeps_existing(self) -> None:
        result = {
            "answer_text_from_choice": "Art. 3.7",
            "answer_explanation": "Art. 4.2",
        }
        answer, source = build_expected_answer(result, "Ma réponse")
        assert answer == "Ma réponse"
        assert source == "existing"

    def test_empty_choice_uses_explanation(self) -> None:
        result = {
            "answer_text_from_choice": "",
            "answer_explanation": "Texte valide",
        }
        answer, source = build_expected_answer(result, "existing")
        assert answer == "Texte valide"
        assert source == "explanation"


# ---------------------------------------------------------------------------
# _clean_question_text
# ---------------------------------------------------------------------------


class TestCleanQuestionText:
    def test_removes_hash(self) -> None:
        assert _clean_question_text("text ## question") == "text question"

    def test_removes_leading_hash(self) -> None:
        assert _clean_question_text("## Question text") == "Question text"

    def test_collapses_whitespace(self) -> None:
        assert _clean_question_text("a  ##  b") == "a b"

    def test_no_hash_unchanged(self) -> None:
        assert _clean_question_text("normal text") == "normal text"

    def test_multiple_hashes(self) -> None:
        result = _clean_question_text("a ## b ## c")
        assert "##" not in result
        assert result == "a b c"

    def test_strips_edges(self) -> None:
        assert _clean_question_text("  ## text  ") == "text"


# ---------------------------------------------------------------------------
# should_update_question
# ---------------------------------------------------------------------------


class TestShouldUpdateQuestion:
    def test_longer_extracted(self) -> None:
        assert should_update_question("short", "this is much longer") is True

    def test_shorter_extracted_rejected(self) -> None:
        assert should_update_question("this is the original", "short") is False

    def test_equal_length_rejected(self) -> None:
        assert should_update_question("abcde", "fghij") is False

    def test_empty_extracted_rejected(self) -> None:
        assert should_update_question("existing", "") is False

    def test_empty_existing_accepts(self) -> None:
        assert should_update_question("", "new text") is True


# ---------------------------------------------------------------------------
# _summarize_changes
# ---------------------------------------------------------------------------


class TestSummarizeChanges:
    def test_counts_fields(self) -> None:
        changes = [
            {"id": "q1", "fields_changed": ["question (50->100 chars)", "choices (3->5)"]},
            {"id": "q2", "fields_changed": ["question (30->80 chars)"]},
        ]
        summary = _summarize_changes(changes)
        assert summary["question"] == 2
        assert summary["choices"] == 1

    def test_empty_changes(self) -> None:
        assert _summarize_changes([]) == {}

    def test_extracts_field_name_before_paren(self) -> None:
        changes = [
            {"id": "q1", "fields_changed": ["expected_answer (10->50 chars, source=choice)"]},
        ]
        summary = _summarize_changes(changes)
        assert summary["expected_answer"] == 1


# ---------------------------------------------------------------------------
# patch_gold_standard — integration
# ---------------------------------------------------------------------------


class TestPatchGoldStandard:
    def test_non_annales_kept(self, tmp_path: Path) -> None:
        """Questions without annales_source are kept unchanged."""
        q = _make_gs_question()
        q["metadata"].pop("annales_source")
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [])
        out_path = tmp_path / "out.json"

        report = patch_gold_standard(gs_path, res_path, out_path)
        assert report["total_output"] == 1

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["question"] == q["question"]

    def test_no_result_kept(self, tmp_path: Path) -> None:
        """Annales question with no matching result is kept unchanged."""
        q = _make_gs_question()
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [])  # empty results
        out_path = tmp_path / "out.json"

        report = patch_gold_standard(gs_path, res_path, out_path)
        assert report["total_output"] == 1

    def test_annulled_excluded(self, tmp_path: Path) -> None:
        """Annulled questions are excluded."""
        q = _make_gs_question()
        r = _make_result(extraction_flags=["annulled"])
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        report = patch_gold_standard(gs_path, res_path, out_path)
        assert report["total_excluded"] == 1
        assert report["total_output"] == 0
        assert report["excluded"][0]["reason"] == "annulled"

    def test_commentary_excluded(self, tmp_path: Path) -> None:
        """Commentary questions are excluded."""
        q = _make_gs_question()
        r = _make_result(extraction_flags=["commentary"])
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        report = patch_gold_standard(gs_path, res_path, out_path)
        assert report["total_excluded"] == 1
        assert report["excluded"][0]["reason"] == "commentary"

    def test_question_updated_when_longer(self, tmp_path: Path) -> None:
        """Question text updated when extracted is longer."""
        q = _make_gs_question(question="Short question")
        long_q = "This is a much longer question text extracted from the docling PDF"
        r = _make_result(question_full=long_q)
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["question"] == long_q

    def test_question_not_updated_when_shorter(self, tmp_path: Path) -> None:
        """Question text NOT updated when extracted is shorter."""
        orig = "This is the original question which is quite long and detailed"
        q = _make_gs_question(question=orig)
        r = _make_result(question_full="Short")
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["question"] == orig

    def test_answer_updated_from_choice(self, tmp_path: Path) -> None:
        """Expected answer updated from answer_text_from_choice."""
        q = _make_gs_question(expected_answer="Art. 3.7")
        r = _make_result(answer_text_from_choice="90 minutes pour 40 coups")
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["expected_answer"] == "90 minutes pour 40 coups"
        assert result["questions"][0]["metadata"]["answer_source"] == "choice"

    def test_answer_updated_from_explanation(self, tmp_path: Path) -> None:
        """Expected answer updated from explanation when no choice."""
        q = _make_gs_question(expected_answer="Old answer")
        r = _make_result(answer_explanation="Detailed explanation from corrigé")
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["expected_answer"] == "Detailed explanation from corrigé"
        assert result["questions"][0]["metadata"]["answer_source"] == "explanation"

    def test_choices_updated_when_more(self, tmp_path: Path) -> None:
        """Choices updated when re-extracted has more choices."""
        q = _make_gs_question(metadata={"choices": {"A": "x", "B": "y"}})
        r = _make_result(choices={"A": "x", "B": "y", "C": "z", "D": "w"})
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert len(result["questions"][0]["metadata"]["choices"]) == 4

    def test_choices_not_updated_when_fewer(self, tmp_path: Path) -> None:
        """Choices NOT updated when re-extracted has fewer."""
        q = _make_gs_question(
            metadata={"choices": {"A": "x", "B": "y", "C": "z"}}
        )
        r = _make_result(choices={"A": "x"})
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert len(result["questions"][0]["metadata"]["choices"]) == 3

    def test_mcq_answer_updated(self, tmp_path: Path) -> None:
        """mcq_answer updated from result."""
        q = _make_gs_question(metadata={"mcq_answer": "A"})
        r = _make_result(mcq_answer="C")
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["metadata"]["mcq_answer"] == "C"

    def test_article_reference_updated_when_longer(self, tmp_path: Path) -> None:
        """article_reference updated when new is longer."""
        q = _make_gs_question(metadata={"article_reference": "Art. 3"})
        r = _make_result(article_reference="Art. 3.7.2 et 4.1")
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["metadata"]["article_reference"] == "Art. 3.7.2 et 4.1"

    def test_success_rate_updated(self, tmp_path: Path) -> None:
        """success_rate updated in annales_source."""
        q = _make_gs_question()
        r = _make_result(success_rate=0.92)
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        src = result["questions"][0]["metadata"]["annales_source"]
        assert src["success_rate"] == 0.92

    def test_extraction_flags_added(self, tmp_path: Path) -> None:
        """extraction_flags, answer_explanation, answer_source added to metadata."""
        q = _make_gs_question()
        r = _make_result(
            extraction_flags=["image_dependent"],
            answer_explanation="Some explanation",
        )
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        meta = result["questions"][0]["metadata"]
        assert meta["extraction_flags"] == ["image_dependent"]
        assert meta["answer_explanation"] == "Some explanation"
        assert "answer_source" in meta

    def test_hash_cleanup_question(self, tmp_path: Path) -> None:
        """## artifacts stripped from question text in final pass."""
        q = _make_gs_question(question="Text ## with fusion artifact")
        r = _make_result()  # no extraction match
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert "##" not in result["questions"][0]["question"]

    def test_hash_cleanup_answer(self, tmp_path: Path) -> None:
        """## artifacts stripped from answer text in final pass."""
        q = _make_gs_question(expected_answer="Answer ## with artifact")
        r = _make_result()
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert "##" not in result["questions"][0]["expected_answer"]

    def test_mcq_answer_corruption_cleaned(self, tmp_path: Path) -> None:
        """Corrupted mcq_answer like 'D 88.1' cleaned to 'D'."""
        q = _make_gs_question(metadata={"mcq_answer": "D 88.1"})
        r = _make_result()
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["metadata"]["mcq_answer"] == "D"

    def test_mcq_answer_multichar_preserved(self, tmp_path: Path) -> None:
        """Valid multi-char mcq_answer like 'ADE' preserved."""
        q = _make_gs_question(metadata={"mcq_answer": "ADE"})
        r = _make_result()
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["metadata"]["mcq_answer"] == "ADE"

    def test_mcq_answer_article_suffix_cleaned(self, tmp_path: Path) -> None:
        """'A ARTICLE 3.6' cleaned to 'A'."""
        q = _make_gs_question(metadata={"mcq_answer": "A ARTICLE 3.6"})
        r = _make_result()
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["metadata"]["mcq_answer"] == "A"

    def test_diff_report_written(self, tmp_path: Path) -> None:
        """Diff report written to diff_path."""
        q = _make_gs_question()
        r = _make_result(mcq_answer="B")
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"
        diff_path = tmp_path / "diff.json"

        patch_gold_standard(gs_path, res_path, out_path, diff_path=diff_path)
        assert diff_path.exists()
        with open(diff_path, encoding="utf-8") as f:
            diff = json.load(f)
        assert "changes_detail" in diff
        assert "changes_summary" in diff

    def test_report_counts(self, tmp_path: Path) -> None:
        """Report counts are accurate."""
        q1 = _make_gs_question(id="ffe:annales:rules:001:aaa")
        q2 = _make_gs_question(id="ffe:annales:rules:002:bbb")
        q3 = _make_gs_question(id="ffe:annales:rules:003:ccc")
        # q3 will be annulled
        r1 = _make_result(id="ffe:annales:rules:001:aaa", mcq_answer="B")
        r2 = _make_result(id="ffe:annales:rules:002:bbb")
        r3 = _make_result(
            id="ffe:annales:rules:003:ccc", extraction_flags=["annulled"]
        )
        gs_path = _write_gs(tmp_path, [q1, q2, q3])
        res_path = _write_results(tmp_path, [r1, r2, r3])
        out_path = tmp_path / "out.json"

        report = patch_gold_standard(gs_path, res_path, out_path)
        assert report["total_input"] == 3  # 2 output + 1 excluded
        assert report["total_output"] == 2
        assert report["total_excluded"] == 1
        assert report["total_changed"] >= 1  # at least q1 changed mcq

    def test_choice_fallback_for_ref_only_answer(self, tmp_path: Path) -> None:
        """When answer is ref-only, fallback to choice text from metadata."""
        q = _make_gs_question(
            expected_answer="Art. 3.7",
            metadata={
                "mcq_answer": "B",
                "choices": {"A": "Non", "B": "La réponse complète ici"},
            },
        )
        # Result also ref-only
        r = _make_result(
            answer_text_from_choice="Art. 4.2",
            answer_explanation="Art. 5.1",
        )
        gs_path = _write_gs(tmp_path, [q])
        res_path = _write_results(tmp_path, [r])
        out_path = tmp_path / "out.json"

        patch_gold_standard(gs_path, res_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["questions"][0]["expected_answer"] == "La réponse complète ici"
        assert result["questions"][0]["metadata"]["answer_source"] == "choice_fallback"


# ---------------------------------------------------------------------------
# validate_patched_gs
# ---------------------------------------------------------------------------


class TestValidatePatchedGs:
    def test_clean_gs_passes(self, tmp_path: Path) -> None:
        gs = {
            "questions": [
                {
                    "id": "q1",
                    "question": "Normal question text",
                    "expected_answer": "Normal answer text",
                }
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        v = validate_patched_gs(gs_path)
        assert v["zero_fusion"] is True
        assert v["zero_empty_answer"] is True
        assert v["ref_only_count"] == 0

    def test_fusion_detected(self, tmp_path: Path) -> None:
        gs = {
            "questions": [
                {
                    "id": "q1",
                    "question": "Text ## fusion",
                    "expected_answer": "answer",
                }
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        v = validate_patched_gs(gs_path)
        assert v["zero_fusion"] is False

    def test_empty_answer_detected(self, tmp_path: Path) -> None:
        gs = {
            "questions": [
                {
                    "id": "q1",
                    "question": "Question",
                    "expected_answer": "",
                }
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        v = validate_patched_gs(gs_path)
        assert v["zero_empty_answer"] is False

    def test_ref_only_answer_detected(self, tmp_path: Path) -> None:
        gs = {
            "questions": [
                {
                    "id": "q1",
                    "question": "Question",
                    "expected_answer": "Art. 3.7",
                }
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        v = validate_patched_gs(gs_path)
        assert v["ref_only_count"] == 1

    def test_total_questions_count(self, tmp_path: Path) -> None:
        gs = {
            "questions": [
                {"id": f"q{i}", "question": f"Question {i}", "expected_answer": f"Answer {i}"}
                for i in range(5)
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        v = validate_patched_gs(gs_path)
        assert v["total_questions"] == 5
