"""Tests for select_page_number_candidates (Phase B-P3a).

ISO Reference: ISO/IEC 29119-3 - Test data quality validation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.annales.select_page_number_candidates import (
    PageCandidate,
    build_chunk_indexes,
    format_report,
    is_page_number_question,
    save_candidates,
    select_page_number_questions,
)
from scripts.evaluation.annales.tests.conftest import make_gs_question

# ---------------------------------------------------------------------------
# is_page_number_question
# ---------------------------------------------------------------------------


class TestIsPageNumberQuestion:
    """Test page-number pattern detection."""

    @pytest.mark.parametrize(
        "text",
        [
            "Quelle regle est enoncee a la page 185?",
            "Que precise le reglement a la page 42?",
            "Selon la page 10, quelle obligation s'applique?",
            "Article de la page 3 du reglement?",
            "Quelle obligation est enoncee a la page 200?",
            "A la Page 15, que dit le texte?",
            "Voir page 99 pour les details?",
            "Regle enoncee page 5?",
            "Sur la page12, quel article?",
        ],
    )
    def test_matches_page_patterns(self, text: str) -> None:
        assert is_page_number_question(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Que doit faire l'arbitre en cas de litige?",
            "Combien de temps pour prendre une decision?",
            "Quelle est la regle du roque?",
            "Le joueur peut-il quitter la salle?",
            "Quel est l'article 5.1 du reglement?",
            "En cas de page blanche, que faire?",  # "page" without number
            "",
        ],
    )
    def test_no_match_non_page_questions(self, text: str) -> None:
        assert is_page_number_question(text) is False


# ---------------------------------------------------------------------------
# select_page_number_questions
# ---------------------------------------------------------------------------


def _make_gs_data(questions: list[dict]) -> dict:
    """Wrap questions in minimal GS structure."""
    answerable = sum(1 for q in questions if not q["content"]["is_impossible"])
    return {
        "version": "1.1",
        "coverage": {
            "total_questions": len(questions),
            "answerable": answerable,
            "unanswerable": len(questions) - answerable,
        },
        "questions": questions,
    }


class TestSelectPageNumberQuestions:
    """Test candidate selection logic."""

    def test_selects_page_number_answerable(self) -> None:
        q1 = make_gs_question(
            qid="q1",
            question_text="Quelle regle est enoncee a la page 185?",
        )
        q2 = make_gs_question(
            qid="q2",
            question_text="Que doit faire l'arbitre en cas de litige?",
        )
        gs = _make_gs_data([q1, q2])
        chunk_index = {"test.pdf-p001-parent001-child00": "some chunk text"}

        result = select_page_number_questions(gs, chunk_index)

        assert len(result) == 1
        assert result[0].old_id == "q1"

    def test_skips_unanswerable(self) -> None:
        q1 = make_gs_question(
            qid="q1",
            is_impossible=True,
            question_text="Quelle regle a la page 99?",
        )
        gs = _make_gs_data([q1])
        chunk_index = {"test.pdf-p001-parent001-child00": "some text"}

        result = select_page_number_questions(gs, chunk_index)

        assert len(result) == 0

    def test_multiple_candidates(self) -> None:
        questions = [
            make_gs_question(
                qid=f"q{i}",
                question_text=f"Quelle regle est a la page {i * 10}?",
            )
            for i in range(5)
        ]
        gs = _make_gs_data(questions)
        chunk_index = {"test.pdf-p001-parent001-child00": "text"}

        result = select_page_number_questions(gs, chunk_index)

        assert len(result) == 5
        assert [c.old_id for c in result] == [f"q{i}" for i in range(5)]

    def test_empty_gs(self) -> None:
        gs = _make_gs_data([])
        result = select_page_number_questions(gs, {})
        assert result == []

    def test_no_page_questions(self) -> None:
        q = make_gs_question(
            qid="q1",
            question_text="Que signifie le roque?",
        )
        gs = _make_gs_data([q])
        result = select_page_number_questions(gs, {})
        assert result == []

    def test_includes_source_when_provided(self) -> None:
        q = make_gs_question(
            qid="q1",
            question_text="Quelle regle a la page 42?",
        )
        gs = _make_gs_data([q])
        chunk_index = {"test.pdf-p001-parent001-child00": "text"}
        source_index = {"test.pdf-p001-parent001-child00": "LA-octobre2025.pdf"}

        result = select_page_number_questions(gs, chunk_index, source_index)

        assert len(result) == 1
        assert result[0].source == "LA-octobre2025.pdf"

    def test_question_preview_truncated(self) -> None:
        long_q = "Quelle regle est enoncee a la page 185 du reglement? " * 5
        q = make_gs_question(qid="q1", question_text=long_q)
        gs = _make_gs_data([q])

        result = select_page_number_questions(gs, {})

        assert len(result[0].question_preview) <= 120


# ---------------------------------------------------------------------------
# build_chunk_indexes
# ---------------------------------------------------------------------------


class TestBuildChunkIndexes:
    """Test chunk index construction."""

    def test_basic_indexing(self) -> None:
        data = {
            "chunks": [
                {"id": "c1", "text": "chunk 1 text", "source": "doc1.pdf"},
                {"id": "c2", "text": "chunk 2 text", "source": "doc2.pdf"},
            ]
        }
        text_idx, source_idx = build_chunk_indexes(data)

        assert text_idx == {"c1": "chunk 1 text", "c2": "chunk 2 text"}
        assert source_idx == {"c1": "doc1.pdf", "c2": "doc2.pdf"}

    def test_missing_fields(self) -> None:
        data = {"chunks": [{"id": "c1"}]}
        text_idx, source_idx = build_chunk_indexes(data)

        assert text_idx == {"c1": ""}
        assert source_idx == {"c1": ""}

    def test_dict_format(self) -> None:
        """Handle case where chunks is a dict instead of list."""
        data = {
            "c1": {"id": "c1", "text": "t1", "source": "s1"},
        }
        text_idx, source_idx = build_chunk_indexes(data)

        assert "c1" in text_idx


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    """Test report formatting."""

    def test_empty_candidates(self) -> None:
        report = format_report([])
        assert "Total candidates: 0" in report

    def test_with_candidates(self) -> None:
        candidates = [
            PageCandidate(
                old_id="q1",
                chunk_id="c1",
                chunk_text="text",
                source="doc.pdf",
                question_preview="Quelle regle a la page 10?",
            ),
        ]
        report = format_report(candidates)
        assert "Total candidates: 1" in report
        assert "doc.pdf" in report


# ---------------------------------------------------------------------------
# save_candidates
# ---------------------------------------------------------------------------


class TestSaveCandidates:
    """Test JSON output."""

    def test_output_structure(self, tmp_path: Path) -> None:
        candidates = [
            PageCandidate(
                old_id="q1",
                chunk_id="c1",
                chunk_text="some text",
                source="doc.pdf",
                question_preview="Quelle regle a la page 10?",
            ),
        ]
        out_path = tmp_path / "candidates.json"
        result = save_candidates(candidates, out_path)

        assert result["phase"] == "B-P3a"
        assert result["total"] == 1
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["old_id"] == "q1"

        # Verify file is valid JSON
        with open(out_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["total"] == 1

    def test_empty_candidates(self, tmp_path: Path) -> None:
        out_path = tmp_path / "empty.json"
        result = save_candidates([], out_path)

        assert result["total"] == 0
        assert result["candidates"] == []
