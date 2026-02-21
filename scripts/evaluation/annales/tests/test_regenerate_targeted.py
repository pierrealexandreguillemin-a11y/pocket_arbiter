"""Tests for regenerate_targeted.py (Phase A-P2).

ISO Reference: ISO/IEC 29119 - Test design
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.evaluation.annales.regenerate_targeted import (  # noqa: E402
    PROFILES,
    _replaceability_score,
    _validate_new_question,
    apply_replacements,
    main,
    select_candidates,
)
from scripts.evaluation.annales.tests.conftest import make_gs_question  # noqa: E402


def _make_gs_data(questions: list[dict]) -> dict:
    """Wrap questions in a minimal GS structure."""
    answerable = sum(
        1 for q in questions if not q.get("content", {}).get("is_impossible", False)
    )
    return {
        "version": "1.1",
        "coverage": {
            "total_questions": len(questions),
            "answerable": answerable,
            "unanswerable": len(questions) - answerable,
        },
        "questions": questions,
    }


# ===========================================================================
# _replaceability_score
# ===========================================================================


class TestReplaceabilityScore:
    """Test the scoring function for replacement candidates."""

    def test_remember_higher_than_understand(self) -> None:
        q_rem = make_gs_question(cognitive_level="Remember")
        q_und = make_gs_question(cognitive_level="Understand")
        assert _replaceability_score(q_rem) > _replaceability_score(q_und)

    def test_understand_higher_than_apply(self) -> None:
        q_und = make_gs_question(cognitive_level="Understand")
        q_app = make_gs_question(cognitive_level="Apply")
        assert _replaceability_score(q_und) > _replaceability_score(q_app)

    def test_factual_higher_than_procedural(self) -> None:
        q_fac = make_gs_question(question_type="factual")
        q_pro = make_gs_question(question_type="procedural")
        assert _replaceability_score(q_fac) > _replaceability_score(q_pro)

    def test_lower_difficulty_more_replaceable(self) -> None:
        q_easy = make_gs_question(difficulty=0.2)
        q_hard = make_gs_question(difficulty=0.8)
        assert _replaceability_score(q_easy) > _replaceability_score(q_hard)

    def test_fact_single_higher_than_summary(self) -> None:
        q_fs = make_gs_question(reasoning_class="fact_single")
        q_sum = make_gs_question(reasoning_class="summary")
        assert _replaceability_score(q_fs) > _replaceability_score(q_sum)


# ===========================================================================
# select_candidates
# ===========================================================================


class TestSelectCandidates:
    """Test candidate selection logic."""

    def test_excludes_unanswerable(self) -> None:
        questions = [
            make_gs_question(qid=f"ans:{i}", is_impossible=False) for i in range(5)
        ] + [make_gs_question(qid=f"unans:{i}", is_impossible=True) for i in range(3)]
        chunk_index = {"test.pdf-p001-parent001-child00": "A" * 100}
        gs = _make_gs_data(questions)
        result = select_candidates(gs, chunk_index, n_per_profile=1)
        all_ids = [t.old_id for tl in result.values() for t in tl]
        assert all(not qid.startswith("unans:") for qid in all_ids)

    def test_excludes_short_chunks(self) -> None:
        questions = [make_gs_question(qid="q1")]
        chunk_index = {"test.pdf-p001-parent001-child00": "short"}
        gs = _make_gs_data(questions)
        result = select_candidates(gs, chunk_index, n_per_profile=1)
        total = sum(len(tl) for tl in result.values())
        assert total == 0

    def test_excludes_missing_chunks(self) -> None:
        questions = [make_gs_question(qid="q1")]
        chunk_index = {}  # chunk not in index
        gs = _make_gs_data(questions)
        result = select_candidates(gs, chunk_index, n_per_profile=1)
        total = sum(len(tl) for tl in result.values())
        assert total == 0

    def test_returns_4_profiles(self) -> None:
        questions = [make_gs_question(qid=f"q:{i}") for i in range(100)]
        chunk_index = {"test.pdf-p001-parent001-child00": "A" * 100}
        gs = _make_gs_data(questions)
        result = select_candidates(gs, chunk_index, n_per_profile=2)
        assert set(result.keys()) == set(PROFILES.keys())

    def test_respects_n_per_profile(self) -> None:
        questions = [make_gs_question(qid=f"q:{i}") for i in range(100)]
        chunk_index = {"test.pdf-p001-parent001-child00": "A" * 100}
        gs = _make_gs_data(questions)
        result = select_candidates(gs, chunk_index, n_per_profile=5)
        for tasks in result.values():
            assert len(tasks) <= 5

    def test_chunk_diversity_pass1(self) -> None:
        """Unique chunks preferred in first pass."""
        questions = []
        chunk_index = {}
        for i in range(8):
            cid = f"src{i}.pdf-p001-parent001-child00"
            q = make_gs_question(qid=f"q:{i}")
            q["provenance"]["chunk_id"] = cid
            questions.append(q)
            chunk_index[cid] = f"Chunk text for source {i} " * 10
        gs = _make_gs_data(questions)
        result = select_candidates(gs, chunk_index, n_per_profile=1)
        chunks_used = [t.chunk_id for tl in result.values() for t in tl]
        assert len(set(chunks_used)) == 4

    def test_pass2_fills_with_shared_chunks(self) -> None:
        """Pass 2 fills remaining slots using shared chunks."""
        # Only 2 distinct chunks but need n_per_profile=3
        # Pass 1 can only pick 2 (one per chunk), Pass 2 fills the rest
        questions = []
        chunk_index = {}
        cid_a = "srcA.pdf-p001-parent001-child00"
        cid_b = "srcB.pdf-p001-parent001-child00"
        chunk_index[cid_a] = "Chunk A text is long enough " * 5
        chunk_index[cid_b] = "Chunk B text is long enough " * 5
        for i in range(10):
            q = make_gs_question(qid=f"q:{i}")
            q["provenance"]["chunk_id"] = cid_a if i % 2 == 0 else cid_b
            questions.append(q)
        gs = _make_gs_data(questions)
        result = select_candidates(gs, chunk_index, n_per_profile=3)
        # First profile should get 3 (2 from pass1 + 1 from pass2)
        first_profile = list(result.keys())[0]
        assert len(result[first_profile]) == 3

    def test_no_duplicate_ids_across_profiles(self) -> None:
        questions = [make_gs_question(qid=f"q:{i}") for i in range(100)]
        chunk_index = {"test.pdf-p001-parent001-child00": "A" * 100}
        gs = _make_gs_data(questions)
        result = select_candidates(gs, chunk_index, n_per_profile=10)
        all_ids = [t.old_id for tl in result.values() for t in tl]
        assert len(all_ids) == len(set(all_ids))


# ===========================================================================
# _validate_new_question
# ===========================================================================


class TestValidateNewQuestion:
    """Test schema validation of replacement questions."""

    def test_valid_question_passes(self) -> None:
        q = make_gs_question(qid="ok")
        assert _validate_new_question(q) == []

    def test_missing_content(self) -> None:
        q = {"classification": {}, "provenance": {}}
        errs = _validate_new_question(q)
        assert any("content" in e for e in errs)

    def test_missing_classification(self) -> None:
        q = make_gs_question(qid="ok")
        del q["classification"]
        errs = _validate_new_question(q)
        assert any("classification" in e for e in errs)

    def test_question_no_mark(self) -> None:
        q = make_gs_question(qid="ok")
        q["content"]["question"] = "No question mark"
        errs = _validate_new_question(q)
        assert any("?" in e for e in errs)

    def test_short_answer(self) -> None:
        q = make_gs_question(qid="ok")
        q["content"]["expected_answer"] = "abc"
        errs = _validate_new_question(q)
        assert any("too short" in e for e in errs)

    def test_difficulty_out_of_range(self) -> None:
        q = make_gs_question(qid="ok", difficulty=1.5)
        errs = _validate_new_question(q)
        assert any("difficulty" in e for e in errs)

    def test_invalid_cognitive_level(self) -> None:
        q = make_gs_question(qid="ok", cognitive_level="Invent")
        errs = _validate_new_question(q)
        assert any("cognitive_level" in e for e in errs)


# ===========================================================================
# apply_replacements
# ===========================================================================


class TestApplyReplacements:
    """Test the replacement/patch logic."""

    @staticmethod
    def _make_replacement(
        old_id: str,
        profile: str = "HARD_APPLY",
    ) -> dict:
        new_q = make_gs_question(qid="new:001")
        return {
            "old_id": old_id,
            "profile": profile,
            "new_question": new_q,
        }

    def test_preserves_original_id(self) -> None:
        q = make_gs_question(qid="old:001")
        gs = _make_gs_data([q])
        repl = self._make_replacement("old:001")
        report = apply_replacements(gs, [repl], date="2026-02-21")
        assert gs["questions"][0]["id"] == "old:001"
        assert report["total_replacements"] == 1

    def test_audit_trail_format(self) -> None:
        q = make_gs_question(qid="old:002")
        gs = _make_gs_data([q])
        repl = self._make_replacement("old:002", profile="HARD_ANALYZE")
        apply_replacements(gs, [repl], date="2026-02-21")
        history = gs["questions"][0]["audit"]["history"]
        assert "[PHASE A-P2]" in history
        assert "HARD_ANALYZE" in history

    def test_preserves_total_count(self) -> None:
        questions = [make_gs_question(qid=f"q:{i}") for i in range(10)]
        gs = _make_gs_data(questions)
        repl = self._make_replacement("q:0")
        report = apply_replacements(gs, [repl], date="2026-02-21")
        assert len(gs["questions"]) == 10
        assert report["total_questions"] == 10

    def test_sets_batch(self) -> None:
        q = make_gs_question(qid="old:003")
        gs = _make_gs_data([q])
        repl = self._make_replacement("old:003")
        apply_replacements(gs, [repl], date="2026-02-21")
        assert gs["questions"][0]["validation"]["batch"] == "gs_v1_step1_p2"

    def test_requires_chunk_match_score_100(self) -> None:
        """Questions must arrive with chunk_match_score=100."""
        q = make_gs_question(qid="old:004")
        gs = _make_gs_data([q])
        repl = self._make_replacement("old:004")
        # Verify the default from make_gs_question has score=100
        report = apply_replacements(gs, [repl], date="2026-02-21")
        assert report["total_replacements"] == 1
        assert gs["questions"][0]["processing"]["chunk_match_score"] == 100

    def test_rejects_missing_chunk_match_score(self) -> None:
        """Reject question without chunk_match_score=100."""
        q = make_gs_question(qid="old:006")
        gs = _make_gs_data([q])
        new_q = make_gs_question(qid="new:006")
        new_q["processing"]["chunk_match_score"] = 80
        repl = {
            "old_id": "old:006",
            "profile": "HARD_APPLY",
            "new_question": new_q,
        }
        report = apply_replacements(gs, [repl], date="2026-02-21")
        assert report["total_replacements"] == 0
        assert any("chunk_match_score" in e for e in report["errors"])

    def test_error_on_missing_id(self) -> None:
        gs = _make_gs_data([make_gs_question(qid="exists")])
        repl = self._make_replacement("nonexistent")
        report = apply_replacements(gs, [repl], date="2026-02-21")
        assert len(report["errors"]) == 1
        assert report["total_replacements"] == 0

    def test_rejects_invalid_question(self) -> None:
        """apply_replacements rejects a question with missing content."""
        q = make_gs_question(qid="old:005")
        gs = _make_gs_data([q])
        bad_q: dict = {"id": "bad", "provenance": {}}
        repl = {
            "old_id": "old:005",
            "profile": "HARD_APPLY",
            "new_question": bad_q,
        }
        report = apply_replacements(gs, [repl], date="2026-02-21")
        assert report["total_replacements"] == 0
        assert len(report["errors"]) == 1
        assert "Invalid question" in report["errors"][0]

    def test_syncs_coverage_header(self) -> None:
        questions = [
            make_gs_question(qid="a1", is_impossible=False),
            make_gs_question(qid="a2", is_impossible=True),
        ]
        gs = _make_gs_data(questions)
        repl = self._make_replacement("a1")
        apply_replacements(gs, [repl], date="2026-02-21")
        assert gs["coverage"]["total_questions"] == 2
        assert gs["coverage"]["answerable"] == 1


# ===========================================================================
# CLI (main)
# ===========================================================================


class TestCLI:
    """Test CLI entry points."""

    def test_select_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        gs = _make_gs_data([make_gs_question(qid=f"q:{i}") for i in range(5)])
        chunks = {
            "chunks": [
                {
                    "id": "test.pdf-p001-parent001-child00",
                    "text": "A" * 100,
                }
            ],
        }
        gs_path = tmp_path / "gs.json"
        chunks_path = tmp_path / "chunks.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")
        chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--gs",
                str(gs_path),
                "--chunks",
                str(chunks_path),
                "--select-only",
            ],
        )
        ret = main()
        assert ret == 0

    def test_apply_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        q = make_gs_question(qid="old:001")
        gs = _make_gs_data([q])
        new_q = make_gs_question(qid="new:001")
        replacements = [
            {
                "old_id": "old:001",
                "profile": "HARD_APPLY",
                "new_question": new_q,
            }
        ]

        gs_path = tmp_path / "gs.json"
        repl_path = tmp_path / "replacements.json"
        out_path = tmp_path / "output.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")
        repl_path.write_text(json.dumps(replacements), encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--gs",
                str(gs_path),
                "--replacements",
                str(repl_path),
                "--apply-only",
                "--output",
                str(out_path),
            ],
        )
        ret = main()
        assert ret == 0
        assert out_path.exists()

    def test_apply_only_dict_format(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replacements file as dict with 'replacements' key."""
        q = make_gs_question(qid="old:001")
        gs = _make_gs_data([q])
        new_q = make_gs_question(qid="new:001")
        replacements = {
            "replacements": [
                {
                    "old_id": "old:001",
                    "profile": "HARD_APPLY",
                    "new_question": new_q,
                }
            ],
        }

        gs_path = tmp_path / "gs.json"
        repl_path = tmp_path / "replacements.json"
        out_path = tmp_path / "output.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")
        repl_path.write_text(json.dumps(replacements), encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--gs",
                str(gs_path),
                "--replacements",
                str(repl_path),
                "--apply-only",
                "--output",
                str(out_path),
            ],
        )
        ret = main()
        assert ret == 0
