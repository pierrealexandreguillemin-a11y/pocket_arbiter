"""Unit tests for recall measurement (no model loading)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pipeline.recall import (
    compute_metrics,
    error_analysis,
    evaluate_question,
    load_gs,
    page_match,
)
from scripts.pipeline.recall_report import write_json, write_markdown

GS_PATH = Path("tests/data/gold_standard_annales_fr_v8_adversarial.json")


class TestLoadGs:
    """Test GS loading and filtering."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_gs(self) -> None:
        if not GS_PATH.exists():
            pytest.skip("GS file not available")

    def test_loads_answerable_only(self) -> None:
        questions = load_gs(GS_PATH)
        assert len(questions) == 298

    def test_question_has_required_fields(self) -> None:
        questions = load_gs(GS_PATH)
        q = questions[0]
        for key in (
            "question",
            "expected_pages",
            "expected_docs",
            "reasoning_class",
            "difficulty",
            "id",
        ):
            assert key in q, f"Missing field: {key}"

    def test_no_impossible_questions(self) -> None:
        questions = load_gs(GS_PATH)
        for q in questions:
            assert len(q["expected_pages"]) >= 1
            assert len(q["expected_docs"]) >= 1


class TestPageMatch:
    """Test page-level matching logic."""

    def test_hit_when_page_matches(self) -> None:
        assert page_match([("doc.pdf", 10)], [("doc.pdf", 10)]) is True

    def test_miss_when_no_page_matches(self) -> None:
        assert page_match([("doc.pdf", 10)], [("doc.pdf", 20)]) is False

    def test_miss_when_empty_retrieved(self) -> None:
        assert page_match([("doc.pdf", 10)], []) is False

    def test_hit_with_multiple_expected(self) -> None:
        expected = [("doc.pdf", 10), ("doc.pdf", 11)]
        assert page_match(expected, [("doc.pdf", 11)]) is True

    def test_miss_wrong_source(self) -> None:
        assert page_match([("doc.pdf", 10)], [("other.pdf", 10)]) is False


class TestEvaluateQuestion:
    """Test single question evaluation."""

    def test_hit_at_rank_1(self) -> None:
        q = {
            "id": "q1",
            "expected_pairs": [("d.pdf", 10)],
            "reasoning_class": "fact_single",
            "difficulty": 0.2,
        }
        pages = [[("d.pdf", 10)], [("d.pdf", 20)]]
        r = evaluate_question(q, pages)
        assert r["hit@1"] is True
        assert r["rank"] == 1

    def test_hit_at_rank_3(self) -> None:
        q = {
            "id": "q2",
            "expected_pairs": [("d.pdf", 10)],
            "reasoning_class": "summary",
            "difficulty": 0.5,
        }
        pages = [[("d.pdf", 20)], [("d.pdf", 30)], [("d.pdf", 10)]]
        r = evaluate_question(q, pages)
        assert r["hit@1"] is False
        assert r["hit@3"] is True
        assert r["hit@5"] is True
        assert r["rank"] == 3

    def test_miss(self) -> None:
        q = {
            "id": "q3",
            "expected_pairs": [("d.pdf", 10)],
            "reasoning_class": "summary",
            "difficulty": 0.8,
        }
        pages = [[("other.pdf", 5)]]
        r = evaluate_question(q, pages)
        assert r["hit@10"] is False
        assert r["rank"] == 0

    def test_empty_contexts(self) -> None:
        q = {
            "id": "q4",
            "expected_pairs": [("d.pdf", 10)],
            "reasoning_class": "summary",
            "difficulty": 0.5,
        }
        r = evaluate_question(q, [])
        assert r["hit@1"] is False
        assert r["rank"] == 0


class TestComputeMetrics:
    """Test metrics aggregation."""

    def test_perfect_recall(self) -> None:
        results = [
            {
                "hit@1": True,
                "hit@3": True,
                "hit@5": True,
                "hit@10": True,
                "rank": 1,
                "reasoning_class": "fact_single",
                "difficulty": 0.2,
            },
            {
                "hit@1": True,
                "hit@3": True,
                "hit@5": True,
                "hit@10": True,
                "rank": 1,
                "reasoning_class": "summary",
                "difficulty": 0.5,
            },
        ]
        m = compute_metrics(results)
        assert m["global"]["recall@1"] == 1.0
        assert m["global"]["mrr"] == 1.0

    def test_partial_recall(self) -> None:
        results = [
            {
                "hit@1": True,
                "hit@3": True,
                "hit@5": True,
                "hit@10": True,
                "rank": 1,
                "reasoning_class": "fact_single",
                "difficulty": 0.2,
            },
            {
                "hit@1": False,
                "hit@3": False,
                "hit@5": False,
                "hit@10": False,
                "rank": 0,
                "reasoning_class": "summary",
                "difficulty": 0.8,
            },
        ]
        m = compute_metrics(results)
        assert m["global"]["recall@1"] == 0.5
        assert m["global"]["recall@5"] == 0.5

    def test_segments_reasoning_class(self) -> None:
        results = [
            {
                "hit@1": True,
                "hit@3": True,
                "hit@5": True,
                "hit@10": True,
                "rank": 1,
                "reasoning_class": "fact_single",
                "difficulty": 0.2,
            },
            {
                "hit@1": False,
                "hit@3": False,
                "hit@5": False,
                "hit@10": False,
                "rank": 0,
                "reasoning_class": "summary",
                "difficulty": 0.8,
            },
        ]
        m = compute_metrics(results)
        assert m["segments"]["reasoning_class"]["fact_single"]["recall@1"] == 1.0
        assert m["segments"]["reasoning_class"]["summary"]["recall@1"] == 0.0

    def test_segments_difficulty(self) -> None:
        results = [
            {
                "hit@1": True,
                "hit@3": True,
                "hit@5": True,
                "hit@10": True,
                "rank": 1,
                "reasoning_class": "fact_single",
                "difficulty": 0.1,
            },
            {
                "hit@1": False,
                "hit@3": False,
                "hit@5": False,
                "hit@10": False,
                "rank": 0,
                "reasoning_class": "summary",
                "difficulty": 0.9,
            },
        ]
        m = compute_metrics(results)
        assert m["segments"]["difficulty"]["easy"]["recall@1"] == 1.0
        assert m["segments"]["difficulty"]["hard"]["recall@1"] == 0.0

    def test_empty_results(self) -> None:
        m = compute_metrics([])
        assert m["global"]["count"] == 0
        assert m["global"]["recall@5"] == 0.0


class TestErrorAnalysis:
    """Test error case extraction."""

    def test_returns_top_n_failures(self) -> None:
        results = [
            {
                "id": f"q{i}",
                "hit@10": i < 5,
                "rank": i + 1 if i < 5 else 0,
                "reasoning_class": "summary",
                "difficulty": 0.5,
            }
            for i in range(10)
        ]
        questions = [
            {
                "id": f"q{i}",
                "question": f"Question {i}",
                "expected_docs": ["d.pdf"],
                "expected_pages": [i],
                "expected_pairs": [("d.pdf", i)],
                "reasoning_class": "summary",
                "difficulty": 0.5,
            }
            for i in range(10)
        ]
        errors = error_analysis(results, questions, n=3)
        assert len(errors) == 3
        assert all(e["hit@10"] is False for e in errors)

    def test_returns_empty_if_no_failures(self) -> None:
        results = [
            {
                "id": "q1",
                "hit@10": True,
                "rank": 1,
                "reasoning_class": "fact_single",
                "difficulty": 0.2,
            }
        ]
        questions = [
            {
                "id": "q1",
                "question": "Q1",
                "expected_docs": ["d.pdf"],
                "expected_pages": [1],
                "expected_pairs": [("d.pdf", 1)],
                "reasoning_class": "fact_single",
                "difficulty": 0.2,
            }
        ]
        assert len(error_analysis(results, questions, n=20)) == 0


class TestWriteReports:
    """Test report file generation."""

    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "test.json"
        data = {"global": {"recall@5": 0.75}, "metadata": {}}
        write_json(data, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["global"]["recall@5"] == 0.75

    def test_write_markdown_has_yaml_header(self, tmp_path: Path) -> None:
        out = tmp_path / "test.md"
        data = {
            "metadata": {"generated": "2026-03-18", "model": "test"},
            "global": {
                "recall@1": 0.5,
                "recall@3": 0.6,
                "recall@5": 0.7,
                "recall@10": 0.8,
                "mrr": 0.55,
            },
            "segments": {"reasoning_class": {}, "difficulty": {}},
            "errors": [],
        }
        write_markdown(data, out)
        content = out.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "recall@5" in content
        assert "Optimisations retrieval" in content

    def test_write_markdown_decision_fine_tune(self, tmp_path: Path) -> None:
        out = tmp_path / "test2.md"
        data = {
            "metadata": {"generated": "2026-03-18"},
            "global": {
                "recall@1": 0.2,
                "recall@3": 0.3,
                "recall@5": 0.4,
                "recall@10": 0.5,
                "mrr": 0.25,
            },
            "segments": {"reasoning_class": {}, "difficulty": {}},
            "errors": [],
        }
        write_markdown(data, out)
        content = out.read_text(encoding="utf-8")
        assert "Fine-tuning" in content


@pytest.mark.slow
class TestRunRecall:
    """Integration test on real corpus."""

    DB_PATH = Path("corpus/processed/corpus_v2_fr.db")

    @pytest.fixture(autouse=True)
    def _skip_if_no_db(self) -> None:
        if not self.DB_PATH.exists():
            pytest.skip("corpus_v2_fr.db not available")
        if not GS_PATH.exists():
            pytest.skip("GS file not available")

    def test_run_produces_reports(self, tmp_path: Path) -> None:
        from scripts.pipeline.recall import run_recall

        data = run_recall(self.DB_PATH, GS_PATH, output_dir=tmp_path)
        assert data["metadata"]["questions_total"] == 298
        assert 0.0 <= data["global"]["recall@5"] <= 1.0
        assert (tmp_path / "recall_baseline.json").exists()
        assert (tmp_path / "recall_baseline.md").exists()
        md = (tmp_path / "recall_baseline.md").read_text(encoding="utf-8")
        assert md.startswith("---\n")
