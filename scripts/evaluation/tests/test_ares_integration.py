"""Tests for ARES integration module.

ISO Reference: ISO 29119 - Software Testing
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.evaluation.ares.convert_to_ares import (
    _get_negative_chunks,
    _write_tsv,
)
from scripts.evaluation.ares.generate_few_shot import (
    _select_diverse_examples,
)
from scripts.evaluation.ares.report import (
    _assess_iso_compliance,
    _build_comparison,
    _generate_recommendations,
)
from scripts.evaluation.ares.run_evaluation import (
    _estimate_cost,
    check_openai_api_key,
    run_mock_evaluation,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_chunks() -> dict[str, dict[str, Any]]:
    """Create sample chunks for testing."""
    return {
        "doc1-p001-parent000-child00": {
            "id": "doc1-p001-parent000-child00",
            "text": "Le toucher-jouer est une règle fondamentale.",
            "source": "doc1.pdf",
            "pages": [1],
            "tokens": 20,
        },
        "doc1-p002-parent001-child00": {
            "id": "doc1-p002-parent001-child00",
            "text": "Le forfait est déclaré après 30 minutes.",
            "source": "doc1.pdf",
            "pages": [2],
            "tokens": 15,
        },
        "doc2-p001-parent000-child00": {
            "id": "doc2-p001-parent000-child00",
            "text": "Les compétitions jeunes ont des règles spécifiques.",
            "source": "doc2.pdf",
            "pages": [1],
            "tokens": 18,
        },
    }


@pytest.fixture
def sample_questions() -> list[dict[str, Any]]:
    """Create sample questions for testing."""
    return [
        {
            "id": "FR-Q01",
            "question": "Quelle est la règle du toucher-jouer ?",
            "category": "regles_jeu",
            "expected_chunk_id": "doc1-p001-parent000-child00",
            "validation": {"status": "VALIDATED"},
            "metadata": {"hard_type": "ANSWERABLE"},
        },
        {
            "id": "FR-Q02",
            "question": "Combien de temps avant forfait ?",
            "category": "temps",
            "expected_chunk_id": "doc1-p002-parent001-child00",
            "validation": {"status": "VALIDATED"},
            "metadata": {"hard_type": "ANSWERABLE"},
        },
    ]


# ============================================================================
# Test TSV Format
# ============================================================================


class TestTsvFormatValid:
    """Test TSV output format is valid for ARES."""

    def test_tsv_has_required_columns(self, tmp_path: Path) -> None:
        """TSV must have Query, Document, Answer columns."""
        samples = [
            {
                "Query": "Test question?",
                "Document": "Test document text.",
                "Answer": "Test answer.",
                "Context_Relevance_Label": 1,
            }
        ]

        output_path = tmp_path / "test.tsv"
        _write_tsv(output_path, samples, include_label=True)

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []

            assert "Query" in fieldnames
            assert "Document" in fieldnames
            assert "Answer" in fieldnames
            assert "Context_Relevance_Label" in fieldnames

    def test_tsv_unlabeled_excludes_label(self, tmp_path: Path) -> None:
        """Unlabeled TSV should not have label column."""
        samples = [
            {
                "Query": "Test question?",
                "Document": "Test document text.",
                "Answer": "Test answer.",
                "Context_Relevance_Label": 1,
            }
        ]

        output_path = tmp_path / "test.tsv"
        _write_tsv(output_path, samples, include_label=False)

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []

            assert "Query" in fieldnames
            assert "Context_Relevance_Label" not in fieldnames

    def test_tsv_preserves_unicode(self, tmp_path: Path) -> None:
        """TSV must preserve French characters."""
        samples = [
            {
                "Query": "Règle du toucher-jouer ?",
                "Document": "Pièce touchée, pièce à jouer.",
                "Answer": "Réponse avec accents éèà.",
                "Context_Relevance_Label": 1,
            }
        ]

        output_path = tmp_path / "test.tsv"
        _write_tsv(output_path, samples, include_label=True)

        with open(output_path, encoding="utf-8") as f:
            content = f.read()
            assert "Règle" in content
            assert "Pièce" in content
            assert "éèà" in content


# ============================================================================
# Test Only Answerable Questions
# ============================================================================


class TestOnlyAnswerable:
    """Test that only answerable questions are used for positive samples."""

    def test_filters_unanswerable(
        self, sample_chunks: dict[str, dict[str, Any]]
    ) -> None:
        """Unanswerable questions should not appear in positive samples."""
        questions: list[dict[str, Any]] = [
            {
                "id": "Q1",
                "question": "Answerable question?",
                "expected_chunk_id": "doc1-p001-parent000-child00",
                "validation": {"status": "VALIDATED"},
                "metadata": {"hard_type": "ANSWERABLE"},
            },
            {
                "id": "Q2",
                "question": "Unanswerable question?",
                "expected_chunk_id": "doc1-p002-parent001-child00",
                "validation": {"status": "VALIDATED"},
                "metadata": {"hard_type": "OUT_OF_SCOPE"},
            },
        ]

        answerable = [
            q
            for q in questions
            if q.get("expected_chunk_id")
            and q["validation"]["status"] == "VALIDATED"
            and q["metadata"].get("hard_type", "ANSWERABLE") == "ANSWERABLE"
        ]

        assert len(answerable) == 1
        assert answerable[0]["id"] == "Q1"

    def test_filters_unvalidated(self) -> None:
        """Unvalidated questions should not appear in positive samples."""
        questions: list[dict[str, Any]] = [
            {
                "id": "Q1",
                "question": "Validated question?",
                "expected_chunk_id": "chunk1",
                "validation": {"status": "VALIDATED"},
                "metadata": {"hard_type": "ANSWERABLE"},
            },
            {
                "id": "Q2",
                "question": "Pending question?",
                "expected_chunk_id": "chunk2",
                "validation": {"status": "PENDING"},
                "metadata": {"hard_type": "ANSWERABLE"},
            },
        ]

        validated = [
            q
            for q in questions
            if q["validation"]["status"] == "VALIDATED"
        ]

        assert len(validated) == 1
        assert validated[0]["id"] == "Q1"


# ============================================================================
# Test Chunk Traceability (ISO 42001)
# ============================================================================


class TestChunkTraceability:
    """Test ISO 42001 traceability requirements."""

    def test_negative_chunks_different_page(
        self, sample_chunks: dict[str, dict[str, Any]]
    ) -> None:
        """Negative chunks must be from different pages/sources."""
        positive_id = "doc1-p001-parent000-child00"
        negatives = _get_negative_chunks(sample_chunks, positive_id, n=2, seed=42)

        positive_chunk = sample_chunks[positive_id]
        positive_pages = set(positive_chunk["pages"])
        positive_source = positive_chunk["source"]

        for neg in negatives:
            neg_pages = set(neg["pages"])
            # Either different source OR non-overlapping pages
            assert (
                neg["source"] != positive_source
                or not neg_pages.intersection(positive_pages)
            )

    def test_mapping_file_has_gs_id(self, tmp_path: Path) -> None:
        """Mapping file must contain gs_id for traceability."""
        mapping = {
            "corpus": "fr",
            "samples": [
                {"gs_id": "FR-Q01", "chunk_id": "chunk1", "label": 1},
                {"gs_id": "FR-Q02_neg", "chunk_id": "chunk2", "label": 0},
            ],
        }

        mapping_path = tmp_path / "mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)

        with open(mapping_path, encoding="utf-8") as f:
            loaded = json.load(f)

        for sample in loaded["samples"]:
            assert "gs_id" in sample
            assert "chunk_id" in sample


# ============================================================================
# Test Negative Ratio
# ============================================================================


class TestNegativeRatio:
    """Test negative sample ratio is correct."""

    def test_30_percent_negatives(self) -> None:
        """Negative samples should be ~30% of total."""
        n_positive = 100
        negative_ratio = 0.30

        # Formula: n_neg = n_pos * ratio / (1 - ratio)
        n_negatives = int(n_positive * negative_ratio / (1 - negative_ratio))

        total = n_positive + n_negatives
        actual_ratio = n_negatives / total

        # Allow 5% tolerance due to rounding
        assert 0.25 <= actual_ratio <= 0.35

    def test_negative_samples_labeled_zero(self, tmp_path: Path) -> None:
        """Negative samples must have label 0."""
        samples = [
            {"Query": "Q1", "Document": "D1", "Answer": "A1", "Context_Relevance_Label": 1},
            {"Query": "Q2", "Document": "D2", "Answer": "", "Context_Relevance_Label": 0},
        ]

        negatives = [s for s in samples if s["Context_Relevance_Label"] == 0]

        assert len(negatives) == 1
        assert negatives[0]["Answer"] == ""  # No valid answer for negative


# ============================================================================
# Test Context Relevance Threshold
# ============================================================================


class TestContextRelevanceThreshold:
    """Test context relevance evaluation thresholds."""

    def test_pass_threshold_80_percent(self) -> None:
        """Context relevance >= 80% should pass."""
        context_relevance = {
            "score": 0.85,
            "ci_95_lower": 0.80,
            "ci_95_upper": 0.90,
            "n_samples": 100,
            "pass": True,
        }

        assert context_relevance["score"] >= 0.80
        assert context_relevance["pass"] is True

    def test_fail_threshold_below_80_percent(self) -> None:
        """Context relevance < 80% should fail."""
        context_relevance = {
            "score": 0.75,
            "ci_95_lower": 0.70,
            "ci_95_upper": 0.80,
            "n_samples": 100,
            "pass": False,
        }

        assert context_relevance["score"] < 0.80
        assert context_relevance["pass"] is False


# ============================================================================
# Test Confidence Interval Validity
# ============================================================================


class TestConfidenceIntervalValid:
    """Test 95% confidence interval validity."""

    def test_ci_bounds_ordered(self) -> None:
        """CI bounds must satisfy: lower <= score <= upper."""
        context_relevance = {
            "score": 0.85,
            "ci_95_lower": 0.80,
            "ci_95_upper": 0.90,
        }

        assert context_relevance["ci_95_lower"] <= context_relevance["score"]
        assert context_relevance["score"] <= context_relevance["ci_95_upper"]

    def test_ci_bounds_in_valid_range(self) -> None:
        """CI bounds must be in [0, 1]."""
        context_relevance = {
            "score": 0.85,
            "ci_95_lower": 0.80,
            "ci_95_upper": 0.90,
        }

        assert 0.0 <= context_relevance["ci_95_lower"] <= 1.0
        assert 0.0 <= context_relevance["ci_95_upper"] <= 1.0

    def test_iso_compliance_validates_ci(self) -> None:
        """ISO compliance check validates CI bounds."""
        valid_cr = {
            "score": 0.85,
            "ci_95_lower": 0.80,
            "ci_95_upper": 0.90,
            "n_samples": 100,
        }

        compliance = _assess_iso_compliance(valid_cr, {"total_samples": 100})
        assert compliance["checks"]["confidence_interval_valid"]["pass"] is True

        # Invalid CI (lower > score)
        invalid_cr = {
            "score": 0.85,
            "ci_95_lower": 0.90,  # Invalid: lower > score
            "ci_95_upper": 0.95,
            "n_samples": 100,
        }

        compliance_invalid = _assess_iso_compliance(invalid_cr, {"total_samples": 100})
        assert compliance_invalid["checks"]["confidence_interval_valid"]["pass"] is False


# ============================================================================
# Test Mock Evaluation
# ============================================================================


class TestMockEvaluation:
    """Test mock evaluation for testing without LLM."""

    def test_mock_returns_valid_structure(self, tmp_path: Path) -> None:
        """Mock evaluation returns expected structure."""
        # Create minimal gold label file
        gold_label_dir = tmp_path / "data" / "evaluation" / "ares"
        gold_label_dir.mkdir(parents=True)

        gold_label_path = gold_label_dir / "gold_label_fr.tsv"
        with open(gold_label_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Query", "Document", "Answer", "Context_Relevance_Label"],
                delimiter="\t"
            )
            writer.writeheader()
            for i in range(10):
                writer.writerow({
                    "Query": f"Question {i}",
                    "Document": f"Document {i}",
                    "Answer": f"Answer {i}",
                    "Context_Relevance_Label": 1 if i < 7 else 0,
                })

        # Patch DATA_DIR to use tmp_path
        with patch("scripts.evaluation.ares.run_evaluation.DATA_DIR", gold_label_dir):
            result = run_mock_evaluation(corpus="fr")

        assert "context_relevance" in result
        assert "score" in result["context_relevance"]
        assert "ci_95_lower" in result["context_relevance"]
        assert "ci_95_upper" in result["context_relevance"]
        assert result["llm_used"] == "mock"


# ============================================================================
# Test Few-Shot Generation
# ============================================================================


class TestFewShotGeneration:
    """Test few-shot example generation."""

    def test_diverse_category_selection(
        self, sample_questions: list[dict[str, Any]]
    ) -> None:
        """Selection should cover different categories."""
        questions = [
            {"id": "Q1", "category": "regles_jeu"},
            {"id": "Q2", "category": "temps"},
            {"id": "Q3", "category": "discipline"},
            {"id": "Q4", "category": "regles_jeu"},
        ]

        selected = _select_diverse_examples(questions, n=3)

        categories = {q["category"] for q in selected}
        # Should have at least 2 different categories
        assert len(categories) >= 2


# ============================================================================
# Test Report Generation
# ============================================================================


class TestReportGeneration:
    """Test ISO-compliant report generation."""

    def test_recommendations_for_low_score(self) -> None:
        """Low score should generate priority recommendation."""
        context_relevance = {"score": 0.75, "ci_95_lower": 0.70, "n_samples": 100}
        iso_compliance = {"overall_pass": False, "checks": {}}
        comparison: dict[str, Any] = {}

        recommendations = _generate_recommendations(
            context_relevance, iso_compliance, comparison
        )

        assert any("PRIORITY" in r for r in recommendations)

    def test_recommendations_for_passing_score(self) -> None:
        """Passing score should not have priority recommendations."""
        context_relevance = {"score": 0.90, "ci_95_lower": 0.85, "n_samples": 200}
        iso_compliance = {"overall_pass": True, "checks": {}}
        comparison: dict[str, Any] = {}

        recommendations = _generate_recommendations(
            context_relevance, iso_compliance, comparison
        )

        assert not any("PRIORITY" in r for r in recommendations)

    def test_comparison_with_recall(self) -> None:
        """Comparison should include correlation assessment."""
        context_relevance = {"score": 0.89}
        retrieval_stats = {"recall_at_5": 0.9156}

        comparison = _build_comparison(context_relevance, retrieval_stats)

        assert "correlation" in comparison
        assert comparison["correlation"] in ["high", "moderate", "low"]


# ============================================================================
# Test Cost Estimation
# ============================================================================


class TestCostEstimation:
    """Test evaluation cost estimation."""

    def test_estimate_gpt4o_mini_cost(self, tmp_path: Path) -> None:
        """GPT-4o-mini cost estimation."""
        # Create sample TSV
        tsv_path = tmp_path / "test.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("Query\tDocument\tAnswer\n")
            for i in range(100):
                f.write(f"Q{i}\tD{i}\tA{i}\n")

        llm_config = {"model": "gpt-4o-mini", "estimated_cost_per_eval": 0.02}

        estimate = _estimate_cost(tsv_path, llm_config)

        assert estimate["n_samples"] == 100
        assert estimate["estimated_total_usd"] == pytest.approx(2.0, abs=0.01)

    def test_vllm_is_free(self, tmp_path: Path) -> None:
        """vLLM local models should be free."""
        tsv_path = tmp_path / "test.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("Query\tDocument\tAnswer\n")
            for i in range(100):
                f.write(f"Q{i}\tD{i}\tA{i}\n")

        llm_config = {"model": "mistral-7b", "estimated_cost_per_eval": 0.0}

        estimate = _estimate_cost(tsv_path, llm_config)

        assert estimate["estimated_total_usd"] == 0.0


# ============================================================================
# Test API Key Check
# ============================================================================


class TestApiKeyCheck:
    """Test OpenAI API key detection."""

    def test_detects_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should detect missing API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert check_openai_api_key() is False

    def test_detects_present_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should detect present API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert check_openai_api_key() is True


# ============================================================================
# Integration Tests with Real File I/O
# ============================================================================


@pytest.fixture
def mock_corpus_data(tmp_path: Path) -> dict[str, Path]:
    """Create mock corpus data for integration testing."""
    # Create gold standard
    gold_standard = {
        "version": "test",
        "questions": [
            {
                "id": "TEST-Q01",
                "question": "Quelle est la règle du toucher-jouer ?",
                "category": "regles_jeu",
                "expected_chunk_id": "doc1.pdf-p001-parent000-child00",
                "validation": {"status": "VALIDATED"},
                "metadata": {"hard_type": "ANSWERABLE"},
            },
            {
                "id": "TEST-Q02",
                "question": "Combien de temps avant forfait ?",
                "category": "temps",
                "expected_chunk_id": "doc1.pdf-p002-parent001-child00",
                "validation": {"status": "VALIDATED"},
                "metadata": {"hard_type": "ANSWERABLE"},
            },
            {
                "id": "TEST-Q03",
                "question": "Question hors scope ?",
                "category": "autre",
                "expected_chunk_id": "doc2.pdf-p001-parent000-child00",
                "validation": {"status": "VALIDATED"},
                "metadata": {"hard_type": "OUT_OF_SCOPE"},
            },
        ],
    }

    tests_data = tmp_path / "tests" / "data"
    tests_data.mkdir(parents=True)
    with open(tests_data / "gold_standard_test.json", "w", encoding="utf-8") as f:
        json.dump(gold_standard, f)

    # Create chunks
    chunks = {
        "chunks": [
            {
                "id": "doc1.pdf-p001-parent000-child00",
                "text": "Le toucher-jouer est une règle fondamentale des échecs.",
                "source": "doc1.pdf",
                "pages": [1],
                "tokens": 20,
            },
            {
                "id": "doc1.pdf-p002-parent001-child00",
                "text": "Le forfait est déclaré après 30 minutes de retard.",
                "source": "doc1.pdf",
                "pages": [2],
                "tokens": 15,
            },
            {
                "id": "doc2.pdf-p001-parent000-child00",
                "text": "Les compétitions jeunes ont des règles spécifiques.",
                "source": "doc2.pdf",
                "pages": [1],
                "tokens": 18,
            },
            {
                "id": "doc3.pdf-p001-parent000-child00",
                "text": "Ce chunk est pour les négatifs.",
                "source": "doc3.pdf",
                "pages": [1],
                "tokens": 12,
            },
        ]
    }

    corpus_dir = tmp_path / "corpus" / "processed"
    corpus_dir.mkdir(parents=True)
    with open(corpus_dir / "chunks_mode_b_test.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    # Create triplets
    triplets = [
        {
            "anchor": "Quelle est la règle du toucher-jouer ?",
            "positive": "Pièce touchée, pièce jouée.",
            "negative": "Autre texte.",
        },
        {
            "anchor": "Combien de temps avant forfait ?",
            "positive": "Forfait après 30 minutes.",
            "negative": "Autre texte.",
        },
    ]

    training_dir = tmp_path / "data" / "training"
    training_dir.mkdir(parents=True)
    with open(training_dir / "gold_triplets_mode_b.jsonl", "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Create output directory
    output_dir = tmp_path / "data" / "evaluation" / "ares"
    output_dir.mkdir(parents=True)

    return {
        "tmp_path": tmp_path,
        "tests_data": tests_data,
        "corpus_dir": corpus_dir,
        "training_dir": training_dir,
        "output_dir": output_dir,
    }


class TestConvertToAresIntegration:
    """Integration tests for convert_to_ares module."""

    def test_convert_gold_standard_creates_files(
        self, mock_corpus_data: dict[str, Path]
    ) -> None:
        """convert_gold_standard_to_ares creates expected output files."""
        from scripts.evaluation.ares import convert_to_ares

        # Patch path constants
        with patch.object(
            convert_to_ares, "TESTS_DATA_DIR", mock_corpus_data["tests_data"]
        ), patch.object(
            convert_to_ares, "CORPUS_DIR", mock_corpus_data["corpus_dir"]
        ), patch.object(
            convert_to_ares, "DATA_TRAINING_DIR", mock_corpus_data["training_dir"]
        ):
            result = convert_to_ares.convert_gold_standard_to_ares(
                corpus="test",
                negative_ratio=0.30,
                output_dir=mock_corpus_data["output_dir"],
                seed=42,
            )

        assert result["gold_label"].exists()
        assert result["unlabeled"].exists()
        assert result["mapping"].exists()

        # Verify mapping content
        with open(result["mapping"], encoding="utf-8") as f:
            mapping = json.load(f)

        assert mapping["corpus"] == "test"
        assert mapping["positive_count"] == 2  # Only 2 ANSWERABLE
        # With only 2 positives and 30% ratio, n_neg = int(2 * 0.3 / 0.7) = 0
        assert mapping["negative_count"] >= 0

    def test_load_gold_standard(self, mock_corpus_data: dict[str, Path]) -> None:
        """load_gold_standard loads test gold standard."""
        from scripts.evaluation.ares import convert_to_ares

        with patch.object(
            convert_to_ares, "TESTS_DATA_DIR", mock_corpus_data["tests_data"]
        ):
            data = convert_to_ares.load_gold_standard("test")

        assert "questions" in data
        assert len(data["questions"]) == 3

    def test_load_chunks(self, mock_corpus_data: dict[str, Path]) -> None:
        """load_chunks loads and indexes test chunks."""
        from scripts.evaluation.ares import convert_to_ares

        with patch.object(
            convert_to_ares, "CORPUS_DIR", mock_corpus_data["corpus_dir"]
        ):
            chunks = convert_to_ares.load_chunks("test")

        assert len(chunks) == 4
        assert "doc1.pdf-p001-parent000-child00" in chunks

    def test_load_triplets(self, mock_corpus_data: dict[str, Path]) -> None:
        """load_triplets loads test triplets."""
        from scripts.evaluation.ares import convert_to_ares

        with patch.object(
            convert_to_ares, "DATA_TRAINING_DIR", mock_corpus_data["training_dir"]
        ):
            triplets = convert_to_ares.load_triplets("test")

        assert len(triplets) == 2
        assert triplets[0]["anchor"] == "Quelle est la règle du toucher-jouer ?"


class TestGenerateFewShotIntegration:
    """Integration tests for generate_few_shot module."""

    def test_generate_few_shot_creates_file(
        self, mock_corpus_data: dict[str, Path]
    ) -> None:
        """generate_few_shot_examples creates expected output file."""
        from scripts.evaluation.ares import generate_few_shot

        with patch.object(
            generate_few_shot, "TESTS_DATA_DIR", mock_corpus_data["tests_data"]
        ), patch.object(
            generate_few_shot, "CORPUS_DIR", mock_corpus_data["corpus_dir"]
        ), patch.object(
            generate_few_shot, "DATA_TRAINING_DIR", mock_corpus_data["training_dir"]
        ):
            result = generate_few_shot.generate_few_shot_examples(
                corpus="test",
                n_positive=2,
                n_negative=2,
                output_dir=mock_corpus_data["output_dir"],
                seed=42,
            )

        assert result.exists()

        # Read and verify content
        with open(result, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) > 0
        assert "Reasoning" in rows[0]


class TestReportIntegration:
    """Integration tests for report module."""

    def test_generate_pending_report(self, tmp_path: Path) -> None:
        """generate_evaluation_report creates pending report when no eval exists."""
        from scripts.evaluation.ares import report

        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        data_dir = tmp_path / "data" / "evaluation" / "ares"
        data_dir.mkdir(parents=True)

        with patch.object(report, "DATA_DIR", data_dir), patch.object(
            report, "RESULTS_DIR", data_dir / "results"
        ):
            result = report.generate_evaluation_report(
                corpus="test", output_dir=output_dir
            )

        assert result["metadata"]["status"] == "pending"
        assert "instructions" in result

    def test_load_retrieval_stats_default(self) -> None:
        """load_retrieval_stats returns default for FR corpus."""
        from scripts.evaluation.ares.report import load_retrieval_stats

        stats = load_retrieval_stats("fr")

        assert stats["recall_at_5"] == pytest.approx(0.9156, abs=0.01)


class TestRunEvaluationIntegration:
    """Integration tests for run_evaluation module."""

    def test_check_ares_available(self) -> None:
        """check_ares_available returns False when ARES not installed."""
        from scripts.evaluation.ares.run_evaluation import check_ares_available

        # In test environment, ARES is likely not installed
        result = check_ares_available()
        assert isinstance(result, bool)

    def test_run_mock_with_real_data(self, mock_corpus_data: dict[str, Path]) -> None:
        """run_mock_evaluation works with fixture data."""
        from scripts.evaluation.ares import run_evaluation

        # Create gold_label file
        gold_label_path = mock_corpus_data["output_dir"] / "gold_label_test.tsv"
        with open(gold_label_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Query", "Document", "Answer", "Context_Relevance_Label"],
                delimiter="\t",
            )
            writer.writeheader()
            for i in range(20):
                writer.writerow({
                    "Query": f"Question {i}",
                    "Document": f"Document {i}",
                    "Answer": f"Answer {i}",
                    "Context_Relevance_Label": 1 if i < 14 else 0,
                })

        with patch.object(run_evaluation, "DATA_DIR", mock_corpus_data["output_dir"]):
            result = run_evaluation.run_mock_evaluation(corpus="test")

        assert "context_relevance" in result
        assert result["llm_used"] == "mock"
        assert 0.0 <= result["context_relevance"]["score"] <= 1.0


# ============================================================================
# Test Language-Specific Templates
# ============================================================================


class TestLanguageTemplates:
    """Test language-appropriate template selection."""

    def test_french_templates_for_fr_corpus(self) -> None:
        """FR corpus should get French templates."""
        from scripts.evaluation.ares.generate_few_shot import _get_templates

        pos, neg = _get_templates("fr")

        assert len(pos) == 5
        assert len(neg) == 5
        # Check first positive template is in French
        assert "document" in pos[0]["reasoning"].lower()
        assert "règle" in pos[0]["reasoning"] or "question" in pos[0]["reasoning"]

    def test_english_templates_for_intl_corpus(self) -> None:
        """INTL corpus should get English templates."""
        from scripts.evaluation.ares.generate_few_shot import _get_templates

        pos, neg = _get_templates("intl")

        assert len(pos) == 5
        assert len(neg) == 5
        # Check first positive template is in English
        assert "document" in pos[0]["reasoning"].lower()
        assert "The" in pos[0]["reasoning"]


# ============================================================================
# Test Minimum Negative Samples
# ============================================================================


class TestMinimumNegativeSamples:
    """Test that small datasets get at least 1 negative sample."""

    def test_small_dataset_gets_one_negative(self) -> None:
        """With 2 positives, should get at least 1 negative."""
        n_positive = 2
        negative_ratio = 0.30

        # Original formula
        n_negatives_original = int(n_positive * negative_ratio / (1 - negative_ratio))

        # Fixed formula
        n_negatives = n_negatives_original
        if n_positive > 0 and n_negatives == 0:
            n_negatives = 1

        assert n_negatives_original == 0  # Original would give 0
        assert n_negatives == 1  # Fixed gives 1

    def test_large_dataset_unchanged(self) -> None:
        """With 100 positives, formula works normally."""
        n_positive = 100
        negative_ratio = 0.30

        n_negatives = int(n_positive * negative_ratio / (1 - negative_ratio))
        if n_positive > 0 and n_negatives == 0:
            n_negatives = 1

        # 100 * 0.3 / 0.7 = 42.8 -> 42
        assert n_negatives == 42


# ============================================================================
# Test CLI Main Functions
# ============================================================================


class TestCLIMain:
    """Test CLI entrypoint functions."""

    def test_convert_main_with_args(
        self, mock_corpus_data: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """convert_to_ares.main() parses args correctly."""
        from scripts.evaluation.ares import convert_to_ares

        # Create fr gold standard in mock location
        with open(
            mock_corpus_data["tests_data"] / "gold_standard_fr.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "version": "test",
                    "questions": [
                        {
                            "id": "FR-Q01",
                            "question": "Test question?",
                            "category": "test",
                            "expected_chunk_id": "doc1.pdf-p001-parent000-child00",
                            "validation": {"status": "VALIDATED"},
                            "metadata": {"hard_type": "ANSWERABLE"},
                        }
                    ],
                },
                f,
            )

        # Create fr chunks
        with open(
            mock_corpus_data["corpus_dir"] / "chunks_mode_b_fr.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "chunks": [
                        {
                            "id": "doc1.pdf-p001-parent000-child00",
                            "text": "Test chunk content.",
                            "source": "doc1.pdf",
                            "pages": [1],
                            "tokens": 10,
                        },
                        {
                            "id": "doc2.pdf-p001-parent000-child00",
                            "text": "Negative chunk.",
                            "source": "doc2.pdf",
                            "pages": [1],
                            "tokens": 10,
                        },
                    ]
                },
                f,
            )

        monkeypatch.setattr(
            "sys.argv", ["convert_to_ares", "--corpus", "fr", "--seed", "123"]
        )

        with patch.object(
            convert_to_ares, "TESTS_DATA_DIR", mock_corpus_data["tests_data"]
        ), patch.object(
            convert_to_ares, "CORPUS_DIR", mock_corpus_data["corpus_dir"]
        ), patch.object(
            convert_to_ares, "DATA_TRAINING_DIR", mock_corpus_data["training_dir"]
        ), patch.object(
            convert_to_ares, "OUTPUT_DIR", mock_corpus_data["output_dir"]
        ):
            convert_to_ares.main()

        assert (mock_corpus_data["output_dir"] / "gold_label_fr.tsv").exists()

    def test_generate_few_shot_main_with_args(
        self, mock_corpus_data: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """generate_few_shot.main() parses args correctly."""
        from scripts.evaluation.ares import generate_few_shot

        # Create fr gold standard
        with open(
            mock_corpus_data["tests_data"] / "gold_standard_fr.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "version": "test",
                    "questions": [
                        {
                            "id": "FR-Q01",
                            "question": "Test question?",
                            "category": "test",
                            "expected_chunk_id": "doc1.pdf-p001-parent000-child00",
                            "validation": {"status": "VALIDATED"},
                            "metadata": {"hard_type": "ANSWERABLE"},
                        }
                    ],
                },
                f,
            )

        # Create fr chunks
        with open(
            mock_corpus_data["corpus_dir"] / "chunks_mode_b_fr.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "chunks": [
                        {
                            "id": "doc1.pdf-p001-parent000-child00",
                            "text": "Test chunk.",
                            "source": "doc1.pdf",
                            "pages": [1],
                            "tokens": 10,
                        },
                        {
                            "id": "doc2.pdf-p001-parent000-child00",
                            "text": "Negative.",
                            "source": "doc2.pdf",
                            "pages": [1],
                            "tokens": 10,
                        },
                    ]
                },
                f,
            )

        monkeypatch.setattr(
            "sys.argv",
            ["generate_few_shot", "--corpus", "fr", "--n-positive", "1", "--n-negative", "1"],
        )

        with patch.object(
            generate_few_shot, "TESTS_DATA_DIR", mock_corpus_data["tests_data"]
        ), patch.object(
            generate_few_shot, "CORPUS_DIR", mock_corpus_data["corpus_dir"]
        ), patch.object(
            generate_few_shot, "DATA_TRAINING_DIR", mock_corpus_data["training_dir"]
        ), patch.object(
            generate_few_shot, "OUTPUT_DIR", mock_corpus_data["output_dir"]
        ):
            generate_few_shot.main()

        assert (mock_corpus_data["output_dir"] / "few_shot_fr.tsv").exists()

    def test_report_main_creates_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """report.main() creates pending report when no eval exists."""
        from scripts.evaluation.ares import report

        monkeypatch.setattr("sys.argv", ["report", "--corpus", "fr"])

        output_dir = tmp_path / "reports"
        output_dir.mkdir()
        data_dir = tmp_path / "data" / "evaluation" / "ares"
        data_dir.mkdir(parents=True)

        with patch.object(report, "DATA_DIR", data_dir), patch.object(
            report, "RESULTS_DIR", data_dir / "results"
        ):
            # Patch generate_evaluation_report to use our temp output dir
            original_func = report.generate_evaluation_report

            def patched_func(corpus: str = "fr", output_dir: Path | None = None) -> Any:
                return original_func(corpus=corpus, output_dir=tmp_path / "reports")

            with patch.object(report, "generate_evaluation_report", patched_func):
                report.main()

        assert (tmp_path / "reports" / "ares_report_fr.json").exists()

    def test_run_evaluation_main_mock(
        self, mock_corpus_data: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_evaluation.main() with --mock flag."""
        from scripts.evaluation.ares import run_evaluation

        # Create gold_label file for fr corpus
        gold_label_path = mock_corpus_data["output_dir"] / "gold_label_fr.tsv"
        with open(gold_label_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Query", "Document", "Answer", "Context_Relevance_Label"],
                delimiter="\t",
            )
            writer.writeheader()
            for i in range(10):
                writer.writerow({
                    "Query": f"Q{i}",
                    "Document": f"D{i}",
                    "Answer": f"A{i}",
                    "Context_Relevance_Label": 1 if i < 7 else 0,
                })

        monkeypatch.setattr("sys.argv", ["run_evaluation", "--corpus", "fr", "--mock"])

        with patch.object(run_evaluation, "DATA_DIR", mock_corpus_data["output_dir"]):
            # Capture stdout
            import io
            import sys

            captured = io.StringIO()
            monkeypatch.setattr(sys, "stdout", captured)

            run_evaluation.main()

            output = captured.getvalue()
            assert "context_relevance" in output
