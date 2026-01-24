"""Tests for generate_hard_negatives.py (Step 3)."""

import pytest
import numpy as np

from scripts.training.unified.generate_hard_negatives import (
    build_chunk_indices,
    cosine_similarity,
    select_random,
    select_same_doc_diff_page,
    validate_quality_gates,
    TARGET_DISTRIBUTION,
)


class TestCosimeSimilarity:
    """Tests for cosine_similarity."""

    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0


class TestBuildChunkIndices:
    """Tests for build_chunk_indices."""

    @pytest.fixture
    def sample_chunks(self) -> list[dict]:
        return [
            {"id": "c1", "source": "doc1.pdf", "section": "Article 1"},
            {"id": "c2", "source": "doc1.pdf", "section": "Competition rules"},
            {"id": "c3", "source": "doc2.pdf", "section": "Interclub rules"},
        ]

    def test_groups_by_source(self, sample_chunks: list[dict]) -> None:
        by_source, _, _ = build_chunk_indices(sample_chunks)
        assert len(by_source["doc1.pdf"]) == 2
        assert len(by_source["doc2.pdf"]) == 1

    def test_groups_by_category(self, sample_chunks: list[dict]) -> None:
        _, by_category, _ = build_chunk_indices(sample_chunks)
        assert "regles_jeu" in by_category  # "Article" keyword
        assert "competitions" in by_category  # "Competition" keyword
        assert "interclubs" in by_category  # "Interclub" keyword

    def test_all_ids_returned(self, sample_chunks: list[dict]) -> None:
        _, _, all_ids = build_chunk_indices(sample_chunks)
        assert len(all_ids) == 3
        assert "c1" in all_ids


class TestSelectSameDocDiffPage:
    """Tests for select_same_doc_diff_page."""

    @pytest.fixture
    def by_source(self) -> dict[str, list[dict]]:
        return {
            "doc.pdf": [
                {"id": "c1", "pages": [1, 2]},
                {"id": "c2", "pages": [3, 4]},
                {"id": "c3", "pages": [5, 6]},
            ]
        }

    def test_selects_different_page(self, by_source: dict) -> None:
        positive = {"id": "c1", "source": "doc.pdf", "pages": [1, 2]}
        result = select_same_doc_diff_page(positive, by_source)
        assert result is not None
        assert result["id"] != "c1"
        assert not set(result["pages"]).intersection(set(positive["pages"]))

    def test_returns_none_when_only_chunk(self) -> None:
        by_source = {"doc.pdf": [{"id": "c1", "pages": [1]}]}
        positive = {"id": "c1", "source": "doc.pdf", "pages": [1]}
        result = select_same_doc_diff_page(positive, by_source)
        assert result is None


class TestSelectRandom:
    """Tests for select_random."""

    def test_excludes_positive(self) -> None:
        all_ids = ["c1", "c2", "c3"]
        for _ in range(10):  # Multiple trials
            result = select_random("c1", all_ids)
            assert result != "c1"
            assert result in ["c2", "c3"]

    def test_returns_none_when_single(self) -> None:
        result = select_random("c1", ["c1"])
        assert result is None


class TestValidateQualityGates:
    """Tests for validate_quality_gates."""

    def test_detects_duplicate_negatives(self) -> None:
        triplets = [
            {"metadata": {"question_id": "q1", "negative_chunk_id": "n1"}},
            {"metadata": {"question_id": "q2", "negative_chunk_id": "n1"}},  # Duplicate
            {"metadata": {"question_id": "q3", "negative_chunk_id": "n2"}},
        ]
        result = validate_quality_gates(triplets, None, [], {})
        assert result["gate_3_duplicate_negatives"]["count"] == 1

    def test_no_duplicates_passes(self) -> None:
        triplets = [
            {"metadata": {"question_id": "q1", "negative_chunk_id": "n1"}},
            {"metadata": {"question_id": "q2", "negative_chunk_id": "n2"}},
        ]
        result = validate_quality_gates(triplets, None, [], {})
        assert result["gate_3_duplicate_negatives"]["passed"] is True


class TestTargetDistribution:
    """Tests for TARGET_DISTRIBUTION."""

    def test_sums_to_one(self) -> None:
        total = sum(TARGET_DISTRIBUTION.values())
        assert total == pytest.approx(1.0)

    def test_same_doc_diff_page_is_40_percent(self) -> None:
        assert TARGET_DISTRIBUTION["same_doc_diff_page"] == 0.40
