"""
Tests for balance_distribution module (Phase 5: Deduplication & Balancing).

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO 29119-3 - Test data balance
    - ISO 25010 - Data quality metrics
"""

from unittest.mock import patch

import numpy as np
import pytest

from scripts.evaluation.annales.balance_distribution import (
    DistributionStats,
    balance_distribution,
    compute_distribution_stats,
    cosine_similarity,
    cosine_similarity_matrix,
    validate_distribution,
)


class TestDistributionStats:
    """Tests for DistributionStats dataclass."""

    def test_unanswerable_ratio(self) -> None:
        """Should compute unanswerable ratio correctly."""
        stats = DistributionStats(
            total=100,
            answerable=70,
            unanswerable=30,
            fact_single=40,
            summary=20,
            reasoning=10,
            arithmetic=0,
            hard=15,
        )
        assert stats.unanswerable_ratio == 0.30

    def test_fact_single_ratio(self) -> None:
        """Should compute fact_single ratio over answerable only."""
        stats = DistributionStats(
            total=100,
            answerable=70,
            unanswerable=30,
            fact_single=35,  # 35/70 = 50%
            summary=20,
            reasoning=10,
            arithmetic=5,
            hard=15,
        )
        assert stats.fact_single_ratio == 0.50

    def test_hard_ratio(self) -> None:
        """Should compute hard ratio over total."""
        stats = DistributionStats(
            total=100,
            answerable=70,
            unanswerable=30,
            fact_single=40,
            summary=20,
            reasoning=10,
            arithmetic=0,
            hard=20,
        )
        assert stats.hard_ratio == 0.20

    def test_zero_division_protection(self) -> None:
        """Should handle zero division gracefully."""
        stats = DistributionStats(
            total=0,
            answerable=0,
            unanswerable=0,
            fact_single=0,
            summary=0,
            reasoning=0,
            arithmetic=0,
            hard=0,
        )
        assert stats.unanswerable_ratio == 0
        assert stats.fact_single_ratio == 0
        assert stats.hard_ratio == 0


class TestComputeDistributionStats:
    """Tests for computing distribution statistics."""

    def test_counts_unanswerable(self) -> None:
        """Should count unanswerable questions correctly."""
        questions = [
            {"content": {"is_impossible": False}, "classification": {}},
            {"content": {"is_impossible": True}, "classification": {}},
            {"content": {"is_impossible": True}, "classification": {}},
        ]
        stats = compute_distribution_stats(questions)
        assert stats.total == 3
        assert stats.unanswerable == 2
        assert stats.answerable == 1

    def test_counts_reasoning_classes(self) -> None:
        """Should count reasoning classes for answerable only."""
        questions = [
            {
                "content": {"is_impossible": False},
                "classification": {"reasoning_class": "fact_single"},
            },
            {
                "content": {"is_impossible": False},
                "classification": {"reasoning_class": "summary"},
            },
            {
                "content": {"is_impossible": True},  # Should not be counted
                "classification": {"reasoning_class": "fact_single"},
            },
        ]
        stats = compute_distribution_stats(questions)
        assert stats.fact_single == 1
        assert stats.summary == 1

    def test_counts_hard_questions(self) -> None:
        """Should count hard questions (difficulty >= 0.7)."""
        questions = [
            {
                "content": {"is_impossible": False},
                "classification": {"difficulty": 0.8},
            },
            {
                "content": {"is_impossible": False},
                "classification": {"difficulty": 0.3},
            },
            {
                "content": {"is_impossible": False},
                "classification": {"difficulty": 0.7},  # Exactly 0.7 is hard
            },
        ]
        stats = compute_distribution_stats(questions)
        assert stats.hard == 2


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Should return 1.0 for identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, rel=0.001)

    def test_orthogonal_vectors(self) -> None:
        """Should return 0.0 for orthogonal vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        sim = cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(0.0, rel=0.001)


class TestCosineSimilarityMatrix:
    """Tests for pairwise similarity matrix."""

    def test_diagonal_is_one(self) -> None:
        """Diagonal should be 1.0 (self-similarity)."""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        matrix = cosine_similarity_matrix(embeddings)
        for i in range(3):
            assert matrix[i, i] == pytest.approx(1.0, rel=0.001)

    def test_symmetric(self) -> None:
        """Matrix should be symmetric."""
        embeddings = np.random.rand(5, 10)
        matrix = cosine_similarity_matrix(embeddings)
        assert np.allclose(matrix, matrix.T)


class TestBalanceDistribution:
    """Tests for distribution balancing."""

    def test_adds_priority_boost(self) -> None:
        """Should add priority boost to under-represented classes."""
        questions = [
            {
                "id": "q1",
                "classification": {"reasoning_class": "reasoning"},
                "processing": {},
            },
            {
                "id": "q2",
                "classification": {"reasoning_class": "fact_single"},
                "processing": {},
            },
        ]
        result = balance_distribution(questions, {})

        # Reasoning should get positive boost
        assert result[0]["processing"]["priority_boost"] == 0.1
        # fact_single should get negative boost
        assert result[1]["processing"]["priority_boost"] == -0.05

    def test_creates_processing_if_missing(self) -> None:
        """Should create processing dict if missing."""
        questions = [{"id": "q1", "classification": {"reasoning_class": "summary"}}]
        result = balance_distribution(questions, {})
        assert "processing" in result[0]


class TestValidateDistribution:
    """Tests for distribution validation against targets."""

    def test_passes_valid_distribution(self) -> None:
        """Should pass when distribution meets targets."""
        stats = DistributionStats(
            total=100,
            answerable=70,
            unanswerable=30,  # 30% - in [25%, 33%]
            fact_single=35,  # 50% of answerable - < 60%
            summary=20,
            reasoning=15,
            arithmetic=0,
            hard=15,  # 15% - >= 10%
        )
        targets = {
            "fact_single": (0.0, 0.60),
            "unanswerable": (0.25, 0.33),
            "hard": (0.10, 1.0),
        }
        passed, errors = validate_distribution(stats, targets)
        assert passed
        assert len(errors) == 0

    def test_fails_high_fact_single(self) -> None:
        """Should fail G5-3 when fact_single >= 60%."""
        stats = DistributionStats(
            total=100,
            answerable=70,
            unanswerable=30,
            fact_single=50,  # 71% of answerable - >= 60%
            summary=10,
            reasoning=10,
            arithmetic=0,
            hard=15,
        )
        targets = {"fact_single": (0.0, 0.60)}
        passed, errors = validate_distribution(stats, targets)
        assert not passed
        assert any("G5-3" in e for e in errors)

    def test_fails_low_hard_ratio(self) -> None:
        """Should fail G5-4 when hard ratio < 10%."""
        stats = DistributionStats(
            total=100,
            answerable=70,
            unanswerable=30,
            fact_single=35,
            summary=20,
            reasoning=15,
            arithmetic=0,
            hard=5,  # 5% - < 10%
        )
        targets = {"hard": (0.10, 1.0)}
        passed, errors = validate_distribution(stats, targets)
        assert not passed
        assert any("G5-4" in e for e in errors)

    def test_fails_bad_unanswerable_ratio(self) -> None:
        """Should fail G5-5 when unanswerable outside [25%, 33%]."""
        # Too low
        stats_low = DistributionStats(
            total=100,
            answerable=85,
            unanswerable=15,  # 15% - < 25%
            fact_single=40,
            summary=25,
            reasoning=20,
            arithmetic=0,
            hard=20,
        )
        targets = {"unanswerable": (0.25, 0.33)}
        passed, errors = validate_distribution(stats_low, targets)
        assert not passed
        assert any("G5-5" in e for e in errors)

        # Too high
        stats_high = DistributionStats(
            total=100,
            answerable=60,
            unanswerable=40,  # 40% - > 33%
            fact_single=30,
            summary=15,
            reasoning=15,
            arithmetic=0,
            hard=20,
        )
        passed, errors = validate_distribution(stats_high, targets)
        assert not passed
        assert any("G5-5" in e for e in errors)


class TestDeduplicationIntegration:
    """Integration tests for deduplication (mocked embeddings)."""

    @pytest.fixture
    def mock_embeddings(self):
        """Mock the embedding functions."""
        with patch(
            "scripts.evaluation.annales.balance_distribution.compute_embeddings_batch"
        ) as mock:
            # Return different embeddings for each question
            def side_effect(texts, batch_size=32):
                n = len(texts)
                # Return orthogonal embeddings (no duplicates)
                embeddings = np.eye(n, min(n, 10))
                if n > 10:
                    embeddings = np.random.rand(n, 10)
                return embeddings

            mock.side_effect = side_effect
            yield mock

    def test_unique_questions_preserved(self, mock_embeddings) -> None:
        """Should preserve all questions when no duplicates."""
        from scripts.evaluation.annales.balance_distribution import (
            deduplicate_questions,
        )

        questions = [
            {"id": f"q{i}", "content": {"question": f"Question {i}?"}} for i in range(5)
        ]
        result = deduplicate_questions(questions, threshold=0.95)
        assert len(result.unique_ids) == 5
        assert len(result.removed_ids) == 0
