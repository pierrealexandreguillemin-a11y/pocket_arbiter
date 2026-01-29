"""Tests for validate_dataset.py (Step 5)."""

import pytest

from scripts.training.unified.validate_dataset import (
    DEDUP_THRESHOLD,
    ENTROPY_THRESHOLD,
    check_exact_duplicates,
    compute_entropy,
    compute_text_hash,
    validate_schema,
)


class TestComputeTextHash:
    """Tests for compute_text_hash."""

    def test_deterministic(self) -> None:
        text = "Hello World"
        h1 = compute_text_hash(text)
        h2 = compute_text_hash(text)
        assert h1 == h2

    def test_different_text_different_hash(self) -> None:
        h1 = compute_text_hash("Text A")
        h2 = compute_text_hash("Text B")
        assert h1 != h2


class TestValidateSchema:
    """Tests for validate_schema."""

    def test_valid_triplet(self) -> None:
        triplets = [
            {
                "anchor": "This is a question about rules",
                "positive": "This is the positive chunk with enough text",
                "negative": "This is the negative chunk with enough text",
            }
        ]
        result = validate_schema(triplets)
        assert result["passed"] is True
        assert result["valid_count"] == 1

    def test_missing_required_field(self) -> None:
        triplets = [
            {
                "anchor": "This is a question",
                "positive": "This is the positive chunk",
                # Missing "negative"
            }
        ]
        result = validate_schema(triplets)
        assert result["passed"] is False
        assert result["error_count"] == 1

    def test_field_too_short(self) -> None:
        triplets = [
            {
                "anchor": "Short",  # Less than 10 chars
                "positive": "This is the positive chunk with enough text",
                "negative": "This is the negative chunk with enough text",
            }
        ]
        result = validate_schema(triplets)
        assert result["passed"] is False

    def test_multiple_triplets(self) -> None:
        triplets = [
            {
                "anchor": f"Question number {i} about chess rules",
                "positive": f"Positive chunk {i} with enough content",
                "negative": f"Negative chunk {i} with enough content",
            }
            for i in range(5)
        ]
        result = validate_schema(triplets)
        assert result["passed"] is True
        assert result["valid_count"] == 5


class TestCheckExactDuplicates:
    """Tests for check_exact_duplicates."""

    def test_no_duplicates(self) -> None:
        triplets = [
            {"anchor": "Question 1", "positive": "Positive 1"},
            {"anchor": "Question 2", "positive": "Positive 2"},
        ]
        result = check_exact_duplicates(triplets)
        assert result["anchor_duplicates"]["count"] == 0
        assert result["positive_duplicates"]["count"] == 0

    def test_anchor_duplicates(self) -> None:
        triplets = [
            {"anchor": "Same question", "positive": "Positive 1"},
            {"anchor": "Same question", "positive": "Positive 2"},
            {"anchor": "Different question", "positive": "Positive 3"},
        ]
        result = check_exact_duplicates(triplets)
        assert result["anchor_duplicates"]["count"] == 1  # 1 duplicate (2 - 1)

    def test_positive_duplicates(self) -> None:
        triplets = [
            {"anchor": "Question 1", "positive": "Same positive"},
            {"anchor": "Question 2", "positive": "Same positive"},
            {"anchor": "Question 3", "positive": "Same positive"},
        ]
        result = check_exact_duplicates(triplets)
        assert result["positive_duplicates"]["count"] == 2  # 2 duplicates (3 - 1)


class TestComputeEntropy:
    """Tests for compute_entropy."""

    def test_uniform_distribution(self) -> None:
        dist = {"a": 10, "b": 10, "c": 10, "d": 10}
        entropy = compute_entropy(dist)
        assert entropy == pytest.approx(1.0)  # Max entropy

    def test_single_class(self) -> None:
        dist = {"a": 100}
        entropy = compute_entropy(dist)
        assert entropy == 1.0  # Trivially uniform

    def test_skewed_distribution(self) -> None:
        dist = {"a": 99, "b": 1}
        entropy = compute_entropy(dist)
        assert entropy < 0.5  # Low entropy

    def test_empty_distribution(self) -> None:
        dist: dict[str, int] = {}
        entropy = compute_entropy(dist)
        assert entropy == 0.0


class TestThresholds:
    """Tests for quality gate thresholds."""

    def test_dedup_threshold(self) -> None:
        assert DEDUP_THRESHOLD == 0.05  # 5%

    def test_entropy_threshold(self) -> None:
        assert ENTROPY_THRESHOLD == 0.8  # 80% normalized entropy
