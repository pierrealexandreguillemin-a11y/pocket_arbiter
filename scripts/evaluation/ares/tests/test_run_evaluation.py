"""
Tests for run_evaluation module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO 42001 - AI management system
"""

from scripts.evaluation.ares.run_evaluation import (
    _ppi_mean_ci,
    run_mock_evaluation,
)


class TestPPIMeanCI:
    """Tests for PPI Mean Estimation confidence interval (ARES-verbatim)."""

    def test_perfect_predictions(self) -> None:
        """When predictions match labels exactly, r_hat = 0."""
        Y_labeled = [1, 1, 0, 0, 1]
        Yhat_labeled = [1, 1, 0, 0, 1]  # Perfect match
        Yhat_unlabeled = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        # Estimate should equal mean of unlabeled predictions (6/10 = 0.6)
        assert abs(estimate - 0.6) < 0.01

    def test_biased_predictor_correction(self) -> None:
        """PPI should correct for systematic bias in predictions."""
        Y_labeled = [1, 1, 1, 1, 1]  # All positive
        Yhat_labeled = [0, 0, 0, 0, 0]  # Predictor says all negative (biased)
        Yhat_unlabeled = [0] * 10  # Predictor says all negative

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        # r_hat = mean(Yhat - Y) = mean(-1, -1, -1, -1, -1) = -1
        # theta_f = 0 (all unlabeled predictions are 0)
        # theta_pp = theta_f - r_hat = 0 - (-1) = 1.0
        assert abs(estimate - 1.0) < 0.01

    def test_empty_labeled_returns_zero(self) -> None:
        """Empty labeled set should return zeros."""
        estimate, ci_lower, ci_upper = _ppi_mean_ci([], [], [1, 0, 1])
        assert estimate == 0.0
        assert ci_lower == 0.0
        assert ci_upper == 0.0

    def test_empty_unlabeled_returns_zero(self) -> None:
        """Empty unlabeled set should return zeros."""
        estimate, ci_lower, ci_upper = _ppi_mean_ci([1, 0], [1, 0], [])
        assert estimate == 0.0
        assert ci_lower == 0.0
        assert ci_upper == 0.0

    def test_all_ones(self) -> None:
        """All positive predictions and labels."""
        Y_labeled = [1, 1, 1, 1, 1]
        Yhat_labeled = [1, 1, 1, 1, 1]
        Yhat_unlabeled = [1] * 20

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        assert estimate == 1.0
        assert ci_upper == 1.0  # Capped at 1.0

    def test_all_zeros(self) -> None:
        """All negative predictions and labels."""
        Y_labeled = [0, 0, 0, 0, 0]
        Yhat_labeled = [0, 0, 0, 0, 0]
        Yhat_unlabeled = [0] * 20

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        assert estimate == 0.0
        assert ci_lower == 0.0  # Capped at 0.0

    def test_ci_width_decreases_with_n(self) -> None:
        """Confidence interval should be narrower with more samples."""
        Y_small = [1, 0, 1, 0, 1]
        Yhat_small = [1, 0, 1, 0, 1]
        Yhat_unlabeled_small = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

        _, ci_lower_small, ci_upper_small = _ppi_mean_ci(
            Y_small, Yhat_small, Yhat_unlabeled_small
        )
        width_small = ci_upper_small - ci_lower_small

        Y_large = [1, 0] * 50
        Yhat_large = [1, 0] * 50
        Yhat_unlabeled_large = [1, 0] * 100

        _, ci_lower_large, ci_upper_large = _ppi_mean_ci(
            Y_large, Yhat_large, Yhat_unlabeled_large
        )
        width_large = ci_upper_large - ci_lower_large

        assert width_large < width_small

    def test_ci_bounds_valid(self) -> None:
        """CI bounds should always be in [0, 1]."""
        # Edge case: predictions could give estimate outside [0, 1]
        Y_labeled = [0, 0, 0, 0, 0]
        Yhat_labeled = [1, 1, 1, 1, 1]  # Over-predicts
        Yhat_unlabeled = [1] * 10

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        assert 0.0 <= ci_lower <= 1.0
        assert 0.0 <= ci_upper <= 1.0
        assert ci_lower <= ci_upper

    def test_known_value_verification(self) -> None:
        """Verify against hand-calculated PPI values."""
        # Simple case: 50% positive rate, perfect predictor
        Y_labeled = [1, 0, 1, 0]
        Yhat_labeled = [1, 0, 1, 0]
        Yhat_unlabeled = [1, 0, 1, 0, 1, 0, 1, 0]

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        # theta_f = 0.5, r_hat = 0, theta_pp = 0.5
        assert abs(estimate - 0.5) < 0.01


class TestRunMockEvaluation:
    """Tests for mock evaluation function."""

    def test_mock_returns_valid_structure(self) -> None:
        """Mock evaluation should return valid result structure."""
        result = run_mock_evaluation(corpus="fr")

        assert "corpus" in result
        assert "llm_used" in result
        assert "timestamp" in result
        assert "context_relevance" in result
        assert result["llm_used"] == "mock"

    def test_mock_context_relevance_structure(self) -> None:
        """Context relevance should have required fields."""
        result = run_mock_evaluation(corpus="fr")

        cr = result["context_relevance"]
        assert "score" in cr
        assert "ci_95_lower" in cr
        assert "ci_95_upper" in cr
        assert "n_samples" in cr
        assert "pass" in cr

    def test_mock_score_in_valid_range(self) -> None:
        """Mock score should be in [0, 1]."""
        result = run_mock_evaluation(corpus="fr")

        score = result["context_relevance"]["score"]
        assert 0.0 <= score <= 1.0

    def test_mock_ci_bounds_valid(self) -> None:
        """Mock CI bounds should be valid."""
        result = run_mock_evaluation(corpus="fr")

        cr = result["context_relevance"]
        assert cr["ci_95_lower"] <= cr["score"] <= cr["ci_95_upper"]
        assert 0.0 <= cr["ci_95_lower"]
        assert cr["ci_95_upper"] <= 1.0
