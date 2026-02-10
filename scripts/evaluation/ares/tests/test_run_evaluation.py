"""Tests for run_evaluation module (3 ARES metrics).

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO 42001 - AI management system
"""

from scripts.evaluation.ares.run_evaluation import (
    ARES_LABEL_COLUMNS,
    ARES_METRICS,
    ARES_SYSTEM_PROMPTS,
    _build_user_prompt,
    _parse_yes_no,
    _ppi_mean_ci,
    run_all_metrics,
    run_mock_evaluation,
)


class TestAresMetricsConstants:
    """Tests for ARES metric constants."""

    def test_three_metrics_defined(self) -> None:
        """ARES should define exactly 3 metrics."""
        assert len(ARES_METRICS) == 3
        assert "context_relevance" in ARES_METRICS
        assert "answer_faithfulness" in ARES_METRICS
        assert "answer_relevance" in ARES_METRICS

    def test_system_prompts_for_all_metrics(self) -> None:
        """Each metric should have a system prompt."""
        for m in ARES_METRICS:
            assert m in ARES_SYSTEM_PROMPTS
            assert len(ARES_SYSTEM_PROMPTS[m]) > 50
            assert "[[Yes]]" in ARES_SYSTEM_PROMPTS[m]
            assert "[[No]]" in ARES_SYSTEM_PROMPTS[m]

    def test_label_columns_for_all_metrics(self) -> None:
        """Each metric should have a label column name."""
        for m in ARES_METRICS:
            assert m in ARES_LABEL_COLUMNS
            assert ARES_LABEL_COLUMNS[m].endswith("_Label")

    def test_context_relevance_prompt_mentions_document(self) -> None:
        """Context relevance prompt should mention document relevance."""
        prompt = ARES_SYSTEM_PROMPTS["context_relevance"]
        assert "document" in prompt.lower()
        assert "relevant" in prompt.lower()

    def test_answer_faithfulness_prompt_mentions_faithful(self) -> None:
        """Answer faithfulness prompt should mention faithfulness."""
        prompt = ARES_SYSTEM_PROMPTS["answer_faithfulness"]
        assert "faithful" in prompt.lower()

    def test_answer_relevance_prompt_mentions_relevant(self) -> None:
        """Answer relevance prompt should mention relevance."""
        prompt = ARES_SYSTEM_PROMPTS["answer_relevance"]
        assert "relevant" in prompt.lower()
        assert "question" in prompt.lower()


class TestParseYesNo:
    """Tests for [[Yes]]/[[No]] response parsing."""

    def test_parse_yes(self) -> None:
        """[[Yes]] should return 1."""
        assert _parse_yes_no("[[Yes]]") == 1

    def test_parse_no(self) -> None:
        """[[No]] should return 0."""
        assert _parse_yes_no("[[No]]") == 0

    def test_parse_yes_with_text(self) -> None:
        """[[Yes]] with surrounding text."""
        assert _parse_yes_no("Based on analysis, [[Yes]]") == 1

    def test_parse_no_with_text(self) -> None:
        """[[No]] with surrounding text."""
        assert _parse_yes_no("The document is not relevant, [[No]]") == 0

    def test_parse_case_insensitive(self) -> None:
        """Parsing should be case insensitive."""
        assert _parse_yes_no("[[yes]]") == 1
        assert _parse_yes_no("[[NO]]") == 0

    def test_both_present_returns_no(self) -> None:
        """If both Yes and No present, No wins."""
        assert _parse_yes_no("[[Yes]] but actually [[No]]") == 0

    def test_fallback_yes(self) -> None:
        """Fallback to 'yes' in text."""
        assert _parse_yes_no("yes it is relevant") == 1

    def test_fallback_yes_mid_text(self) -> None:
        """Fallback finds 'yes' beyond first 20 chars."""
        assert _parse_yes_no("Based on my analysis, yes it is relevant") == 1

    def test_fallback_no(self) -> None:
        """Fallback to no-yes = 0."""
        assert _parse_yes_no("no it is not") == 0

    def test_fallback_empty(self) -> None:
        """Empty response returns 0."""
        assert _parse_yes_no("") == 0


class TestBuildUserPrompt:
    """Tests for user prompt building per metric."""

    def test_context_relevance_includes_question_and_document(self) -> None:
        """Context relevance prompt has question and document."""
        prompt = _build_user_prompt(
            "context_relevance", "What is castling?", "Castling is...", "answer"
        )
        assert "Question:" in prompt
        assert "Document:" in prompt

    def test_answer_faithfulness_includes_document_and_answer(self) -> None:
        """Answer faithfulness prompt has document and answer."""
        prompt = _build_user_prompt(
            "answer_faithfulness", "query", "Document text", "Answer text"
        )
        assert "Document:" in prompt
        assert "Answer:" in prompt

    def test_answer_relevance_includes_question_and_answer(self) -> None:
        """Answer relevance prompt has question and answer."""
        prompt = _build_user_prompt(
            "answer_relevance", "What is castling?", "doc", "Answer text"
        )
        assert "Question:" in prompt
        assert "Answer:" in prompt


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

        assert abs(estimate - 0.6) < 0.01

    def test_biased_predictor_correction(self) -> None:
        """PPI should correct for systematic bias in predictions."""
        Y_labeled = [1, 1, 1, 1, 1]
        Yhat_labeled = [0, 0, 0, 0, 0]
        Yhat_unlabeled = [0] * 10

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        assert abs(estimate - 1.0) < 0.01

    def test_empty_labeled_returns_zero(self) -> None:
        """Empty labeled set should return zeros."""
        estimate, ci_lower, ci_upper = _ppi_mean_ci([], [], [1, 0, 1])
        assert estimate == 0.0

    def test_empty_unlabeled_returns_zero(self) -> None:
        """Empty unlabeled set should return zeros."""
        estimate, ci_lower, ci_upper = _ppi_mean_ci([1, 0], [1, 0], [])
        assert estimate == 0.0

    def test_all_ones(self) -> None:
        """All positive predictions and labels."""
        Y_labeled = [1, 1, 1, 1, 1]
        Yhat_labeled = [1, 1, 1, 1, 1]
        Yhat_unlabeled = [1] * 20

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        assert estimate == 1.0
        assert ci_upper == 1.0

    def test_all_zeros(self) -> None:
        """All negative predictions and labels."""
        Y_labeled = [0, 0, 0, 0, 0]
        Yhat_labeled = [0, 0, 0, 0, 0]
        Yhat_unlabeled = [0] * 20

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        assert estimate == 0.0
        assert ci_lower == 0.0

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
        Y_labeled = [0, 0, 0, 0, 0]
        Yhat_labeled = [1, 1, 1, 1, 1]
        Yhat_unlabeled = [1] * 10

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

        assert 0.0 <= ci_lower <= 1.0
        assert 0.0 <= ci_upper <= 1.0
        assert ci_lower <= ci_upper

    def test_known_value_verification(self) -> None:
        """Verify against hand-calculated PPI values."""
        Y_labeled = [1, 0, 1, 0]
        Yhat_labeled = [1, 0, 1, 0]
        Yhat_unlabeled = [1, 0, 1, 0, 1, 0, 1, 0]

        estimate, ci_lower, ci_upper = _ppi_mean_ci(
            Y_labeled, Yhat_labeled, Yhat_unlabeled
        )

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
        assert cr["ci_95_lower"] >= 0.0
        assert cr["ci_95_upper"] <= 1.0

    def test_mock_with_metric_parameter(self) -> None:
        """Mock evaluation accepts metric parameter."""
        result = run_mock_evaluation(corpus="fr", metric="answer_faithfulness")

        assert result["metric"] == "answer_faithfulness"
        assert "answer_faithfulness" in result

    def test_mock_all_metrics(self) -> None:
        """Mock evaluation works for all 3 metrics."""
        for m in ARES_METRICS:
            result = run_mock_evaluation(corpus="fr", metric=m)
            assert m in result
            assert "score" in result[m]

    def test_mock_unknown_metric_raises_valueerror(self) -> None:
        """Unknown metric should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown metric"):
            run_mock_evaluation(corpus="fr", metric="nonexistent_metric")

    def test_mock_n_samples_matches_gold_label_rows(self) -> None:
        """Mock n_samples must reflect actual gold label file row count."""
        result = run_mock_evaluation(corpus="fr", metric="context_relevance")
        n = result["context_relevance"]["n_samples"]
        # Real gold_label_fr.tsv has 60 samples (from convert_to_ares)
        assert n > 0, "n_samples should be > 0 (reads real TSV)"
        assert n == 60, f"Expected 60 gold label rows, got {n}"

    def test_mock_score_reflects_positive_rate(self) -> None:
        """Mock score should be close to gold label positive rate (not arbitrary)."""
        result = run_mock_evaluation(corpus="fr", metric="context_relevance")
        score = result["context_relevance"]["score"]
        # Gold label has ~70% positives (42 pos / 18 neg from 30% ratio)
        # Score = positive_rate +/- noise(0.05) so should be in [0.55, 0.85]
        assert (
            0.50 <= score <= 0.90
        ), f"Score {score:.3f} should reflect gold label positive rate (~0.70)"

    def test_mock_metrics_produce_different_scores(self) -> None:
        """Different metrics must produce different mock scores (independent seeds)."""
        scores = {}
        for m in ARES_METRICS:
            result = run_mock_evaluation(corpus="fr", metric=m)
            scores[m] = result[m]["score"]

        # With hashlib-based seeds, at least 2 of 3 metrics should differ
        unique_scores = len(set(f"{s:.6f}" for s in scores.values()))
        assert unique_scores >= 2, (
            f"Expected at least 2 distinct scores across 3 metrics, "
            f"got {unique_scores}: {scores}"
        )


class TestRunAllMetrics:
    """Tests for run_all_metrics function."""

    def test_mock_all_returns_three_metrics(self) -> None:
        """run_all_metrics with mock returns 3 metric results."""
        result = run_all_metrics(backend="mock", corpus="fr")

        assert "metrics" in result
        assert len(result["metrics"]) == 3
        for m in ARES_METRICS:
            assert m in result["metrics"]
            assert "score" in result["metrics"][m]
            assert "pass" in result["metrics"][m]

    def test_all_pass_field(self) -> None:
        """Result should have all_pass boolean."""
        result = run_all_metrics(backend="mock", corpus="fr")
        assert "all_pass" in result
        assert isinstance(result["all_pass"], bool)

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown backend"):
            run_all_metrics(backend="invalid_backend")

    def test_all_metrics_scores_are_independent(self) -> None:
        """Each metric in run_all_metrics should have its own score, not copies."""
        result = run_all_metrics(backend="mock", corpus="fr")
        scores = [result["metrics"][m]["score"] for m in ARES_METRICS]
        # At least 2 of 3 must differ (independent seeds per metric)
        unique = len(set(f"{s:.6f}" for s in scores))
        assert unique >= 2, f"Metrics should be independent, got: {scores}"

    def test_all_pass_is_conjunction(self) -> None:
        """all_pass must be True only if ALL metrics pass."""
        result = run_all_metrics(backend="mock", corpus="fr")
        individual = [result["metrics"][m]["pass"] for m in ARES_METRICS]
        assert result["all_pass"] == all(individual)
