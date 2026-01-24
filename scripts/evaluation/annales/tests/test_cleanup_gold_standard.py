"""
Tests for cleanup_gold_standard module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

from scripts.evaluation.annales.cleanup_gold_standard import (
    assess_answer_quality,
    cleanup_gold_standard,
    derive_answer_text,
)


class TestDeriveAnswerText:
    """Tests for derive_answer_text function."""

    def test_derives_single_choice(self) -> None:
        """Should derive text from single choice."""
        question = {
            "choices": {"A": "Réponse A", "B": "Réponse B"},
            "expected_answer": "A",
        }
        result = derive_answer_text(question)
        assert result.answer_text == "Réponse A"
        assert result.answer_complete is True

    def test_derives_multiple_choices(self) -> None:
        """Should join multiple choice texts with separator."""
        question = {
            "choices": {"A": "Premier", "B": "Deuxième"},
            "expected_answer": "AB",
        }
        result = derive_answer_text(question)
        assert result.answer_text == "Premier | Deuxième"

    def test_handles_missing_letter(self) -> None:
        """Should track missing letters."""
        question = {
            "choices": {"A": "Only A"},
            "expected_answer": "ABC",
        }
        result = derive_answer_text(question)
        assert result.answer_text == "Only A"
        assert result.answer_complete is False
        assert "B" in result.answer_missing_letters
        assert "C" in result.answer_missing_letters

    def test_handles_no_choices(self) -> None:
        """Should return empty result if no choices."""
        question = {"expected_answer": "A"}
        result = derive_answer_text(question)
        assert result.answer_text is None

    def test_handles_no_expected_answer(self) -> None:
        """Should return empty result if no expected_answer."""
        question = {"choices": {"A": "Test"}}
        result = derive_answer_text(question)
        assert result.answer_text is None

    def test_uses_mcq_answer_field(self) -> None:
        """Should also check mcq_answer field."""
        question = {
            "choices": {"A": "From mcq_answer"},
            "mcq_answer": "A",
        }
        result = derive_answer_text(question)
        assert result.answer_text == "From mcq_answer"


class TestAssessAnswerQuality:
    """Tests for assess_answer_quality function."""

    def test_good_answer_score(self) -> None:
        """Good answer should have score 1.0."""
        result = assess_answer_quality("Une réponse complète et détaillée")
        assert result.score == 1.0
        assert result.warning is None

    def test_empty_answer_warning(self) -> None:
        """Empty answer should have score 0.0 and warning."""
        result = assess_answer_quality("")
        assert result.score == 0.0
        assert result.warning == "empty_answer"

    def test_short_answer_warning(self) -> None:
        """Short answer should have score 0.5."""
        result = assess_answer_quality("Vrai")
        assert result.score == 0.5
        assert result.warning == "short_answer"

    def test_reference_only_voir(self) -> None:
        """'Voir ' prefix should be reference_only."""
        result = assess_answer_quality("Voir article 5.3 pour plus de détails")
        assert result.score == 0.3
        assert result.warning == "reference_only"

    def test_reference_only_ref(self) -> None:
        """'Réf:' prefix should be reference_only."""
        result = assess_answer_quality("Réf: R01 - 2.3")
        assert result.score == 0.3
        assert result.warning == "reference_only"

    def test_reference_only_cf(self) -> None:
        """'cf. ' prefix should be reference_only."""
        result = assess_answer_quality("cf. article 1.3 des règles")
        assert result.score == 0.3
        assert result.warning == "reference_only"

    def test_reference_only_se_referer(self) -> None:
        """'Se référer' prefix should be reference_only."""
        result = assess_answer_quality("Se référer au chapitre 5")
        assert result.score == 0.3
        assert result.warning == "reference_only"

    def test_borderline_length_30_chars(self) -> None:
        """Answer with exactly 30 chars should be good."""
        result = assess_answer_quality("A" * 30)  # Exactly 30 chars
        assert result.score == 1.0
        assert result.warning is None

    def test_short_answer_29_chars(self) -> None:
        """Answer with 29 chars should be short."""
        result = assess_answer_quality("A" * 29)  # 29 chars
        assert result.score == 0.5
        assert result.warning == "short_answer"


class TestCleanupGoldStandard:
    """Tests for cleanup_gold_standard function."""

    def test_derives_answer_text(self) -> None:
        """Should derive answer_text from choices."""
        gs_data = {
            "questions": [
                {
                    "choices": {"A": "Correct answer"},
                    "expected_answer": "A",
                }
            ]
        }
        result = cleanup_gold_standard(gs_data)
        assert result["questions"][0]["answer_text"] == "Correct answer"

    def test_moves_expected_answer_to_mcq_answer(self) -> None:
        """Should rename expected_answer to mcq_answer."""
        gs_data = {
            "questions": [
                {
                    "choices": {"A": "Test"},
                    "expected_answer": "A",
                }
            ]
        }
        result = cleanup_gold_standard(gs_data)
        assert "expected_answer" not in result["questions"][0]
        assert result["questions"][0]["mcq_answer"] == "A"

    def test_adds_quality_score(self) -> None:
        """Should add quality_score field."""
        gs_data = {
            "questions": [
                {
                    "choices": {"A": "Une réponse assez longue pour être valide"},
                    "expected_answer": "A",
                }
            ]
        }
        result = cleanup_gold_standard(gs_data)
        assert "quality_score" in result["questions"][0]

    def test_adds_quality_warning_for_weak_answers(self) -> None:
        """Should add quality_warning for weak answers."""
        gs_data = {
            "questions": [
                {
                    "choices": {"A": "Voir article"},
                    "expected_answer": "A",
                }
            ]
        }
        result = cleanup_gold_standard(gs_data)
        assert result["questions"][0]["quality_warning"] == "reference_only"

    def test_generates_cleanup_stats(self) -> None:
        """Should generate cleanup statistics."""
        gs_data = {
            "questions": [
                {"choices": {"A": "Test"}, "expected_answer": "A"},
                {"choices": {"B": "Long answer text here"}, "expected_answer": "B"},
            ]
        }
        result = cleanup_gold_standard(gs_data)
        assert "cleanup_stats" in result
        assert result["cleanup_stats"]["total"] == 2

    def test_fallback_to_article_reference(self) -> None:
        """Should use article_reference as fallback."""
        gs_data = {
            "questions": [
                {
                    "choices": {},
                    "article_reference": "Article 1.3 des règles du jeu",
                }
            ]
        }
        result = cleanup_gold_standard(gs_data)
        assert result["questions"][0]["answer_text"] == "Article 1.3 des règles du jeu"
        assert result["questions"][0]["answer_source"] == "article_reference"

    def test_handles_partial_derivation(self) -> None:
        """Should mark incomplete derivation."""
        gs_data = {
            "questions": [
                {
                    "choices": {"A": "Only A present"},
                    "expected_answer": "AB",  # B is missing
                }
            ]
        }
        result = cleanup_gold_standard(gs_data)
        assert result["questions"][0].get("answer_incomplete") is True
        assert "B" in result["questions"][0].get("answer_missing_letters", [])

    def test_mcq_letter_only_fallback(self) -> None:
        """Should fallback to expected_answer when no derivation possible."""
        gs_data = {
            "questions": [
                {
                    "choices": {},
                    "expected_answer": "C",
                }
            ]
        }
        result = cleanup_gold_standard(gs_data)
        assert result["questions"][0].get("answer_type_mcq_letter") is True

    def test_quality_stats_aggregation(self) -> None:
        """Should aggregate quality warnings by type."""
        gs_data = {
            "questions": [
                {"choices": {"A": "Voir article 1"}, "expected_answer": "A"},
                {"choices": {"A": "Voir article 2"}, "expected_answer": "A"},
                # Long enough answer to be considered good quality (>30 chars)
                {"choices": {"A": "Une réponse normale et suffisamment longue"}, "expected_answer": "A"},
            ]
        }
        result = cleanup_gold_standard(gs_data)
        stats = result["cleanup_stats"]
        assert stats["quality_warnings"]["reference_only"] == 2
        assert stats["quality_good"] >= 1

    def test_empty_questions_list(self) -> None:
        """Should handle empty questions list."""
        gs_data = {"questions": []}
        result = cleanup_gold_standard(gs_data)
        assert result["cleanup_stats"]["total"] == 0

    def test_comprehensive_cleanup(self) -> None:
        """Comprehensive test covering multiple scenarios."""
        gs_data = {
            "questions": [
                # Good answer (derived complete)
                {"choices": {"A": "Une réponse complète et détaillée"}, "expected_answer": "A"},
                # Multiple choice answer
                {"choices": {"A": "Premier choix", "B": "Deuxième choix"}, "expected_answer": "AB"},
                # Short answer (warning)
                {"choices": {"A": "Oui"}, "expected_answer": "A"},
                # Reference only (warning)
                {"choices": {"A": "Voir article 5.3"}, "expected_answer": "A"},
                # Missing letter (partial)
                {"choices": {"A": "Only A"}, "expected_answer": "AC"},
                # Uses mcq_answer instead of expected_answer
                {"choices": {"B": "From mcq field"}, "mcq_answer": "B"},
            ]
        }
        result = cleanup_gold_standard(gs_data)

        # Check all questions processed
        assert result["cleanup_stats"]["total"] == 6

        # Check stats are recorded
        assert result["cleanup_stats"]["mcq_letters_removed"] >= 5  # All expected_answer moved

        # Check quality assessment ran
        assert result["cleanup_stats"]["quality_good"] >= 1
        assert "short_answer" in result["cleanup_stats"]["quality_warnings"]

    def test_preserves_existing_fields(self) -> None:
        """Should preserve existing fields while adding new ones."""
        gs_data = {
            "questions": [
                {
                    "id": "FR-ANN-001",
                    "choices": {"A": "Une réponse"},
                    "expected_answer": "A",
                    "custom_field": "should_persist",
                }
            ]
        }
        result = cleanup_gold_standard(gs_data)
        q = result["questions"][0]
        assert q["id"] == "FR-ANN-001"
        assert q["custom_field"] == "should_persist"
        assert "answer_text" in q
        assert "quality_score" in q
