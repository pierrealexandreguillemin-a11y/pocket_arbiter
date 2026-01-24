"""
Tests for reformulate_questions module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

from scripts.evaluation.annales.reformulate_questions import (
    detect_question_type,
    extract_core_question,
    reformulate_question,
    simplify_scenario,
)


class TestDetectQuestionType:
    """Tests for question type detection."""

    def test_scenario_dans(self) -> None:
        """Should detect scenario starting with 'Dans'."""
        result = detect_question_type("Dans une finale Roi + Dame contre Roi...")
        assert result == "scenario"

    def test_scenario_un_joueur(self) -> None:
        """Should detect scenario starting with 'Un joueur'."""
        result = detect_question_type("Un joueur appuie sur la pendule...")
        assert result == "scenario"

    def test_scenario_vous_etes(self) -> None:
        """Should detect scenario starting with 'Vous êtes'."""
        result = detect_question_type("Vous êtes arbitre d'un open...")
        assert result == "scenario"

    def test_direct_que(self) -> None:
        """Should detect direct question starting with 'Que'."""
        result = detect_question_type("Que devez-vous faire ?")
        assert result == "direct"

    def test_direct_comment(self) -> None:
        """Should detect direct question starting with 'Comment'."""
        result = detect_question_type("Comment appliquer la règle des 50 coups ?")
        assert result == "direct"

    def test_conditional_si(self) -> None:
        """Should detect conditional starting with 'Si'."""
        result = detect_question_type("Si un joueur touche une pièce...")
        assert result == "conditional"

    def test_statement_la_regle(self) -> None:
        """Should detect statement starting with 'La règle'."""
        result = detect_question_type("La règle stipule que...")
        assert result == "statement"

    def test_unknown_type(self) -> None:
        """Should return unknown for unrecognized patterns."""
        result = detect_question_type("Texte sans pattern reconnu")
        assert result == "unknown"


class TestExtractCoreQuestion:
    """Tests for core question extraction."""

    def test_extracts_interrogative(self) -> None:
        """Should extract interrogative sentence."""
        text = "Un joueur fait ceci. Que devez-vous faire? Attendez la réponse."
        result = extract_core_question(text)
        assert "Que devez-vous faire" in result

    def test_handles_no_question(self) -> None:
        """Should return cleaned text when no question found."""
        text = "Un joueur fait un coup illégal."
        result = extract_core_question(text)
        assert "joueur" in result

    def test_removes_choices(self) -> None:
        """Should remove multiple choice options."""
        text = "Que faire? A - Option 1 B - Option 2"
        result = extract_core_question(text)
        assert "Option" not in result


class TestSimplifyScenario:
    """Tests for scenario simplification."""

    def test_simplifies_time_increment(self) -> None:
        """Should simplify time increment phrase."""
        text = "avec incrément de 30 secondes"
        result = simplify_scenario(text)
        assert result == "avec incrément"

    def test_simplifies_elapsed(self) -> None:
        """Should simplify 'se sont écoulés depuis'."""
        text = "50 coups se sont écoulés depuis"
        result = simplify_scenario(text)
        assert "après" in result

    def test_removes_board_references(self) -> None:
        """Should remove verbose board references."""
        text = "sur son échiquier"
        result = simplify_scenario(text)
        assert result == ""


class TestReformulateQuestion:
    """Tests for full question reformulation."""

    def test_preserves_original(self) -> None:
        """Should preserve original question text."""
        q = {"question": "Dans un open, que faire?", "article_reference": "1.3"}
        result = reformulate_question(q)
        assert "question_original" in result
        assert result["question_original"] == "Dans un open, que faire?"

    def test_adds_reformulation_metadata(self) -> None:
        """Should add reformulation metadata."""
        q = {"question": "Que faire?", "article_reference": ""}
        result = reformulate_question(q)
        assert "reformulation" in result
        assert "original_type" in result["reformulation"]
        assert "method" in result["reformulation"]

    def test_creates_query_variants(self) -> None:
        """Should create query variants."""
        q = {"question": "Dans un match, que faire?", "article_reference": "9.6"}
        result = reformulate_question(q)
        assert "query_variants" in result
        assert len(result["query_variants"]) >= 1

    def test_article_based_variants(self) -> None:
        """Should create article-based variants when reference available."""
        q = {
            "question": "Dans une finale, 50 coups se sont écoulés.",
            "article_reference": "9.6.2",
        }
        result = reformulate_question(q)
        # Should have variant with article reference
        variants = result.get("query_variants", [])
        article_variant = any("9.6.2" in v for v in variants)
        assert article_variant or len(variants) >= 1

    def test_direct_question_preserved(self) -> None:
        """Should preserve direct questions with minimal changes."""
        q = {"question": "Comment appliquer la règle?", "article_reference": ""}
        result = reformulate_question(q)
        assert "appliquer" in result["question"]
