"""
Tests for parse_annales module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.evaluation.annales.parse_annales import (
    _clean_text,
    _extract_all_questions_from_markdown,
    _extract_choices_from_block,
    _group_questions_by_sequence,
    _is_correction_table,
    _parse_correction_table,
    classify_question_taxonomy,
    parse_annales_file,
)


class TestClassifyQuestionTaxonomy:
    """Tests for question taxonomy classification."""

    def test_scenario_question(self) -> None:
        """Should classify scenario questions correctly."""
        text = "Vous êtes l'arbitre d'une partie. Que faites-vous?"
        result = classify_question_taxonomy(text, "UVR")
        assert result["question_type"] == "scenario"
        assert result["cognitive_level"] == "APPLY"
        assert result["reasoning_type"] == "multi-hop"

    def test_factual_question(self) -> None:
        """Should classify factual questions correctly."""
        text = "Quelle est la durée minimale d'une partie rapide?"
        result = classify_question_taxonomy(text, "UVR")
        assert result["question_type"] == "factual"
        assert result["cognitive_level"] == "RECALL"

    def test_procedural_question(self) -> None:
        """Should classify procedural questions correctly."""
        text = "Quelle procédure suivre pour homologuer un tournoi?"
        result = classify_question_taxonomy(text, "UVC")
        assert result["question_type"] == "procedural"
        assert result["cognitive_level"] == "UNDERSTAND"

    def test_uvt_default_scenario(self) -> None:
        """Should default to scenario for long UVT questions."""
        text = "x" * 150  # Long text without keywords
        result = classify_question_taxonomy(text, "UVT")
        assert result["question_type"] == "scenario"

    def test_multiple_choice_answer_type(self) -> None:
        """Should detect multiple choice when has_choices is True."""
        result = classify_question_taxonomy("Test question", "UVR", has_choices=True)
        assert result["answer_type"] == "multiple_choice"

    def test_temporal_reasoning(self) -> None:
        """Should detect temporal reasoning."""
        text = "Quand peut-on demander la nulle?"  # No scenario keywords
        result = classify_question_taxonomy(text, "UVR", has_choices=False)
        assert result["reasoning_type"] == "temporal"

    def test_multi_hop_with_multiple_refs(self) -> None:
        """Should detect multi-hop reasoning with multiple refs."""
        result = classify_question_taxonomy("Question", "UVR", has_multiple_refs=True)
        assert result["reasoning_type"] == "multi-hop"

    def test_yes_no_answer_type(self) -> None:
        """Should detect yes/no answer type with vrai/faux keywords."""
        result = classify_question_taxonomy("Cette affirmation est vrai ou faux?", "UVR", has_choices=False)
        assert result["answer_type"] == "yes_no"

    def test_list_answer_type(self) -> None:
        """Should detect list answer type with list keywords."""
        result = classify_question_taxonomy("Listez les conditions requises", "UVR", has_choices=False)
        assert result["answer_type"] == "list"

    def test_abstractive_for_scenario(self) -> None:
        """Should return abstractive answer type for scenarios without choices."""
        text = "Vous êtes l'arbitre. Que faites-vous dans cette situation?"
        result = classify_question_taxonomy(text, "UVR", has_choices=False)
        assert result["answer_type"] == "abstractive"

    def test_extractive_default(self) -> None:
        """Should default to extractive for simple questions without choices."""
        result = classify_question_taxonomy("Question simple", "UVR", has_choices=False)
        assert result["answer_type"] == "extractive"

    def test_comparative_question_type(self) -> None:
        """Should detect comparative questions."""
        text = "Quelle différence entre une partie rapide et une partie éclair?"
        result = classify_question_taxonomy(text, "UVR")
        assert result["question_type"] == "comparative"
        assert result["cognitive_level"] == "ANALYZE"

    def test_enumeration_keyword_list(self) -> None:
        """Should detect list answer type with énumérez keyword."""
        result = classify_question_taxonomy("Énumérez les règles applicables", "UVR", has_choices=False)
        assert result["answer_type"] == "list"

    def test_quels_sont_keyword_list(self) -> None:
        """Should detect list answer type with 'quels sont' keyword."""
        result = classify_question_taxonomy("Quels sont les critères requis?", "UVR", has_choices=False)
        assert result["answer_type"] == "list"


class TestCleanText:
    """Tests for text cleaning function."""

    def test_normalize_whitespace(self) -> None:
        """Should normalize multiple spaces."""
        result = _clean_text("hello   world")
        assert "  " not in result

    def test_strip_text(self) -> None:
        """Should strip leading/trailing whitespace."""
        result = _clean_text("  hello  ")
        assert result == "hello"


class TestExtractChoices:
    """Tests for choice extraction from question blocks."""

    def test_extract_abcd_choices(self) -> None:
        """Should extract A/B/C/D choices."""
        block = """
        - a) First choice
        - b) Second choice
        - c) Third choice
        - d) Fourth choice
        """
        choices = _extract_choices_from_block(block)
        assert len(choices) == 4
        assert "A" in choices
        assert "First choice" in choices["A"]

    def test_partial_choices(self) -> None:
        """Should handle partial choice sets."""
        block = """
        - a) Only A
        - b) Only B
        """
        choices = _extract_choices_from_block(block)
        assert len(choices) == 2


class TestIsCorrectionTable:
    """Tests for correction table identification."""

    def test_valid_correction_table(self) -> None:
        """Should identify valid correction table."""
        table = {
            "headers": ["Question", "Réponse", "Articles", "Taux Réussite"],
            "rows": [["1", "A", "Art. 1.1", "80%"]],
        }
        assert _is_correction_table(table) is True

    def test_non_correction_table(self) -> None:
        """Should reject non-correction table."""
        table = {
            "headers": ["Name", "Score"],
            "rows": [["Player 1", "100"]],
        }
        assert _is_correction_table(table) is False

    def test_headers_in_first_row(self) -> None:
        """Should detect headers in first row (UVT format)."""
        table = {
            "headers": ["UVT session info", "UVT session info"],
            "rows": [
                ["Question", "Réponse", "Articles", "Taux Réussite"],
                ["1", "B", "Art. 2.1", "75%"],
            ],
        }
        assert _is_correction_table(table) is True


class TestParseCorrectionTable:
    """Tests for correction table parsing."""

    def test_parse_standard_table(self) -> None:
        """Should parse standard correction table."""
        table = {
            "headers": ["Question", "Réponse", "Articles de référence", "Taux Réussite"],
            "rows": [
                ["1", "A", "Article 1.1", "80%"],
                ["2", "B", "Article 2.1", "65%"],
            ],
        }
        corrections = _parse_correction_table(table)
        assert len(corrections) == 2
        assert corrections[0]["num"] == 1
        assert corrections[0]["correct_answer"] == "A"
        assert corrections[0]["success_rate"] == 0.80
        assert corrections[0]["difficulty"] == 0.20

    def test_parse_table_with_header_in_first_row(self) -> None:
        """Should parse table with headers in first row."""
        table = {
            "headers": ["UV info", "UV info", "UV info", "UV info"],
            "rows": [
                ["Question", "Réponse", "Articles", "Taux Réussite"],
                ["1", "C", "Art. 3.1", "90%"],
            ],
        }
        corrections = _parse_correction_table(table)
        assert len(corrections) == 1
        assert corrections[0]["correct_answer"] == "C"


class TestExtractQuestionsFromMarkdown:
    """Tests for question extraction from markdown."""

    def test_extract_single_question(self) -> None:
        """Should extract single question with choices."""
        markdown = """
## Question 1 : What is the answer?

- a) Option A
- b) Option B
- c) Option C
- d) Option D

## Next section
"""
        questions = _extract_all_questions_from_markdown(markdown)
        assert len(questions) >= 1
        assert questions[0]["num"] == 1
        assert "answer" in questions[0]["text"].lower()

    def test_extract_multiple_questions(self) -> None:
        """Should extract multiple sequential questions."""
        markdown = """
## Question 1 : First question?

- a) A1
- b) B1

## Question 2 : Second question?

- a) A2
- b) B2
"""
        questions = _extract_all_questions_from_markdown(markdown)
        assert len(questions) == 2
        assert questions[0]["num"] == 1
        assert questions[1]["num"] == 2


class TestGroupQuestionsBySequence:
    """Tests for question grouping by UV sequence."""

    def test_single_sequence(self) -> None:
        """Should keep single sequence together."""
        questions = [
            {"num": 1, "text": "Q1"},
            {"num": 2, "text": "Q2"},
            {"num": 3, "text": "Q3"},
        ]
        groups = _group_questions_by_sequence(questions)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_multiple_sequences(self) -> None:
        """Should split at Q1 restart."""
        questions = [
            {"num": 1, "text": "UV1-Q1"},
            {"num": 2, "text": "UV1-Q2"},
            {"num": 1, "text": "UV2-Q1"},  # New sequence starts
            {"num": 2, "text": "UV2-Q2"},
        ]
        groups = _group_questions_by_sequence(questions)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2


class TestParseAnnalesFile:
    """Integration tests for full file parsing."""

    def test_parse_minimal_json(self) -> None:
        """Should parse minimal valid JSON structure with UV indicator."""
        data = {
            "filename": "test_UVR_annales.pdf",
            "markdown": """
## Question 1 : Test question?

- a) A
- b) B
- c) C
- d) D
""",
            "tables": [
                {
                    "headers": ["UVR Question", "Réponse", "Articles", "Taux Réussite"],
                    "rows": [["1", "A", "Art. 1.1", "75%"]],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            result = parse_annales_file(Path(f.name))

        # Should extract the question from markdown
        assert result["total_questions"] >= 1
        assert len(result["units"]) >= 1
        # First unit should have the question merged with correction
        first_unit = result["units"][0]
        assert first_unit["statistics"]["with_text"] >= 1

    def test_file_not_found(self) -> None:
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_annales_file(Path("/nonexistent/file.json"))

    def test_invalid_json_structure(self) -> None:
        """Should raise error for invalid JSON structure."""
        data = {"invalid": "structure"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()

            with pytest.raises(ValueError):
                parse_annales_file(Path(f.name))
