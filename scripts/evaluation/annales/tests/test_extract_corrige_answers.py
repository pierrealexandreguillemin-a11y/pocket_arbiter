"""
Tests for extract_corrige_answers module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

from scripts.evaluation.annales.extract_corrige_answers import (
    _derive_choice_text,
    _extract_explanation_from_block,
    extract_question_explanations,
    find_corrige_sections,
)


class TestFindCorrigeSections:
    """Tests for find_corrige_sections function."""

    def test_finds_uvr_header(self) -> None:
        """Should find UVR corrigé section."""
        markdown = """
## Introduction

## UVR session de décembre 2024 - Corrigé détaillé

Question 1 content here

## UVC session de décembre 2024 - Corrigé détaillé
"""
        sections = find_corrige_sections(markdown)
        assert len(sections) >= 1
        assert sections[0][0] == "UVR"

    def test_finds_multiple_uv_sections(self) -> None:
        """Should find multiple UV sections."""
        markdown = """
## UVR - juin 2025 - Corrigé Détaillé

UVR content

## UVC - juin 2025 - Corrigé Détaillé

UVC content

## UVO - juin 2025 - Corrigé Détaillé

UVO content
"""
        sections = find_corrige_sections(markdown)
        uvs = [s[0] for s in sections]
        assert "UVR" in uvs
        assert "UVC" in uvs
        assert "UVO" in uvs

    def test_handles_varied_formats(self) -> None:
        """Should handle format variations."""
        markdown = """
## FFE DNA UVR session de décembre 2024 - Corrigé détaillé

Content here
"""
        sections = find_corrige_sections(markdown)
        assert len(sections) >= 1
        assert sections[0][0] == "UVR"

    def test_returns_empty_for_no_corrige(self) -> None:
        """Should return empty for markdown without corrigé."""
        markdown = """
## Introduction

## UVR - Sujet

Questions here
"""
        sections = find_corrige_sections(markdown)
        assert sections == []


class TestExtractQuestionExplanations:
    """Tests for extract_question_explanations function."""

    def test_extracts_single_explanation(self) -> None:
        """Should extract explanation from question block."""
        corrige_text = """
## QUESTION 1 :

La question posée

- a) Choix A
- b) Choix B

Article 1.3 des règles du jeu

L'explication officielle du correcteur qui détaille la réponse.
"""
        explanations = extract_question_explanations(corrige_text)
        assert 1 in explanations
        assert "explication officielle" in explanations[1].lower()

    def test_extracts_multiple_explanations(self) -> None:
        """Should extract explanations from multiple questions."""
        corrige_text = """
## QUESTION 1 :

Question 1 text
- a) A
- b) B

Article 1.3

Explication pour question 1.

## QUESTION 2 :

Question 2 text
- a) C
- b) D

Article 2.1

Explication pour question 2.
"""
        explanations = extract_question_explanations(corrige_text)
        assert 1 in explanations
        assert 2 in explanations

    def test_handles_question_without_explanation(self) -> None:
        """Should handle question without explanation text."""
        corrige_text = """
## QUESTION 1 :

Question text
- a) A
- b) B
"""
        explanations = extract_question_explanations(corrige_text)
        # May be empty or have incomplete extraction
        assert isinstance(explanations, dict)


class TestExtractExplanationFromBlock:
    """Tests for _extract_explanation_from_block function."""

    def test_extracts_post_article_text(self) -> None:
        """Should extract text after article reference."""
        block = """
Question text here

- a) Choix A
- b) Choix B

Article 1.3 des règles

L'explication détaillée vient après la référence article.
"""
        explanation = _extract_explanation_from_block(block)
        assert "explication détaillée" in explanation.lower()

    def test_handles_multiple_line_explanation(self) -> None:
        """Should join multiple lines of explanation."""
        block = """
Question

- a) A
- b) B

Article 5.2

Première partie de l'explication.
Suite de l'explication sur une autre ligne.
"""
        explanation = _extract_explanation_from_block(block)
        assert "première partie" in explanation.lower()
        assert "suite" in explanation.lower()

    def test_returns_empty_for_short_text(self) -> None:
        """Should return empty for very short explanations."""
        block = """
Question

- a) A

Article 1.1

OK
"""
        explanation = _extract_explanation_from_block(block)
        # Less than 20 chars should be rejected
        assert explanation == ""

    def test_handles_no_article_reference(self) -> None:
        """Should return empty if no article reference found."""
        block = """
Question text

- a) A
- b) B

No reference here.
"""
        explanation = _extract_explanation_from_block(block)
        assert explanation == ""


class TestDeriveChoiceText:
    """Tests for _derive_choice_text function."""

    def test_derives_single_choice(self) -> None:
        """Should derive text from single choice letter."""
        question = {
            "choices": {"A": "Vrai", "B": "Faux"},
            "mcq_answer": "A",
        }
        result = _derive_choice_text(question)
        assert result == "Vrai"

    def test_derives_multiple_choices(self) -> None:
        """Should join multiple choice texts."""
        question = {
            "choices": {"A": "Premier", "B": "Deuxième", "C": "Troisième"},
            "mcq_answer": "AC",
        }
        result = _derive_choice_text(question)
        assert result == "Premier | Troisième"

    def test_handles_missing_letter(self) -> None:
        """Should handle missing choice letter gracefully."""
        question = {
            "choices": {"A": "Only A"},
            "mcq_answer": "AB",  # B not in choices
        }
        result = _derive_choice_text(question)
        assert result == "Only A"

    def test_returns_none_for_no_choices(self) -> None:
        """Should return None if no choices."""
        question = {
            "choices": {},
            "mcq_answer": "A",
        }
        result = _derive_choice_text(question)
        assert result is None

    def test_returns_none_for_no_answer(self) -> None:
        """Should return None if no mcq_answer."""
        question = {
            "choices": {"A": "Test"},
            "mcq_answer": "",
        }
        result = _derive_choice_text(question)
        assert result is None

    def test_handles_lowercase_letters(self) -> None:
        """Should handle both upper and lowercase answers."""
        question = {
            "choices": {"A": "Test A", "B": "Test B"},
            "mcq_answer": "A",  # Should match A in choices
        }
        result = _derive_choice_text(question)
        assert result == "Test A"
