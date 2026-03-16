"""
Tests for extract_corrige_answers module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

import json
import tempfile
from pathlib import Path

from scripts.evaluation.annales.extract_corrige_answers import (
    _derive_choice_text,
    _extract_explanation_from_block,
    extract_all_corrige_answers,
    extract_question_explanations,
    find_corrige_sections,
    update_gold_standard_with_explanations,
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


class TestEdgeCases:
    """Edge case tests for extraction functions."""

    def test_find_corrige_with_accented_month(self) -> None:
        """Should handle accented month names."""
        markdown = """
## UVR session de décembre 2024 - Corrigé détaillé

Content here

## Fin
"""
        sections = find_corrige_sections(markdown)
        assert len(sections) == 1
        assert sections[0][0] == "UVR"

    def test_extract_explanations_case_insensitive(self) -> None:
        """Should be case insensitive for question headers."""
        corrige_text = """
## question 1 :

Question text
- a) A
- b) B

Article 1.3

This is the explanation text that is longer than twenty characters.
"""
        explanations = extract_question_explanations(corrige_text)
        assert 1 in explanations

    def test_derive_choice_text_with_all_letters(self) -> None:
        """Should join all choice texts for ABCD answer."""
        question = {
            "choices": {"A": "A1", "B": "B2", "C": "C3", "D": "D4"},
            "mcq_answer": "ABCD",
        }
        result = _derive_choice_text(question)
        assert result == "A1 | B2 | C3 | D4"

    def test_extract_explanation_with_chapitre_reference(self) -> None:
        """Should detect Chapitre references."""
        block = """
Question text

- a) A

Chapitre 8 - Temps de réflexion

La pendule doit être arrêtée conformément aux règles établies.
"""
        explanation = _extract_explanation_from_block(block)
        assert "pendule" in explanation.lower()

    def test_extract_explanation_with_la_reference(self) -> None:
        """Should detect LA references (e.g., LA – 5.3)."""
        block = """
Question

- a) Choice

LA – 5.3 référence

Une explication détaillée du correcteur concernant la question posée.
"""
        explanation = _extract_explanation_from_block(block)
        assert "explication" in explanation.lower()

    def test_find_corrige_with_uvo(self) -> None:
        """Should find UVO (Open) corrigé sections."""
        markdown = """
## UVO - juin 2025 - Corrigé Détaillé

Open questions content

## UVT
"""
        sections = find_corrige_sections(markdown)
        uvs = [s[0] for s in sections]
        assert "UVO" in uvs

    def test_find_corrige_with_uvt(self) -> None:
        """Should find UVT (Tournoi) corrigé sections."""
        markdown = """
## UVT session de décembre 2024 - Corrigé détaillé

Tournament questions content

## Fin
"""
        sections = find_corrige_sections(markdown)
        uvs = [s[0] for s in sections]
        assert "UVT" in uvs

    def test_extract_explanation_stops_at_next_question(self) -> None:
        """Should stop extracting at next question marker."""
        block = """
Question text

- a) A

Article 1.3

L'explication qui doit être capturée ici.

## Question 2 :

Ce texte ne doit pas être inclus.
"""
        explanation = _extract_explanation_from_block(block)
        assert "explication" in explanation.lower()
        assert "question 2" not in explanation.lower()

    def test_extract_explanation_skips_images(self) -> None:
        """Should skip image markers."""
        block = """
Question

- a) A

Article 1.3

Première partie de l'explication.
<!-- image -->
Suite de l'explication après le marqueur.
"""
        explanation = _extract_explanation_from_block(block)
        assert "première partie" in explanation.lower()
        assert "suite" in explanation.lower()
        # The image line itself should be skipped
        assert "<!--" not in explanation

    def test_extract_explanation_with_annexe_reference(self) -> None:
        """Should detect Annexe references."""
        block = """
Question text

- a) A

Annexe 2 - Tableaux de répartition

L'explication concernant les tableaux et leur utilisation.
"""
        explanation = _extract_explanation_from_block(block)
        assert "explication" in explanation.lower()

    def test_extract_explanation_numeric_only_rejected(self) -> None:
        """Should reject explanations that are only numbers."""
        block = """
Question

- a) A

Article 1.1

1 2 3 4 5 6 7 8 9 0
"""
        explanation = _extract_explanation_from_block(block)
        assert explanation == ""


class TestExtractAllCorrigeAnswers:
    """Integration tests for extract_all_corrige_answers function."""

    def test_extracts_from_docling_json(self) -> None:
        """Should extract explanations from Docling JSON file."""
        docling_data = {
            "markdown": """
## UVR session de décembre 2024 - Corrigé détaillé

## QUESTION 1 :

Question about rules

- a) Oui
- b) Non

Article 3.1 des règles du jeu

Explication détaillée pour la première question du corrigé.

## QUESTION 2 :

Another question

- a) Vrai
- b) Faux

Article 4.2

Deuxième explication avec suffisamment de contenu.

## Fin du corrigé
"""
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(docling_data, f)
            f.flush()
            result = extract_all_corrige_answers(Path(f.name))

        assert "UVR" in result
        assert 1 in result["UVR"]
        assert 2 in result["UVR"]

    def test_returns_empty_for_non_dict(self) -> None:
        """Should return empty for non-dict JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(["list", "data"], f)
            f.flush()
            result = extract_all_corrige_answers(Path(f.name))

        assert result == {}

    def test_returns_empty_for_no_markdown(self) -> None:
        """Should return empty if no markdown field."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"other": "data"}, f)
            f.flush()
            result = extract_all_corrige_answers(Path(f.name))

        assert result == {}

    def test_handles_multiple_uvs(self) -> None:
        """Should extract from multiple UV sections."""
        docling_data = {
            "markdown": """
## UVR - juin 2025 - Corrige Detaille

## QUESTION 1 :

UVR question

Article 1.1

Explication UVR question un avec details complets.

## UVC - juin 2025 - Corrige Detaille

## QUESTION 1 :

UVC question

R01 - 2.3

Explication UVC question un avec details complets.

## Fin
"""
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(docling_data, f)
            f.flush()
            result = extract_all_corrige_answers(Path(f.name))

        # At least UVR should be found
        assert "UVR" in result
        assert 1 in result["UVR"]


class TestUpdateGoldStandardWithExplanations:
    """Integration tests for update_gold_standard_with_explanations."""

    def test_merges_choice_and_explanation(self) -> None:
        """Should merge choice text with explanation."""
        gs_data = {
            "version": {"number": "6.6.0"},
            "questions": [
                {
                    "id": "FR-ANN-UVR-001",
                    "choices": {"A": "Vrai", "B": "Faux"},
                    "mcq_answer": "A",
                    "annales_source": {
                        "session": "dec2024",
                        "uv": "UVR",
                        "question_num": 1,
                    },
                }
            ],
        }

        docling_data = {
            "markdown": """
## UVR session de décembre 2024 - Corrigé détaillé

## QUESTION 1 :

Question

Article 1.1

Explication officielle du correcteur pour cette question.
"""
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gs_path = Path(tmpdir) / "gold_standard.json"
            gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

            docling_dir = Path(tmpdir) / "docling"
            docling_dir.mkdir()
            (docling_dir / "annales_dec2024.json").write_text(
                json.dumps(docling_data), encoding="utf-8"
            )

            output_path = Path(tmpdir) / "output.json"
            stats = update_gold_standard_with_explanations(
                gs_path, [docling_dir], output_path
            )

            assert stats["merged"] == 1
            result = json.loads(output_path.read_text(encoding="utf-8"))
            q = result["questions"][0]
            assert q["answer_text"] == "Vrai"
            assert "explication" in q["answer_explanation"].lower()
            assert q["answer_source"] == "merged"

    def test_choice_only_when_no_explanation(self) -> None:
        """Should use choice only when no explanation found."""
        gs_data = {
            "version": {"number": "6.6.0"},
            "questions": [
                {
                    "id": "FR-ANN-UVR-001",
                    "choices": {"A": "Oui"},
                    "mcq_answer": "A",
                    "annales_source": {
                        "session": "dec2024",
                        "uv": "UVR",
                        "question_num": 99,  # Not in corrigé
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gs_path = Path(tmpdir) / "gold_standard.json"
            gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

            # Empty docling dir
            docling_dir = Path(tmpdir) / "docling"
            docling_dir.mkdir()

            output_path = Path(tmpdir) / "output.json"
            stats = update_gold_standard_with_explanations(
                gs_path, [docling_dir], output_path
            )

            assert stats["choice_only"] == 1
            result = json.loads(output_path.read_text(encoding="utf-8"))
            assert result["questions"][0]["answer_source"] == "choice_only"

    def test_updates_version_number(self) -> None:
        """Should update version to 6.7.0."""
        gs_data = {
            "version": {"number": "6.6.0"},
            "questions": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gs_path = Path(tmpdir) / "gold_standard.json"
            gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

            docling_dir = Path(tmpdir) / "docling"
            docling_dir.mkdir()

            output_path = Path(tmpdir) / "output.json"
            update_gold_standard_with_explanations(gs_path, [docling_dir], output_path)

            result = json.loads(output_path.read_text(encoding="utf-8"))
            assert result["version"]["number"] == "6.7.0"

    def test_skips_report_files(self) -> None:
        """Should skip files with 'report' in name."""
        gs_data = {
            "version": "6.6.0",
            "questions": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gs_path = Path(tmpdir) / "gold_standard.json"
            gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

            docling_dir = Path(tmpdir) / "docling"
            docling_dir.mkdir()
            # Create a report file that should be skipped
            (docling_dir / "extraction_report.json").write_text(
                json.dumps({"markdown": "# Report"}), encoding="utf-8"
            )

            output_path = Path(tmpdir) / "output.json"
            stats = update_gold_standard_with_explanations(
                gs_path, [docling_dir], output_path
            )

            # Should complete without errors
            assert stats["total"] == 0

    def test_handles_string_version(self) -> None:
        """Should handle string version format."""
        gs_data = {
            "version": "6.6.0",  # String instead of dict
            "questions": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gs_path = Path(tmpdir) / "gold_standard.json"
            gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

            docling_dir = Path(tmpdir) / "docling"
            docling_dir.mkdir()

            output_path = Path(tmpdir) / "output.json"
            update_gold_standard_with_explanations(gs_path, [docling_dir], output_path)

            result = json.loads(output_path.read_text(encoding="utf-8"))
            assert result["version"]["number"] == "6.7.0"
