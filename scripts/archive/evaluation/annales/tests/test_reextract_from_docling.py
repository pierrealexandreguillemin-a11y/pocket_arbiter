"""
Tests for reextract_from_docling module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Data quality validation
"""

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluation.annales.reextract_from_docling import (
    _aggregate_by_session,
    clean_text,
    detect_extraction_flags,
    extract_article_reference,
    extract_choices,
    extract_explanation,
    extract_question_block,
    extract_question_text,
    find_uv_corrige,
    find_uv_grille,
    find_uv_sujet,
    get_docling_path,
    load_docling_markdown,
    parse_grille_table,
    reextract_all,
    reextract_question,
)

# ---- clean_text ----


class TestCleanText:
    """Tests for text cleaning function."""

    def test_removes_image_markers(self) -> None:
        """Should remove <!-- image --> markers."""
        assert "image" not in clean_text("Before <!-- image --> After")

    def test_removes_internal_headers(self) -> None:
        """Should remove ## markdown headers."""
        result = clean_text("Some text ## Header here")
        assert "##" not in result
        assert "Some text" in result
        assert "Header here" in result

    def test_normalizes_whitespace(self) -> None:
        """Should collapse multiple spaces."""
        result = clean_text("hello    world")
        assert result == "hello world"

    def test_strips_text(self) -> None:
        """Should strip leading/trailing whitespace."""
        result = clean_text("  hello  ")
        assert result == "hello"

    def test_empty_string(self) -> None:
        """Should handle empty input."""
        assert clean_text("") == ""

    def test_combined_artifacts(self) -> None:
        """Should handle multiple artifacts at once."""
        result = clean_text("  text <!-- image -->  ## header  more  ")
        assert "<!--" not in result
        assert "##" not in result
        assert result == "text header more"


# ---- get_docling_path ----


class TestGetDoclingPath:
    """Tests for docling path resolution."""

    def test_known_session(self) -> None:
        """Should return path for known session."""
        path = get_docling_path("dec2024", Path("/base"))
        assert path is not None
        assert "Annales-Decembre-2024.json" in str(path)
        assert "annales_dec_2024" in str(path)

    def test_unknown_session(self) -> None:
        """Should return None for unknown session."""
        assert get_docling_path("jan2030") is None

    def test_default_corpus_base(self) -> None:
        """Should use default corpus base when not specified."""
        path = get_docling_path("dec2019")
        assert path is not None
        assert "corpus" in str(path)

    def test_all_mapped_sessions(self) -> None:
        """Should have path for all 10 GS sessions."""
        sessions = [
            "dec2019",
            "jun2021",
            "dec2021",
            "jun2022",
            "dec2022",
            "jun2023",
            "dec2023",
            "jun2024",
            "dec2024",
            "jun2025",
        ]
        for s in sessions:
            path = get_docling_path(s, Path("/base"))
            assert path is not None, f"No mapping for session {s}"


# ---- load_docling_markdown ----


class TestLoadDoclingMarkdown:
    """Tests for markdown loading."""

    def test_loads_from_valid_file(self, tmp_path: Path) -> None:
        """Should load markdown from valid JSON file."""
        # Create directory structure
        subdir = tmp_path / "annales_dec_2024"
        subdir.mkdir()
        data = {"markdown": "# Test markdown\n\nQuestion 1"}
        json_path = subdir / "Annales-Decembre-2024.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_docling_markdown("dec2024", tmp_path)
        assert result is not None
        assert "Test markdown" in result

    def test_returns_none_missing_session(self) -> None:
        """Should return None for unmapped session."""
        assert load_docling_markdown("jan2030") is None

    def test_returns_none_missing_file(self, tmp_path: Path) -> None:
        """Should return None when file doesn't exist."""
        result = load_docling_markdown("dec2024", tmp_path)
        assert result is None


# ---- find_uv_sujet ----


class TestFindUvSujet:
    """Tests for UV sujet section detection."""

    def test_new_format_explicit_sujet(self) -> None:
        """Should find section with explicit 'Sujet' keyword (dec2024+)."""
        markdown = """
## FFE DNA UVR session décembre 2024 – Sujet sans réponse

Question 1 : What is this?

- a) A
- b) B

## FFE DNA UVR session décembre 2024 – Fin du sujet
"""
        result = find_uv_sujet(markdown, "UVR")
        assert result is not None
        start, end = result
        section = markdown[start:end]
        assert "Question 1" in section

    def test_old_format_session_header(self) -> None:
        """Should find section with 'session' keyword (dec2019-jun2024)."""
        markdown = """
## UVR - session de décembre 2019

Question 1 : First question?

- a) A
- b) B

## UVR - Grille des réponses
"""
        result = find_uv_sujet(markdown, "UVR")
        assert result is not None
        start, end = result
        section = markdown[start:end]
        assert "Question 1" in section
        assert "Grille" not in section

    def test_no_uv_section(self) -> None:
        """Should return None when UV not found."""
        markdown = "## Some other content\n\nNo UV sections here."
        assert find_uv_sujet(markdown, "UVR") is None

    def test_stops_at_grille(self) -> None:
        """Should not include grille section content."""
        markdown = """
## UVR - session de juin 2022

Question 1 : Test?

## UVR session de juin 2022 - Grille des réponses

| 1 | A | Art. 1 | 80% |
"""
        result = find_uv_sujet(markdown, "UVR")
        assert result is not None
        start, end = result
        section = markdown[start:end]
        assert "Art. 1" not in section


# ---- find_uv_grille ----


class TestFindUvGrille:
    """Tests for UV grille section detection."""

    def test_finds_grille_section(self) -> None:
        """Should find the grille des réponses section."""
        markdown = """
## UVC - session de juin 2023 - Sujet

Questions here

## UVC - session de juin 2023 - Grille des réponses

| Q | R | Art | Taux |
| 1 | B | Art. 1 | 75% |

## UVC - session de juin 2023 - Corrigé détaillé
"""
        result = find_uv_grille(markdown, "UVC")
        assert result is not None
        start, end = result
        section = markdown[start:end]
        assert "Art. 1" in section

    def test_no_grille(self) -> None:
        """Should return None when no grille section."""
        markdown = "## UVR - Sujet\n\nQuestions only."
        assert find_uv_grille(markdown, "UVR") is None


# ---- find_uv_corrige ----


class TestFindUvCorrige:
    """Tests for UV corrigé section detection."""

    def test_finds_corrige_section(self) -> None:
        """Should find the corrigé section."""
        markdown = """
## UVR - session de décembre 2021 - Grille

Table here

## UVR - session de décembre 2021 - Corrigé détaillé

Question 1 : Explanation here

## UVC - session de décembre 2021
"""
        result = find_uv_corrige(markdown, "UVR")
        assert result is not None
        start, end = result
        section = markdown[start:end]
        assert "Explanation here" in section

    def test_no_corrige(self) -> None:
        """Should return None when no corrigé section."""
        markdown = "## UVO - Sujet\n\nOnly questions."
        assert find_uv_corrige(markdown, "UVO") is None


# ---- extract_question_block ----


class TestExtractQuestionBlock:
    """Tests for question block extraction from sections."""

    def test_extract_single_question(self) -> None:
        """Should extract a single question block."""
        section = """
Question 1 : What is the answer?

- a) First
- b) Second
"""
        result = extract_question_block(section, 1)
        assert result is not None
        assert "answer" in result.lower()

    def test_extract_between_questions(self) -> None:
        """Should extract block bounded by next question."""
        section = """
Question 1 : First question?

- a) A1
- b) B1

Question 2 : Second question?

- a) A2
- b) B2
"""
        result = extract_question_block(section, 1)
        assert result is not None
        assert "First" in result
        assert "Second" not in result

    def test_extract_question_with_hash_prefix(self) -> None:
        """Should handle ## before Question (Docling artifact)."""
        section = """
## Question 3 : With hash prefix?

- a) Yes
- b) No
"""
        result = extract_question_block(section, 3)
        assert result is not None
        assert "hash prefix" in result.lower()

    def test_question_not_found(self) -> None:
        """Should return None for missing question number."""
        section = "Question 1 : Only this one.\n"
        assert extract_question_block(section, 99) is None

    def test_preserves_full_text(self) -> None:
        """Should not truncate long question text."""
        long_text = "This is a very long question text. " * 20
        section = f"Question 5 : {long_text}\n"
        result = extract_question_block(section, 5)
        assert result is not None
        assert len(result) > 200


# ---- extract_choices ----


class TestExtractChoices:
    """Tests for QCM choice extraction across format variants."""

    def test_dash_letter_paren(self) -> None:
        """Should extract '- a) text' format."""
        block = """
- a) First choice
- b) Second choice
- c) Third choice
- d) Fourth choice
"""
        choices = extract_choices(block)
        assert len(choices) == 4
        assert choices["A"] == "First choice"
        assert choices["D"] == "Fourth choice"

    def test_letter_colon(self) -> None:
        """Should extract 'A : text' format (dec2019)."""
        block = """
A : La réponse est 42
B : La réponse est 43
C : La réponse est 44
"""
        choices = extract_choices(block)
        assert len(choices) >= 3
        assert "42" in choices.get("A", "")

    def test_dash_letter_dash(self) -> None:
        """Should extract '- A - text' format (jun2021)."""
        block = """
- A - Premier choix
- B - Deuxième choix
- C - Troisième choix
"""
        choices = extract_choices(block)
        assert len(choices) >= 3
        assert "Premier" in choices.get("A", "")

    def test_no_choices(self) -> None:
        """Should return empty dict when no choices found."""
        block = "This is a question without any choices."
        choices = extract_choices(block)
        assert len(choices) == 0

    def test_uppercase_letters(self) -> None:
        """Should normalize letters to uppercase."""
        block = """
- a) lower case
- b) also lower
"""
        choices = extract_choices(block)
        assert "A" in choices
        assert "B" in choices
        assert "a" not in choices


# ---- extract_question_text ----


class TestExtractQuestionText:
    """Tests for question text extraction (before choices)."""

    def test_text_before_choices(self) -> None:
        """Should extract text before choice markers."""
        block = """The question is about rules.

- a) Answer A
- b) Answer B
"""
        result = extract_question_text(block)
        assert "rules" in result
        assert "Answer A" not in result

    def test_text_without_choices(self) -> None:
        """Should return full text when no choices present."""
        block = "This is a question with no choices at all."
        result = extract_question_text(block)
        assert "no choices" in result

    def test_cleans_artifacts(self) -> None:
        """Should clean text artifacts."""
        block = "Text with <!-- image --> and   extra   spaces"
        result = extract_question_text(block)
        assert "<!--" not in result
        assert "  " not in result


# ---- extract_article_reference ----


class TestExtractArticleReference:
    """Tests for article reference extraction from corrigé."""

    def test_la_format(self) -> None:
        """Should extract 'LA - Article' format."""
        block = "Réponse correcte: A\nLA - Article 6.7.2 du Livre de l'Arbitre\nExplication."
        ref = extract_article_reference(block)
        assert ref is not None
        assert "LA" in ref
        assert "6.7.2" in ref

    def test_article_format(self) -> None:
        """Should extract 'Article X.Y' format."""
        block = "Voir Article 3.10.1 pour les détails."
        ref = extract_article_reference(block)
        assert ref is not None
        assert "Article 3.10.1" in ref

    def test_art_dot_format(self) -> None:
        """Should extract 'Art. X.Y' format."""
        block = "Référence: Art. 5.2.3 du règlement."
        ref = extract_article_reference(block)
        assert ref is not None
        assert "Art." in ref

    def test_no_reference(self) -> None:
        """Should return None when no reference found."""
        block = "No article reference in this text."
        assert extract_article_reference(block) is None

    def test_short_ref_rejected(self) -> None:
        """Should reject references shorter than 8 chars."""
        block = "Art. 1"
        assert extract_article_reference(block) is None


# ---- extract_explanation ----


class TestExtractExplanation:
    """Tests for explanation extraction from corrigé blocks."""

    def test_extracts_after_article(self) -> None:
        """Should extract explanation text after article reference."""
        block = """Question about something

- a) Choice A
- b) Choice B

Article 3.7.1 du règlement

L'explication détaillée est que le joueur doit demander la nulle avant d'appuyer sur la pendule."""
        expl = extract_explanation(block)
        assert expl is not None
        assert "joueur" in expl

    def test_no_article_returns_none(self) -> None:
        """Should return None when no article line found."""
        block = "Just some text without any article reference."
        assert extract_explanation(block) is None

    def test_short_explanation_rejected(self) -> None:
        """Should reject explanations shorter than 20 chars."""
        block = "Text\nArticle 1.2.3 reference\nShort."
        assert extract_explanation(block) is None

    def test_skips_image_markers(self) -> None:
        """Should skip lines with image markers in explanation."""
        block = """- a) A
- b) B

Article 5.1.2 du règlement

<!-- image -->
L'explication complète de la situation est la suivante."""
        expl = extract_explanation(block)
        assert expl is not None
        assert "image" not in expl


# ---- parse_grille_table ----


class TestParseGrilleTable:
    """Tests for grille answer table parsing."""

    def test_standard_table(self) -> None:
        """Should parse standard markdown table."""
        table = """
| Question | Réponse | Articles | Taux Réussite |
|----------|---------|----------|---------------|
| 1 | A | Art. 6.7 | 80% |
| 2 | C | Art. 3.1 | 65% |
| 3 | B | Art. 9.2 | 42% |
"""
        result = parse_grille_table(table)
        assert len(result) == 3
        assert result[1]["answer"] == "A"
        assert result[1]["rate"] == 0.80
        assert result[2]["answer"] == "C"
        assert result[3]["rate"] == 0.42

    def test_article_reference(self) -> None:
        """Should extract article reference from table."""
        table = """
| Question | Réponse | Articles | Taux Réussite |
|----------|---------|----------|---------------|
| 5 | D | LA Article 2.3 | 55% |
"""
        result = parse_grille_table(table)
        assert 5 in result
        assert "Article 2.3" in result[5]["article"]

    def test_empty_table(self) -> None:
        """Should return empty dict for no rows."""
        assert parse_grille_table("No table content here") == {}

    def test_six_column_format(self) -> None:
        """Should parse 6-column format (jun2021 style)."""
        table = """
| QUESTION | REGLE | CADENCE | REPONSE | ARTICLE | Taux Réussite (%) |
|----------|-------|---------|---------|---------|-------------------|
| 1 | Nulle | Rapide | A | Art. 9.2.1 | 89% |
| 2 | Nulle | Lente | C | Art. 9.2.1 | 26% |
"""
        result = parse_grille_table(table)
        assert len(result) == 2
        assert result[1]["answer"] == "A"
        assert result[1]["rate"] == 0.89
        assert result[2]["answer"] == "C"
        assert result[2]["rate"] == 0.26

    def test_five_column_uvc_format(self) -> None:
        """Should parse 5-column UVC format (dec2019 style)."""
        table = """
| N° | QUESTION | ARTICLE | REPONSE | Taux Réussite (%) |
|----|----------|---------|---------|-------------------|
| 1 | Constitution | Art. 5.1 | B | 72% |
| 2 | Budget | Art. 8.3 | A | 58% |
"""
        result = parse_grille_table(table)
        assert len(result) == 2
        assert result[1]["answer"] == "B"
        assert result[2]["answer"] == "A"

    def test_merged_num_answer_cells(self) -> None:
        """Should parse OCR-shifted cells with 'num answer' merged."""
        table = """
| Question | Réponse | Document de référence : Article | Taux Réussite |
|----------|---------|--------------------------------|---------------|
| 1 C | LA - Art. 11 | 88 (%) |  |
| 2 A | LA - Art. 6.2 | 96 (%) |  |
"""
        result = parse_grille_table(table)
        assert len(result) == 2
        assert result[1]["answer"] == "C"
        assert result[1]["rate"] == 0.88
        assert result[2]["answer"] == "A"

    def test_merged_question_reponse_header(self) -> None:
        """Should parse 'Question Réponse' merged header (jun2023/jun2024)."""
        table = """
| Question Réponse | Document de référence : Article | Taux Réussite |
|------------------|--------------------------------|---------------|
| 1 B | LA - Art. 3.1 | 75% |
| 2 D | LA - Art. 5.2 | 62% |
"""
        result = parse_grille_table(table)
        assert len(result) == 2
        assert result[1]["answer"] == "B"
        assert result[2]["answer"] == "D"


# ---- detect_extraction_flags ----


class TestDetectExtractionFlags:
    """Tests for extraction flag detection."""

    def test_no_question_found(self) -> None:
        """Should flag when question not found."""
        flags = detect_extraction_flags(None, {}, question_found=False)
        assert "no_question_found" in flags

    def test_no_choices(self) -> None:
        """Should flag when no choices extracted."""
        flags = detect_extraction_flags("Some text", {}, question_found=True)
        assert "no_choices" in flags

    def test_image_dependent(self) -> None:
        """Should flag image-dependent questions."""
        flags = detect_extraction_flags(
            "See the diagram <!-- image --> below",
            {"A": "x"},
            question_found=True,
        )
        assert "image_dependent" in flags

    def test_commentary_detection(self) -> None:
        """Should flag commentary blocks."""
        flags = detect_extraction_flags(
            "45% des candidats ont répondu correctement",
            {"A": "x"},
            question_found=True,
        )
        assert "commentary" in flags

    def test_annulled_detection(self) -> None:
        """Should flag annulled questions."""
        flags = detect_extraction_flags(
            "Cette question annulée ne compte pas",
            {"A": "x"},
            question_found=True,
        )
        assert "annulled" in flags

    def test_multi_block_detection(self) -> None:
        """Should flag blocks with internal ## Question N headers."""
        flags = detect_extraction_flags(
            "Text\n## Question 5 : Another question\nMore text",
            {"A": "x"},
            question_found=True,
        )
        assert "multi_block" in flags

    def test_no_multi_block_for_subheadings(self) -> None:
        """Should NOT flag Docling sub-headings (## Quelle...)."""
        flags = detect_extraction_flags(
            "Scenario text\n## Quelle UV doit-il valider ?\n- a) A",
            {"A": "x"},
            question_found=True,
        )
        assert "multi_block" not in flags

    def test_clean_block_no_flags(self) -> None:
        """Should return empty flags for clean block."""
        flags = detect_extraction_flags(
            "Normal question text with choices",
            {"A": "x", "B": "y"},
            question_found=True,
        )
        assert len(flags) == 0


# ---- reextract_question ----


class TestReextractQuestion:
    """Tests for single question re-extraction."""

    @pytest.fixture()
    def sample_markdown(self) -> str:
        """Markdown simulating a full annales document."""
        return """
## UVR - session de décembre 2021

Question 1 : Lors d'une partie d'échecs, un joueur réclame la nulle. Que décidez-vous?

- a) La partie continue
- b) La nulle est accordée
- c) L'arbitre doit vérifier
- d) Le joueur est pénalisé

Question 2 : Quelle est la durée minimale?

- a) 30 minutes
- b) 60 minutes
- c) 15 minutes

## UVR session de décembre 2021 - Grille des réponses

| Question | Réponse | Articles | Taux |
|----------|---------|----------|------|
| 1 | C | Art. 9.2.1 | 72% |
| 2 | B | Art. 3.1 | 85% |

## UVR session de décembre 2021 - Corrigé détaillé

Question 1 : Lors d'une partie d'échecs...

- a) La partie continue
- b) La nulle est accordée
- c) L'arbitre doit vérifier
- d) Le joueur est pénalisé

Article 9.2.1 du Livre de l'Arbitre

L'arbitre doit vérifier la position car la réclamation de nulle nécessite une vérification selon les règles.

Question 2 : Quelle est la durée minimale?

- a) 30 minutes
- b) 60 minutes
- c) 15 minutes

Article 3.1 du règlement FFE

La durée minimale est de 60 minutes par joueur pour une partie classique.
"""

    def test_extracts_question_full(self, sample_markdown: str) -> None:
        """Should extract full question text."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        assert result["question_full"] != ""
        assert "nulle" in result["question_full"].lower()

    def test_extracts_choices(self, sample_markdown: str) -> None:
        """Should extract all QCM choices."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        assert len(result["choices"]) == 4
        assert "A" in result["choices"]
        assert "D" in result["choices"]

    def test_extracts_mcq_answer(self, sample_markdown: str) -> None:
        """Should extract correct answer letter from grille."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        assert result["mcq_answer"] == "C"

    def test_extracts_answer_text(self, sample_markdown: str) -> None:
        """Should derive answer text from correct choice."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        assert result["answer_text_from_choice"] is not None
        assert "vérifier" in result["answer_text_from_choice"].lower()

    def test_extracts_article_reference(self, sample_markdown: str) -> None:
        """Should extract article reference from corrigé."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        assert result["article_reference"] is not None
        assert "9.2.1" in result["article_reference"]

    def test_extracts_explanation(self, sample_markdown: str) -> None:
        """Should extract explanation from corrigé."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        assert result["answer_explanation"] is not None
        assert "position" in result["answer_explanation"].lower()

    def test_extracts_success_rate(self, sample_markdown: str) -> None:
        """Should extract success rate from grille."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        assert result["success_rate"] == 0.72

    def test_no_flags_for_clean_question(self, sample_markdown: str) -> None:
        """Should have no flags for clean extraction."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        assert result["extraction_flags"] == []

    def test_missing_question_flags(self, sample_markdown: str) -> None:
        """Should flag missing question number."""
        result = reextract_question("dec2021", "UVR", 99, "test:id:99", sample_markdown)
        assert "no_question_found" in result["extraction_flags"]

    def test_schema_output_fields(self, sample_markdown: str) -> None:
        """Should return all 9 required fields."""
        result = reextract_question("dec2021", "UVR", 1, "test:id:1", sample_markdown)
        required_fields = [
            "id",
            "question_full",
            "choices",
            "mcq_answer",
            "answer_text_from_choice",
            "answer_explanation",
            "article_reference",
            "success_rate",
            "extraction_flags",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_question_2(self, sample_markdown: str) -> None:
        """Should correctly extract second question."""
        result = reextract_question("dec2021", "UVR", 2, "test:id:2", sample_markdown)
        assert "durée" in result["question_full"].lower()
        assert result["mcq_answer"] == "B"
        assert result["success_rate"] == 0.85


# ---- reextract_all ----


class TestReextractAll:
    """Tests for batch re-extraction."""

    @pytest.fixture()
    def gs_with_markdown(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create minimal GS + docling file for testing."""
        # Docling JSON
        corpus_base = tmp_path / "corpus"
        subdir = corpus_base / "annales_all"
        subdir.mkdir(parents=True)

        markdown = """
## UVR - session de décembre 2021

Question 1 : Test question about rules?

- a) Choice A
- b) Choice B
- c) Choice C
- d) Choice D

## UVR session de décembre 2021 - Grille des réponses

| Question | Réponse | Articles | Taux |
|----------|---------|----------|------|
| 1 | A | Art. 1.1 | 80% |

## UVR session de décembre 2021 - Corrigé détaillé

Question 1 : Test question about rules?

- a) Choice A
- b) Choice B
- c) Choice C
- d) Choice D

Article 1.1 du règlement des compétitions

L'explication détaillée de la réponse correcte et la justification complète.
"""
        docling_file = subdir / "Annales-Session-decembre-2021.json"
        docling_file.write_text(
            json.dumps({"markdown": markdown}),
            encoding="utf-8",
        )

        # Gold standard JSON
        gs_data = {
            "questions": [
                {
                    "id": "ffe:annales:UVR:1:abc",
                    "question": "Test?",
                    "expected_answer": "A",
                    "metadata": {
                        "annales_source": {
                            "session": "dec2021",
                            "uv": "UVR",
                            "question_num": 1,
                        },
                    },
                },
                {
                    "id": "ffe:human:UVR:1:xyz",
                    "question": "Human question",
                    "expected_answer": "Human answer",
                    "metadata": {},
                },
            ],
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        return gs_path, corpus_base

    def test_processes_annales_only(self, gs_with_markdown: tuple[Path, Path]) -> None:
        """Should process only annales questions (skip human)."""
        gs_path, corpus_base = gs_with_markdown
        results, report = reextract_all(gs_path, corpus_base)
        assert report["total_processed"] == 1
        assert results[0]["id"] == "ffe:annales:UVR:1:abc"

    def test_report_structure(self, gs_with_markdown: tuple[Path, Path]) -> None:
        """Should generate report with required fields."""
        gs_path, corpus_base = gs_with_markdown
        _, report = reextract_all(gs_path, corpus_base)
        assert "total_processed" in report
        assert "total_skipped" in report
        assert "total_with_flags" in report
        assert "flag_counts" in report
        assert "by_session" in report

    def test_dry_run_limit(self, gs_with_markdown: tuple[Path, Path]) -> None:
        """Should respect dry-run limit."""
        gs_path, corpus_base = gs_with_markdown
        results, report = reextract_all(gs_path, corpus_base, dry_run_limit=1)
        assert report["total_processed"] == 1

    def test_extracts_valid_data(self, gs_with_markdown: tuple[Path, Path]) -> None:
        """Should extract actual question data."""
        gs_path, corpus_base = gs_with_markdown
        results, _ = reextract_all(gs_path, corpus_base)
        r = results[0]
        assert r["question_full"] != ""
        assert r["mcq_answer"] == "A"
        assert r["success_rate"] == 0.80
        assert len(r["choices"]) == 4


# ---- _aggregate_by_session ----


class TestAggregateBySession:
    """Tests for session-level aggregation."""

    def test_aggregates_correctly(self) -> None:
        """Should aggregate stats by session."""
        results: list[dict[str, Any]] = [
            {
                "id": "q1",
                "question_full": "text",
                "choices": {"A": "a"},
                "mcq_answer": "A",
                "answer_explanation": "expl",
                "extraction_flags": [],
            },
            {
                "id": "q2",
                "question_full": "",
                "choices": {},
                "mcq_answer": None,
                "answer_explanation": None,
                "extraction_flags": ["no_question_found"],
            },
        ]
        gs_questions: list[dict[str, Any]] = [
            {
                "id": "q1",
                "metadata": {"annales_source": {"session": "dec2021"}},
            },
            {
                "id": "q2",
                "metadata": {"annales_source": {"session": "dec2021"}},
            },
        ]
        agg = _aggregate_by_session(results, gs_questions)
        assert "dec2021" in agg
        assert agg["dec2021"]["total"] == 2
        assert agg["dec2021"]["with_full_text"] == 1
        assert agg["dec2021"]["flagged"] == 1

    def test_multiple_sessions(self) -> None:
        """Should separate stats across sessions."""
        results: list[dict[str, Any]] = [
            {
                "id": "q1",
                "question_full": "t",
                "choices": {},
                "mcq_answer": None,
                "answer_explanation": None,
                "extraction_flags": [],
            },
            {
                "id": "q2",
                "question_full": "t",
                "choices": {},
                "mcq_answer": None,
                "answer_explanation": None,
                "extraction_flags": [],
            },
        ]
        gs_questions: list[dict[str, Any]] = [
            {"id": "q1", "metadata": {"annales_source": {"session": "dec2021"}}},
            {"id": "q2", "metadata": {"annales_source": {"session": "jun2022"}}},
        ]
        agg = _aggregate_by_session(results, gs_questions)
        assert len(agg) == 2
        assert agg["dec2021"]["total"] == 1
        assert agg["jun2022"]["total"] == 1


# ---- Missing session handling ----


class TestMissingSessionHandling:
    """Tests for handling sessions with no docling file."""

    def test_missing_docling_file(self, tmp_path: Path) -> None:
        """Should flag questions when docling file is missing."""
        gs_data = {
            "questions": [
                {
                    "id": "ffe:annales:UVR:1:abc",
                    "question": "Q?",
                    "expected_answer": "A",
                    "metadata": {
                        "annales_source": {
                            "session": "dec2021",
                            "uv": "UVR",
                            "question_num": 1,
                        },
                    },
                },
            ],
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        # No docling file exists
        results, report = reextract_all(gs_path, tmp_path)
        assert len(results) == 1
        assert "no_question_found" in results[0]["extraction_flags"]
