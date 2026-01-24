"""
Tests for validate_answers module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

from scripts.evaluation.annales.validate_answers import (
    ARTICLE_PATTERN,
    CHAPTER_PATTERN,
    PREAMBULE_PATTERN,
    R01_ARTICLE_PATTERN,
    R01_SECTION_PATTERN,
    REGULATION_PATTERN,
    build_article_page_index,
    extract_article_from_reference,
    find_answer_in_chunks,
)


class TestPatterns:
    """Tests for regex patterns."""

    def test_article_pattern_matches_dotted(self) -> None:
        """Should match articles like 1.3, 4.2.1."""
        assert ARTICLE_PATTERN.search("Article 1.3") is not None
        assert ARTICLE_PATTERN.search("Voir 4.2.1").group(1) == "4.2.1"

    def test_article_pattern_no_match_single(self) -> None:
        """Should not match single numbers."""
        # Returns None if no match found
        match = ARTICLE_PATTERN.search("Article 5 du règlement")
        assert match is None

    def test_chapter_pattern_matches(self) -> None:
        """Should match chapter references."""
        assert CHAPTER_PATTERN.search("Chapitre 8").group(1) == "8"
        assert CHAPTER_PATTERN.search("Chap. 3").group(1) == "3"

    def test_preambule_pattern_matches(self) -> None:
        """Should match préambule variations."""
        assert PREAMBULE_PATTERN.search("Préambule") is not None
        assert PREAMBULE_PATTERN.search("Preambule") is not None

    def test_r01_section_pattern_matches(self) -> None:
        """Should match R01 section patterns."""
        match = R01_SECTION_PATTERN.search("R01 - 2. Statut")
        assert match is not None
        assert match.group(1) == "2"

    def test_r01_article_pattern_matches(self) -> None:
        """Should match R01 article patterns."""
        match = R01_ARTICLE_PATTERN.search("R01 article 5.3")
        assert match is not None
        assert match.group(1) == "5.3"

    def test_r01_article_pattern_with_dot(self) -> None:
        """Should match R.01 format."""
        match = R01_ARTICLE_PATTERN.search("R.01 - art. 3")
        assert match is not None
        assert match.group(1) == "3"

    def test_regulation_pattern_matches_r02(self) -> None:
        """Should match R02 references."""
        match = REGULATION_PATTERN.search("R02 - 5.3")
        assert match is not None
        assert match.group(1).upper() == "R"
        assert match.group(2) == "02"
        assert match.group(3) == "5.3"

    def test_regulation_pattern_matches_a02(self) -> None:
        """Should match A02 (annexe) references."""
        match = REGULATION_PATTERN.search("A02 - 3.1")
        assert match is not None
        assert match.group(1).upper() == "A"


class TestBuildArticlePageIndex:
    """Tests for build_article_page_index function."""

    def test_indexes_single_article(self) -> None:
        """Should index a single article reference."""
        chunks = [
            {"source": "test.pdf", "text": "Article 1.3 des règles", "pages": [5]}
        ]
        index = build_article_page_index(chunks)
        assert "test.pdf|1.3" in index
        assert 5 in index["test.pdf|1.3"]["pages"]

    def test_indexes_multiple_articles(self) -> None:
        """Should index multiple articles from same chunk."""
        chunks = [
            {"source": "test.pdf", "text": "Articles 1.3 et 2.4", "pages": [10, 11]}
        ]
        index = build_article_page_index(chunks)
        assert "test.pdf|1.3" in index
        assert "test.pdf|2.4" in index

    def test_indexes_chapter_pattern(self) -> None:
        """Should index chapter references."""
        chunks = [
            {"source": "test.pdf", "text": "Chapitre 8 - Temps", "pages": [20]}
        ]
        index = build_article_page_index(chunks)
        assert "test.pdf|Ch.8" in index

    def test_indexes_preambule(self) -> None:
        """Should index préambule reference."""
        chunks = [
            {"source": "test.pdf", "text": "Le Préambule établit", "pages": [1]}
        ]
        index = build_article_page_index(chunks)
        assert "test.pdf|Préambule" in index

    def test_indexes_r01_section(self) -> None:
        """Should index R01 section reference."""
        chunks = [
            {"source": "test.pdf", "text": "R01 - 2. Statut des joueurs", "pages": [15]}
        ]
        index = build_article_page_index(chunks)
        assert "test.pdf|R01.Sec.2" in index

    def test_indexes_r01_article(self) -> None:
        """Should index R01 article reference."""
        chunks = [
            {"source": "test.pdf", "text": "R01 article 5.3 indique", "pages": [22]}
        ]
        index = build_article_page_index(chunks)
        assert "test.pdf|R01.5.3" in index

    def test_merges_pages_from_multiple_chunks(self) -> None:
        """Should merge pages from multiple chunks with same article."""
        chunks = [
            {"source": "test.pdf", "text": "Article 1.3", "pages": [5]},
            {"source": "test.pdf", "text": "Voir article 1.3", "pages": [10]},
        ]
        index = build_article_page_index(chunks)
        assert "test.pdf|1.3" in index
        assert sorted(index["test.pdf|1.3"]["pages"]) == [5, 10]

    def test_handles_empty_chunks(self) -> None:
        """Should handle empty chunks list."""
        index = build_article_page_index([])
        assert index == {}


class TestExtractArticleFromReference:
    """Tests for extract_article_from_reference function."""

    def test_extracts_dotted_article(self) -> None:
        """Should extract dotted article number."""
        keys = extract_article_from_reference("Article 1.3 des règles")
        assert "1.3" in keys

    def test_extracts_chapter(self) -> None:
        """Should extract chapter reference."""
        keys = extract_article_from_reference("Chapitre 8")
        assert "Ch.8" in keys

    def test_extracts_preambule(self) -> None:
        """Should extract préambule reference."""
        keys = extract_article_from_reference("Préambule du règlement")
        assert "Préambule" in keys

    def test_extracts_r01_section(self) -> None:
        """Should extract R01 section."""
        keys = extract_article_from_reference("R01 - 2. Statut")
        assert "R01.Sec.2" in keys

    def test_extracts_r01_article(self) -> None:
        """Should extract R01 article."""
        keys = extract_article_from_reference("R01 article 5.3")
        assert "R01.5.3" in keys

    def test_extracts_regulation_pattern(self) -> None:
        """Should extract generic regulation pattern."""
        keys = extract_article_from_reference("R02 - 3.1")
        assert "R02.3.1" in keys

    def test_extracts_multiple_keys(self) -> None:
        """Should extract multiple possible keys."""
        keys = extract_article_from_reference("Article 1.3, Chapitre 8")
        assert "1.3" in keys
        assert "Ch.8" in keys

    def test_handles_empty_reference(self) -> None:
        """Should handle empty reference."""
        keys = extract_article_from_reference("")
        assert keys == []

    def test_handles_no_match(self) -> None:
        """Should handle reference with no recognizable pattern."""
        keys = extract_article_from_reference("Voir le document")
        assert keys == []


class TestFindAnswerInChunks:
    """Tests for find_answer_in_chunks function."""

    def test_finds_exact_match(self) -> None:
        """Should find exact answer in chunk."""
        chunks = [{"text": "La réponse est vrai car l'article le confirme."}]
        result = find_answer_in_chunks("vrai car l'article", chunks)
        assert result["found"] is True
        assert result["confidence"] == 1.0

    def test_returns_not_found(self) -> None:
        """Should return not found for missing answer."""
        chunks = [{"text": "Texte sans rapport avec la question."}]
        result = find_answer_in_chunks("réponse attendue", chunks)
        assert result["found"] is False

    def test_handles_letter_only_answer(self) -> None:
        """Should handle MCQ letter-only answers."""
        chunks = [{"text": "Quelque texte"}]
        result = find_answer_in_chunks("A", chunks)
        assert result["found"] is None
        assert result["reason"] == "letter_only_answer"

    def test_case_insensitive_search(self) -> None:
        """Should match case-insensitively."""
        chunks = [{"text": "LA RÉPONSE EST ICI"}]
        result = find_answer_in_chunks("la réponse est ici", chunks)
        assert result["found"] is True

    def test_handles_empty_chunks(self) -> None:
        """Should handle empty chunks list."""
        result = find_answer_in_chunks("test", [])
        assert result["found"] is False
