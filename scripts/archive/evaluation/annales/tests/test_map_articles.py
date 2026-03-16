"""
Tests for map_articles_to_corpus module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

from scripts.evaluation.annales.map_articles_to_corpus import (
    _extract_article_number,
    _extract_section_info,
    map_article_to_document,
)


class TestExtractArticleNumber:
    """Tests for article number extraction."""

    def test_simple_article(self) -> None:
        """Should extract simple article number."""
        result = _extract_article_number("Article 1.3 des règles du jeu")
        assert result == "1.3"

    def test_nested_article(self) -> None:
        """Should extract nested article number."""
        result = _extract_article_number("Article 4.2.1 des règles")
        assert result == "4.2.1"

    def test_deep_nested_article(self) -> None:
        """Should extract deeply nested article number."""
        result = _extract_article_number("Article 6.11.4 du règlement")
        assert result == "6.11.4"

    def test_chapter_reference(self) -> None:
        """Should extract chapter reference."""
        result = _extract_article_number("Chapitre 2.1 du LA")
        assert result == "Ch.2.1"

    def test_no_article_number(self) -> None:
        """Should return None for text without article number."""
        result = _extract_article_number("General rules")
        assert result is None


class TestExtractSectionInfo:
    """Tests for section info extraction."""

    def test_commentary_detection(self) -> None:
        """Should detect commentary references."""
        result = _extract_section_info("Commentaire Article 3.7.3.5")
        assert result["is_commentary"] is True

    def test_multiple_refs_detection(self) -> None:
        """Should detect multiple references."""
        result = _extract_section_info("Article 4.2.1 et 4.3.1")
        assert result["has_multiple_refs"] is True

    def test_chapter_extraction(self) -> None:
        """Should extract chapter number."""
        result = _extract_section_info("LA - Chapitre 5.2 - Préparation")
        assert result["chapter"] == "5.2"


class TestMapArticleToDocument:
    """Tests for article to document mapping."""

    def test_regles_du_jeu_maps_to_la(self) -> None:
        """Should map règles du jeu to LA."""
        result = map_article_to_document("Article 1.3 des règles du jeu")
        assert result["document"] == "LA-octobre2025.pdf"
        assert result["category"] == "regles_jeu"
        assert result["confidence"] >= 0.7

    def test_r01_maps_to_regles_generales(self) -> None:
        """Should map R01 to Règles générales."""
        result = map_article_to_document("R01 - Article 5.2")
        assert result["document"] == "R01_2025_26_Regles_generales.pdf"
        assert result["category"] == "regles_ffe"

    def test_r03_maps_to_competitions(self) -> None:
        """Should map R03 to Compétitions homologuées."""
        result = map_article_to_document("R03 - Compétitions homologuées - Article 2.2")
        assert result["document"] == "R03_2025_26_Competitions_homologuees.pdf"
        assert result["category"] == "competitions"

    def test_a02_maps_to_clubs(self) -> None:
        """Should map A02 to Championnat des clubs."""
        result = map_article_to_document("A02 - 3.7 Composition des équipes")
        assert result["document"] == "A02_2025_26_Championnat_de_France_des_Clubs.pdf"
        assert result["category"] == "interclubs"

    def test_c01_maps_to_coupe(self) -> None:
        """Should map C01 to Coupe de France."""
        result = map_article_to_document("C01 - 3.2. Couleurs")
        assert result["document"] == "C01_2025_26_Coupe_de_France.pdf"
        assert result["category"] == "coupes"

    def test_la_chapitre_maps_to_la(self) -> None:
        """Should map LA Chapitre to LA."""
        result = map_article_to_document("LA - Chapitre 1.3 Le Règlement intérieur")
        assert result["document"] == "LA-octobre2025.pdf"

    def test_commentary_maps_to_la(self) -> None:
        """Should map commentary to LA."""
        result = map_article_to_document("Commentaire article 6.11.4 des règles du jeu")
        assert result["document"] == "LA-octobre2025.pdf"
        assert result["article_num"] == "6.11.4"

    def test_empty_reference(self) -> None:
        """Should handle empty reference."""
        result = map_article_to_document("")
        assert result["document"] is None
        assert result["confidence"] == 0.0

    def test_unknown_document_code(self) -> None:
        """Should handle unknown document codes with fallback."""
        result = map_article_to_document("X99 - Some unknown article")
        assert result["confidence"] <= 0.5

    def test_annexe_maps_to_la(self) -> None:
        """Should map Annexe to LA."""
        result = map_article_to_document("Annexe A - Notation algébrique")
        assert result["document"] == "LA-octobre2025.pdf"

    def test_article_only_fallback(self) -> None:
        """Should fallback to LA for generic article mention."""
        result = map_article_to_document("Article général sans précision")
        # Should still attempt LA as fallback
        assert result["document"] is not None or result["confidence"] > 0

    def test_bare_article_number(self) -> None:
        """Should map bare article numbers to LA."""
        result = map_article_to_document("9.6.2")
        assert result["document"] == "LA-octobre2025.pdf"
        assert result["article_num"] == "9.6.2"

    def test_abbreviated_art_format(self) -> None:
        """Should map 'Art 3.8.2' format to LA."""
        result = map_article_to_document("Art 3.8.2.")
        assert result["document"] == "LA-octobre2025.pdf"
        assert result["article_num"] == "3.8.2"

    def test_multiple_bare_articles(self) -> None:
        """Should map first article from multiple bare numbers."""
        result = map_article_to_document("7.5.4 4.4.2 4.6.2")
        assert result["document"] == "LA-octobre2025.pdf"
        assert result["article_num"] == "7.5.4"
