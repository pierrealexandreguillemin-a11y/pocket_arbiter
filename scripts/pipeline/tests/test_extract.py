"""
Tests unitaires pour extract_pdf.py

ISO Reference: ISO/IEC 29119 - Test execution

Note: Ce fichier teste UNIQUEMENT les fonctions implementees.
Les fonctions stub (NotImplementedError) n'ont PAS de tests car elles
ne sont pas encore implementees. Les tests seront ajoutes lors de
l'implementation reelle en Phase 1A.

Fonctions testees:
- detect_section() - Implementation complete
- validate_pdf() - Implementation complete

Fonctions NON testees (stubs):
- extract_pdf() - A implementer
- extract_corpus() - A implementer
"""

from pathlib import Path

import pytest

from scripts.pipeline.extract_pdf import (
    detect_section,
    validate_pdf,
)


class TestDetectSection:
    """Tests pour detect_section() - fonction implementee."""

    def test_detect_article(self):
        """Detecte un titre d'article."""
        text = "Article 4.1 - Le toucher-jouer\nContenu..."
        result = detect_section(text)
        assert result == "Article 4.1 - Le toucher-jouer"

    def test_detect_chapitre(self):
        """Detecte un titre de chapitre."""
        text = "Chapitre 3 - Les regles du jeu"
        result = detect_section(text)
        assert result == "Chapitre 3 - Les regles du jeu"

    def test_detect_numbered_section(self):
        """Detecte une section numerotee."""
        text = "4.1.2 Cas particuliers"
        result = detect_section(text)
        assert result == "4.1.2 Cas particuliers"

    def test_no_section_detected(self):
        """Retourne None si pas de section."""
        text = "Ceci est un texte normal sans section."
        result = detect_section(text)
        assert result is None

    def test_empty_text(self):
        """Gere le texte vide."""
        assert detect_section("") is None
        assert detect_section(None) is None


class TestValidatePdf:
    """Tests pour validate_pdf() - fonction implementee."""

    def test_nonexistent_file(self, tmp_path: Path):
        """Retourne False pour fichier inexistant."""
        fake_path = tmp_path / "nonexistent.pdf"
        assert validate_pdf(fake_path) is False

    def test_non_pdf_extension(self, tmp_path: Path):
        """Retourne False pour extension non-PDF."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("Not a PDF")
        assert validate_pdf(txt_file) is False

    def test_valid_pdf_magic_number(self, tmp_path: Path):
        """Retourne True pour fichier avec magic number PDF valide."""
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        assert validate_pdf(pdf_file) is True

    def test_invalid_pdf_magic_number(self, tmp_path: Path):
        """Retourne False pour fichier .pdf sans magic number valide."""
        pdf_file = tmp_path / "fake.pdf"
        pdf_file.write_bytes(b"This is not a PDF file")
        assert validate_pdf(pdf_file) is False


class TestExtractPdf:
    """Tests pour extract_pdf() - fonction implementee."""

    @pytest.mark.slow
    def test_extract_real_pdf(self):
        """Extrait un PDF reel du corpus (ISO 29119 integration test)."""
        from scripts.pipeline.extract_pdf import extract_pdf

        pdf_path = Path("corpus/fr/LA-octobre2025.pdf")
        if not pdf_path.exists():
            pytest.skip("corpus/fr/LA-octobre2025.pdf not found")

        result = extract_pdf(pdf_path)

        assert result["filename"] == "LA-octobre2025.pdf"
        assert result["total_pages"] >= 200  # Flexible, not hardcoded
        assert len(result["pages"]) > 0
        assert "extraction_date" in result

    def test_extract_pdf_not_found(self, tmp_path: Path):
        """Leve FileNotFoundError si PDF n'existe pas."""
        from scripts.pipeline.extract_pdf import extract_pdf

        with pytest.raises(FileNotFoundError):
            extract_pdf(tmp_path / "nonexistent.pdf")

    def test_extract_pdf_not_pdf(self, tmp_path: Path):
        """Leve ValueError si fichier n'est pas PDF."""
        from scripts.pipeline.extract_pdf import extract_pdf

        txt_file = tmp_path / "file.txt"
        txt_file.write_text("Not a PDF")

        with pytest.raises(ValueError):
            extract_pdf(txt_file)

    @pytest.mark.slow
    def test_extract_page_structure(self):
        """Verifie structure des pages extraites (ISO 29119)."""
        from scripts.pipeline.extract_pdf import extract_pdf

        pdf_path = Path("corpus/fr/LA-octobre2025.pdf")
        if not pdf_path.exists():
            pytest.skip("corpus/fr/LA-octobre2025.pdf not found")

        result = extract_pdf(pdf_path)
        page = result["pages"][0]

        assert "page_num" in page
        assert "text" in page
        assert "section" in page
        assert page["page_num"] >= 1
