"""
Tests unitaires pour extract_docling.py

ISO Reference: ISO/IEC 29119 - Test execution

Tests pour l'extraction PDF avec Docling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExtractPdfDocling:
    """Tests pour extract_pdf_docling()."""

    def test_file_not_found(self):
        """Leve FileNotFoundError pour fichier inexistant."""
        from scripts.pipeline.extract_docling import extract_pdf_docling

        with pytest.raises(FileNotFoundError, match="PDF not found"):
            extract_pdf_docling(Path("/nonexistent/file.pdf"))

    @patch("scripts.pipeline.extract_docling.DocumentConverter")
    def test_extraction_success(self, mock_converter_class: MagicMock):
        """Teste une extraction reussie avec mock."""
        from scripts.pipeline.extract_docling import extract_pdf_docling

        # Setup mock
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Test\nContent"
        mock_doc.tables = []

        mock_result = MagicMock()
        mock_result.document = mock_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        # Create temp file
        tmp_file = Path("test_temp.pdf")
        tmp_file.write_bytes(b"%PDF-1.4 test")

        try:
            result = extract_pdf_docling(tmp_file)

            assert result["filename"] == "test_temp.pdf"
            assert "# Test" in result["markdown"]
            assert result["total_tables"] == 0
            assert result["extractor"] == "docling"
            assert "extraction_date" in result
        finally:
            tmp_file.unlink(missing_ok=True)

    @patch("scripts.pipeline.extract_docling.DocumentConverter")
    def test_extraction_with_tables(self, mock_converter_class: MagicMock):
        """Teste l'extraction avec tables."""
        from scripts.pipeline.extract_docling import extract_pdf_docling

        # Setup mock with tables
        mock_table = MagicMock()
        mock_table.export_to_markdown.return_value = "| Col1 | Col2 |\n|---|---|\n| A | B |"

        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Doc with table"
        mock_doc.tables = [mock_table]

        mock_result = MagicMock()
        mock_result.document = mock_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        tmp_file = Path("test_table.pdf")
        tmp_file.write_bytes(b"%PDF-1.4 test")

        try:
            result = extract_pdf_docling(tmp_file)

            assert result["total_tables"] == 1
            assert len(result["tables"]) == 1
            assert "test_table-table0" in result["tables"][0]["id"]
        finally:
            tmp_file.unlink(missing_ok=True)

    @patch("scripts.pipeline.extract_docling.DocumentConverter")
    def test_extraction_failure(self, mock_converter_class: MagicMock):
        """Teste la gestion des erreurs d'extraction."""
        from scripts.pipeline.extract_docling import extract_pdf_docling

        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("PDF parsing failed")
        mock_converter_class.return_value = mock_converter

        tmp_file = Path("test_error.pdf")
        tmp_file.write_bytes(b"%PDF-1.4 test")

        try:
            with pytest.raises(RuntimeError, match="Docling extraction failed"):
                extract_pdf_docling(tmp_file)
        finally:
            tmp_file.unlink(missing_ok=True)


class TestExtractCorpusDocling:
    """Tests pour extract_corpus_docling()."""

    def test_input_dir_not_found(self, tmp_path: Path):
        """Leve FileNotFoundError pour dossier inexistant."""
        from scripts.pipeline.extract_docling import extract_corpus_docling

        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            extract_corpus_docling(
                input_dir=tmp_path / "nonexistent",
                output_dir=tmp_path / "output",
            )

    @patch("scripts.pipeline.extract_docling.extract_pdf_docling")
    def test_corpus_extraction(
        self, mock_extract: MagicMock, tmp_path: Path
    ):
        """Teste l'extraction d'un corpus."""
        from scripts.pipeline.extract_docling import extract_corpus_docling

        # Create test input directory with PDF
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test1.pdf").write_bytes(b"%PDF-1.4 test")
        (input_dir / "test2.pdf").write_bytes(b"%PDF-1.4 test")

        output_dir = tmp_path / "output"

        # Mock extraction result
        mock_extract.return_value = {
            "filename": "test.pdf",
            "markdown": "# Test",
            "tables": [],
            "total_tables": 0,
            "extraction_date": "2026-01-19T12:00:00",
            "extractor": "docling",
        }

        report = extract_corpus_docling(
            input_dir=input_dir,
            output_dir=output_dir,
            corpus_name="test",
        )

        assert report["corpus"] == "test"
        assert report["files_processed"] == 2
        assert report["extractor"] == "docling"
        assert len(report["errors"]) == 0
        assert (output_dir / "extraction_report.json").exists()

    @patch("scripts.pipeline.extract_docling.extract_pdf_docling")
    def test_corpus_extraction_with_errors(
        self, mock_extract: MagicMock, tmp_path: Path
    ):
        """Teste la gestion des erreurs pendant l'extraction corpus."""
        from scripts.pipeline.extract_docling import extract_corpus_docling

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "good.pdf").write_bytes(b"%PDF-1.4 test")
        (input_dir / "bad.pdf").write_bytes(b"%PDF-1.4 test")

        output_dir = tmp_path / "output"

        # First call succeeds, second fails
        mock_extract.side_effect = [
            {
                "filename": "good.pdf",
                "markdown": "# Good",
                "tables": [],
                "total_tables": 0,
                "extraction_date": "2026-01-19",
                "extractor": "docling",
            },
            RuntimeError("Extraction failed for bad.pdf"),
        ]

        report = extract_corpus_docling(
            input_dir=input_dir,
            output_dir=output_dir,
        )

        assert report["files_processed"] == 1
        assert len(report["errors"]) == 1
        assert "bad.pdf" in report["errors"][0]


class TestIntegration:
    """Tests d'integration avec vrais PDFs."""

    @pytest.mark.slow
    def test_real_pdf_extraction(self):
        """Extrait un PDF reel du corpus (test lent)."""
        from scripts.pipeline.extract_docling import extract_pdf_docling

        pdf_path = Path("corpus/fr/Compétitions/E02-Le_classement_rapide.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not found")

        result = extract_pdf_docling(pdf_path)

        assert result["filename"] == "E02-Le_classement_rapide.pdf"
        assert len(result["markdown"]) > 100
        assert result["extractor"] == "docling"
        assert "CLASSEMENT RAPIDE" in result["markdown"]
