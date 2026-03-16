"""
Tests unitaires pour extract_docling.py

ISO Reference: ISO/IEC 29119 - Test execution

Tests pour l'extraction PDF avec Docling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExtractTableContent:
    """Tests pour _extract_table_content()."""

    def test_table_with_no_data(self):
        """Retourne structure vide si table.data est None."""
        from scripts.pipeline.extract_docling import _extract_table_content

        mock_table = MagicMock()
        mock_table.data = None
        mock_doc = MagicMock()

        result = _extract_table_content(mock_table, mock_doc)

        assert result["num_rows"] == 0
        assert result["num_cols"] == 0
        assert result["headers"] == []
        assert result["rows"] == []
        assert result["text"] == ""

    def test_table_with_empty_grid(self):
        """Retourne structure vide si grid est vide."""
        from scripts.pipeline.extract_docling import _extract_table_content

        mock_data = MagicMock()
        mock_data.num_rows = 0
        mock_data.num_cols = 0
        mock_data.grid = None

        mock_table = MagicMock()
        mock_table.data = mock_data

        result = _extract_table_content(mock_table, MagicMock())

        assert result["num_rows"] == 0
        assert result["num_cols"] == 0
        assert result["text"] == ""

    def test_table_with_simple_grid(self):
        """Extrait correctement une grille simple."""
        from scripts.pipeline.extract_docling import _extract_table_content

        # Create mock cells
        cell1 = MagicMock()
        cell1.text = "Header1"
        cell1.column_header = True

        cell2 = MagicMock()
        cell2.text = "Header2"
        cell2.column_header = True

        cell3 = MagicMock()
        cell3.text = "Value1"
        cell3.column_header = False

        cell4 = MagicMock()
        cell4.text = "Value2"
        cell4.column_header = False

        mock_data = MagicMock()
        mock_data.num_rows = 2
        mock_data.num_cols = 2
        mock_data.grid = [[cell1, cell2], [cell3, cell4]]

        mock_table = MagicMock()
        mock_table.data = mock_data

        result = _extract_table_content(mock_table, MagicMock())

        assert result["num_rows"] == 2
        assert result["num_cols"] == 2
        assert result["headers"] == ["Header1", "Header2"]
        assert result["rows"] == [["Value1", "Value2"]]
        assert "Header1 | Header2" in result["text"]
        assert "Value1 | Value2" in result["text"]

    def test_table_without_header_markers(self):
        """Traite la premiere ligne comme header si pas de markers."""
        from scripts.pipeline.extract_docling import _extract_table_content

        cell1 = MagicMock()
        cell1.text = "Col1"
        cell1.column_header = False

        cell2 = MagicMock()
        cell2.text = "Col2"
        cell2.column_header = False

        cell3 = MagicMock()
        cell3.text = "Data1"
        cell3.column_header = False

        cell4 = MagicMock()
        cell4.text = "Data2"
        cell4.column_header = False

        mock_data = MagicMock()
        mock_data.num_rows = 2
        mock_data.num_cols = 2
        mock_data.grid = [[cell1, cell2], [cell3, cell4]]

        mock_table = MagicMock()
        mock_table.data = mock_data

        result = _extract_table_content(mock_table, MagicMock())

        # First row is treated as headers by default
        assert result["headers"] == ["Col1", "Col2"]
        assert result["rows"] == [["Data1", "Data2"]]

    def test_table_with_none_cell_text(self):
        """Gere les cellules avec text=None."""
        from scripts.pipeline.extract_docling import _extract_table_content

        cell = MagicMock()
        cell.text = None
        cell.column_header = False

        mock_data = MagicMock()
        mock_data.num_rows = 1
        mock_data.num_cols = 1
        mock_data.grid = [[cell]]

        mock_table = MagicMock()
        mock_table.data = mock_data

        result = _extract_table_content(mock_table, MagicMock())

        # Should not crash, should return empty string for None
        assert result["headers"] == [""]


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
        """Teste l'extraction avec tables utilisant la nouvelle logique grid."""
        from scripts.pipeline.extract_docling import extract_pdf_docling

        # Setup mock table with proper data structure
        mock_cell1 = MagicMock()
        mock_cell1.text = "Header"
        mock_cell1.column_header = True

        mock_cell2 = MagicMock()
        mock_cell2.text = "Value"
        mock_cell2.column_header = False

        mock_data = MagicMock()
        mock_data.num_rows = 2
        mock_data.num_cols = 1
        mock_data.grid = [[mock_cell1], [mock_cell2]]

        mock_table = MagicMock()
        mock_table.data = mock_data

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
            table = result["tables"][0]
            assert table["id"] == "test_table-table0"
            assert table["source"] == "test_table.pdf"
            assert table["num_rows"] == 2
            assert table["num_cols"] == 1
            assert table["headers"] == ["Header"]
            assert table["rows"] == [["Value"]]
            assert "Header" in table["text"]
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
    def test_corpus_extraction(self, mock_extract: MagicMock, tmp_path: Path):
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

    @patch("scripts.pipeline.extract_docling.extract_pdf_docling")
    def test_corpus_extraction_counts_tables(
        self, mock_extract: MagicMock, tmp_path: Path
    ):
        """Verifie que total_tables est accumule correctement."""
        from scripts.pipeline.extract_docling import extract_corpus_docling

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.pdf").write_bytes(b"%PDF-1.4")
        (input_dir / "file2.pdf").write_bytes(b"%PDF-1.4")

        mock_extract.side_effect = [
            {
                "filename": "file1.pdf",
                "markdown": "",
                "tables": [{}],
                "total_tables": 1,
                "extraction_date": "",
                "extractor": "docling",
            },
            {
                "filename": "file2.pdf",
                "markdown": "",
                "tables": [{}, {}],
                "total_tables": 2,
                "extraction_date": "",
                "extractor": "docling",
            },
        ]

        report = extract_corpus_docling(input_dir, tmp_path / "out")

        assert report["total_tables"] == 3


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

    @pytest.mark.slow
    def test_real_pdf_with_tables(self):
        """Extrait un PDF avec tables reelles."""
        from scripts.pipeline.extract_docling import extract_pdf_docling

        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not found")

        result = extract_pdf_docling(pdf_path)

        assert result["total_tables"] >= 1
        if result["tables"]:
            table = result["tables"][0]
            assert "num_rows" in table
            assert "num_cols" in table
            assert "headers" in table
            assert "rows" in table
            assert "text" in table
            # Verify content is not empty (the bug we fixed)
            assert len(table["text"]) > 0 or table["num_rows"] == 0
