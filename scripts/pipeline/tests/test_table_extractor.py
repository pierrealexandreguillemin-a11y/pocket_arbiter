"""
Tests for table_extractor.py - Camelot Table Extraction

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements

Note: PDF extraction tests are marked slow and require actual PDF files.
Unit tests use mocks for the Camelot dependency.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestDetectTableType:
    """Tests for detect_table_type function."""

    def test_cadence_detection(self):
        """Should detect time control tables."""
        from scripts.pipeline.table_extractor import detect_table_type

        headers = ["Cadence", "Temps", "Increment"]
        rows = [["Rapide", "15 min", "10 sec"]]
        assert detect_table_type(headers, rows) == "cadence"

        # Also with 'minutes' keyword
        headers = ["Type", "Minutes"]
        rows = [["Blitz", "3"]]
        assert detect_table_type(headers, rows) == "cadence"

    def test_penalty_detection(self):
        """Should detect penalty tables."""
        from scripts.pipeline.table_extractor import detect_table_type

        headers = ["Infraction", "Penalite"]
        rows = [["Retard", "Avertissement"]]
        assert detect_table_type(headers, rows) == "penalty"

        headers = ["Type", "Sanction", "Duree"]
        rows = [["Triche", "Exclusion", "1 an"]]
        assert detect_table_type(headers, rows) == "penalty"

    def test_elo_detection(self):
        """Should detect Elo rating tables."""
        from scripts.pipeline.table_extractor import detect_table_type

        headers = ["Joueur", "Elo", "Federation"]
        rows = [["Dupont", "2100", "FRA"]]
        assert detect_table_type(headers, rows) == "elo"

        headers = ["Rang", "Classement", "Points"]
        rows = [["1", "2450", "8.5"]]
        assert detect_table_type(headers, rows) == "elo"

    def test_tiebreak_detection(self):
        """Should detect tiebreak tables."""
        from scripts.pipeline.table_extractor import detect_table_type

        headers = ["Critere", "Description"]
        rows = [["Buchholz", "Somme adversaires"]]  # Avoid 'points' (elo keyword)
        assert detect_table_type(headers, rows) == "tiebreak"

        headers = ["Departage", "Ordre"]
        rows = [["Sonneborn-Berger", "1"]]
        assert detect_table_type(headers, rows) == "tiebreak"

        headers = ["Cumulatif", "Valeur"]
        rows = [["Total", "12"]]
        assert detect_table_type(headers, rows) == "tiebreak"

    def test_other_type(self):
        """Should return 'other' for unrecognized tables."""
        from scripts.pipeline.table_extractor import detect_table_type

        headers = ["Nom", "Adresse", "Telephone"]
        rows = [["Club A", "Paris", "01..."]]
        assert detect_table_type(headers, rows) == "other"


class TestTableToText:
    """Tests for table_to_text function."""

    def test_cadence_table_format(self):
        """Should format cadence table with context."""
        from scripts.pipeline.table_extractor import table_to_text

        headers = ["Type", "Temps"]
        rows = [["Rapide", "15+10"]]
        result = table_to_text(headers, rows, "cadence")

        assert "Table de cadence" in result
        assert "Type" in result
        assert "Temps" in result
        assert "Rapide" in result
        assert "15+10" in result

    def test_penalty_table_format(self):
        """Should format penalty table with context."""
        from scripts.pipeline.table_extractor import table_to_text

        headers = ["Faute", "Sanction"]
        rows = [["Retard", "Perte"]]
        result = table_to_text(headers, rows, "penalty")

        assert "penalite" in result.lower()
        assert "Faute" in result

    def test_empty_rows_handled(self):
        """Should handle tables with no rows."""
        from scripts.pipeline.table_extractor import table_to_text

        headers = ["A", "B"]
        rows = []
        result = table_to_text(headers, rows, "other")

        assert "A" in result
        assert "B" in result

    def test_multirow_format(self):
        """Should format multiple rows correctly."""
        from scripts.pipeline.table_extractor import table_to_text

        headers = ["Col1", "Col2"]
        rows = [["A", "B"], ["C", "D"], ["E", "F"]]
        result = table_to_text(headers, rows, "other")

        # All values should be present
        for val in ["A", "B", "C", "D", "E", "F"]:
            assert val in result


class TestExtractTablesFromPdf:
    """Tests for extract_tables_from_pdf function (mocked)."""

    @patch("scripts.pipeline.table_extractor.camelot")
    def test_lattice_method_used_first(self, mock_camelot):
        """Should try lattice method before stream."""
        from scripts.pipeline.table_extractor import extract_tables_from_pdf

        # Mock empty result from lattice
        mock_camelot.read_pdf.return_value = []

        result = extract_tables_from_pdf(Path("test.pdf"))

        # Should have called read_pdf at least once with lattice
        calls = mock_camelot.read_pdf.call_args_list
        assert len(calls) >= 1
        assert calls[0][1]["flavor"] == "lattice"

    @patch("scripts.pipeline.table_extractor.camelot")
    def test_stream_fallback(self, mock_camelot):
        """Should fallback to stream if lattice finds nothing."""
        from scripts.pipeline.table_extractor import extract_tables_from_pdf

        # Lattice returns nothing, stream returns tables
        mock_camelot.read_pdf.side_effect = [
            [],  # Lattice
            [Mock(df=MagicMock(), page=1, accuracy=95.0, whitespace=5.0)],  # Stream
        ]

        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.empty = True  # Will be skipped
        mock_camelot.read_pdf.return_value[0].df = mock_df if len(mock_camelot.read_pdf.return_value) > 0 else None

        result = extract_tables_from_pdf(Path("test.pdf"))

        # Should have called stream as fallback
        calls = mock_camelot.read_pdf.call_args_list
        assert len(calls) == 2
        assert calls[1][1]["flavor"] == "stream"

    @patch("scripts.pipeline.table_extractor.camelot")
    def test_table_structure_returned(self, mock_camelot):
        """Should return correct table structure."""
        from scripts.pipeline.table_extractor import extract_tables_from_pdf
        import pandas as pd

        # Create mock table with DataFrame
        mock_table = Mock()
        mock_table.df = pd.DataFrame({
            "Col1": ["Header1", "Value1"],
            "Col2": ["Header2", "Value2"],
        })
        mock_table.page = 5
        mock_table.accuracy = 95.5
        mock_table.whitespace = 3.2

        mock_camelot.read_pdf.return_value = [mock_table]

        result = extract_tables_from_pdf(Path("document.pdf"))

        assert len(result) == 1
        table = result[0]
        assert table["source"] == "document.pdf"
        assert table["page"] == 5
        assert table["accuracy"] == 95.5
        assert "headers" in table
        assert "rows" in table
        assert "text" in table
        assert "table_type" in table

    @patch("scripts.pipeline.table_extractor.camelot")
    def test_camelot_exception_handled(self, mock_camelot):
        """Should handle Camelot exceptions gracefully."""
        from scripts.pipeline.table_extractor import extract_tables_from_pdf

        mock_camelot.read_pdf.side_effect = Exception("PDF parsing failed")

        result = extract_tables_from_pdf(Path("broken.pdf"))

        assert result == []

    @patch("scripts.pipeline.table_extractor.camelot")
    def test_small_tables_filtered(self, mock_camelot):
        """Should filter out tiny tables (< 2 rows or < 2 cols)."""
        from scripts.pipeline.table_extractor import extract_tables_from_pdf
        import pandas as pd

        # Create tiny table (1 row)
        mock_table = Mock()
        mock_table.df = pd.DataFrame({"A": ["Only one row"]})
        mock_table.page = 1
        mock_table.accuracy = 90.0
        mock_table.whitespace = 5.0

        mock_camelot.read_pdf.return_value = [mock_table]

        result = extract_tables_from_pdf(Path("test.pdf"))

        # Should be filtered out
        assert result == []


class TestProcessCorpusTables:
    """Tests for process_corpus_tables function."""

    @patch("scripts.pipeline.table_extractor.extract_tables_from_pdf")
    def test_processes_all_pdfs(self, mock_extract, tmp_path):
        """Should process all PDF files in directory."""
        from scripts.pipeline.table_extractor import process_corpus_tables

        # Create dummy PDF files
        (tmp_path / "doc1.pdf").touch()
        (tmp_path / "doc2.pdf").touch()
        (tmp_path / "not_pdf.txt").touch()  # Should be ignored

        mock_extract.return_value = []  # No tables found

        output_file = tmp_path / "output" / "tables.json"
        report = process_corpus_tables(tmp_path, output_file, corpus="fr")

        # Should have processed 2 PDFs
        assert mock_extract.call_count == 2
        assert report["total_files"] == 2
        assert output_file.exists()

    @patch("scripts.pipeline.table_extractor.extract_tables_from_pdf")
    def test_aggregates_tables(self, mock_extract, tmp_path):
        """Should aggregate tables from multiple PDFs."""
        from scripts.pipeline.table_extractor import process_corpus_tables

        (tmp_path / "doc1.pdf").touch()
        (tmp_path / "doc2.pdf").touch()

        # Each PDF has some tables
        mock_extract.side_effect = [
            [{"id": "t1", "table_type": "elo"}],
            [{"id": "t2", "table_type": "cadence"}, {"id": "t3", "table_type": "elo"}],
        ]

        output_file = tmp_path / "tables.json"
        report = process_corpus_tables(tmp_path, output_file, corpus="fr")

        assert report["total_tables"] == 3
        assert report["files_with_tables"] == 2
        assert report["by_type"]["elo"] == 2
        assert report["by_type"]["cadence"] == 1

    @patch("scripts.pipeline.table_extractor.extract_tables_from_pdf")
    def test_adds_corpus_metadata(self, mock_extract, tmp_path):
        """Should add corpus field to each table."""
        from scripts.pipeline.table_extractor import process_corpus_tables
        import json

        (tmp_path / "doc.pdf").touch()
        mock_extract.return_value = [
            {"id": "t1", "table_type": "other"},
        ]

        output_file = tmp_path / "tables.json"
        process_corpus_tables(tmp_path, output_file, corpus="intl")

        # Check output file
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["corpus"] == "intl"
        assert data["tables"][0]["corpus"] == "intl"
