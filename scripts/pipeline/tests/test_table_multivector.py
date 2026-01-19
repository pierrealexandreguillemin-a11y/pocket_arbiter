"""
Tests for table_multivector.py - Multi-Vector Table Retrieval

API requires Claude Code summaries - no rule-based or API fallback.

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements
    - ISO/IEC 42001 - AI traceability (Claude Code summaries)
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestTableToMarkdown:
    """Tests for table_to_markdown function."""

    def test_simple_table(self):
        """Should convert simple table to markdown."""
        from scripts.pipeline.table_multivector import table_to_markdown

        table = {
            "headers": ["Nom", "Elo"],
            "rows": [["Dupont", "2100"], ["Martin", "1950"]],
        }

        md = table_to_markdown(table)

        assert "| Nom | Elo |" in md
        assert "| --- | --- |" in md
        assert "| Dupont | 2100 |" in md
        assert "| Martin | 1950 |" in md

    def test_empty_table(self):
        """Should handle empty table."""
        from scripts.pipeline.table_multivector import table_to_markdown

        table = {"headers": [], "rows": []}

        md = table_to_markdown(table)

        assert md == ""

    def test_pads_short_rows(self):
        """Should pad rows shorter than headers."""
        from scripts.pipeline.table_multivector import table_to_markdown

        table = {
            "headers": ["A", "B", "C"],
            "rows": [["1", "2"]],  # Missing C
        }

        md = table_to_markdown(table)

        assert "| 1 | 2 |" in md


class TestTableToText:
    """Tests for table_to_text function."""

    def test_simple_table(self):
        """Should convert table to text format."""
        from scripts.pipeline.table_multivector import table_to_text

        headers = ["Type", "Temps"]
        rows = [["Rapide", "15 min"]]

        text = table_to_text(headers, rows)

        assert "Type | Temps" in text
        assert "Rapide | 15 min" in text

    def test_empty_headers(self):
        """Should handle empty headers."""
        from scripts.pipeline.table_multivector import table_to_text

        text = table_to_text([], [])

        assert text == ""


class TestLoadTablesFromDoclingDir:
    """Tests for load_tables_from_docling_dir function."""

    def test_loads_tables_from_directory(self, tmp_path):
        """Should load tables from all JSON files in directory."""
        from scripts.pipeline.table_multivector import load_tables_from_docling_dir

        # Create mock docling extraction files
        doc1 = {
            "filename": "doc1.pdf",
            "tables": [
                {"id": "doc1-table0", "headers": ["A", "B"], "rows": [["1", "2"]]},
            ],
        }
        doc2 = {
            "filename": "doc2.pdf",
            "tables": [
                {"id": "doc2-table0", "headers": ["X", "Y"], "rows": [["a", "b"]]},
                {"id": "doc2-table1", "headers": ["P", "Q"], "rows": [["p", "q"]]},
            ],
        }

        (tmp_path / "doc1.json").write_text(json.dumps(doc1), encoding="utf-8")
        (tmp_path / "doc2.json").write_text(json.dumps(doc2), encoding="utf-8")

        tables = load_tables_from_docling_dir(tmp_path)

        assert len(tables) == 3
        assert tables[0]["id"] == "doc1-table0"
        assert tables[1]["id"] == "doc2-table0"
        assert tables[2]["id"] == "doc2-table1"

    def test_skips_extraction_report(self, tmp_path):
        """Should skip extraction_report.json file."""
        from scripts.pipeline.table_multivector import load_tables_from_docling_dir

        doc = {"tables": [{"id": "t1", "headers": ["A"]}]}
        report = {"files_processed": 1, "errors": []}

        (tmp_path / "doc.json").write_text(json.dumps(doc), encoding="utf-8")
        (tmp_path / "extraction_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

        tables = load_tables_from_docling_dir(tmp_path)

        assert len(tables) == 1

    def test_sets_default_table_type(self, tmp_path):
        """Should set table_type to 'other' if not present."""
        from scripts.pipeline.table_multivector import load_tables_from_docling_dir

        doc = {"tables": [{"id": "t1", "headers": ["A", "B"]}]}
        (tmp_path / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

        tables = load_tables_from_docling_dir(tmp_path)

        assert tables[0]["table_type"] == "other"


class TestProcessTablesMultivector:
    """Tests for process_tables_multivector function."""

    def test_fails_without_summaries_file(self, tmp_path):
        """Should exit with error when summaries file missing (ISO 42001)."""
        from scripts.pipeline.table_multivector import process_tables_multivector

        # Create input directory with tables
        input_dir = tmp_path / "docling"
        input_dir.mkdir()
        doc = {"tables": [{"id": "t1", "headers": ["A", "B"], "rows": [["1", "2"]]}]}
        (input_dir / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

        summaries_file = tmp_path / "nonexistent_summaries.json"
        output_file = tmp_path / "output.json"

        with pytest.raises(SystemExit) as exc_info:
            process_tables_multivector(
                input_path=input_dir,
                summaries_file=summaries_file,
                output_file=output_file,
            )

        assert exc_info.value.code == 1

    def test_fails_with_empty_summaries(self, tmp_path):
        """Should exit when summaries file has wrong format."""
        from scripts.pipeline.table_multivector import process_tables_multivector

        input_dir = tmp_path / "docling"
        input_dir.mkdir()
        doc = {"tables": [{"id": "t1", "headers": ["A", "B"], "rows": []}]}
        (input_dir / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

        # Empty summaries dict
        summaries_file = tmp_path / "summaries.json"
        summaries_file.write_text('{"summaries": {}}', encoding="utf-8")

        output_file = tmp_path / "output.json"

        with pytest.raises(SystemExit) as exc_info:
            process_tables_multivector(
                input_path=input_dir,
                summaries_file=summaries_file,
                output_file=output_file,
            )

        assert exc_info.value.code == 1

    def test_processes_tables_with_summaries(self, tmp_path):
        """Should process tables when Claude Code summaries provided."""
        from scripts.pipeline.table_multivector import process_tables_multivector

        # Create input directory
        input_dir = tmp_path / "docling"
        input_dir.mkdir()
        doc = {
            "tables": [
                {
                    "id": "t1",
                    "headers": ["Type", "Temps"],
                    "rows": [["Rapide", "15 min"]],
                    "source": "doc.pdf",
                    "page": 1,
                    "table_type": "cadence",
                },
                {
                    "id": "t2",
                    "headers": ["Critere", "Description"],
                    "rows": [["Buchholz", "Somme Elo adversaires"]],
                    "source": "doc.pdf",
                    "page": 2,
                    "table_type": "tiebreak",
                },
            ]
        }
        (input_dir / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

        # Create summaries (Claude Code generated)
        summaries = {
            "summaries": {
                "t1": "Table de cadences officielles FFE avec temps et increments.",
                "t2": "Criteres de departage pour tournois suisses.",
            }
        }
        summaries_file = tmp_path / "summaries.json"
        summaries_file.write_text(json.dumps(summaries), encoding="utf-8")

        output_file = tmp_path / "output.json"

        report = process_tables_multivector(
            input_path=input_dir,
            summaries_file=summaries_file,
            output_file=output_file,
            corpus="fr",
        )

        # Check report
        assert report["corpus"] == "fr"
        assert report["total_tables"] == 2
        assert report["tables_with_summary"] == 2
        assert report["summary_source"] == "claude_code"

        # Check output
        output = json.loads(output_file.read_text(encoding="utf-8"))
        assert output["summary_source"] == "claude_code"
        assert len(output["children"]) == 2
        assert len(output["parents"]) == 2

        # Verify child has Claude summary
        child = output["children"][0]
        assert "cadences" in child["text"].lower()

    def test_skips_tables_without_summary(self, tmp_path):
        """Should skip tables that don't have a Claude Code summary."""
        from scripts.pipeline.table_multivector import process_tables_multivector

        input_dir = tmp_path / "docling"
        input_dir.mkdir()
        doc = {
            "tables": [
                {"id": "t1", "headers": ["A", "B"], "rows": [["1", "2"]]},
                {"id": "t2", "headers": ["X", "Y"], "rows": [["a", "b"]]},
            ]
        }
        (input_dir / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

        # Only provide summary for t1
        summaries = {"summaries": {"t1": "Summary for table 1 only."}}
        summaries_file = tmp_path / "summaries.json"
        summaries_file.write_text(json.dumps(summaries), encoding="utf-8")

        output_file = tmp_path / "output.json"

        report = process_tables_multivector(
            input_path=input_dir,
            summaries_file=summaries_file,
            output_file=output_file,
        )

        assert report["total_tables"] == 2
        assert report["tables_with_summary"] == 1
        assert report["skipped_no_summary"] == 1

    def test_skips_tables_with_insufficient_headers(self, tmp_path):
        """Should skip tables with fewer than 2 valid headers."""
        from scripts.pipeline.table_multivector import process_tables_multivector

        input_dir = tmp_path / "docling"
        input_dir.mkdir()
        doc = {
            "tables": [
                {"id": "t1", "headers": ["OnlyOne"], "rows": [["val"]]},
                {"id": "t2", "headers": ["", "  "], "rows": [[]]},
                {"id": "t3", "headers": ["Valid1", "Valid2"], "rows": [["a", "b"]]},
            ]
        }
        (input_dir / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

        summaries = {
            "summaries": {
                "t1": "Summary 1",
                "t2": "Summary 2",
                "t3": "Summary 3",
            }
        }
        summaries_file = tmp_path / "summaries.json"
        summaries_file.write_text(json.dumps(summaries), encoding="utf-8")

        output_file = tmp_path / "output.json"

        report = process_tables_multivector(
            input_path=input_dir,
            summaries_file=summaries_file,
            output_file=output_file,
        )

        assert report["total_tables"] == 3
        assert report["skipped_invalid_headers"] == 2
        assert report["tables_with_summary"] == 1


class TestPydanticModels:
    """Tests for pydantic schema validation (ISO 25010 data quality)."""

    def test_child_document_requires_text(self):
        """Should require non-empty text for embedding."""
        from pydantic import ValidationError

        from scripts.pipeline.table_multivector import ChildDocument

        with pytest.raises(ValidationError):
            ChildDocument(id="c1", doc_id="p1", text="")

    def test_child_document_valid(self):
        """Should accept valid child document."""
        from scripts.pipeline.table_multivector import ChildDocument

        child = ChildDocument(
            id="c1",
            doc_id="p1",
            text="Table summary text",
            source="doc.pdf",
            page=5,
        )

        assert child.id == "c1"
        assert child.doc_id == "p1"
        assert child.type == "table_summary"

    def test_parent_document_valid(self):
        """Should accept valid parent document."""
        from scripts.pipeline.table_multivector import ParentDocument

        parent = ParentDocument(
            id="p1",
            headers=["A", "B"],
            rows=[["1", "2"]],
            markdown="| A | B |",
        )

        assert parent.id == "p1"
        assert parent.type == "table"
        assert parent.headers == ["A", "B"]
