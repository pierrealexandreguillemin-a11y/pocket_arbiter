"""
Tests for table_multivector.py - Multi-Vector Table Retrieval

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements
"""

import importlib.util
import json
import os

import pytest


class TestGenerateRuleBasedSummary:
    """Tests for generate_rule_based_summary function."""

    def test_cadence_table_summary(self):
        """Should generate summary for cadence table."""
        from scripts.pipeline.table_multivector import generate_rule_based_summary

        table = {
            "table_type": "cadence",
            "headers": ["Type", "Temps", "Increment"],
            "source": "LA-2025.pdf",
            "page": 15,
        }

        summary = generate_rule_based_summary(table)

        assert "cadence" in summary.lower()
        assert "temps" in summary.lower() or "Temps" in summary
        assert "LA-2025.pdf" in summary
        assert "15" in summary

    def test_tiebreak_table_summary(self):
        """Should generate summary for tiebreak table."""
        from scripts.pipeline.table_multivector import generate_rule_based_summary

        table = {
            "table_type": "tiebreak",
            "headers": ["Ordre", "Critere", "Description"],
            "source": "reglement.pdf",
            "page": 42,
        }

        summary = generate_rule_based_summary(table)

        assert "departage" in summary.lower()
        assert "Ordre" in summary or "ordre" in summary.lower()

    def test_truncates_many_headers(self):
        """Should truncate when more than 5 headers."""
        from scripts.pipeline.table_multivector import generate_rule_based_summary

        table = {
            "table_type": "other",
            "headers": ["A", "B", "C", "D", "E", "F", "G"],
            "source": "doc.pdf",
            "page": 1,
        }

        summary = generate_rule_based_summary(table)

        assert "+2 colonnes" in summary


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

        assert "| 1 | 2 |  |" in md or "| 1 | 2 | |" in md


class TestCreateMultivectorEntry:
    """Tests for create_multivector_entry function."""

    def test_creates_child_and_parent(self):
        """Should create linked child and parent entries."""
        from scripts.pipeline.table_multivector import create_multivector_entry

        table = {
            "id": "test-table0",
            "table_type": "elo",
            "headers": ["Rang", "Joueur", "Elo"],
            "rows": [["1", "Kasparov", "2851"]],
            "source": "rankings.pdf",
            "page": 5,
            "text": "Table Elo...",
            "accuracy": 95.0,
        }

        entry = create_multivector_entry(table, use_llm=False)

        # Check structure
        assert "doc_id" in entry
        assert "child" in entry
        assert "parent" in entry

        # Check doc_id linkage
        assert entry["child"]["doc_id"] == entry["doc_id"]
        assert entry["parent"]["id"] == entry["doc_id"]

        # Check child (for embedding)
        child = entry["child"]
        assert child["type"] == "table_summary"
        assert "text" in child
        assert len(child["text"]) > 0

        # Check parent (for docstore)
        parent = entry["parent"]
        assert parent["type"] == "table"
        assert parent["headers"] == ["Rang", "Joueur", "Elo"]
        assert "markdown" in parent
        assert "| Kasparov |" in parent["markdown"]


class TestGenerateLlmSummary:
    """Tests for generate_llm_summary function."""

    def test_raises_import_error_without_genai(self):
        """Should raise ImportError when google-generativeai not installed."""
        # Skip if google-generativeai is actually installed
        if importlib.util.find_spec("google.generativeai") is not None:
            pytest.skip("google-generativeai is installed")

        from scripts.pipeline.table_multivector import generate_llm_summary

        table = {"text": "Test table", "source": "doc.pdf", "page": 1}

        with pytest.raises(ImportError, match="google-generativeai required"):
            generate_llm_summary(table, api_key="fake-key")

    def test_raises_value_error_without_api_key(self):
        """Should raise ValueError without API key (when genai installed)."""
        # Skip if google-generativeai not installed
        pytest.importorskip("google.generativeai")

        from scripts.pipeline.table_multivector import generate_llm_summary

        table = {"text": "Test table", "source": "doc.pdf", "page": 1}

        # Clear env
        original = os.environ.pop("GOOGLE_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="API key required"):
                generate_llm_summary(table, api_key=None)
        finally:
            if original:
                os.environ["GOOGLE_API_KEY"] = original


class TestProcessTablesMultivector:
    """Tests for process_tables_multivector function."""

    def test_processes_tables_file(self, tmp_path):
        """Should process tables JSON into multi-vector format."""
        from scripts.pipeline.table_multivector import process_tables_multivector

        # Create input file (tables must have >= 2 valid headers to pass filter)
        input_data = {
            "corpus": "fr",
            "tables": [
                {
                    "id": "t1",
                    "table_type": "cadence",
                    "headers": ["Type", "Temps"],
                    "rows": [["Rapide", "15 min"]],
                    "source": "doc.pdf",
                    "page": 1,
                    "text": "Table cadence",
                    "accuracy": 90.0,
                },
                {
                    "id": "t2",
                    "table_type": "tiebreak",
                    "headers": ["Ordre", "Critere"],
                    "rows": [["1", "Buchholz"]],
                    "source": "doc.pdf",
                    "page": 2,
                    "text": "Table departage",
                    "accuracy": 85.0,
                },
            ],
        }

        input_file = tmp_path / "tables.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        output_file = tmp_path / "output" / "multivector.json"

        report = process_tables_multivector(
            input_file=input_file,
            output_file=output_file,
            use_llm=False,
        )

        # Check report
        assert report["corpus"] == "fr"
        assert report["total_tables"] == 2
        assert report["children_created"] == 2
        assert report["parents_created"] == 2
        assert report["use_llm"] is False

        # Check output file
        assert output_file.exists()

        with open(output_file, encoding="utf-8") as f:
            output = json.load(f)

        assert output["corpus"] == "fr"
        assert output["strategy"] == "multi_vector"
        assert len(output["children"]) == 2
        assert len(output["parents"]) == 2
        assert "parent_lookup" in output

        # Verify linkage
        child = output["children"][0]
        doc_id = child["doc_id"]
        parent_idx = output["parent_lookup"][doc_id]
        parent = output["parents"][parent_idx]
        assert parent["id"] == doc_id

    def test_empty_tables(self, tmp_path):
        """Should handle empty tables list."""
        from scripts.pipeline.table_multivector import process_tables_multivector

        input_data = {"corpus": "intl", "tables": []}

        input_file = tmp_path / "empty.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        output_file = tmp_path / "output.json"

        report = process_tables_multivector(
            input_file=input_file,
            output_file=output_file,
            use_llm=False,
        )

        assert report["total_tables"] == 0
        assert report["children_created"] == 0

    def test_filters_tables_with_insufficient_headers(self, tmp_path):
        """Should skip tables with fewer than 2 valid headers (quality gate)."""
        from scripts.pipeline.table_multivector import process_tables_multivector

        input_data = {
            "corpus": "fr",
            "tables": [
                {
                    "id": "t1",
                    "table_type": "other",
                    "headers": ["OnlyOne"],  # Only 1 header - should be skipped
                    "rows": [["Value"]],
                    "source": "doc.pdf",
                    "page": 1,
                },
                {
                    "id": "t2",
                    "table_type": "other",
                    "headers": ["", "  ", "Valid"],  # Only 1 valid header - skipped
                    "rows": [["", "", "Data"]],
                    "source": "doc.pdf",
                    "page": 2,
                },
                {
                    "id": "t3",
                    "table_type": "cadence",
                    "headers": ["Type", "Temps"],  # 2 valid headers - kept
                    "rows": [["Rapide", "15 min"]],
                    "source": "doc.pdf",
                    "page": 3,
                },
            ],
        }

        input_file = tmp_path / "mixed.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        output_file = tmp_path / "output.json"

        report = process_tables_multivector(
            input_file=input_file,
            output_file=output_file,
            use_llm=False,
        )

        assert report["total_tables"] == 3
        assert report["tables_skipped"] == 2
        assert report["children_created"] == 1
        assert report["parents_created"] == 1


class TestGenerateTableSummary:
    """Tests for generate_table_summary dispatcher."""

    def test_uses_rule_based_by_default(self):
        """Should use rule-based when use_llm=False."""
        from scripts.pipeline.table_multivector import generate_table_summary

        table = {
            "table_type": "penalty",
            "headers": ["Faute", "Sanction"],
            "source": "doc.pdf",
            "page": 10,
        }

        summary = generate_table_summary(table, use_llm=False)

        assert "penalite" in summary.lower() or "sanction" in summary.lower()
        assert "doc.pdf" in summary
