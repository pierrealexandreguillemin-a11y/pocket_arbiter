"""
Tests for migrate_table_summaries.py.

ISO Reference:
    - ISO/IEC 29119 - Software Testing
    - ISO/IEC 42001 A.6.2.2 - AI traceability
"""

import json
from pathlib import Path


class TestBuildTableToPageMapping:
    """Tests for build_table_to_page_mapping function."""

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path):
        """Empty directory returns empty mapping."""
        from scripts.pipeline.migrate_table_summaries import build_table_to_page_mapping

        result = build_table_to_page_mapping(tmp_path)
        assert result == {}

    def test_skips_extraction_report(self, tmp_path: Path):
        """Skips extraction_report.json file."""
        from scripts.pipeline.migrate_table_summaries import build_table_to_page_mapping

        # Create extraction_report.json
        report_file = tmp_path / "extraction_report.json"
        report_file.write_text('{"files": 1}', encoding="utf-8")

        result = build_table_to_page_mapping(tmp_path)
        assert result == {}

    def test_extracts_page_from_docling_document_dict(self, tmp_path: Path):
        """Extracts page numbers from docling_document tables (dict format)."""
        from scripts.pipeline.migrate_table_summaries import build_table_to_page_mapping

        # Create docling file with tables in dict format
        docling_data = {
            "filename": "test.pdf",
            "docling_document": {
                "tables": {
                    "#/tables/0": {"prov": [{"page_no": 5}]},
                    "#/tables/1": {"prov": [{"page_no": 10}]},
                }
            },
        }
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(docling_data), encoding="utf-8")

        result = build_table_to_page_mapping(tmp_path)
        assert "test-table0" in result
        assert result["test-table0"] == 5
        assert "test-table1" in result
        assert result["test-table1"] == 10

    def test_extracts_page_from_docling_document_list(self, tmp_path: Path):
        """Extracts page numbers from docling_document tables (list format)."""
        from scripts.pipeline.migrate_table_summaries import build_table_to_page_mapping

        # Create docling file with tables in list format
        docling_data = {
            "filename": "doc.pdf",
            "docling_document": {
                "tables": [
                    {"prov": [{"page_no": 3}]},
                    {"prov": [{"page_no": 7}]},
                ]
            },
        }
        json_file = tmp_path / "doc.json"
        json_file.write_text(json.dumps(docling_data), encoding="utf-8")

        result = build_table_to_page_mapping(tmp_path)
        assert "doc-table0" in result
        assert result["doc-table0"] == 3

    def test_handles_missing_docling_document(self, tmp_path: Path):
        """Handles files without docling_document gracefully."""
        from scripts.pipeline.migrate_table_summaries import build_table_to_page_mapping

        docling_data = {
            "filename": "test.pdf",
            "markdown": "# Test",
        }
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(docling_data), encoding="utf-8")

        result = build_table_to_page_mapping(tmp_path)
        assert result == {}


class TestMigrateTableSummaries:
    """Tests for migrate_table_summaries function."""

    def test_migrates_table_with_mapping(self, tmp_path: Path):
        """Migrates table summaries when mapping exists."""
        from scripts.pipeline.migrate_table_summaries import migrate_table_summaries

        # Create tables file
        tables_data = {
            "children": [
                {"id": "t1", "doc_id": "test-table0", "page": None},
                {"id": "t2", "doc_id": "test-table1", "page": None},
            ]
        }
        tables_file = tmp_path / "tables.json"
        tables_file.write_text(json.dumps(tables_data), encoding="utf-8")

        # Create docling dir with mapping
        docling_dir = tmp_path / "docling"
        docling_dir.mkdir()
        docling_data = {
            "filename": "test.pdf",
            "docling_document": {
                "tables": {
                    "#/tables/0": {"prov": [{"page_no": 5}]},
                    "#/tables/1": {"prov": [{"page_no": 10}]},
                }
            },
        }
        (docling_dir / "test.json").write_text(
            json.dumps(docling_data), encoding="utf-8"
        )

        # Migrate
        stats = migrate_table_summaries(tables_file, docling_dir)

        # Check stats
        assert stats["migrated"] == 2
        assert stats["no_mapping"] == 0

        # Check file was updated
        with open(tables_file, encoding="utf-8") as f:
            updated = json.load(f)
        assert updated["children"][0]["page"] == 5
        assert updated["children"][1]["page"] == 10

    def test_skips_already_migrated(self, tmp_path: Path):
        """Skips tables that already have page numbers."""
        from scripts.pipeline.migrate_table_summaries import migrate_table_summaries

        tables_data = {
            "children": [
                {"id": "t1", "doc_id": "test-table0", "page": 3},  # Already has page
            ]
        }
        tables_file = tmp_path / "tables.json"
        tables_file.write_text(json.dumps(tables_data), encoding="utf-8")

        docling_dir = tmp_path / "docling"
        docling_dir.mkdir()

        stats = migrate_table_summaries(tables_file, docling_dir)
        assert stats["already_has_page"] == 1
        assert stats["migrated"] == 0

    def test_reports_no_mapping(self, tmp_path: Path):
        """Reports tables without mapping."""
        from scripts.pipeline.migrate_table_summaries import migrate_table_summaries

        tables_data = {
            "children": [
                {"id": "t1", "doc_id": "unknown-table0", "page": None},
            ]
        }
        tables_file = tmp_path / "tables.json"
        tables_file.write_text(json.dumps(tables_data), encoding="utf-8")

        docling_dir = tmp_path / "docling"
        docling_dir.mkdir()

        stats = migrate_table_summaries(tables_file, docling_dir)
        assert stats["no_mapping"] == 1
        assert len(stats["errors"]) == 1


class TestValidateTableSummaries:
    """Tests for validate_table_summaries function."""

    def test_valid_all_have_page(self, tmp_path: Path):
        """Returns valid=True when all have page >= 1."""
        from scripts.pipeline.migrate_table_summaries import validate_table_summaries

        tables_data = {
            "children": [
                {"id": "t1", "page": 5},
                {"id": "t2", "page": 10},
            ]
        }
        tables_file = tmp_path / "tables.json"
        tables_file.write_text(json.dumps(tables_data), encoding="utf-8")

        report = validate_table_summaries(tables_file)
        assert report["valid"] is True
        assert report["with_page"] == 2
        assert report["coverage_pct"] == 100.0

    def test_invalid_with_null_page(self, tmp_path: Path):
        """Returns valid=False when some have null page."""
        from scripts.pipeline.migrate_table_summaries import validate_table_summaries

        tables_data = {
            "children": [
                {"id": "t1", "page": 5},
                {"id": "t2", "page": None},
            ]
        }
        tables_file = tmp_path / "tables.json"
        tables_file.write_text(json.dumps(tables_data), encoding="utf-8")

        report = validate_table_summaries(tables_file)
        assert report["valid"] is False
        assert report["null_page"] == 1
        assert report["coverage_pct"] == 50.0

    def test_invalid_with_zero_page(self, tmp_path: Path):
        """Returns valid=False when some have page=0."""
        from scripts.pipeline.migrate_table_summaries import validate_table_summaries

        tables_data = {
            "children": [
                {"id": "t1", "page": 0},
            ]
        }
        tables_file = tmp_path / "tables.json"
        tables_file.write_text(json.dumps(tables_data), encoding="utf-8")

        report = validate_table_summaries(tables_file)
        assert report["valid"] is False
        assert report["zero_page"] == 1
