"""Tests for Kaggle dataset preparation."""

import json
from pathlib import Path

from scripts.training.prepare_kaggle_dataset import extract_paragraphs


class TestExtractParagraphs:
    def test_splits_on_double_newline(self, tmp_path: Path) -> None:
        doc = {
            "markdown": ("First paragraph here.\n\nSecond paragraph here.\n\nTiny."),
            "source": "test.pdf",
        }
        (tmp_path / "test.json").write_text(json.dumps(doc), encoding="utf-8")
        result = extract_paragraphs(tmp_path)
        # "Tiny." is < 20 chars, should be filtered
        assert len(result) == 2
        assert result[0]["source"] == "test.pdf"

    def test_skips_tiny_fragments(self, tmp_path: Path) -> None:
        doc = {
            "markdown": ("Short.\n\nA long enough paragraph with real content here."),
            "source": "t.pdf",
        }
        (tmp_path / "t.json").write_text(json.dumps(doc), encoding="utf-8")
        result = extract_paragraphs(tmp_path)
        assert len(result) == 1

    def test_preserves_source(self, tmp_path: Path) -> None:
        doc = {
            "markdown": "A paragraph with enough text here for testing.",
            "source": "R01.pdf",
        }
        (tmp_path / "r01.json").write_text(json.dumps(doc), encoding="utf-8")
        result = extract_paragraphs(tmp_path)
        assert result[0]["source"] == "R01.pdf"

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = extract_paragraphs(tmp_path)
        assert result == []

    def test_multiple_docs(self, tmp_path: Path) -> None:
        for i, name in enumerate(["a.json", "b.json"]):
            doc = {
                "markdown": f"Paragraph from document {i} with content.",
                "source": f"doc{i}.pdf",
            }
            (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
        result = extract_paragraphs(tmp_path)
        assert len(result) == 2
        sources = {r["source"] for r in result}
        assert len(sources) == 2

    def test_fallback_source_from_stem(self, tmp_path: Path) -> None:
        """When 'source' key absent, fall back to file stem."""
        doc = {"markdown": "A paragraph with enough content to pass filter."}
        (tmp_path / "mystem.json").write_text(json.dumps(doc), encoding="utf-8")
        result = extract_paragraphs(tmp_path)
        assert len(result) == 1
        assert result[0]["source"] == "mystem"

    def test_text_field_present(self, tmp_path: Path) -> None:
        doc = {
            "markdown": "First paragraph.\n\nSecond paragraph is long enough.",
            "source": "src.pdf",
        }
        (tmp_path / "src.json").write_text(json.dumps(doc), encoding="utf-8")
        result = extract_paragraphs(tmp_path)
        for item in result:
            assert "text" in item
            assert "source" in item
