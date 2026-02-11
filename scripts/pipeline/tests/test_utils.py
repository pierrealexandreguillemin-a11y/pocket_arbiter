"""
Tests for scripts.pipeline.utils shared utilities.

Covers: cosine_similarity, load_json, save_json, get_timestamp, get_date,
        list_pdf_files, validate_chunk_schema.

ISO Reference:
    - ISO/IEC 29119 - Test coverage
    - ISO/IEC 12207 - Reusability validation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.utils import (
    cosine_similarity,
    get_date,
    get_timestamp,
    list_pdf_files,
    load_json,
    save_json,
    validate_chunk_schema,
)

# ---------------------------------------------------------------------------
# TestCosineSimilarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for cosine_similarity (PURE)."""

    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_zero(self) -> None:
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_both_zero_vectors(self) -> None:
        v1 = np.array([0.0, 0.0])
        v2 = np.array([0.0, 0.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_opposite_vectors(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=1e-6)

    def test_returns_float(self) -> None:
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        result = cosine_similarity(v1, v2)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TestLoadJson
# ---------------------------------------------------------------------------


class TestLoadJson:
    """Tests for load_json (PURE I/O)."""

    def test_valid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        data = load_json(p)
        assert data["key"] == "value"

    def test_valid_json_str_path(self, tmp_path: Path) -> None:
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"k": 42}), encoding="utf-8")
        data = load_json(str(p))
        assert data["k"] == 42

    def test_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{invalid", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_json(p)

    def test_utf8_content(self, tmp_path: Path) -> None:
        p = tmp_path / "utf8.json"
        p.write_text(json.dumps({"text": "éàü"}), encoding="utf-8")
        data = load_json(p)
        assert data["text"] == "éàü"


# ---------------------------------------------------------------------------
# TestSaveJson
# ---------------------------------------------------------------------------


class TestSaveJson:
    """Tests for save_json (PURE I/O)."""

    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        save_json({"a": 1}, p)
        assert p.exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["a"] == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "out.json"
        save_json({"nested": True}, p)
        assert p.exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["nested"] is True

    def test_ensure_ascii_false(self, tmp_path: Path) -> None:
        p = tmp_path / "utf8.json"
        save_json({"text": "éàü"}, p)
        raw = p.read_text(encoding="utf-8")
        assert "éàü" in raw  # Not escaped

    def test_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "rt.json"
        original = {"key": "value", "list": [1, 2, 3]}
        save_json(original, p)
        loaded = load_json(p)
        assert loaded == original


# ---------------------------------------------------------------------------
# TestGetTimestamp / TestGetDate
# ---------------------------------------------------------------------------


class TestGetTimestamp:
    """Tests for get_timestamp (PURE)."""

    def test_iso_format(self) -> None:
        ts = get_timestamp()
        # ISO 8601: YYYY-MM-DDTHH:MM:SS
        assert "T" in ts
        assert len(ts) == 19

    def test_returns_str(self) -> None:
        assert isinstance(get_timestamp(), str)


class TestGetDate:
    """Tests for get_date (PURE)."""

    def test_iso_date_format(self) -> None:
        d = get_date()
        # YYYY-MM-DD
        assert len(d) == 10
        assert d[4] == "-"
        assert d[7] == "-"

    def test_returns_str(self) -> None:
        assert isinstance(get_date(), str)


# ---------------------------------------------------------------------------
# TestListPdfFiles
# ---------------------------------------------------------------------------


class TestListPdfFiles:
    """Tests for list_pdf_files (I/O)."""

    def test_finds_pdfs(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "c.txt").write_text("not a pdf")
        result = list_pdf_files(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".pdf" for p in result)

    def test_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.pdf").write_bytes(b"%PDF-1.4")
        result = list_pdf_files(tmp_path)
        assert len(result) == 1

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = list_pdf_files(tmp_path)
        assert result == []

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list_pdf_files(tmp_path / "nonexistent")

    def test_sorted_output(self, tmp_path: Path) -> None:
        (tmp_path / "z.pdf").write_bytes(b"%PDF")
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        result = list_pdf_files(tmp_path)
        assert result[0].name == "a.pdf"
        assert result[1].name == "z.pdf"


# ---------------------------------------------------------------------------
# TestValidateChunkSchema
# ---------------------------------------------------------------------------


class TestValidateChunkSchema:
    """Tests for validate_chunk_schema (PURE)."""

    def _valid_chunk(self) -> dict:
        return {
            "id": "FR-001-001-01",
            "text": "x" * 60,
            "source": "test.pdf",
            "page": 1,
            "tokens": 100,
            "metadata": {
                "corpus": "fr",
                "extraction_date": "2026-01-01",
                "version": "1.0",
            },
        }

    def test_valid_chunk_no_errors(self) -> None:
        errors = validate_chunk_schema(self._valid_chunk())
        assert errors == []

    def test_missing_required_field(self) -> None:
        chunk = self._valid_chunk()
        del chunk["text"]
        errors = validate_chunk_schema(chunk)
        assert any("Missing required field: text" in e for e in errors)

    def test_invalid_chunk_id_format(self) -> None:
        chunk = self._valid_chunk()
        chunk["id"] = "bad-id"
        errors = validate_chunk_schema(chunk)
        assert any("Invalid chunk ID format" in e for e in errors)

    def test_valid_intl_chunk_id(self) -> None:
        chunk = self._valid_chunk()
        chunk["id"] = "INTL-002-003-04"
        errors = validate_chunk_schema(chunk)
        assert errors == []

    def test_text_too_short(self) -> None:
        chunk = self._valid_chunk()
        chunk["text"] = "short"
        errors = validate_chunk_schema(chunk)
        assert any("Text too short" in e for e in errors)

    def test_too_many_tokens(self) -> None:
        chunk = self._valid_chunk()
        chunk["tokens"] = 600
        errors = validate_chunk_schema(chunk)
        assert any("Too many tokens" in e for e in errors)

    def test_missing_metadata_field(self) -> None:
        chunk = self._valid_chunk()
        del chunk["metadata"]["corpus"]
        errors = validate_chunk_schema(chunk)
        assert any("Missing metadata field: corpus" in e for e in errors)

    def test_multiple_errors(self) -> None:
        chunk = {"id": "bad", "text": "x", "tokens": 999}
        errors = validate_chunk_schema(chunk)
        # Missing fields + invalid id + short text + too many tokens
        assert len(errors) >= 4
