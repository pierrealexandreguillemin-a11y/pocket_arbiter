"""
Tests for scripts.pipeline.utils shared utilities.

Covers: cosine_similarity, load_json, save_json.

ISO Reference:
    - ISO/IEC 29119 - Test coverage
    - ISO/IEC 12207 - Reusability validation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.utils import cosine_similarity, load_json, save_json

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
