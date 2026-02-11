"""Tests for convert_to_ragas.py Schema V2 support.

ISO Reference: ISO 29119 - Software Testing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.evaluation.ragas.convert_to_ragas import (
    convert_to_ragas,
    load_chunks,
    load_gold_standard,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _make_gs_v2(questions: list[dict]) -> dict:
    """Wrap questions in a minimal GS V2 envelope."""
    return {"version": "1.1", "schema": "GS_SCHEMA_V2", "questions": questions}


def _make_chunks(chunks: list[dict]) -> dict:
    """Wrap chunks in the standard envelope."""
    return {"chunks": chunks}


def _v2_question(
    qid: str = "gs:001",
    question: str = "Q?",
    answer: str = "A.",
    chunk_id: str = "c1",
    hard_type: str = "ANSWERABLE",
    status: str = "VALIDATED",
) -> dict:
    """Build a Schema V2 question with all nested fields."""
    return {
        "id": qid,
        "content": {
            "question": question,
            "expected_answer": answer,
            "is_impossible": False,
        },
        "provenance": {"chunk_id": chunk_id, "docs": ["test.pdf"], "pages": [1]},
        "classification": {"hard_type": hard_type},
        "validation": {"status": status},
    }


def _v1_question(
    qid: str = "FR-Q01",
    question: str = "Q?",
    answer: str = "A.",
    chunk_id: str = "c1",
) -> dict:
    """Build a legacy Schema V1 (flat) question."""
    return {
        "id": qid,
        "question": question,
        "expected_answer": answer,
        "expected_chunk_id": chunk_id,
        "metadata": {"hard_type": "ANSWERABLE"},
        "validation": {"status": "VALIDATED"},
    }


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def chunks_dir(tmp_path: Path) -> Path:
    """Create a chunks directory with one chunk."""
    d = tmp_path / "corpus"
    d.mkdir()
    data = _make_chunks([{"id": "c1", "text": "Chunk text about chess rules."}])
    (d / "chunks_mode_b_fr.json").write_text(json.dumps(data), encoding="utf-8")
    return d


# ── load_gold_standard ────────────────────────────────────────────────


class TestLoadGoldStandard:
    def test_loads_explicit_path(self, tmp_path: Path) -> None:
        gs = _make_gs_v2([_v2_question()])
        path = tmp_path / "gs.json"
        path.write_text(json.dumps(gs), encoding="utf-8")
        result = load_gold_standard(path)
        assert len(result["questions"]) == 1

    def test_missing_file_raises(self) -> None:
        with (
            patch(
                "scripts.evaluation.ragas.convert_to_ragas.TESTS_DATA_DIR",
                Path("/nonexistent"),
            ),
            pytest.raises(FileNotFoundError),
        ):
            load_gold_standard(Path("/nonexistent/gs.json"))


# ── convert_to_ragas: Schema V2 ──────────────────────────────────────


class TestConvertSchemaV2:
    """Test convert_to_ragas with Schema V2 (nested) questions."""

    def test_v2_produces_correct_record(self, tmp_path: Path, chunks_dir: Path) -> None:
        gs = _make_gs_v2([_v2_question(question="What is X?", answer="X is a rule.")])
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            stats = convert_to_ragas(gs_path=gs_path, output_dir=tmp_path)

        assert stats["written"] == 1
        output = tmp_path / "ragas_evaluation.jsonl"
        record = json.loads(output.read_text(encoding="utf-8").strip())
        assert record["question"] == "What is X?"
        assert record["ground_truth"] == "X is a rule."
        assert record["answer"] == ""
        assert len(record["contexts"]) == 1

    def test_v2_filters_non_answerable(self, tmp_path: Path, chunks_dir: Path) -> None:
        gs = _make_gs_v2(
            [
                _v2_question(qid="gs:001", hard_type="ANSWERABLE"),
                _v2_question(qid="gs:002", hard_type="OUT_OF_DATABASE"),
            ]
        )
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            stats = convert_to_ragas(gs_path=gs_path, output_dir=tmp_path)

        assert stats["answerable"] == 1
        assert stats["written"] == 1

    def test_v2_filters_non_validated(self, tmp_path: Path, chunks_dir: Path) -> None:
        gs = _make_gs_v2([_v2_question(status="PENDING")])
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            stats = convert_to_ragas(gs_path=gs_path, output_dir=tmp_path)

        assert stats["answerable"] == 0

    def test_v2_skips_missing_chunk(self, tmp_path: Path, chunks_dir: Path) -> None:
        gs = _make_gs_v2([_v2_question(chunk_id="nonexistent")])
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            stats = convert_to_ragas(gs_path=gs_path, output_dir=tmp_path)

        assert stats["answerable"] == 1
        assert stats["written"] == 0  # Chunk not found


# ── convert_to_ragas: Schema V1 (flat) ───────────────────────────────


class TestConvertSchemaV1:
    """Test backward compatibility with legacy flat schema."""

    def test_v1_produces_correct_record(self, tmp_path: Path, chunks_dir: Path) -> None:
        gs = {"questions": [_v1_question(question="Old Q?", answer="Old A.")]}
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            stats = convert_to_ragas(gs_path=gs_path, output_dir=tmp_path)

        assert stats["written"] == 1
        record = json.loads(
            (tmp_path / "ragas_evaluation.jsonl").read_text(encoding="utf-8").strip()
        )
        assert record["question"] == "Old Q?"
        assert record["ground_truth"] == "Old A."


# ── None-safety (content: null, provenance: null) ─────────────────────


class TestNoneSafety:
    """Test that None nested dicts don't crash."""

    def test_content_none_skipped(self, tmp_path: Path, chunks_dir: Path) -> None:
        """Question with content: null should not crash."""
        gs = _make_gs_v2(
            [
                {
                    "id": "gs:bad",
                    "content": None,
                    "provenance": {"chunk_id": "c1"},
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                }
            ]
        )
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            stats = convert_to_ragas(gs_path=gs_path, output_dir=tmp_path)

        # Should not crash, item has no question so written record has empty fields
        assert stats["answerable"] == 1

    def test_provenance_none_skipped(self, tmp_path: Path, chunks_dir: Path) -> None:
        """Question with provenance: null should be filtered out (no chunk_id)."""
        gs = _make_gs_v2(
            [
                {
                    "id": "gs:bad",
                    "content": {"question": "Q?", "expected_answer": "A."},
                    "provenance": None,
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                }
            ]
        )
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            stats = convert_to_ragas(gs_path=gs_path, output_dir=tmp_path)

        assert stats["answerable"] == 0  # No chunk_id

    def test_classification_none_uses_default(
        self, tmp_path: Path, chunks_dir: Path
    ) -> None:
        """Question with classification: null falls back to ANSWERABLE default."""
        gs = _make_gs_v2(
            [
                {
                    "id": "gs:001",
                    "content": {"question": "Q?", "expected_answer": "A."},
                    "provenance": {"chunk_id": "c1"},
                    "classification": None,
                    "metadata": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                }
            ]
        )
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs), encoding="utf-8")

        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            stats = convert_to_ragas(gs_path=gs_path, output_dir=tmp_path)

        assert stats["written"] == 1


# ── load_chunks ───────────────────────────────────────────────────────


class TestLoadChunks:
    def test_loads_chunks_indexed_by_id(self, chunks_dir: Path) -> None:
        with patch("scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR", chunks_dir):
            result = load_chunks("fr")
        assert "c1" in result
        assert result["c1"]["text"] == "Chunk text about chess rules."

    def test_missing_file_raises(self) -> None:
        with patch(
            "scripts.evaluation.ragas.convert_to_ragas.CORPUS_DIR",
            Path("/nonexistent"),
        ):
            with pytest.raises(FileNotFoundError):
                load_chunks("fr")
