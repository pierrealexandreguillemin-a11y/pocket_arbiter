"""Tests for export_formats.py (Step 4).

Critical: RAGAS ground_truth and ARES Answer must be the expected answer,
NOT the question. RAGAS paper (arXiv:2309.15217) Section 3: "ground truth
answer that serves as reference for evaluation."

ISO Reference: ISO 29119 - Software testing
"""

import csv
import json
import tempfile
from pathlib import Path

from scripts.training.unified.export_formats import (
    export_ares_format,
    export_ragas_format,
)


class TestExportRagasFormat:
    """Tests for RAGAS JSONL export."""

    def _make_triplets(self) -> list[dict]:
        """Create sample triplets with metadata."""
        return [
            {
                "anchor": "What is castling?",
                "positive": "Castling is a special move involving the king and rook.",
                "negative": "The pawn moves one square forward.",
                "metadata": {
                    "original_question": "What is castling?",
                    "expected_answer": "Castling involves moving the king two squares "
                    "toward a rook, then moving the rook to the other side of the king.",
                    "gs_id": "q001",
                },
            },
            {
                "anchor": "How does en passant work?",
                "positive": "En passant is a special pawn capture.",
                "negative": "The queen moves in any direction.",
                "metadata": {
                    "original_question": "How does en passant work?",
                    "expected_answer": "A pawn that has advanced two squares can be "
                    "captured by an adjacent enemy pawn as if it had moved one square.",
                    "gs_id": "q002",
                },
            },
        ]

    def test_ground_truth_is_expected_answer_not_question(self) -> None:
        """CRITICAL: ground_truth must be the expected answer, not the question.

        Bug history: ground_truth was set to metadata.get("original_question")
        which is the question itself — semantically wrong for RAGAS evaluation.
        """
        triplets = self._make_triplets()
        with tempfile.TemporaryDirectory() as tmp:
            export_ragas_format(triplets, Path(tmp))
            ragas_path = Path(tmp) / "ragas_evaluation.jsonl"

            with open(ragas_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

        for i, record in enumerate(records):
            expected_answer = triplets[i]["metadata"]["expected_answer"]
            question = triplets[i]["anchor"]

            # ground_truth MUST be the expected answer
            assert (
                record["ground_truth"] == expected_answer
            ), f"ground_truth should be expected_answer, got: {record['ground_truth']!r}"
            # ground_truth MUST NOT be the question
            assert (
                record["ground_truth"] != question
            ), f"ground_truth must not equal the question: {record['ground_truth']!r}"

    def test_ragas_output_has_required_fields(self) -> None:
        """Each RAGAS record must have question, answer, contexts, ground_truth."""
        triplets = self._make_triplets()
        with tempfile.TemporaryDirectory() as tmp:
            export_ragas_format(triplets, Path(tmp))
            ragas_path = Path(tmp) / "ragas_evaluation.jsonl"

            with open(ragas_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

        assert len(records) == 2
        for record in records:
            assert "question" in record
            assert "answer" in record
            assert "contexts" in record
            assert "ground_truth" in record
            assert isinstance(record["contexts"], list)

    def test_question_is_anchor(self) -> None:
        """RAGAS question field must be the triplet anchor."""
        triplets = self._make_triplets()
        with tempfile.TemporaryDirectory() as tmp:
            export_ragas_format(triplets, Path(tmp))
            ragas_path = Path(tmp) / "ragas_evaluation.jsonl"

            with open(ragas_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

        assert records[0]["question"] == "What is castling?"
        assert records[1]["question"] == "How does en passant work?"

    def test_contexts_is_positive_chunk(self) -> None:
        """RAGAS contexts must contain the positive (ground truth) chunk."""
        triplets = self._make_triplets()
        with tempfile.TemporaryDirectory() as tmp:
            export_ragas_format(triplets, Path(tmp))
            ragas_path = Path(tmp) / "ragas_evaluation.jsonl"

            with open(ragas_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

        assert records[0]["contexts"] == [
            "Castling is a special move involving the king and rook."
        ]

    def test_answer_is_empty_for_rag_generation(self) -> None:
        """RAGAS answer field must be empty (to be filled by RAG system)."""
        triplets = self._make_triplets()
        with tempfile.TemporaryDirectory() as tmp:
            export_ragas_format(triplets, Path(tmp))
            ragas_path = Path(tmp) / "ragas_evaluation.jsonl"

            with open(ragas_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

        for record in records:
            assert record["answer"] == ""

    def test_missing_expected_answer_yields_empty_ground_truth(self) -> None:
        """If metadata has no expected_answer, ground_truth should be empty."""
        triplets = [
            {
                "anchor": "Q?",
                "positive": "Context text.",
                "negative": "Other text.",
                "metadata": {"original_question": "Q?"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            export_ragas_format(triplets, Path(tmp))
            ragas_path = Path(tmp) / "ragas_evaluation.jsonl"

            with open(ragas_path, encoding="utf-8") as f:
                record = json.loads(f.readline())

        # Must NOT fallback to original_question
        assert record["ground_truth"] == ""
        assert record["ground_truth"] != "Q?"

    def test_return_value(self) -> None:
        """export_ragas_format returns path and count."""
        triplets = self._make_triplets()
        with tempfile.TemporaryDirectory() as tmp:
            result = export_ragas_format(triplets, Path(tmp))

        assert "ragas_path" in result
        assert result["count"] == 2


class TestExportAresFormat:
    """Tests for ARES TSV export."""

    def _make_triplets(self) -> list[dict]:
        """Create sample triplets with metadata."""
        return [
            {
                "anchor": "What is castling?",
                "positive": "Castling is a special move involving the king and rook.",
                "negative": "The pawn moves one square forward.",
                "metadata": {
                    "original_question": "What is castling?",
                    "expected_answer": "Castling involves moving the king two squares.",
                },
            },
        ]

    def test_ares_answer_is_expected_answer_not_question(self) -> None:
        """CRITICAL: ARES Answer column must be expected_answer, not question.

        Same class of bug as the RAGAS ground_truth fix.
        """
        triplets = self._make_triplets()
        with tempfile.TemporaryDirectory() as tmp:
            export_ares_format(triplets, Path(tmp))
            gold_path = Path(tmp) / "ares_gold_label.tsv"

            with open(gold_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)

        # First row is the positive example
        positive_row = rows[0]
        assert (
            positive_row["Answer"] == "Castling involves moving the king two squares."
        )
        assert positive_row["Answer"] != "What is castling?"

    def test_ares_unlabeled_answer_is_expected_answer(self) -> None:
        """Unlabeled ARES TSV should also use expected_answer."""
        triplets = self._make_triplets()
        with tempfile.TemporaryDirectory() as tmp:
            export_ares_format(triplets, Path(tmp))
            unlabeled_path = Path(tmp) / "ares_unlabeled.tsv"

            with open(unlabeled_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)

        assert rows[0]["Answer"] == "Castling involves moving the king two squares."

    def test_ares_missing_expected_answer_yields_empty(self) -> None:
        """If no expected_answer in metadata, Answer should be empty."""
        triplets = [
            {
                "anchor": "Q?",
                "positive": "Context.",
                "negative": "Other.",
                "metadata": {"original_question": "Q?"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            export_ares_format(triplets, Path(tmp))
            gold_path = Path(tmp) / "ares_gold_label.tsv"

            with open(gold_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)

        # Must NOT fallback to original_question
        assert rows[0]["Answer"] == ""
        assert rows[0]["Answer"] != "Q?"
