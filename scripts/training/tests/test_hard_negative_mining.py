"""
Tests pour hard_negative_mining.py

ISO Reference: ISO/IEC 29119 - Test coverage >= 80%
"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.training.hard_negative_mining import (
    convert_to_training_format,
    load_pairs_jsonl,
    save_triplets_jsonl,
    select_hard_negative,
)


class TestLoadPairsJSONL:
    """Tests pour load_pairs_jsonl()."""

    def test_load_valid_file(self):
        """Charge un fichier JSONL valide."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pairs.jsonl"
            path.write_text('{"query": "Q1"}\n{"query": "Q2"}\n')

            pairs = load_pairs_jsonl(path)
            assert len(pairs) == 2
            assert pairs[0]["query"] == "Q1"

    def test_file_not_found(self):
        """Leve FileNotFoundError si fichier manquant."""
        with pytest.raises(FileNotFoundError):
            load_pairs_jsonl(Path("/nonexistent/file.jsonl"))

    def test_skips_empty_lines(self):
        """Ignore les lignes vides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pairs.jsonl"
            path.write_text('{"query": "Q1"}\n\n{"query": "Q2"}\n\n')

            pairs = load_pairs_jsonl(path)
            assert len(pairs) == 2


class TestSaveTripletsJSONL:
    """Tests pour save_triplets_jsonl()."""

    def test_save_triplets(self):
        """Sauvegarde des triplets en JSONL."""
        triplets = [
            {"anchor": "A1", "positive": "P1", "negative": "N1"},
            {"anchor": "A2", "positive": "P2", "negative": "N2"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "triplets.jsonl"
            save_triplets_jsonl(triplets, path)

            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2

            loaded = json.loads(lines[0])
            assert loaded["anchor"] == "A1"

    def test_creates_parent_dirs(self):
        """Cree les repertoires parents."""
        triplets = [{"anchor": "A1", "positive": "P1", "negative": "N1"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "triplets.jsonl"
            save_triplets_jsonl(triplets, path)
            assert path.exists()


class TestSelectHardNegative:
    """Tests pour select_hard_negative()."""

    def test_selects_highest_in_range(self):
        """Selectionne le score le plus eleve dans la plage."""
        candidates = [
            {"id": "1", "text": "T1", "score": 0.5},
            {"id": "2", "text": "T2", "score": 0.7},
            {"id": "3", "text": "T3", "score": 0.3},
        ]
        result = select_hard_negative(candidates, min_score=0.3, max_score=0.9)
        assert result is not None
        assert result["id"] == "2"  # Score 0.7 est le plus eleve

    def test_excludes_above_max(self):
        """Exclut les scores au-dessus du max."""
        candidates = [
            {"id": "1", "text": "T1", "score": 0.95},  # Trop eleve
            {"id": "2", "text": "T2", "score": 0.7},
        ]
        result = select_hard_negative(candidates, min_score=0.3, max_score=0.9)
        assert result is not None
        assert result["id"] == "2"

    def test_fallback_if_none_in_range(self):
        """Fallback au plus proche si aucun dans la plage."""
        candidates = [
            {"id": "1", "text": "T1", "score": 0.1},  # En dessous du min
            {"id": "2", "text": "T2", "score": 0.2},  # En dessous du min
        ]
        result = select_hard_negative(candidates, min_score=0.3, max_score=0.9)
        # Prend le plus eleve en dessous du max
        assert result is not None
        assert result["id"] == "2"

    def test_returns_none_if_all_above_max(self):
        """Retourne None si tous au-dessus du max."""
        candidates = [
            {"id": "1", "text": "T1", "score": 0.95},
            {"id": "2", "text": "T2", "score": 0.92},
        ]
        result = select_hard_negative(candidates, min_score=0.3, max_score=0.9)
        assert result is None

    def test_empty_candidates(self):
        """Retourne None si liste vide."""
        result = select_hard_negative([], min_score=0.3, max_score=0.9)
        assert result is None


class TestConvertToTrainingFormat:
    """Tests pour convert_to_training_format()."""

    def test_converts_triplets(self):
        """Convertit les triplets au format simplifie."""
        triplets = [
            {
                "anchor": "A1",
                "positive": "P1",
                "negative": "N1",
                "positive_chunk_id": "C1",
                "negative_chunk_id": "C2",
                "negative_score": 0.7,
            },
        ]
        result = convert_to_training_format(triplets)

        assert len(result) == 1
        assert result[0] == {
            "anchor": "A1",
            "positive": "P1",
            "negative": "N1",
        }
        # Pas de metadata
        assert "positive_chunk_id" not in result[0]

    def test_empty_list(self):
        """Gere une liste vide."""
        result = convert_to_training_format([])
        assert result == []
