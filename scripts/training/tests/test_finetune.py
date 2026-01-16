"""
Tests pour finetune_embeddinggemma.py

ISO Reference: ISO/IEC 29119 - Test coverage >= 80%
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.training.finetune_embeddinggemma import (
    check_resources,
    get_training_args,
    load_triplets_jsonl,
)


class TestLoadTripletsJSONL:
    """Tests pour load_triplets_jsonl()."""

    def test_load_valid_file(self):
        """Charge un fichier JSONL valide."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "triplets.jsonl"
            content = [
                {"anchor": "A1", "positive": "P1", "negative": "N1"},
                {"anchor": "A2", "positive": "P2", "negative": "N2"},
            ]
            with open(path, "w") as f:
                for item in content:
                    f.write(json.dumps(item) + "\n")

            triplets = load_triplets_jsonl(path)
            assert len(triplets) == 2
            assert triplets[0]["anchor"] == "A1"

    def test_file_not_found(self):
        """Leve FileNotFoundError si fichier manquant."""
        with pytest.raises(FileNotFoundError):
            load_triplets_jsonl(Path("/nonexistent/triplets.jsonl"))

    def test_skips_empty_lines(self):
        """Ignore les lignes vides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "triplets.jsonl"
            path.write_text('{"anchor": "A1"}\n\n{"anchor": "A2"}\n')

            triplets = load_triplets_jsonl(path)
            assert len(triplets) == 2


class TestGetTrainingArgs:
    """Tests pour get_training_args()."""

    def test_cpu_mode(self):
        """Arguments pour CPU."""
        args = get_training_args("output_dir", use_cpu=True)
        assert args["fp16"] is False
        assert args["dataloader_num_workers"] == 0

    def test_gpu_mode(self):
        """Arguments pour GPU."""
        args = get_training_args("output_dir", use_cpu=False)
        assert args["fp16"] is True

    def test_custom_params(self):
        """Parametres personnalises."""
        args = get_training_args(
            output_dir="my_output",
            epochs=5,
            batch_size=8,
            learning_rate=1e-5,
        )
        assert args["output_dir"] == "my_output"
        assert args["num_train_epochs"] == 5
        assert args["per_device_train_batch_size"] == 8
        assert args["learning_rate"] == 1e-5

    def test_default_values(self):
        """Valeurs par defaut."""
        args = get_training_args("output")
        assert args["num_train_epochs"] == 3
        assert args["per_device_train_batch_size"] == 4
        assert args["gradient_accumulation_steps"] == 4
        assert args["warmup_ratio"] == 0.1
        assert args["logging_steps"] == 50
        assert args["save_strategy"] == "epoch"


class TestCheckResources:
    """Tests pour check_resources()."""

    def test_normal_memory(self):
        """Pas d'erreur si memoire normale."""
        # Mock psutil pour simuler 50% RAM
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.percent = 50.0
            # Ne doit pas lever d'erreur
            check_resources()

    def test_high_memory_raises(self):
        """Leve MemoryError si RAM trop elevee."""
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.percent = 90.0
            with pytest.raises(MemoryError, match="RAM usage"):
                check_resources()

    def test_boundary_memory(self):
        """Comportement a la limite (85%)."""
        with patch("psutil.virtual_memory") as mock_mem:
            # 85% exactement ne devrait pas lever d'erreur
            mock_mem.return_value.percent = 85.0
            check_resources()

            # 85.1% devrait lever une erreur
            mock_mem.return_value.percent = 85.1
            with pytest.raises(MemoryError):
                check_resources()
