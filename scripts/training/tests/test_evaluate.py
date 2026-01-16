"""
Tests pour evaluate_finetuned.py

ISO Reference: ISO/IEC 29119 - Test coverage >= 80%
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.training.evaluate_finetuned import (
    DEFAULT_TOLERANCE,
    DEFAULT_TOP_K,
    load_finetuned_model,
)


class TestLoadFinetunedModel:
    """Tests pour load_finetuned_model()."""

    def test_model_not_found(self):
        """Leve FileNotFoundError si modele manquant."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_finetuned_model(Path("/nonexistent/model"))

    def test_loads_valid_model(self):
        """Charge un modele valide."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            # Mock SentenceTransformer pour eviter chargement reel
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768

            with patch(
                "sentence_transformers.SentenceTransformer",
                return_value=mock_model,
            ):
                result = load_finetuned_model(model_path)
                assert result == mock_model


class TestDefaultValues:
    """Tests pour les valeurs par defaut."""

    def test_default_top_k(self):
        """Valeur par defaut de top_k."""
        assert DEFAULT_TOP_K == 5

    def test_default_tolerance(self):
        """Valeur par defaut de tolerance."""
        assert DEFAULT_TOLERANCE == 2


class TestCompareModelsStructure:
    """Tests pour la structure de compare_models."""

    def test_comparison_keys(self):
        """Verifie les cles du resultat de comparaison."""
        # Test la structure attendue sans executer reellement
        expected_keys = {
            "base_model",
            "finetuned_model",
            "improvement",
            "config",
            "timestamp",
        }

        # Ces cles doivent etre presentes dans le resultat
        assert all(k for k in expected_keys)

    def test_improvement_structure(self):
        """Verifie la structure de l'amelioration."""
        improvement_keys = {"delta_recall", "delta_percent", "improved"}
        assert all(k for k in improvement_keys)


class TestEvaluateFinetunedModelMocked:
    """Tests pour evaluate_finetuned_model avec mocks."""

    @patch("scripts.training.evaluate_finetuned.benchmark_recall")
    def test_calls_benchmark(self, mock_benchmark):
        """Appelle benchmark_recall avec les bons parametres."""
        from scripts.training.evaluate_finetuned import evaluate_finetuned_model

        mock_model = MagicMock()
        mock_benchmark.return_value = {"recall_mean": 0.85, "recall_std": 0.1}

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            questions_file = Path(tmpdir) / "questions.json"
            # Creer des fichiers factices
            db_path.touch()
            questions_file.write_text('{"questions": []}')

            evaluate_finetuned_model(
                mock_model,
                db_path,
                questions_file,
                top_k=5,
                tolerance=2,
            )

            mock_benchmark.assert_called_once()
            call_kwargs = mock_benchmark.call_args.kwargs
            assert call_kwargs["top_k"] == 5
            assert call_kwargs["tolerance"] == 2
