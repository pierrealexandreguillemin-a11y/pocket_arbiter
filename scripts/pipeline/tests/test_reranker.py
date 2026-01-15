"""
Tests pour le module reranker.

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Performance efficiency

Ce module teste le cross-encoder reranking pour ameliorer
la precision du retrieval RAG.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    """Mock CrossEncoder pour tests unitaires."""
    mock = MagicMock()
    # Simulate different scores for different pairs
    mock.predict.return_value = np.array([0.9, 0.3, 0.7, 0.5, 0.1])
    return mock


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Chunks de test."""
    return [
        {"text": "Article 4.1 toucher-jouer", "page": 41, "id": 1},
        {"text": "Regles du roque", "page": 40, "id": 2},
        {"text": "Article 4.2 j'adoube", "page": 42, "id": 3},
        {"text": "Pendule et temps", "page": 45, "id": 4},
        {"text": "Meteo du jour", "page": 99, "id": 5},
    ]


# =============================================================================
# Unit Tests - load_reranker
# =============================================================================


class TestLoadReranker:
    """Tests pour load_reranker()."""

    def test_load_with_mock(self):
        """Test load_reranker avec mock."""
        from scripts.pipeline.reranker import load_reranker

        with patch("sentence_transformers.CrossEncoder") as mock_cls:
            mock_cls.return_value = MagicMock()
            _reranker = load_reranker("test-model")  # noqa: F841
            mock_cls.assert_called_once_with("test-model", max_length=512)

    def test_load_with_custom_max_length(self):
        """Test load_reranker avec max_length custom."""
        from scripts.pipeline.reranker import load_reranker

        with patch("sentence_transformers.CrossEncoder") as mock_cls:
            mock_cls.return_value = MagicMock()
            _reranker = load_reranker("test-model", max_length=256)  # noqa: F841
            mock_cls.assert_called_once_with("test-model", max_length=256)

    def test_load_fallback_on_error(self):
        """Test fallback si modele principal echoue."""
        from scripts.pipeline.reranker import FALLBACK_MODEL, load_reranker

        with patch("sentence_transformers.CrossEncoder") as mock_cls:
            # First call fails, second succeeds
            mock_cls.side_effect = [OSError("Model not found"), MagicMock()]
            _reranker = load_reranker("bad-model", use_fallback=True)  # noqa: F841
            assert mock_cls.call_count == 2
            # Second call should use fallback model
            mock_cls.assert_called_with(FALLBACK_MODEL, max_length=512)

    def test_load_no_fallback_raises(self):
        """Test sans fallback raise l'erreur."""
        from scripts.pipeline.reranker import load_reranker

        with patch("sentence_transformers.CrossEncoder") as mock_cls:
            mock_cls.side_effect = OSError("Model not found")
            with pytest.raises(OSError, match="Could not load reranker"):
                load_reranker("bad-model", use_fallback=False)


# =============================================================================
# Unit Tests - rerank
# =============================================================================


class TestRerank:
    """Tests pour rerank()."""

    def test_rerank_sorts_by_score(
        self, mock_cross_encoder: MagicMock, sample_chunks: list[dict]
    ):
        """Rerank trie par score decroissant."""
        from scripts.pipeline.reranker import rerank

        result = rerank("toucher-jouer", sample_chunks, mock_cross_encoder, top_k=5)

        # Scores: [0.9, 0.3, 0.7, 0.5, 0.1]
        # Expected order: 0.9, 0.7, 0.5, 0.3, 0.1
        scores = [r["rerank_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_returns_top_k(
        self, mock_cross_encoder: MagicMock, sample_chunks: list[dict]
    ):
        """Rerank retourne exactement top_k resultats."""
        from scripts.pipeline.reranker import rerank

        result = rerank("toucher-jouer", sample_chunks, mock_cross_encoder, top_k=3)
        assert len(result) == 3

    def test_rerank_adds_score_to_chunks(
        self, mock_cross_encoder: MagicMock, sample_chunks: list[dict]
    ):
        """Rerank ajoute rerank_score aux chunks."""
        from scripts.pipeline.reranker import rerank

        result = rerank("toucher-jouer", sample_chunks, mock_cross_encoder, top_k=5)

        for chunk in result:
            assert "rerank_score" in chunk
            assert isinstance(chunk["rerank_score"], float)

    def test_rerank_preserves_chunk_metadata(
        self, mock_cross_encoder: MagicMock, sample_chunks: list[dict]
    ):
        """Rerank preserve les metadata des chunks."""
        from scripts.pipeline.reranker import rerank

        result = rerank("toucher-jouer", sample_chunks, mock_cross_encoder, top_k=5)

        for chunk in result:
            assert "page" in chunk
            assert "id" in chunk
            assert "text" in chunk

    def test_rerank_empty_chunks(self, mock_cross_encoder: MagicMock):
        """Rerank avec liste vide retourne liste vide."""
        from scripts.pipeline.reranker import rerank

        result = rerank("query", [], mock_cross_encoder, top_k=5)
        assert result == []
        mock_cross_encoder.predict.assert_not_called()

    def test_rerank_single_chunk(self, mock_cross_encoder: MagicMock):
        """Rerank avec un seul chunk retourne ce chunk avec score 1.0."""
        from scripts.pipeline.reranker import rerank

        chunks = [{"text": "Single chunk", "page": 1}]
        result = rerank("query", chunks, mock_cross_encoder, top_k=5)

        assert len(result) == 1
        assert result[0]["rerank_score"] == 1.0
        mock_cross_encoder.predict.assert_not_called()

    def test_rerank_custom_content_key(self, mock_cross_encoder: MagicMock):
        """Rerank avec cle de contenu custom."""
        from scripts.pipeline.reranker import rerank

        chunks = [
            {"content": "Chunk A", "id": 1},
            {"content": "Chunk B", "id": 2},
        ]
        mock_cross_encoder.predict.return_value = np.array([0.8, 0.2])

        _result = rerank(  # noqa: F841
            "query", chunks, mock_cross_encoder, top_k=2, content_key="content"
        )

        # Verify pairs were built with "content" key
        pairs = mock_cross_encoder.predict.call_args[0][0]
        assert pairs[0][1] == "Chunk A"
        assert pairs[1][1] == "Chunk B"

    def test_rerank_calls_predict_with_pairs(
        self, mock_cross_encoder: MagicMock, sample_chunks: list[dict]
    ):
        """Rerank appelle predict avec les bonnes paires."""
        from scripts.pipeline.reranker import rerank

        query = "toucher-jouer"
        rerank(query, sample_chunks, mock_cross_encoder, top_k=5)

        pairs = mock_cross_encoder.predict.call_args[0][0]
        assert len(pairs) == 5
        for pair in pairs:
            assert pair[0] == query  # Query in first position
            assert isinstance(pair[1], str)  # Chunk text in second


# =============================================================================
# Unit Tests - rerank_with_scores
# =============================================================================


class TestRerankWithScores:
    """Tests pour rerank_with_scores()."""

    def test_returns_tuple(
        self, mock_cross_encoder: MagicMock, sample_chunks: list[dict]
    ):
        """rerank_with_scores retourne un tuple (chunks, scores)."""
        from scripts.pipeline.reranker import rerank_with_scores

        chunks, scores = rerank_with_scores(
            "query", sample_chunks, mock_cross_encoder, top_k=3
        )

        assert isinstance(chunks, list)
        assert isinstance(scores, list)
        assert len(chunks) == 3
        assert len(scores) == 3

    def test_scores_match_chunks(
        self, mock_cross_encoder: MagicMock, sample_chunks: list[dict]
    ):
        """Scores correspondent aux chunks rerankes."""
        from scripts.pipeline.reranker import rerank_with_scores

        chunks, scores = rerank_with_scores(
            "query", sample_chunks, mock_cross_encoder, top_k=5
        )

        for chunk, score in zip(chunks, scores):
            assert chunk["rerank_score"] == score


# =============================================================================
# Integration Tests (requires real model - slow)
# =============================================================================


class TestRerankIntegration:
    """Tests d'integration avec vrai modele."""

    @pytest.mark.slow
    def test_real_reranker_loads(self):
        """Test que le modele par defaut se charge."""
        from scripts.pipeline.reranker import FALLBACK_MODEL, load_reranker

        # Use fallback model (smaller, faster)
        reranker = load_reranker(FALLBACK_MODEL)
        assert reranker is not None

    @pytest.mark.slow
    def test_real_reranker_scores(self):
        """Test que le reranker produit des scores sensibles."""
        from scripts.pipeline.reranker import FALLBACK_MODEL, load_reranker, rerank

        reranker = load_reranker(FALLBACK_MODEL)

        chunks = [
            {"text": "Le joueur qui touche une piece doit la jouer.", "id": 1},
            {"text": "La meteo est ensoleillee aujourd'hui.", "id": 2},
            {"text": "Article 4.1 definit la regle du toucher-jouer.", "id": 3},
        ]

        result = rerank(
            "Quelle est la regle du toucher-jouer ?", chunks, reranker, top_k=3
        )

        # Le chunk sur toucher-jouer devrait avoir un meilleur score
        # que le chunk sur la meteo
        scores = {r["id"]: r["rerank_score"] for r in result}
        assert scores[1] > scores[2]  # Toucher-jouer > meteo
        assert scores[3] > scores[2]  # Article 4.1 > meteo
