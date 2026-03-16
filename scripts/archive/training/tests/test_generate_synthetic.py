"""
Tests pour generate_synthetic_data.py

ISO Reference: ISO/IEC 29119 - Test coverage >= 80%
"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.training.generate_synthetic_data import (
    MockLLMClient,
    create_llm_client,
    generate_questions_for_chunk,
    generate_synthetic_dataset,
    load_pairs_jsonl,
    parse_questions_response,
    save_pairs_jsonl,
)


class TestParseQuestionsResponse:
    """Tests pour parse_questions_response()."""

    def test_valid_json_array(self):
        """Parse un tableau JSON valide."""
        response = '["Question 1 ?", "Question 2 ?"]'
        result = parse_questions_response(response)
        assert result == ["Question 1 ?", "Question 2 ?"]

    def test_json_with_surrounding_text(self):
        """Parse JSON entoure de texte."""
        response = 'Here are the questions: ["Q1 ?", "Q2 ?"] Hope this helps!'
        result = parse_questions_response(response)
        assert result == ["Q1 ?", "Q2 ?"]

    def test_filters_empty_strings(self):
        """Filtre les strings vides."""
        response = '["Q1 ?", "", "Q2 ?", "  "]'
        result = parse_questions_response(response)
        assert result == ["Q1 ?", "Q2 ?"]

    def test_invalid_json_raises(self):
        """Leve une erreur si JSON invalide."""
        with pytest.raises(ValueError, match="No JSON array found"):
            parse_questions_response("No JSON here")

    def test_not_a_list_raises(self):
        """Leve une erreur si pas une liste."""
        # JSON object without array brackets raises "No JSON array found"
        with pytest.raises(ValueError, match="No JSON array found"):
            parse_questions_response('{"key": "value"}')


class TestMockLLMClient:
    """Tests pour MockLLMClient."""

    def test_generate_returns_json(self):
        """MockLLMClient retourne du JSON valide."""
        client = MockLLMClient()
        result = client.generate("test prompt")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2


class TestCreateLLMClient:
    """Tests pour create_llm_client()."""

    def test_mock_provider(self):
        """Cree un client mock."""
        client = create_llm_client("mock")
        assert isinstance(client, MockLLMClient)

    def test_unknown_provider_raises(self):
        """Leve une erreur pour provider inconnu."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client("unknown")

    def test_anthropic_without_key_raises(self):
        """Anthropic sans cle API leve une erreur."""
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            create_llm_client("anthropic")

    def test_openai_without_key_raises(self):
        """OpenAI sans cle API leve une erreur."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_llm_client("openai")


class TestGenerateQuestionsForChunk:
    """Tests pour generate_questions_for_chunk()."""

    def test_generates_questions(self):
        """Genere des questions avec mock."""
        llm = MockLLMClient()
        result = generate_questions_for_chunk("Chunk text", llm, num_questions=2)
        assert len(result) == 2
        assert all(isinstance(q, str) for q in result)

    def test_respects_num_questions(self):
        """Respecte le nombre de questions demande."""
        llm = MockLLMClient()
        result = generate_questions_for_chunk("Chunk text", llm, num_questions=1)
        assert len(result) == 1


class TestGenerateSyntheticDataset:
    """Tests pour generate_synthetic_dataset()."""

    def test_generates_pairs(self):
        """Genere des paires pour chaque chunk."""
        chunks = [
            {"id": "FR-001-001-01", "text": "Chunk 1", "source": "doc.pdf", "page": 1},
            {"id": "FR-001-002-01", "text": "Chunk 2", "source": "doc.pdf", "page": 2},
        ]
        llm = MockLLMClient()
        pairs = generate_synthetic_dataset(chunks, llm, num_questions=2)
        # 2 chunks * 2 questions = 4 pairs
        assert len(pairs) == 4

    def test_respects_max_chunks(self):
        """Respecte la limite de chunks."""
        chunks = [
            {
                "id": f"FR-001-{i:03d}-01",
                "text": f"Chunk {i}",
                "source": "doc.pdf",
                "page": i,
            }
            for i in range(10)
        ]
        llm = MockLLMClient()
        pairs = generate_synthetic_dataset(chunks, llm, num_questions=2, max_chunks=3)
        # 3 chunks * 2 questions = 6 pairs
        assert len(pairs) == 6

    def test_pair_structure(self):
        """Verifie la structure des paires."""
        chunks = [
            {
                "id": "FR-001-001-01",
                "text": "Chunk text",
                "source": "doc.pdf",
                "page": 5,
            },
        ]
        llm = MockLLMClient()
        pairs = generate_synthetic_dataset(chunks, llm, num_questions=1)

        assert len(pairs) == 1
        pair = pairs[0]
        assert "query" in pair
        assert "positive" in pair
        assert "chunk_id" in pair
        assert "source" in pair
        assert "page" in pair
        assert pair["positive"] == "Chunk text"
        assert pair["page"] == 5


class TestSaveLoadPairsJSONL:
    """Tests pour save/load pairs JSONL."""

    def test_save_and_load(self):
        """Sauvegarde et charge des paires."""
        pairs = [
            {"query": "Q1 ?", "positive": "A1", "chunk_id": "C1"},
            {"query": "Q2 ?", "positive": "A2", "chunk_id": "C2"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pairs.jsonl"
            save_pairs_jsonl(pairs, output)

            assert output.exists()

            loaded = load_pairs_jsonl(output)
            assert len(loaded) == 2
            assert loaded[0]["query"] == "Q1 ?"

    def test_load_creates_parent_dirs(self):
        """Save cree les repertoires parents."""
        pairs = [{"query": "Q1 ?", "positive": "A1"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "subdir" / "pairs.jsonl"
            save_pairs_jsonl(pairs, output)
            assert output.exists()


class TestAnthropicClientInit:
    """Tests pour AnthropicClient initialisation."""

    def test_uses_env_api_key(self, monkeypatch):
        """Utilise ANTHROPIC_API_KEY de l'environnement."""
        from scripts.training.generate_synthetic_data import AnthropicClient

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = AnthropicClient()
        assert client.api_key == "test-key"

    def test_custom_model(self, monkeypatch):
        """Accepte un modele personnalise."""
        from scripts.training.generate_synthetic_data import AnthropicClient

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = AnthropicClient(model="claude-3-opus")
        assert client.model == "claude-3-opus"


class TestOpenAIClientInit:
    """Tests pour OpenAIClient initialisation."""

    def test_uses_env_api_key(self, monkeypatch):
        """Utilise OPENAI_API_KEY de l'environnement."""
        from scripts.training.generate_synthetic_data import OpenAIClient

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = OpenAIClient()
        assert client.api_key == "test-key"

    def test_custom_model(self, monkeypatch):
        """Accepte un modele personnalise."""
        from scripts.training.generate_synthetic_data import OpenAIClient

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = OpenAIClient(model="gpt-4")
        assert client.model == "gpt-4"


class TestCreateLLMClientWithEnv:
    """Tests pour create_llm_client avec env vars."""

    def test_anthropic_with_env(self, monkeypatch):
        """Cree AnthropicClient avec env var."""
        from scripts.training.generate_synthetic_data import AnthropicClient

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = create_llm_client("anthropic")
        assert isinstance(client, AnthropicClient)

    def test_openai_with_env(self, monkeypatch):
        """Cree OpenAIClient avec env var."""
        from scripts.training.generate_synthetic_data import OpenAIClient

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = create_llm_client("openai")
        assert isinstance(client, OpenAIClient)
