"""Tests for AdaptLLM regex mining on French regulatory text."""

from scripts.training.mine_reading_tasks import (
    compute_mining_stats,
    mine_completion,
    mine_connectors,
    mine_summarization,
)

FIXTURE_TEXT = """## Chapitre 3 : Forfaits

### Section 3.1 : Principes generaux

Le forfait est prononce lorsque le joueur ne se presente pas.
Par consequent, le joueur perd la partie par defaut.

Cependant, si le joueur previent l'arbitre, un delai peut etre accorde.

En application de l'article 6.7 des Lois des Echecs, l'arbitre peut
accorder un delai supplementaire sous reserve de circonstances exceptionnelles.

De plus, le reglement prevoit que le joueur doit signer la feuille de partie.
"""

FIXTURE_SOURCE = "R01_test.json"


class TestMineConnectors:
    def test_finds_nli_consequent(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "nli_consequent" in types

    def test_finds_nli_contrast(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "nli_contrast" in types

    def test_finds_reference(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "reference" in types

    def test_finds_conditional(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "conditional" in types

    def test_finds_addition(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        types = [r["task_type"] for r in results]
        assert "addition" in types

    def test_finds_causal(self):
        text = "Le joueur est elimine car il a depasse le temps."
        results = mine_connectors(text, "test.json")
        types = [r["task_type"] for r in results]
        assert "causal" in types

    def test_exercise_has_required_fields(self):
        results = mine_connectors(FIXTURE_TEXT, FIXTURE_SOURCE)
        assert len(results) > 0
        ex = results[0]
        assert "messages" in ex
        assert "task_type" in ex
        assert "source" in ex
        assert len(ex["messages"]) == 2
        assert ex["messages"][0]["role"] == "user"
        assert ex["messages"][1]["role"] == "assistant"


class TestMineSummarization:
    def test_extracts_from_headings(self):
        results = mine_summarization(FIXTURE_TEXT, FIXTURE_SOURCE)
        assert len(results) >= 2  # Chapitre 3 + Section 3.1

    def test_passage_is_section_content(self):
        results = mine_summarization(FIXTURE_TEXT, FIXTURE_SOURCE)
        assert any("forfait" in r["messages"][0]["content"].lower() for r in results)


class TestMineCompletion:
    def test_only_near_connectors(self):
        results = mine_completion(FIXTURE_TEXT, FIXTURE_SOURCE)
        assert len(results) <= 10  # bounded by connector count * radius

    def test_has_masked_sentence(self):
        results = mine_completion(FIXTURE_TEXT, FIXTURE_SOURCE)
        if results:
            assert "___" in results[0]["messages"][0]["content"]


class TestComputeMiningStats:
    def test_stats_structure(self):
        exercises = [
            {"task_type": "nli_consequent", "source": "R01.json"},
            {"task_type": "nli_contrast", "source": "R01.json"},
            {"task_type": "summarization", "source": "LA.json"},
        ]
        stats = compute_mining_stats(exercises)
        assert "total" in stats
        assert "by_type" in stats
        assert "by_source" in stats
        assert stats["total"] == 3
        assert stats["by_type"]["nli_consequent"] == 1
        assert stats["by_source"]["LA.json"] == 1

    def test_la_bias_flag(self):
        exercises = [{"task_type": "nli", "source": "LA.json"}] * 9 + [
            {"task_type": "nli", "source": "R01.json"}
        ]
        stats = compute_mining_stats(exercises)
        assert "la_bias_pct" in stats
        assert stats["la_bias_pct"] == 90.0
        assert stats["la_bias_warning"] is True

    def test_no_bias_flag_when_balanced(self):
        exercises = [{"task_type": "nli", "source": "R01.json"}] * 5 + [
            {"task_type": "nli", "source": "LA.json"}
        ] * 3
        stats = compute_mining_stats(exercises)
        assert stats["la_bias_warning"] is False
