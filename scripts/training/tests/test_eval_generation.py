"""Tests for generation evaluation."""

from scripts.training.eval_generation import (
    check_citation,
    load_annales_questions,
    load_human_questions,
)
from scripts.training.generation_prompt import (
    SYSTEM_PROMPT,
    build_rag_prompt,
)

GS_PATH = "tests/data/gold_standard_annales_fr_v8_adversarial.json"


class TestBuildRagPrompt:
    def test_includes_system_prompt(self):
        messages = build_rag_prompt("Quelle cadence ?", "Art 5.1")
        assert SYSTEM_PROMPT in messages[0]["content"]

    def test_includes_question_and_context(self):
        messages = build_rag_prompt("Ma question", "Mon contexte")
        assert "Ma question" in messages[0]["content"]
        assert "Mon contexte" in messages[0]["content"]

    def test_returns_valid_messages_format(self):
        messages = build_rag_prompt("Q", "C")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestCheckCitation:
    def test_detects_la_mention(self):
        response = "Selon le Livre de l'Arbitre, article 6.2"
        assert check_citation(response, ["LA-octobre2025.pdf"], [45])

    def test_rejects_no_citation(self):
        response = "Le joueur doit se presenter."
        assert not check_citation(response, ["LA-octobre2025.pdf"], [45])

    def test_detects_r01_mention(self):
        response = "D'apres les regles generales, section 3"
        assert check_citation(response, ["R01_2025_26_Regles_generales.pdf"], [3])

    def test_detects_page_number(self):
        response = "Voir page 185 du reglement."
        assert check_citation(response, ["LA-octobre2025.pdf"], [185])

    def test_detects_a01_mention(self):
        response = "Le championnat de France stipule que..."
        assert check_citation(response, ["A01_2025_26_Championnat_de_France.pdf"], [12])

    def test_detects_a02_mention(self):
        response = "Le championnat de clubs interdit cela."
        assert check_citation(
            response, ["A02_2025_26_Championnat_de_France_par_equipes.pdf"], [7]
        )

    def test_detects_p_abbreviation(self):
        response = "Cf. p. 185 du document."
        assert check_citation(response, ["LA-octobre2025.pdf"], [185])


class TestLoadQuestions:
    def test_load_human_questions_count(self):
        qs = load_human_questions(GS_PATH)
        assert len(qs) == 34

    def test_load_human_questions_all_human_id(self):
        qs = load_human_questions(GS_PATH)
        assert all(q["id"].startswith("ffe:human:") for q in qs)

    def test_load_human_questions_none_impossible(self):
        qs = load_human_questions(GS_PATH)
        assert all(not q.get("content", {}).get("is_impossible", False) for q in qs)

    def test_load_annales_questions_count(self):
        qs = load_annales_questions(GS_PATH)
        assert len(qs) == 264

    def test_load_annales_questions_none_impossible(self):
        qs = load_annales_questions(GS_PATH)
        assert all(not q.get("content", {}).get("is_impossible", False) for q in qs)

    def test_load_annales_questions_have_annales_source(self):
        qs = load_annales_questions(GS_PATH)
        assert all(
            q.get("provenance", {}).get("annales_source") is not None for q in qs
        )
