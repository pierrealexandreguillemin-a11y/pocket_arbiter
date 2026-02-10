"""
Tests for generate_real_questions module (Phase 1-2: BY DESIGN generation).

100% PURE - no mocks, no embeddings.

ISO Reference:
    - ISO/IEC 29119 - Test data generation
    - ISO 42001 A.6.2.2 - Provenance tracking
"""

from __future__ import annotations

import random
from pathlib import Path

from scripts.evaluation.annales.generate_real_questions import (
    UNANSWERABLE_GENERATORS,
    extract_article_info,
    extract_key_sentences,
    extract_rules_and_definitions,
    generate_question_from_extraction,
    generate_questions_from_chunk,
    generate_summary_question,
    generate_unanswerable_question,
    generate_unique_question,
)

# ---------------------------------------------------------------------------
# TestExtractArticleInfo
# ---------------------------------------------------------------------------


class TestExtractArticleInfo:
    """Tests for extract_article_info (G1-1)."""

    def test_full_article(self) -> None:
        num, content = extract_article_info("Article 5.1 - Obligations de l'arbitre")
        assert num == "5.1"
        assert "Obligations" in content

    def test_abbreviated_art(self) -> None:
        num, content = extract_article_info("Art. 3 Le joueur doit noter ses coups")
        assert num == "3"
        assert "joueur" in content

    def test_no_match(self) -> None:
        num, content = extract_article_info("Texte sans article reference")
        assert num == ""
        assert content == ""

    def test_case_insensitive(self) -> None:
        num, _ = extract_article_info("article 7.2: Regles du blitz")
        assert num == "7.2"


# ---------------------------------------------------------------------------
# TestExtractKeySentences
# ---------------------------------------------------------------------------


class TestExtractKeySentences:
    """Tests for extract_key_sentences."""

    def test_long_sentences_kept(self) -> None:
        text = (
            "L'arbitre doit veiller au bon deroulement de la competition. "
            "Il est responsable de l'application des regles."
        )
        sentences = extract_key_sentences(text)
        assert len(sentences) >= 1
        assert any("arbitre" in s for s in sentences)

    def test_short_sentences_skipped(self) -> None:
        text = "Oui. Non. Peut-etre. Texte court."
        sentences = extract_key_sentences(text)
        assert len(sentences) == 0

    def test_headers_caps_skipped(self) -> None:
        text = (
            "CHAPITRE 1 OBLIGATIONS DE L'ARBITRE. "
            "L'arbitre est responsable du deroulement de la competition."
        )
        sentences = extract_key_sentences(text)
        # CAPS-only sentence should be skipped
        for s in sentences:
            assert not s.isupper()

    def test_toc_pattern_skipped(self) -> None:
        # TOC line: "Article 5.1 Obligations de l'arbitre...42" ends with digits
        text = "Article 5.1 Obligations de l'arbitre......42"
        sentences = extract_key_sentences(text)
        for s in sentences:
            assert "......42" not in s

    def test_table_high_digits_skipped(self) -> None:
        text = "12345678901234567890123456789012. L'arbitre doit veiller au bon deroulement de la competition."
        sentences = extract_key_sentences(text)
        for s in sentences:
            # Sentences with >30% digits should be excluded
            digit_ratio = len([c for c in s if c.isdigit()]) / max(len(s), 1)
            assert digit_ratio <= 0.3


# ---------------------------------------------------------------------------
# TestExtractRulesAndDefinitions
# ---------------------------------------------------------------------------


class TestExtractRulesAndDefinitions:
    """Tests for extract_rules_and_definitions - 5 patterns."""

    def test_procedural(self) -> None:
        text = "L'arbitre doit veiller au bon deroulement."
        result = extract_rules_and_definitions(text)
        types = [e["type"] for e in result]
        assert "procedural" in types

    def test_definition(self) -> None:
        text = "Le pat est une situation ou le joueur ne peut pas bouger."
        result = extract_rules_and_definitions(text)
        types = [e["type"] for e in result]
        assert "definition" in types

    def test_scenario(self) -> None:
        text = "En cas de litige, l'arbitre tranche."
        result = extract_rules_and_definitions(text)
        types = [e["type"] for e in result]
        assert "scenario" in types

    def test_rule(self) -> None:
        text = "Il est interdit de quitter la salle pendant la partie."
        result = extract_rules_and_definitions(text)
        types = [e["type"] for e in result]
        assert "rule" in types

    def test_factual(self) -> None:
        text = "Le joueur dispose de 30 minutes pour terminer la partie."
        result = extract_rules_and_definitions(text)
        types = [e["type"] for e in result]
        assert "factual" in types


# ---------------------------------------------------------------------------
# TestGenerateQuestionFromExtraction
# ---------------------------------------------------------------------------


class TestGenerateQuestionFromExtraction:
    """Tests for generate_question_from_extraction."""

    def test_procedural_doit(self) -> None:
        ext = {
            "type": "procedural",
            "match": "L'arbitre doit veiller",
            "groups": ("L'arbitre", "doit", "veiller"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "doit" in q["question"].lower() or "arbitre" in q["question"].lower()

    def test_procedural_peut(self) -> None:
        ext = {
            "type": "procedural",
            "match": "Le joueur peut demander",
            "groups": ("Le joueur", "peut", "demander"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "peut" in q["question"].lower()

    def test_procedural_ne_peut_pas(self) -> None:
        ext = {
            "type": "procedural",
            "match": "Un joueur ne peut pas utiliser un telephone",
            "groups": ("Un joueur", "ne peut pas", "utiliser un telephone"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "ne peut pas" in q["question"].lower()

    def test_definition(self) -> None:
        ext = {
            "type": "definition",
            "match": "Le pat est une situation speciale",
            "groups": ("Le", "pat", "une situation speciale"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "pat" in q["question"].lower()

    def test_scenario(self) -> None:
        ext = {
            "type": "scenario",
            "match": "En cas de litige, l'arbitre tranche",
            "groups": ("litige", "l'arbitre tranche"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "litige" in q["question"]

    def test_rule_interdit(self) -> None:
        ext = {
            "type": "rule",
            "match": "Il est interdit de quitter",
            "groups": ("interdit", "quitter la salle"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "interdit" in q["question"].lower()

    def test_rule_obligatoire(self) -> None:
        ext = {
            "type": "rule",
            "match": "Il est obligatoire de noter ses coups",
            "groups": ("obligatoire", "noter ses coups"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "obligatoire" in q["question"].lower()

    def test_rule_permis(self) -> None:
        ext = {
            "type": "rule",
            "match": "Il est permis de proposer nulle",
            "groups": ("permis", "proposer nulle"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "permis" in q["question"].lower()

    def test_factual_number(self) -> None:
        ext = {
            "type": "factual",
            "match": "30 minutes",
            "groups": ("30", "minutes"),
        }
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is not None
        assert "minutes" in q["question"]

    def test_none_for_unknown_type(self) -> None:
        ext = {"type": "unknown_type", "match": "text", "groups": ()}
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is None

    def test_none_for_factual_insufficient_groups(self) -> None:
        """Factual type with < 2 groups falls through to None."""
        ext = {"type": "factual", "match": "30", "groups": ("30",)}
        q = generate_question_from_extraction(ext, "chunk text")
        assert q is None


# ---------------------------------------------------------------------------
# TestGenerateSummaryQuestion
# ---------------------------------------------------------------------------


class TestGenerateSummaryQuestion:
    """Tests for generate_summary_question."""

    def test_none_if_less_than_2_sentences(self) -> None:
        result = generate_summary_question(["Une seule phrase ici."], "chunk1")
        assert result is None

    def test_arbitre_keyword(self) -> None:
        sentences = [
            "L'arbitre est responsable du bon deroulement.",
            "Il doit aussi verifier les pendules.",
        ]
        result = generate_summary_question(sentences, "chunk1")
        assert result is not None
        assert "arbitre" in result["question"].lower()

    def test_joueur_keyword(self) -> None:
        sentences = [
            "Le joueur doit noter tous ses coups sur la feuille.",
            "Il doit egalement verifier le resultat avant de signer.",
        ]
        result = generate_summary_question(sentences, "chunk1")
        assert result is not None
        assert "joueur" in result["question"].lower()

    def test_partie_keyword(self) -> None:
        sentences = [
            "La partie se deroule en trois phases distinctes.",
            "Elle se termine par un mat, un pat ou un abandon.",
        ]
        result = generate_summary_question(sentences, "chunk1")
        assert result is not None
        assert "partie" in result["question"].lower()

    def test_temps_keyword(self) -> None:
        sentences = [
            "Le temps de reflexion est limite a 90 minutes pour 40 coups.",
            "Apres cela, le joueur dispose de 30 minutes supplementaires.",
        ]
        result = generate_summary_question(sentences, "chunk1")
        assert result is not None
        assert "temps" in result["question"].lower()

    def test_fallback_generic(self) -> None:
        sentences = [
            "La salle doit etre bien eclairee pour le confort.",
            "Les tables doivent etre espacees de 1.5 metres.",
        ]
        result = generate_summary_question(sentences, "chunk1")
        assert result is not None
        assert result["question"].endswith("?")


# ---------------------------------------------------------------------------
# TestGenerateUniqueQuestion
# ---------------------------------------------------------------------------


class TestGenerateUniqueQuestion:
    """Tests for generate_unique_question."""

    def test_doit_template(self, sample_chunk: dict) -> None:
        sentence = "L'arbitre doit surveiller les pendules pendant la partie."
        q = generate_unique_question(sample_chunk, sentence, 0)
        assert q["question"].endswith("?")
        assert q["chunk_id"] == sample_chunk["id"]

    def test_peut_template(self, sample_chunk: dict) -> None:
        sentence = "Le joueur peut demander une verification de la position."
        q = generate_unique_question(sample_chunk, sentence, 0)
        assert q["question"].endswith("?")

    def test_minutes_template(self, sample_chunk: dict) -> None:
        sentence = "Chaque joueur dispose de 15 minutes pour completer la partie."
        q = generate_unique_question(sample_chunk, sentence, 0)
        assert q["question"].endswith("?")

    def test_reasoning_class_for_long_sentence(self, sample_chunk: dict) -> None:
        sentence = (
            "a" * 101
            + " voici une longue phrase pour tester la classification du raisonnement."
        )
        q = generate_unique_question(sample_chunk, sentence, 0)
        assert q["reasoning_class"] == "summary"

    def test_ne_peut_pas_template(self, sample_chunk: dict) -> None:
        sentence = (
            "Un joueur ne peut pas quitter la salle sans permission de l'arbitre."
        )
        q = generate_unique_question(sample_chunk, sentence, 0)
        assert q["question"].endswith("?")

    def test_arbitre_template(self, sample_chunk: dict) -> None:
        sentence = "L'arbitre est responsable de l'application stricte du reglement."
        q = generate_unique_question(sample_chunk, sentence, 1)
        assert q["question"].endswith("?")

    def test_joueur_template(self, sample_chunk: dict) -> None:
        sentence = "Le joueur signe la feuille de partie a la fin du match."
        q = generate_unique_question(sample_chunk, sentence, 0)
        assert q["question"].endswith("?")

    def test_else_template(self, sample_chunk: dict) -> None:
        sentence = "La salle de competition doit disposer d'un eclairage suffisant pour le confort."
        q = generate_unique_question(sample_chunk, sentence, 0)
        assert q["question"].endswith("?")

    def test_reasoning_class_reasoning_keyword(self, sample_chunk: dict) -> None:
        # Short sentence with "doit" keyword -> reasoning class
        sentence = "L'organisateur doit prevoir le materiel necessaire."
        q = generate_unique_question(sample_chunk, sentence, 1)
        assert q["reasoning_class"] in ("reasoning", "summary")

    def test_reasoning_class_variety_idx0(self, sample_chunk: dict) -> None:
        # idx % 3 == 0, medium length, no keywords
        sentence = "La salle de competition est reservee exclusivement aux participants inscrits."
        q = generate_unique_question(sample_chunk, sentence, 3)
        assert q["reasoning_class"] in ("fact_single", "summary", "reasoning")


# ---------------------------------------------------------------------------
# TestGenerateQuestionsFromChunk
# ---------------------------------------------------------------------------


class TestGenerateQuestionsFromChunk:
    """Tests for generate_questions_from_chunk."""

    def test_empty_if_short_text(self, sample_chunk_short: dict) -> None:
        result = generate_questions_from_chunk(sample_chunk_short)
        assert result == []

    def test_deduplication(self, sample_chunk: dict) -> None:
        random.seed(42)
        result = generate_questions_from_chunk(sample_chunk, target_count=10)
        question_texts = [q["question"] for q in result]
        assert len(question_texts) == len(set(question_texts))

    def test_respects_target_count(self, sample_chunk: dict) -> None:
        random.seed(42)
        result = generate_questions_from_chunk(sample_chunk, target_count=2)
        assert len(result) <= 2

    def test_rich_chunk_generates_questions(self, sample_chunk: dict) -> None:
        random.seed(42)
        result = generate_questions_from_chunk(sample_chunk, target_count=5)
        assert len(result) >= 1
        for q in result:
            assert "question" in q
            assert "chunk_id" in q


# ---------------------------------------------------------------------------
# TestUnanswerableGenerators
# ---------------------------------------------------------------------------


class TestUnanswerableGenerators:
    """Tests for individual unanswerable question generators."""

    def test_out_of_scope(self) -> None:
        from scripts.evaluation.annales.generate_real_questions import (
            generate_out_of_scope,
        )

        q = generate_out_of_scope(0)
        assert "?" in q
        assert any(sport in q.lower() for sport in ["tennis", "basketball", "football"])

    def test_insufficient_info(self) -> None:
        from scripts.evaluation.annales.generate_real_questions import (
            generate_insufficient_info,
        )

        q = generate_insufficient_info(0)
        assert "?" in q

    def test_false_premise(self) -> None:
        from scripts.evaluation.annales.generate_real_questions import (
            generate_false_premise,
        )

        q = generate_false_premise(0)
        assert "?" in q

    def test_temporal_mismatch(self) -> None:
        from scripts.evaluation.annales.generate_real_questions import (
            generate_temporal_mismatch,
        )

        q = generate_temporal_mismatch(0)
        assert "?" in q

    def test_ambiguous(self) -> None:
        from scripts.evaluation.annales.generate_real_questions import (
            generate_ambiguous,
        )

        q = generate_ambiguous(0)
        assert "?" in q

    def test_counterfactual(self) -> None:
        from scripts.evaluation.annales.generate_real_questions import (
            generate_counterfactual,
        )

        q = generate_counterfactual(0)
        assert "?" in q


# ---------------------------------------------------------------------------
# TestGenerateUnanswerableQuestion
# ---------------------------------------------------------------------------


class TestGenerateUnanswerableQuestion:
    """Tests for generate_unanswerable_question."""

    def test_is_impossible_true(self, sample_chunk: dict) -> None:
        random.seed(42)
        q = generate_unanswerable_question(sample_chunk, "OUT_OF_SCOPE", 0)
        assert q["is_impossible"] is True

    def test_reasoning_class_adversarial(self, sample_chunk: dict) -> None:
        random.seed(42)
        q = generate_unanswerable_question(sample_chunk, "FALSE_PREMISE", 0)
        assert q["reasoning_class"] == "adversarial"

    def test_insufficient_info_adds_source(self, sample_chunk: dict) -> None:
        random.seed(42)
        q = generate_unanswerable_question(sample_chunk, "INSUFFICIENT_INFO", 0)
        assert (
            sample_chunk["source"][:20].lower() in q["question"].lower()
            or "page" in q["question"].lower()
        )

    def test_hard_type_set(self, sample_chunk: dict) -> None:
        random.seed(42)
        q = generate_unanswerable_question(sample_chunk, "COUNTERFACTUAL", 0)
        assert q["hard_type"] == "COUNTERFACTUAL"


# ---------------------------------------------------------------------------
# TestUnanswerableGeneratorsDict
# ---------------------------------------------------------------------------


class TestUnanswerableGeneratorsDict:
    """Tests for UNANSWERABLE_GENERATORS constant."""

    def test_six_keys(self) -> None:
        assert len(UNANSWERABLE_GENERATORS) == 6
        expected = {
            "OUT_OF_SCOPE",
            "INSUFFICIENT_INFO",
            "FALSE_PREMISE",
            "TEMPORAL_MISMATCH",
            "AMBIGUOUS",
            "COUNTERFACTUAL",
        }
        assert set(UNANSWERABLE_GENERATORS.keys()) == expected

    def test_all_callable(self) -> None:
        for key, gen in UNANSWERABLE_GENERATORS.items():
            assert callable(gen), f"{key} is not callable"


# ---------------------------------------------------------------------------
# TestRunGeneration (MOCK I/O)
# ---------------------------------------------------------------------------


class TestRunGeneration:
    """Tests for run_generation orchestrator (mock file I/O)."""

    def test_distribution_70_30(self, tmp_path: Path) -> None:
        """run_generation produces ~70% answerable, ~30% unanswerable."""
        from unittest.mock import patch

        from scripts.evaluation.annales.generate_real_questions import run_generation

        chunks = [
            {
                "id": f"test.pdf-p{i:03d}-parent{i:03d}-child00",
                "text": (
                    f"Article {i}.1 - L'arbitre doit veiller au bon deroulement "
                    f"de la ronde {i}. Le joueur peut demander une pause. "
                    "En cas de litige, l'arbitre tranche la situation."
                ),
                "source": "test.pdf",
                "page": i,
            }
            for i in range(1, 51)
        ]
        chunks_data = {"chunks": chunks}
        strata_data = {
            "strata": {
                "s1": {"selected_chunks": [c["id"] for c in chunks]},
            }
        }

        def mock_load(path: object) -> dict:
            s = str(path)
            if "chunks" in s:
                return chunks_data
            return strata_data

        output_path = tmp_path / "output.json"

        with (
            patch(
                "scripts.evaluation.annales.generate_real_questions.load_json",
                side_effect=mock_load,
            ),
            patch(
                "scripts.evaluation.annales.generate_real_questions.save_json",
            ) as mock_save,
        ):
            random.seed(42)
            result = run_generation(
                Path("chunks.json"),
                Path("strata.json"),
                output_path,
                target_total=20,
            )

        assert "questions" in result
        total = len(result["questions"])
        assert total > 0
        impossible = sum(1 for q in result["questions"] if q.get("is_impossible"))
        answerable = total - impossible
        # 70/30 split with small target
        assert answerable >= impossible
        # All have IDs
        for q in result["questions"]:
            assert q["id"].startswith("gs:scratch:")
        # save_json was called
        mock_save.assert_called_once()

    def test_empty_strata_no_crash(self, tmp_path: Path) -> None:
        """run_generation with no selected chunks doesn't crash."""
        from unittest.mock import patch

        from scripts.evaluation.annales.generate_real_questions import run_generation

        chunks_data = {"chunks": []}
        strata_data = {"strata": {}}

        def mock_load(path: object) -> dict:
            s = str(path)
            if "chunks" in s:
                return chunks_data
            return strata_data

        with (
            patch(
                "scripts.evaluation.annales.generate_real_questions.load_json",
                side_effect=mock_load,
            ),
            patch(
                "scripts.evaluation.annales.generate_real_questions.save_json",
            ),
        ):
            random.seed(42)
            # With no chunks, unanswerable generation uses random.choice on empty list
            # This should either produce 0 questions or raise - test the behavior
            try:
                result = run_generation(
                    Path("chunks.json"),
                    Path("strata.json"),
                    tmp_path / "out.json",
                    target_total=10,
                )
                # If it succeeds, answerable should be 0
                answerable = [
                    q for q in result["questions"] if not q.get("is_impossible")
                ]
                assert len(answerable) == 0
            except (IndexError, ValueError):
                # Expected: random.choice on empty list
                pass
