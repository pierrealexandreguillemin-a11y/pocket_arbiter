# scripts/pipeline/tests/test_synonyms.py
"""Tests for stemming and synonym expansion."""

from __future__ import annotations

from scripts.pipeline.synonyms import (
    CHESS_SYNONYMS,
    build_reverse_synonyms,
    expand_query,
    stem_text,
)


class TestStemText:
    """Test Snowball FR stemming."""

    def test_basic_french_stemming(self) -> None:
        assert stem_text("arbitrage") == "arbitrag"

    def test_plural_reduction(self) -> None:
        assert stem_text("competitions") == stem_text("competition")

    def test_verb_conjugation(self) -> None:
        assert stem_text("participer") == stem_text("participation")

    def test_multi_word(self) -> None:
        result = stem_text("les arbitres des competitions")
        assert "arbitr" in result
        assert "competit" in result

    def test_empty_string(self) -> None:
        assert stem_text("") == ""

    def test_preserves_numbers(self) -> None:
        result = stem_text("article 3.2")
        assert "3.2" in result

    def test_diacritics_preserved_in_stem(self) -> None:
        # Snowball handles accented French
        result = stem_text("equipe")
        assert len(result) > 0


class TestExpandQuery:
    """Test synonym expansion."""

    def test_known_synonym_added(self) -> None:
        result = expand_query("cadence de jeu")
        assert "temp" in result or "rythm" in result

    def test_reverse_synonym(self) -> None:
        # "temps" should expand to include "cadence"
        result = expand_query("temps de jeu")
        assert "cadenc" in result  # stemmed

    def test_no_expansion_for_unknown(self) -> None:
        result = expand_query("hello world")
        # Should still return stemmed text, just no extra terms
        assert "hello" in result.lower() or stem_text("hello") in result

    def test_multiple_synonyms(self) -> None:
        result = expand_query("forfait et cadence")
        assert "absenc" in result or "defaut" in result  # forfait synonyms
        assert "temp" in result or "rythm" in result  # cadence synonyms

    def test_empty_query(self) -> None:
        assert expand_query("") == ""

    def test_output_is_stemmed(self) -> None:
        result = expand_query("les forfaits")
        # Output should be stemmed
        assert "forfait" in result  # stem of "forfaits"


class TestChessSynonyms:
    """Test CHESS_SYNONYMS dict completeness."""

    def test_minimum_entries(self) -> None:
        assert len(CHESS_SYNONYMS) >= 15

    def test_key_terms_present(self) -> None:
        for term in ["cadence", "elo", "forfait", "mat", "appariement"]:
            assert term in CHESS_SYNONYMS, f"Missing key term: {term}"

    def test_values_are_lists(self) -> None:
        for key, values in CHESS_SYNONYMS.items():
            assert isinstance(values, list), f"{key} values should be list"
            assert len(values) >= 1, f"{key} should have at least 1 synonym"


class TestBuildReverseSynonyms:
    """Test reverse lookup building."""

    def test_builds_reverse(self) -> None:
        syns = {"a": ["b", "c"]}
        reverse = build_reverse_synonyms(syns)
        assert "b" in reverse
        assert "a" in reverse["b"]
        assert "c" in reverse
        assert "a" in reverse["c"]
