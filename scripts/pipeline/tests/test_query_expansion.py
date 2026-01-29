"""
Tests for Query Expansion Module

ISO Reference:
    - ISO/IEC 29119 - Test execution
"""

from scripts.pipeline.query_expansion import (
    CHESS_SYNONYMS,
    expand_query,
    expand_query_bm25,
    get_query_keywords,
    normalize_text,
)


class TestNormalizeText:
    """Tests for normalize_text()."""

    def test_removes_accents(self):
        """Accents are removed."""
        assert normalize_text("réclamation") == "reclamation"
        assert normalize_text("dépassé") == "depasse"
        assert normalize_text("départage") == "departage"
        assert normalize_text("échecs") == "echecs"

    def test_lowercase(self):
        """Text is lowercased."""
        assert normalize_text("BLITZ") == "blitz"
        assert normalize_text("Buchholz") == "buchholz"

    def test_combined(self):
        """Combined normalization."""
        assert normalize_text("Règles du BLITZ") == "regles du blitz"


class TestExpandQuery:
    """Tests for expand_query()."""

    def test_expands_blitz(self):
        """Blitz is expanded with synonyms."""
        result = expand_query("Quelles sont les règles du blitz ?")
        assert "blitz" in result.lower()
        assert "rapide" in result.lower()
        assert "eclair" in result.lower()

    def test_expands_reclamation(self):
        """Réclamation is expanded with synonyms."""
        result = expand_query("Comment gérer une réclamation ?")
        assert "réclamation" in result
        assert "contestation" in result.lower()
        assert "plainte" in result.lower()

    def test_expands_temps_depasse(self):
        """Temps dépassé is expanded with drapeau."""
        result = expand_query("Réclamation pour temps dépassé")
        assert "drapeau" in result.lower()

    def test_expands_departage(self):
        """Départage is expanded with tie-break."""
        result = expand_query("Comment se déroule un départage ?")
        assert "tie-break" in result.lower()
        assert "buchholz" in result.lower()

    def test_no_false_positive_mat(self):
        """'mat' in 'matériel' should not be expanded."""
        result = expand_query("Conditions matérielles minimales")
        # Should not contain chess "mat" synonyms
        assert "echec et mat" not in result.lower()
        assert "mater" not in result.lower()

    def test_no_expansion_for_unknown(self):
        """Unknown terms are not expanded."""
        original = "Quelle est la météo ?"
        result = expand_query(original)
        assert result == original

    def test_max_expansions_limit(self):
        """Respects max_expansions limit."""
        result = expand_query("règles du blitz", max_expansions=1)
        # Should only have 1 expansion per term
        assert "rapide" in result.lower()
        # With max=1, should not have all synonyms
        expansion_count = sum(
            1 for syn in CHESS_SYNONYMS["blitz"] if syn.lower() in result.lower()
        )
        assert expansion_count <= 1


class TestExpandQueryBm25:
    """Tests for expand_query_bm25()."""

    def test_returns_keywords_only(self):
        """Returns keywords without full question."""
        result = expand_query_bm25("Quelles sont les règles du blitz ?")
        # Should be keywords, not full sentences
        assert "quelles" not in result.lower()
        assert "blitz" in result.lower()
        assert "rapide" in result.lower()

    def test_deduplicates(self):
        """Exact duplicates are removed."""
        result = expand_query_bm25("blitz blitz")
        words = result.lower().split()
        # "blitz" should appear only once (deduped)
        assert words.count("blitz") == 1
        # Synonyms should also be deduped
        assert words.count("rapide") == 1

    def test_empty_for_no_match(self):
        """Returns empty for unknown terms."""
        result = expand_query_bm25("Quelle météo ?")
        # May have some matches from common words, but chess terms absent
        assert "blitz" not in result


class TestGetQueryKeywords:
    """Tests for get_query_keywords()."""

    def test_removes_stopwords(self):
        """French stopwords are removed."""
        result = get_query_keywords("Quelle est la règle du toucher-jouer ?")
        assert "est" not in result
        assert "la" not in result
        assert "du" not in result

    def test_keeps_meaningful_words(self):
        """Meaningful words are kept."""
        result = get_query_keywords("Quelle est la règle du toucher-jouer ?")
        assert "regle" in result or "règle" in result
        assert "toucher" in result


class TestChessSynonyms:
    """Tests for CHESS_SYNONYMS dictionary."""

    def test_has_key_terms(self):
        """Dictionary contains key chess terms."""
        assert "blitz" in CHESS_SYNONYMS
        assert "reclamation" in CHESS_SYNONYMS
        assert "departage" in CHESS_SYNONYMS
        assert "temps" in CHESS_SYNONYMS

    def test_synonyms_are_lists(self):
        """Each value is a non-empty list."""
        for term, synonyms in CHESS_SYNONYMS.items():
            assert isinstance(synonyms, list), f"{term} should have list"
            assert len(synonyms) > 0, f"{term} should have synonyms"

    def test_no_duplicate_synonyms(self):
        """No duplicate synonyms per term."""
        for term, synonyms in CHESS_SYNONYMS.items():
            normalized = [normalize_text(s) for s in synonyms]
            assert len(normalized) == len(set(normalized)), (
                f"{term} has duplicate synonyms"
            )


class TestIntegrationWithSearch:
    """Integration tests with search pipeline."""

    def test_expansion_formats_correctly(self):
        """Expanded query is properly formatted."""
        result = expand_query("réclamation temps dépassé", include_original=True)
        # Should start with original
        assert result.startswith("réclamation temps dépassé")
        # Should have expansions after
        assert len(result) > len("réclamation temps dépassé")

    def test_bm25_expansion_is_or_friendly(self):
        """BM25 expansion produces OR-friendly keywords."""
        result = expand_query_bm25("blitz")
        # Keywords should be space-separated (for OR in FTS)
        assert " " in result
        # No special FTS characters
        assert "AND" not in result
        assert "OR" not in result
