"""Tests for enrichment module (OPT 1-2-4: context, abbreviations, chapter overrides)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.pipeline.enrichment import (
    ABBREVIATIONS,
    CHAPTER_OVERRIDES,
    apply_chapter_override,
    enrich_chunks,
    expand_abbreviations,
    load_contexts,
)

# === OPT-2: Abbreviation expansion ===


class TestExpandAbbreviations:
    """Tests for abbreviation expansion in chunk text."""

    def test_expands_known_abbreviation(self) -> None:
        text = "Le DNA organise les examens."
        result = expand_abbreviations(text)
        assert "DNA (Direction Nationale de l'Arbitrage)" in result

    def test_word_boundary_only(self) -> None:
        """Must not expand inside words (e.g. 'FDNA' should not match 'DNA')."""
        text = "Le protocole FDNA est different."
        result = expand_abbreviations(text)
        assert "FDNA" in result
        assert "(Direction" not in result

    def test_skip_already_expanded(self) -> None:
        """If abbreviation is already followed by '(', skip it."""
        text = "La FFE (Federation Francaise des Echecs) organise."
        result = expand_abbreviations(text)
        # Should not double-expand
        assert result.count("(Federation") == 1

    def test_multiple_abbreviations(self) -> None:
        text = "La FFE et la FIDE organisent les UV."
        result = expand_abbreviations(text)
        assert "FFE (Federation Francaise des Echecs)" in result
        assert "FIDE (Federation Internationale des Echecs)" in result
        assert "UV (Unite de Valeur)" in result

    def test_no_match_returns_unchanged(self) -> None:
        text = "Le joueur avance son pion en e4."
        result = expand_abbreviations(text)
        assert result == text

    def test_case_sensitive(self) -> None:
        """Abbreviations are uppercase — 'fide' should not match."""
        text = "Le mot fide n'est pas une abbreviation."
        result = expand_abbreviations(text)
        assert result == text

    def test_all_keys_have_nonempty_expansion(self) -> None:
        for key, val in ABBREVIATIONS.items():
            assert len(key) >= 2, f"Key too short: {key}"
            assert len(val) > len(key), f"Expansion not longer: {key} -> {val}"


class TestAbbreviationsCorpus:
    """Verify abbreviation dict against real corpus (gate E2)."""

    @pytest.mark.slow
    def test_all_abbreviations_match_corpus(self) -> None:
        """Each abbreviation key must match >= 1 chunk in the corpus DB."""
        import sqlite3

        db_path = Path("corpus/processed/corpus_v2_fr.db")
        if not db_path.exists():
            pytest.skip("corpus DB not available")

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT text FROM children").fetchall()
        all_text = " ".join(r[0] for r in rows)
        conn.close()

        for abbr in ABBREVIATIONS:
            count = len(re.findall(rf"\b{abbr}\b", all_text))
            assert count > 0, f"Abbreviation '{abbr}' has 0 matches in corpus"


# === OPT-4: Chapter overrides ===


class TestApplyChapterOverride:
    """Tests for chapter title overrides (LA-specific pages)."""

    def test_override_elo_standard(self) -> None:
        result = apply_chapter_override("LA-octobre2025.pdf", 183, "Some CCH")
        assert result == "Classement Elo Standard FIDE"

    def test_override_boundary_start(self) -> None:
        result = apply_chapter_override("LA-octobre2025.pdf", 182, "Original")
        assert result == "Classement Elo Standard FIDE"

    def test_override_boundary_end(self) -> None:
        result = apply_chapter_override("LA-octobre2025.pdf", 186, "Original")
        assert result == "Classement Elo Standard FIDE"

    def test_no_override_outside_range(self) -> None:
        result = apply_chapter_override("LA-octobre2025.pdf", 181, "Original CCH")
        assert result == "Original CCH"

    def test_no_override_other_source(self) -> None:
        """Overrides only apply to LA-octobre2025.pdf."""
        result = apply_chapter_override("R01_2025_26_Regles_generales.pdf", 183, "X")
        assert result == "X"

    def test_none_page_returns_original(self) -> None:
        result = apply_chapter_override("LA-octobre2025.pdf", None, "Original")
        assert result == "Original"

    def test_all_override_ranges_no_overlap(self) -> None:
        """Override page ranges must not overlap."""
        ranges = list(CHAPTER_OVERRIDES.keys())
        for i, (s1, e1) in enumerate(ranges):
            for s2, e2 in ranges[i + 1 :]:
                assert e1 < s2 or e2 < s1, f"Overlap: ({s1},{e1}) vs ({s2},{e2})"


# === OPT-1: Context loader ===


class TestLoadContexts:
    """Tests for loading chunk_contexts.json."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        ctx = {"chunk-001": "Context one.", "chunk-002": "Context two."}
        f = tmp_path / "contexts.json"
        f.write_text(json.dumps(ctx), encoding="utf-8")

        result = load_contexts(f)
        assert result == ctx
        assert len(result) == 2

    def test_load_empty_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "contexts.json"
        f.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="empty"):
            load_contexts(f)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_contexts(tmp_path / "missing.json")


# === enrich_chunks (orchestration) ===


class TestEnrichChunks:
    """Tests for enrich_chunks orchestration."""

    def test_prepends_context_and_expands(self) -> None:
        children = [
            {"id": "c001", "text": "Le DNA organise."},
            {"id": "c002", "text": "Un simple texte."},
        ]
        contexts = {
            "c001": "Section arbitrage du reglement FFE.",
            "c002": "Contexte du chunk deux.",
        }
        result = enrich_chunks(children, contexts)

        # OPT-1: context prepended
        assert result[0]["text"].startswith("Section arbitrage")
        # OPT-2: abbreviation expanded in both context and chunk
        assert "DNA (Direction Nationale de l'Arbitrage)" in result[0]["text"]
        assert "FFE (Federation Francaise des Echecs)" in result[0]["text"]
        # Original text still present after context
        assert "organise." in result[0]["text"]

    def test_missing_context_no_prepend(self) -> None:
        children = [{"id": "c999", "text": "Texte original."}]
        contexts = {"c001": "Autre chunk."}
        enrich_chunks(children, contexts)
        # No context prepended, abbreviations still run
        assert children[0]["text"] == "Texte original."

    def test_mutates_in_place(self) -> None:
        children = [{"id": "c1", "text": "La FFE decide."}]
        contexts = {"c1": "Contexte."}
        result = enrich_chunks(children, contexts)
        assert result is children  # Same list object
        assert "Contexte." in children[0]["text"]

    def test_context_separator(self) -> None:
        """Context and text separated by double newline (Anthropic pattern)."""
        children = [{"id": "x", "text": "Contenu."}]
        contexts = {"x": "Mon contexte."}
        enrich_chunks(children, contexts)
        assert "Mon contexte.\n\nContenu." in children[0]["text"]
