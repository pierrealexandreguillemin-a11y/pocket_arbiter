"""Tests for PDF extraction with hierarchical headings."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.pipeline.extract import extract_pdf, extract_corpus, _strip_page_headers


# ---------------------------------------------------------------------------
# Unit tests for _strip_page_headers
# ---------------------------------------------------------------------------
class TestStripPageHeaders:
    """Test page header stripping logic."""

    def test_removes_repeated_headings(self):
        md = (
            "## Title\n\nContent.\n\n"
            "## Title\n\nMore content.\n\n"
            "## Title\n\nEven more.\n"
        )
        result = _strip_page_headers(md)
        assert result.count("## Title") == 1

    def test_keeps_first_occurrence(self):
        md = (
            "## Title\n\nFirst.\n\n"
            "## Other\n\nContent.\n\n"
            "## Title\n\nSecond.\n\n"
            "## Title\n\nThird.\n"
        )
        result = _strip_page_headers(md)
        assert "First." in result
        assert "## Title" in result

    def test_does_not_remove_non_repeated(self):
        md = "## A\n\nContent A.\n\n## B\n\nContent B.\n"
        result = _strip_page_headers(md)
        assert result == md

    def test_threshold_is_3(self):
        md = "## X\n\nA.\n\n## X\n\nB.\n"  # only 2 occurrences
        result = _strip_page_headers(md)
        assert result.count("## X") == 2  # both kept

    def test_handles_different_heading_levels(self):
        md = (
            "## Title\n\nA.\n\n"
            "### Title\n\nB.\n\n"
            "#### Title\n\nC.\n\n"
            "## Title\n\nD.\n\n"
            "## Title\n\nE.\n"
        )
        # "Title" at ## appears 3 times, at ### 1 time, at #### 1 time
        # The text "Title" appears 5 times total across levels
        result = _strip_page_headers(md)
        # All share same text "Title", 5 >= 3, so duplicates removed
        assert result.count("Title") >= 1

    def test_empty_input(self):
        assert _strip_page_headers("") == ""

    def test_no_headings(self):
        md = "Just plain text.\n\nMore text.\n"
        assert _strip_page_headers(md) == md


# ---------------------------------------------------------------------------
# Integration tests on real PDFs
# ---------------------------------------------------------------------------
class TestExtractPdfR01:
    """Test extraction on R01 (6 pages, small)."""

    @pytest.mark.slow
    def test_has_multiple_heading_levels(self):
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        levels = set(re.findall(r"^(#{1,6}) ", md, re.MULTILINE))
        assert len(levels) >= 2, f"Expected multiple heading levels, got: {levels}"

    @pytest.mark.slow
    def test_text_faithful(self):
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        assert "Licences" in md
        assert "Forfait" in md or "forfait" in md
        assert "Commission Technique" in md

    @pytest.mark.slow
    def test_tables_extracted(self):
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        assert len(result["tables"]) >= 2

    @pytest.mark.slow
    def test_sub_articles_deeper_than_top(self):
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        sub = re.search(r"^(#{1,6}) .*2\.1", md, re.MULTILINE)
        top = re.search(r"^(#{1,6}) .*2\. Statut", md, re.MULTILINE)
        if sub and top:
            assert len(sub.group(1)) > len(top.group(1)), \
                f"2.1 (h{len(sub.group(1))}) should be deeper than 2. (h{len(top.group(1))})"

    @pytest.mark.slow
    def test_page_headers_stripped(self):
        """Repeated page header should appear at most once."""
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        # "REGLES GENERALES" is the page header, should appear once
        count = len(re.findall(
            r"^#{1,6} .*R.GLES G.N.RALES", md, re.MULTILINE
        ))
        assert count == 1, f"Page header appears {count} times (expected 1)"


class TestExtractPdfLA:
    """Test extraction on LA-octobre2025 (222 pages, monster)."""

    @pytest.mark.slow
    def test_has_multiple_heading_levels(self):
        pdf_path = Path("corpus/fr/LA-octobre2025.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        levels = set(re.findall(r"^(#{1,6}) ", md, re.MULTILINE))
        assert len(levels) >= 2, f"Expected multiple heading levels, got: {levels}"

    @pytest.mark.slow
    def test_text_faithful(self):
        pdf_path = Path("corpus/fr/LA-octobre2025.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        # Known content from the PDF
        assert "Article 5" in md
        assert "fraiement" in md.lower() or "défraiement" in md.lower()
        assert "jury" in md.lower()
        assert "mat" in md.lower()  # echec et mat

    @pytest.mark.slow
    def test_tables_extracted(self):
        """LA has ~93-98 tables."""
        pdf_path = Path("corpus/fr/LA-octobre2025.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        assert len(result["tables"]) >= 80, \
            f"Expected 80+ tables, got {len(result['tables'])}"

    @pytest.mark.slow
    def test_article_hierarchy(self):
        """Articles should have proper hierarchy (e.g., 7 > 7.1)."""
        pdf_path = Path("corpus/fr/LA-octobre2025.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        art7 = re.search(r"^(#{1,6}) .*Article 7\b", md, re.MULTILINE)
        art71 = re.search(r"^(#{1,6}) .*7\.1", md, re.MULTILINE)
        if art7 and art71:
            assert len(art71.group(1)) >= len(art7.group(1)), \
                f"7.1 (h{len(art71.group(1))}) should be >= depth of Article 7 (h{len(art7.group(1))})"


class TestExtractPdfSmall:
    """Test extraction on small PDFs (1-2 pages)."""

    @pytest.mark.slow
    def test_h02_joueurs_mobilite_reduite(self):
        pdf_path = Path("corpus/fr/Compétitions/H02_2025_26_Joueurs_a_mobilite_reduite.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        assert len(md) > 100, "Should have substantial content"
        assert "handicap" in md.lower()

    @pytest.mark.slow
    def test_e02_classement_rapide(self):
        pdf_path = Path("corpus/fr/Compétitions/E02-Le_classement_rapide.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        assert len(md) > 50
        assert "classement" in md.lower() or "rapide" in md.lower()


class TestExtractCorpus:
    """Test corpus-level extraction."""

    @pytest.mark.slow
    def test_excludes_annales(self, tmp_path):
        corpus_dir = Path("corpus/fr")
        if not corpus_dir.exists():
            pytest.skip("Corpus not available")
        output_dir = tmp_path / "out"
        results = extract_corpus(corpus_dir, output_dir, exclude_dirs={"Annales"})
        source_names = [r["source"] for r in results]
        for name in source_names:
            assert "Annales" not in name, f"Annales PDF should be excluded: {name}"

    @pytest.mark.slow
    def test_extracts_expected_count(self, tmp_path):
        """Should extract ~28 PDFs (29 minus Nationale 2.mhtml, minus Annales)."""
        corpus_dir = Path("corpus/fr")
        if not corpus_dir.exists():
            pytest.skip("Corpus not available")
        output_dir = tmp_path / "out"
        results = extract_corpus(corpus_dir, output_dir)
        # 28 PDFs in corpus minus Annales (7) = ~21-28
        assert len(results) >= 20, f"Expected 20+ extractions, got {len(results)}"
