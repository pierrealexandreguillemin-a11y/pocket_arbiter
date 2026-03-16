"""Tests for PDF extraction with hierarchical headings."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.pipeline.extract import extract_pdf


class TestExtractPdf:
    """Test single PDF extraction."""

    @pytest.mark.slow
    def test_r01_has_multiple_heading_levels(self):
        """R01 should produce at least 2 different heading levels."""
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        levels = set(re.findall(r"^(#{1,6}) ", md, re.MULTILINE))
        assert len(levels) >= 2, f"Expected multiple heading levels, got: {levels}"

    @pytest.mark.slow
    def test_r01_text_faithful(self):
        """Extracted text should contain known R01 content."""
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        assert "Licences" in md
        assert "Forfait" in md or "forfait" in md
        assert "Commission Technique" in md

    @pytest.mark.slow
    def test_r01_tables_extracted(self):
        """R01 has 2 tables (categories + cadences)."""
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        assert len(result["tables"]) >= 2

    @pytest.mark.slow
    def test_heading_levels_match_structure(self):
        """Sub-articles should have deeper heading levels than top articles."""
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        # Find heading levels for a top article and its sub-article
        sub = re.search(r"^(#{1,6}) .*2\.1", md, re.MULTILINE)
        top = re.search(r"^(#{1,6}) .*2\. Statut", md, re.MULTILINE)
        if sub and top:
            assert len(sub.group(1)) > len(top.group(1)), \
                f"Sub-article 2.1 (h{len(sub.group(1))}) should be deeper than 2. (h{len(top.group(1))})"
