"""
Tests for fix_gs_iso_compliance module (Phase 4: ISO enrichment).

100% PURE - no mocks, no embeddings.

ISO Reference:
    - ISO/IEC 29119 - Validation
    - ISO 42001 A.6.2.2 - Provenance tracking
"""

from __future__ import annotations

import copy
from pathlib import Path

from scripts.evaluation.annales.fix_gs_iso_compliance import (
    FFE_DOMAIN_TERMS,
    enrich_article_reference,
    extract_domain_keywords,
    fix_question,
    generate_answer_explanation,
    reformulate_question,
)

# ---------------------------------------------------------------------------
# TestFFEDomainTerms
# ---------------------------------------------------------------------------


class TestFFEDomainTerms:
    """Tests for FFE_DOMAIN_TERMS constant."""

    def test_exact_term_count(self) -> None:
        # Exact count prevents silent regression or unreviewed additions
        assert len(FFE_DOMAIN_TERMS) == 79, (
            f"FFE_DOMAIN_TERMS has {len(FFE_DOMAIN_TERMS)} terms, expected 79. "
            "Update this test if terms were intentionally added/removed."
        )

    def test_contains_essential(self) -> None:
        essentials = {"roi", "dame", "tour", "arbitre", "elo"}
        assert essentials.issubset(FFE_DOMAIN_TERMS)


# ---------------------------------------------------------------------------
# TestExtractDomainKeywords
# ---------------------------------------------------------------------------


class TestExtractDomainKeywords:
    """Tests for extract_domain_keywords."""

    def test_finds_ffe_terms(self) -> None:
        text = "L'arbitre doit surveiller le roi et la dame pendant le tournoi."
        keywords = extract_domain_keywords(text)
        assert "arbitre" in keywords
        assert "roi" in keywords

    def test_excludes_stopwords(self) -> None:
        text = "Le joueur doit faire une action dans la salle avec un arbitre."
        keywords = extract_domain_keywords(text)
        stopwords = {"le", "la", "une", "dans", "avec"}
        for kw in keywords:
            assert kw not in stopwords

    def test_respects_max_keywords(self) -> None:
        text = "Le roi, la dame, le cavalier, le fou, la tour, le pion, l'arbitre, le tournoi, le elo, le blitz, la pendule, le roque"
        keywords = extract_domain_keywords(text, max_keywords=3)
        assert len(keywords) <= 3

    def test_empty_text(self) -> None:
        keywords = extract_domain_keywords("")
        assert keywords == []

    def test_text_without_domain_terms(self) -> None:
        text = "The weather today is very nice and sunny for a walk outside."
        keywords = extract_domain_keywords(text)
        # English text must not yield any FFE chess domain terms
        for kw in keywords:
            assert (
                kw not in FFE_DOMAIN_TERMS
            ), f"Unexpected domain term '{kw}' in English text"


# ---------------------------------------------------------------------------
# TestReformulateQuestion
# ---------------------------------------------------------------------------


class TestReformulateQuestion:
    """Tests for reformulate_question."""

    def test_applies_patterns(self) -> None:
        q = "Que stipule l'article 5.1 du reglement?"
        result = reformulate_question(q, "chunk text", "source.pdf")
        assert "stipule" not in result.lower()

    def test_removes_pdf(self) -> None:
        q = "Quelle regle est enoncee dans reglement_FFE.pdf?"
        result = reformulate_question(q, "chunk text", "reglement_FFE.pdf")
        assert ".pdf" not in result

    def test_adds_question_mark(self) -> None:
        q = "Quelle est la regle applicable"
        result = reformulate_question(q, "chunk text", "source.pdf")
        assert result.endswith("?")

    def test_cleans_spaces(self) -> None:
        q = "Quelle   regle   est   applicable?"
        result = reformulate_question(q, "chunk text", "source.pdf")
        assert "  " not in result

    def test_removes_page_reference(self) -> None:
        q = "Que precise le reglement a la page 42?"
        result = reformulate_question(q, "chunk text", "source.pdf")
        assert "page 42" not in result


# ---------------------------------------------------------------------------
# TestGenerateAnswerExplanation
# ---------------------------------------------------------------------------


class TestGenerateAnswerExplanation:
    """Tests for generate_answer_explanation."""

    def test_includes_source(self) -> None:
        result = generate_answer_explanation(
            "L'arbitre doit verifier",
            "L'arbitre doit verifier les pendules avant la partie.",
            "Art. 5.1",
            "reglement_FFE.pdf",
        )
        assert "reglement" in result.lower()

    def test_includes_article_ref(self) -> None:
        result = generate_answer_explanation(
            "reponse",
            "chunk text avec la reponse incluse",
            "Art. 3.2",
            "source.pdf",
        )
        assert "Art. 3.2" in result

    def test_includes_extrait(self) -> None:
        result = generate_answer_explanation(
            "reponse",
            "Un chunk de texte contenant la reponse attendue.",
            "Art. 1",
            "source.pdf",
        )
        assert "Extrait:" in result


# ---------------------------------------------------------------------------
# TestEnrichArticleReference
# ---------------------------------------------------------------------------


class TestEnrichArticleReference:
    """Tests for enrich_article_reference."""

    def test_keeps_good_ref(self) -> None:
        current = "LA-octobre2025 Art. 5.1 - Obligations de l'arbitre"
        result = enrich_article_reference(current, "chunk text", "source.pdf")
        assert result == current

    def test_extracts_art_from_chunk(self) -> None:
        result = enrich_article_reference(
            "", "Article 7.2 - Les cadences de jeu sont definies.", "regles.pdf"
        )
        assert "Art." in result or "7.2" in result

    def test_extracts_chapitre(self) -> None:
        result = enrich_article_reference(
            "", "Chapitre 3 du reglement interieur.", "reglement.pdf"
        )
        assert "Chap." in result or "3" in result

    def test_fallback_source(self) -> None:
        result = enrich_article_reference(
            "", "Pas de reference d'article ici.", "LA-octobre2025.pdf"
        )
        assert "LA-octobre2025" in result


# ---------------------------------------------------------------------------
# TestFixQuestion
# ---------------------------------------------------------------------------


class TestFixQuestion:
    """Tests for fix_question."""

    def _make_fixable_question(self) -> dict:
        """Create a question needing ISO fixes."""
        return {
            "content": {
                "question": "Que stipule l'article du reglement_test.pdf?",
                "expected_answer": "L'arbitre doit verifier les pendules.",
                "is_impossible": False,
            },
            "mcq": {
                "original_question": "Que stipule l'article du reglement_test.pdf?",
            },
            "provenance": {
                "chunk_id": "test.pdf-p001-parent001-child00",
                "docs": ["test.pdf"],
                "article_reference": "",
                "answer_explanation": "",
            },
            "classification": {
                "keywords": [],
            },
            "audit": {
                "history": "[BY DESIGN] test",
            },
        }

    def test_reformulates_if_stipule(self) -> None:
        q = copy.deepcopy(self._make_fixable_question())
        chunks = {
            "test.pdf-p001-parent001-child00": {
                "text": "L'arbitre doit verifier les pendules avant chaque ronde.",
                "source": "test.pdf",
            }
        }
        fixed = fix_question(q, chunks)
        assert "stipule" not in fixed["content"]["question"].lower()

    def test_fills_empty_explanation(self) -> None:
        q = copy.deepcopy(self._make_fixable_question())
        chunks = {
            "test.pdf-p001-parent001-child00": {
                "text": "L'arbitre doit verifier les pendules.",
                "source": "test.pdf",
            }
        }
        fixed = fix_question(q, chunks)
        assert fixed["provenance"]["answer_explanation"] != ""

    def test_domain_keywords(self) -> None:
        q = copy.deepcopy(self._make_fixable_question())
        chunks = {
            "test.pdf-p001-parent001-child00": {
                "text": "L'arbitre surveille le tournoi de blitz avec pendule.",
                "source": "test.pdf",
            }
        }
        fixed = fix_question(q, chunks)
        assert len(fixed["classification"]["keywords"]) > 0

    def test_audit_iso_fix(self) -> None:
        q = copy.deepcopy(self._make_fixable_question())
        chunks = {
            "test.pdf-p001-parent001-child00": {
                "text": "L'arbitre doit verifier les pendules.",
                "source": "test.pdf",
            }
        }
        fixed = fix_question(q, chunks)
        assert "[ISO FIX]" in fixed["audit"]["history"]


# ---------------------------------------------------------------------------
# TestCreateValidationReport
# ---------------------------------------------------------------------------


class TestCreateValidationReport:
    """Tests for create_validation_report."""

    def _make_gs_question(
        self,
        *,
        is_impossible: bool = False,
        chunk_match_score: int = 100,
    ) -> dict:
        return {
            "content": {
                "question": "Q?",
                "expected_answer": "A" if not is_impossible else "",
                "is_impossible": is_impossible,
            },
            "provenance": {
                "chunk_id": "c1",
                "answer_explanation": "expl",
                "article_reference": "Art. 1",
            },
            "classification": {
                "keywords": ["arbitre", "roi"],
            },
            "processing": {
                "chunk_match_score": chunk_match_score,
            },
        }

    def test_report_structure(self, tmp_path: Path) -> None:
        from scripts.evaluation.annales.fix_gs_iso_compliance import (
            create_validation_report,
        )

        questions = [self._make_gs_question() for _ in range(5)]
        output = tmp_path / "report.json"
        report = create_validation_report(questions, output)
        assert "report_id" in report
        assert "coverage" in report
        assert "provenance_compliance" in report
        assert "status" in report
        assert output.exists()

    def test_unanswerable_ratio(self, tmp_path: Path) -> None:
        from scripts.evaluation.annales.fix_gs_iso_compliance import (
            create_validation_report,
        )

        questions = [self._make_gs_question() for _ in range(7)]
        questions.extend(self._make_gs_question(is_impossible=True) for _ in range(3))
        output = tmp_path / "report.json"
        report = create_validation_report(questions, output)
        assert report["coverage"]["unanswerable_ratio"] == 0.3

    def test_domain_keywords(self, tmp_path: Path) -> None:
        from scripts.evaluation.annales.fix_gs_iso_compliance import (
            create_validation_report,
        )

        questions = [self._make_gs_question()]
        output = tmp_path / "report.json"
        report = create_validation_report(questions, output)
        assert report["semantic_quality"]["domain_keywords_count"] >= 0
