"""
Tests for enrich_schema_v2 module (Phase 4: Schema v2.0 enrichment).

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO 42001 A.6.2.2 - Provenance tracking
"""

import pytest

from scripts.evaluation.annales.enrich_schema_v2 import (
    count_schema_fields,
    enrich_to_schema_v2,
    extract_article_reference,
    extract_keywords,
    generate_question_id,
    infer_category,
    infer_reasoning_type,
    validate_schema_compliance,
)


class TestGenerateQuestionId:
    """Tests for question ID generation."""

    def test_la_category(self) -> None:
        """Should generate ID with arbitrage category for LA chunks."""
        qid = generate_question_id("Question?", "LA-octobre2025.pdf-p001")
        assert qid.startswith("gs:scratch:arbitrage:")

    def test_r01_category(self) -> None:
        """Should generate ID with reglements category for R01 chunks."""
        qid = generate_question_id("Question?", "R01_reglement-p001")
        assert qid.startswith("gs:scratch:reglements:")

    def test_interclubs_category(self) -> None:
        """Should generate ID with interclubs category."""
        qid = generate_question_id("Question?", "Interclubs_2024-p001")
        assert qid.startswith("gs:scratch:interclubs:")

    def test_unique_hash(self) -> None:
        """Should generate unique hashes for different questions."""
        qid1 = generate_question_id("Question 1?", "chunk1")
        qid2 = generate_question_id("Question 2?", "chunk1")
        assert qid1 != qid2


class TestExtractArticleReference:
    """Tests for article reference extraction."""

    def test_article_in_section(self) -> None:
        """Should extract article number from section."""
        chunk = {"section": "ARTICLE 5.1 - Regles"}
        ref = extract_article_reference(chunk)
        assert "Article 5.1" in ref

    def test_article_in_text(self) -> None:
        """Should extract article from text if not in section."""
        chunk = {"section": "", "text": "Selon l'article 3.2, le joueur..."}
        ref = extract_article_reference(chunk)
        assert "Article 3.2" in ref

    def test_chapter_fallback(self) -> None:
        """Should extract chapter if no article found."""
        chunk = {"section": "", "text": "Chapitre 7 traite de..."}
        ref = extract_article_reference(chunk)
        assert "Chapitre 7" in ref

    def test_section_fallback(self) -> None:
        """Should fall back to section name."""
        chunk = {"section": "Introduction generale", "text": "Lorem ipsum"}
        ref = extract_article_reference(chunk)
        assert "Introduction" in ref

    def test_empty_result(self) -> None:
        """Should return empty string when nothing found."""
        chunk = {"section": "", "text": ""}
        ref = extract_article_reference(chunk)
        assert ref == ""


class TestInferCategory:
    """Tests for category inference."""

    def test_arbitrage_keywords(self) -> None:
        """Should detect arbitrage category."""
        chunk = {"text": "L'arbitre doit verifier..."}
        cat = infer_category(chunk, "Comment l'arbitre doit-il agir?")
        assert cat == "arbitrage"

    def test_regles_jeu_keywords(self) -> None:
        """Should detect regles_jeu category."""
        chunk = {"text": "Le roque est permis si..."}
        cat = infer_category(chunk, "Quand peut-on roquer?")
        assert cat == "regles_jeu"

    def test_competition_keywords(self) -> None:
        """Should detect competition category."""
        chunk = {"text": "Le tournoi se deroule en 9 rondes..."}
        cat = infer_category(chunk, "Combien de rondes?")
        assert cat == "competition"

    def test_default_general(self) -> None:
        """Should default to general category."""
        chunk = {"text": "Lorem ipsum dolor sit amet"}
        cat = infer_category(chunk, "Question generale?")
        assert cat == "general"


class TestInferReasoningType:
    """Tests for reasoning type inference."""

    def test_single_hop_what(self) -> None:
        """Should detect single-hop for 'qu'est-ce' questions."""
        rtype = infer_reasoning_type("Qu'est-ce que le pat?")
        assert rtype == "single-hop"

    def test_multi_hop_why(self) -> None:
        """Should detect multi-hop for 'pourquoi' questions."""
        rtype = infer_reasoning_type("Pourquoi la partie est-elle nulle?")
        assert rtype == "multi-hop"

    def test_temporal_when(self) -> None:
        """Should detect temporal for 'quand' questions."""
        rtype = infer_reasoning_type("Quand le roque est-il interdit?")
        assert rtype == "temporal"

    def test_comparative(self) -> None:
        """Should detect comparative questions."""
        # Use a question that doesn't start with "Quel" to avoid single-hop match
        rtype = infer_reasoning_type(
            "En quoi se distingue la difference entre pat et mat?"
        )
        assert rtype == "comparative"

    def test_default_single_hop(self) -> None:
        """Should default to single-hop."""
        rtype = infer_reasoning_type("Une question simple.")
        assert rtype == "single-hop"


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_removes_stopwords(self) -> None:
        """Should remove French stopwords."""
        keywords = extract_keywords("Le joueur dans la partie avec les pieces")
        assert "joueur" in keywords
        assert "partie" in keywords
        assert "pieces" in keywords
        # Stopwords should be excluded
        assert "dans" not in keywords

    def test_min_length_filter(self) -> None:
        """Should filter short words."""
        keywords = extract_keywords("Un jeu de test")
        # Words < 4 chars should be excluded
        assert "jeu" not in keywords
        assert "test" in keywords

    def test_max_keywords_limit(self) -> None:
        """Should respect max keywords limit."""
        text = " ".join([f"keyword{i}" for i in range(20)])
        keywords = extract_keywords(text, max_keywords=5)
        assert len(keywords) <= 5


class TestEnrichToSchemaV2:
    """Tests for Schema v2.0 enrichment."""

    @pytest.fixture
    def sample_question(self) -> dict:
        """Sample question for enrichment tests."""
        return {
            "question": "Combien de secondes pour un coup rapide?",
            "expected_answer": "60 secondes selon le reglement.",
            "difficulty": 0.5,
            "question_type": "factual",
            "cognitive_level": "Remember",
            "reasoning_class": "fact_single",
        }

    @pytest.fixture
    def sample_chunk(self) -> dict:
        """Sample chunk for enrichment tests."""
        return {
            "id": "LA-octobre2025.pdf-p010-parent01",
            "source": "LA-octobre2025.pdf",
            "page": 10,
            "pages": [10, 11],
            "text": "Le temps de reflexion est de 60 secondes par coup.",
            "section": "Article 5.2 - Cadences",
        }

    def test_creates_all_groups(
        self, sample_question: dict, sample_chunk: dict
    ) -> None:
        """Should create all 8 required groups."""
        result = enrich_to_schema_v2(sample_question, sample_chunk)

        required_groups = [
            "content",
            "mcq",
            "provenance",
            "classification",
            "validation",
            "processing",
            "audit",
        ]
        for group in required_groups:
            assert group in result, f"Missing group: {group}"

    def test_by_design_markers(self, sample_question: dict, sample_chunk: dict) -> None:
        """Should mark as BY DESIGN generation."""
        result = enrich_to_schema_v2(sample_question, sample_chunk)

        assert result["processing"]["chunk_match_score"] == 100
        assert result["processing"]["chunk_match_method"] == "by_design_input"
        assert "by_design" in result["processing"]["extraction_flags"]
        assert "BY DESIGN" in result["audit"]["history"]

    def test_provenance_tracking(
        self, sample_question: dict, sample_chunk: dict
    ) -> None:
        """Should track provenance correctly (ISO 42001)."""
        result = enrich_to_schema_v2(sample_question, sample_chunk)

        assert result["provenance"]["chunk_id"] == sample_chunk["id"]
        assert sample_chunk["source"] in result["provenance"]["docs"]
        assert result["provenance"]["pages"] == [10, 11]

    def test_unanswerable_handling(self, sample_chunk: dict) -> None:
        """Should handle unanswerable questions correctly."""
        unanswerable_q = {
            "question": "Quel est le salaire d'un arbitre?",
            "expected_answer": "",
            "is_impossible": True,
            "hard_type": "OUT_OF_SCOPE",
        }

        result = enrich_to_schema_v2(unanswerable_q, sample_chunk)

        assert result["content"]["is_impossible"] is True
        assert result["classification"]["hard_type"] == "OUT_OF_SCOPE"
        assert result["processing"]["triplet_ready"] is False


class TestCountSchemaFields:
    """Tests for schema field counting."""

    def test_counts_root_fields(self) -> None:
        """Should count root fields."""
        question = {"id": "test", "legacy_id": ""}
        count = count_schema_fields(question)
        assert count >= 2

    def test_counts_group_fields(self) -> None:
        """Should count fields in groups."""
        question = {
            "id": "test",
            "legacy_id": "",
            "content": {
                "question": "Q?",
                "expected_answer": "A",
                "is_impossible": False,
            },
            "mcq": {},
            "provenance": {},
            "classification": {},
            "validation": {},
            "processing": {},
            "audit": {},
        }
        count = count_schema_fields(question)
        assert count >= 5  # At least root (2) + content (3)


class TestValidateSchemaCompliance:
    """Tests for schema validation."""

    @pytest.fixture
    def valid_question(self) -> dict:
        """A valid Schema v2.0 question."""
        return {
            "id": "gs:scratch:test:001:abc12345",
            "legacy_id": "",
            "content": {
                "question": "Test question?",
                "expected_answer": "Test answer",
                "is_impossible": False,
            },
            "mcq": {
                "original_question": "Test?",
                "choices": {},
                "mcq_answer": "",
                "correct_answer": "Test answer",
                "original_answer": "Test answer",
            },
            "provenance": {
                "chunk_id": "chunk-001",
                "docs": ["test.pdf"],
                "pages": [1],
                "article_reference": "Article 1",
                "answer_explanation": "",
                "annales_source": None,
            },
            "classification": {
                "category": "general",
                "keywords": ["test"],
                "difficulty": 0.5,
                "question_type": "factual",
                "cognitive_level": "Remember",
                "reasoning_type": "single-hop",
                "reasoning_class": "fact_single",
                "answer_type": "extractive",
                "hard_type": "ANSWERABLE",
            },
            "validation": {
                "status": "VALIDATED",
                "method": "by_design_generation",
                "reviewer": "claude_code",
                "answer_current": True,
                "verified_date": "2026-02-05",
                "pages_verified": True,
                "batch": "test",
            },
            "processing": {
                "chunk_match_score": 100,
                "chunk_match_method": "by_design_input",
                "reasoning_class_method": "generation_prompt",
                "triplet_ready": True,
                "extraction_flags": ["by_design"],
                "answer_source": "chunk_extraction",
                "quality_score": 0.8,
            },
            "audit": {
                "history": "[BY DESIGN] Test",
                "qat_revalidation": None,
                "requires_inference": False,
            },
        }

    def test_valid_question_passes(self, valid_question: dict) -> None:
        """Should pass validation for valid question."""
        passed, errors = validate_schema_compliance(valid_question)
        assert passed, f"Validation failed: {errors}"

    def test_missing_id_fails(self, valid_question: dict) -> None:
        """Should fail when ID is missing."""
        valid_question["id"] = ""
        passed, errors = validate_schema_compliance(valid_question)
        assert not passed
        assert any("id" in e.lower() for e in errors)

    def test_missing_chunk_id_fails(self, valid_question: dict) -> None:
        """Should fail when provenance.chunk_id is empty."""
        valid_question["provenance"]["chunk_id"] = ""
        passed, errors = validate_schema_compliance(valid_question)
        assert not passed
        assert any("chunk_id" in e for e in errors)

    def test_wrong_match_method_fails(self, valid_question: dict) -> None:
        """Should fail when chunk_match_method is not by_design_input."""
        valid_question["processing"]["chunk_match_method"] = "post_hoc"
        passed, errors = validate_schema_compliance(valid_question)
        assert not passed
        assert any("by_design_input" in e for e in errors)
