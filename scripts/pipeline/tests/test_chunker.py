"""
Tests unitaires pour chunker.py

ISO Reference: ISO/IEC 29119 - Test execution

Note: Ce fichier teste UNIQUEMENT les fonctions implementees.
Les fonctions stub (NotImplementedError) n'ont PAS de tests car elles
ne sont pas encore implementees. Les tests seront ajoutes lors de
l'implementation reelle en Phase 1A.

Fonctions testees:
- generate_chunk_id() - Implementation complete
- validate_chunk_schema() (dans utils.py) - Implementation complete

Fonctions NON testees (stubs):
- chunk_text() - A implementer
- count_tokens() - A implementer
- chunk_document() - A implementer
- chunk_corpus() - A implementer
"""

import pytest

from scripts.pipeline.chunker import generate_chunk_id


class TestGenerateChunkId:
    """Tests pour generate_chunk_id() - fonction implementee."""

    def test_generate_fr_id(self):
        """Genere un ID pour corpus FR."""
        chunk_id = generate_chunk_id("fr", 1, 15, 1)
        assert chunk_id == "FR-001-015-01"

    def test_generate_intl_id(self):
        """Genere un ID pour corpus INTL."""
        chunk_id = generate_chunk_id("intl", 1, 42, 3)
        assert chunk_id == "INTL-001-042-03"

    def test_generate_large_numbers(self):
        """Gere les grands numeros."""
        chunk_id = generate_chunk_id("fr", 123, 456, 78)
        assert chunk_id == "FR-123-456-78"

    def test_invalid_corpus_raises(self):
        """Leve ValueError pour corpus invalide."""
        with pytest.raises(ValueError):
            generate_chunk_id("invalid", 1, 1, 1)

    def test_case_insensitive_corpus(self):
        """Accepte corpus en minuscules."""
        chunk_id = generate_chunk_id("FR", 1, 1, 1)
        assert chunk_id.startswith("FR-")


class TestChunkValidation:
    """Tests de validation des chunks via utils.validate_chunk_schema()."""

    def test_chunk_schema_valid(self, sample_chunk: dict):
        """Valide un chunk conforme."""
        from scripts.pipeline.utils import validate_chunk_schema

        errors = validate_chunk_schema(sample_chunk)
        assert errors == []

    def test_chunk_schema_missing_field(self, sample_chunk: dict):
        """Detecte un champ manquant."""
        from scripts.pipeline.utils import validate_chunk_schema

        del sample_chunk["id"]
        errors = validate_chunk_schema(sample_chunk)
        assert any("id" in e for e in errors)

    def test_chunk_schema_invalid_id(self, sample_chunk: dict):
        """Detecte un ID invalide."""
        from scripts.pipeline.utils import validate_chunk_schema

        sample_chunk["id"] = "invalid-id-format"
        errors = validate_chunk_schema(sample_chunk)
        assert any("ID format" in e for e in errors)


class TestCountTokens:
    """Tests pour count_tokens() - fonction implementee."""

    def test_count_simple(self):
        """Compte tokens d'un texte simple."""
        from scripts.pipeline.chunker import count_tokens

        tokens = count_tokens("Hello world")
        assert tokens >= 2

    def test_count_french_text(self):
        """Compte tokens de texte francais."""
        from scripts.pipeline.chunker import count_tokens

        tokens = count_tokens("Article 4.3 - Le toucher-jouer")
        assert tokens > 0

    def test_count_empty(self):
        """Compte tokens d'un texte vide."""
        from scripts.pipeline.chunker import count_tokens

        tokens = count_tokens("")
        assert tokens == 0


class TestChunkText:
    """Tests pour chunk_text() - fonction implementee."""

    def test_chunk_short_text(self):
        """Texte court = un seul chunk."""
        from scripts.pipeline.chunker import chunk_text

        text = "Article 4.1 - Le toucher-jouer. Texte de test suffisamment long."
        chunks = chunk_text(
            text, max_tokens=256, metadata={"source": "test.pdf", "page": 1}
        )

        assert len(chunks) == 1
        assert chunks[0]["tokens"] <= 256

    def test_chunk_long_text(self):
        """Texte long = plusieurs chunks."""
        from scripts.pipeline.chunker import chunk_text

        text = " ".join(["Phrase de test numero {}.".format(i) for i in range(100)])
        chunks = chunk_text(
            text,
            max_tokens=100,
            overlap_tokens=20,
            metadata={"source": "test.pdf", "page": 1},
        )

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["tokens"] <= 120  # With tolerance

    def test_chunk_empty_raises(self):
        """Leve ValueError pour texte vide."""
        from scripts.pipeline.chunker import chunk_text

        with pytest.raises(ValueError):
            chunk_text("", max_tokens=256)

    def test_chunk_invalid_params_raises(self):
        """Leve ValueError si max_tokens <= overlap."""
        from scripts.pipeline.chunker import chunk_text

        with pytest.raises(ValueError):
            chunk_text("Test text here.", max_tokens=50, overlap_tokens=100)

    def test_chunk_metadata_propagation(self):
        """Metadonnees propagees aux chunks."""
        from scripts.pipeline.chunker import chunk_text

        metadata = {"source": "LA.pdf", "page": 41, "corpus": "fr", "section": "Art 4"}
        text = "Texte de test suffisamment long pour creer un chunk valide."
        chunks = chunk_text(text, max_tokens=256, metadata=metadata)

        assert chunks[0]["source"] == "LA.pdf"
        assert chunks[0]["page"] == 41
        assert chunks[0]["metadata"]["corpus"] == "fr"


class TestChunkDocument:
    """Tests pour chunk_document() - fonction implementee."""

    def test_chunk_extracted_document(self):
        """Chunke un document extrait."""
        from scripts.pipeline.chunker import chunk_document

        extracted_data = {
            "filename": "test.pdf",
            "pages": [
                {
                    "page_num": 1,
                    "text": "Page 1 contenu suffisamment long pour etre valide.",
                    "section": None,
                },
                {
                    "page_num": 2,
                    "text": "Page 2 contenu suffisamment long pour etre valide.",
                    "section": "Art 1",
                },
            ],
        }

        chunks = chunk_document(extracted_data, corpus="fr", doc_num=1)

        assert len(chunks) >= 2
        assert chunks[0]["id"].startswith("FR-001-001-")
        assert chunks[0]["source"] == "test.pdf"
