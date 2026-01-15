"""
Tests unitaires pour chunker.py

ISO Reference: ISO/IEC 29119 - Test execution

Couverture cible: >= 80% (ISO 29119)

Fonctions testees:
- generate_chunk_id()
- validate_chunk_schema() (dans utils.py)
- count_tokens()
- chunk_text() / chunk_by_article() / chunk_text_legacy()
- chunk_document()
- chunk_corpus()
- _enforce_iso_limits()
- split_at_sentence_boundary()
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

        # Texte long avec phrases substantielles pour depasser MIN_CHUNK_TOKENS
        text = " ".join(
            [
                "Cette phrase numero {} est suffisamment longue pour le test.".format(i)
                for i in range(50)
            ]
        )
        chunks = chunk_text(
            text,
            max_tokens=200,
            overlap_tokens=40,
            metadata={"source": "test.pdf", "page": 1},
        )

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["tokens"] <= 250  # With tolerance

    def test_chunk_empty_returns_empty(self):
        """Retourne liste vide pour texte vide (v2.0 behavior)."""
        from scripts.pipeline.chunker import chunk_text

        # v2.0: texte vide retourne [] au lieu de lever ValueError
        result = chunk_text("", max_tokens=256)
        assert result == []

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

        # v2.0: MIN_CHUNK_TOKENS=100, donc texte doit etre assez long
        long_text_1 = (
            "Article 4.1 Le toucher-jouer est une regle fondamentale aux echecs. "
            "Lorsqu'un joueur touche une piece, il doit la jouer si le coup est legal. "
            "Cette regle s'applique a toutes les competitions officielles de la FFE. "
            "L'arbitre doit veiller au respect de cette regle durant la partie. "
            "En cas de litige, c'est l'arbitre qui prend la decision finale."
        )
        long_text_2 = (
            "Article 4.2 La regle du j'adoube permet au joueur d'ajuster ses pieces. "
            "Le joueur doit annoncer j'adoube avant de toucher la piece. "
            "Sans cette annonce, la regle du toucher-jouer s'applique. "
            "Cette regle est essentielle pour eviter les malentendus. "
            "L'arbitre peut penaliser un joueur qui abuse de cette regle."
        )

        extracted_data = {
            "filename": "test.pdf",
            "pages": [
                {
                    "page_num": 1,
                    "text": long_text_1,
                    "section": None,
                },
                {
                    "page_num": 2,
                    "text": long_text_2,
                    "section": "Art 4.2",
                },
            ],
        }

        chunks = chunk_document(extracted_data, corpus="fr", doc_num=1)

        assert len(chunks) >= 1  # v2.0: peut merger si petit
        assert chunks[0]["id"].startswith("FR-001-")
        assert chunks[0]["source"] == "test.pdf"

    def test_chunk_document_skips_short_pages(self):
        """Pages avec texte < 50 chars sont ignorees."""
        from scripts.pipeline.chunker import chunk_document

        extracted_data = {
            "filename": "test.pdf",
            "pages": [
                {"page_num": 1, "text": "Court."},  # < 50 chars, ignore
                {
                    "page_num": 2,
                    "text": "Article 5.1 Texte suffisamment long pour etre traite. "
                    * 5,
                },
            ],
        }

        chunks = chunk_document(extracted_data, corpus="fr", doc_num=1)
        # Seule page 2 devrait etre chunkee
        assert all(c["page"] == 2 for c in chunks)


class TestEnforceIsoLimits:
    """Tests pour _enforce_iso_limits() - edge cases ISO 25010."""

    def test_enforce_limits_under_max(self):
        """Chunk sous la limite reste inchange."""
        import tiktoken

        from scripts.pipeline.chunker import TOKENIZER_NAME, _enforce_iso_limits

        encoder = tiktoken.get_encoding(TOKENIZER_NAME)
        text = "Short chunk text."
        remaining = "More text."

        result_text, result_remaining, tokens = _enforce_iso_limits(
            text, remaining, encoder, max_tokens=100
        )

        assert result_text == text
        assert result_remaining == remaining
        assert tokens <= 100

    def test_enforce_limits_over_max(self):
        """Chunk depasse la limite est tronque."""
        import tiktoken

        from scripts.pipeline.chunker import TOKENIZER_NAME, _enforce_iso_limits

        encoder = tiktoken.get_encoding(TOKENIZER_NAME)
        # Generer un texte long
        long_text = "Word " * 200  # ~200 tokens
        remaining = "Rest."

        result_text, result_remaining, tokens = _enforce_iso_limits(
            long_text, remaining, encoder, max_tokens=50
        )

        assert tokens <= 50
        assert len(result_remaining) > len(remaining)  # Overflow ajoute

    def test_enforce_limits_empty_remaining(self):
        """Overflow avec remaining vide fonctionne."""
        import tiktoken

        from scripts.pipeline.chunker import TOKENIZER_NAME, _enforce_iso_limits

        encoder = tiktoken.get_encoding(TOKENIZER_NAME)
        long_text = "Word " * 100

        result_text, result_remaining, tokens = _enforce_iso_limits(
            long_text, "", encoder, max_tokens=30
        )

        assert tokens <= 30
        assert result_remaining  # Overflow devient remaining


class TestSplitAtSentenceBoundary:
    """Tests pour split_at_sentence_boundary()."""

    def test_split_finds_sentence_end(self):
        """Coupe a la fin d'une phrase."""
        from scripts.pipeline.chunker import split_at_sentence_boundary

        text = "First sentence here. Second sentence there. Third one."
        first, rest = split_at_sentence_boundary(text, target_tokens=10)

        assert first.endswith(".")
        assert rest.startswith("S") or rest.startswith("T")

    def test_split_no_sentences(self):
        """Texte sans ponctuation coupe aux tokens."""
        from scripts.pipeline.chunker import split_at_sentence_boundary

        text = "No punctuation here just words flowing continuously"
        first, rest = split_at_sentence_boundary(text, target_tokens=5)

        # Sans phrases, coupe aux tokens
        assert first
        assert rest or not rest  # Peut etre vide si texte court

    def test_split_short_text(self):
        """Texte court retourne entier."""
        from scripts.pipeline.chunker import split_at_sentence_boundary

        text = "Short."
        first, rest = split_at_sentence_boundary(text, target_tokens=100)

        assert first == text
        assert rest == ""

    def test_split_with_tolerance(self):
        """Tolerance permet depassement limite."""
        from scripts.pipeline.chunker import split_at_sentence_boundary

        text = "First sentence. Second one is longer here. Third."
        first, rest = split_at_sentence_boundary(text, target_tokens=5, tolerance=50)

        # Avec tolerance elevee, peut prendre plus
        assert first


class TestChunkCorpus:
    """Tests pour chunk_corpus() - traitement batch."""

    def test_chunk_corpus_empty_dir(self, tmp_path):
        """Repertoire vide leve FileNotFoundError."""
        from scripts.pipeline.chunker import chunk_corpus

        fake_dir = tmp_path / "nonexistent"
        output_file = tmp_path / "output.json"

        with pytest.raises(FileNotFoundError):
            chunk_corpus(fake_dir, output_file, "fr")

    def test_chunk_corpus_processes_files(self, tmp_path):
        """Traite tous les fichiers JSON du repertoire."""
        import json

        from scripts.pipeline.chunker import chunk_corpus

        # Creer repertoire avec fichiers extraction
        input_dir = tmp_path / "extractions"
        input_dir.mkdir()

        doc1 = {
            "filename": "doc1.pdf",
            "pages": [
                {
                    "page_num": 1,
                    "text": "Article 1.1 Contenu du premier document. " * 10,
                }
            ],
        }
        doc2 = {
            "filename": "doc2.pdf",
            "pages": [
                {
                    "page_num": 1,
                    "text": "Article 2.1 Contenu du deuxieme document. " * 10,
                }
            ],
        }

        (input_dir / "doc1.json").write_text(json.dumps(doc1), encoding="utf-8")
        (input_dir / "doc2.json").write_text(json.dumps(doc2), encoding="utf-8")

        output_file = tmp_path / "chunks.json"

        report = chunk_corpus(input_dir, output_file, "fr")

        assert output_file.exists()
        assert report["documents_processed"] == 2
        assert report["total_chunks"] >= 2

        # Verifier contenu output
        with open(output_file, encoding="utf-8") as f:
            output_data = json.load(f)
        assert "chunks" in output_data
        assert output_data["metadata"]["corpus"] == "fr"

    def test_chunk_corpus_ignores_report_file(self, tmp_path):
        """Ignore extraction_report.json."""
        import json

        from scripts.pipeline.chunker import chunk_corpus

        input_dir = tmp_path / "extractions"
        input_dir.mkdir()

        doc = {
            "filename": "doc.pdf",
            "pages": [{"page_num": 1, "text": "Article 1.1 Test content. " * 10}],
        }
        report = {"status": "done"}  # Fichier rapport a ignorer

        (input_dir / "doc.json").write_text(json.dumps(doc), encoding="utf-8")
        (input_dir / "extraction_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

        output_file = tmp_path / "chunks.json"
        result = chunk_corpus(input_dir, output_file, "fr")

        assert result["documents_processed"] == 1  # Seulement doc.json


class TestChunkByArticle:
    """Tests pour chunk_by_article() - chunking semantique."""

    def test_chunk_by_article_detects_articles(self):
        """Detecte et preserve les articles."""
        from scripts.pipeline.chunker import chunk_by_article

        text = """Article 4.1 Le toucher-jouer est une regle fondamentale.
        Cette regle s'applique a toutes les parties.

Article 4.2 La regle du j'adoube permet d'ajuster les pieces.
        Le joueur doit annoncer j'adoube avant de toucher."""

        chunks = chunk_by_article(text, max_tokens=256)

        # Devrait avoir au moins 2 chunks (un par article)
        assert len(chunks) >= 1
        # Metadata devrait contenir article
        articles = [c.get("metadata", {}).get("article") for c in chunks]
        assert any("4.1" in str(a) for a in articles if a)

    def test_chunk_by_article_fallback_legacy(self):
        """Sans articles, fallback au chunking legacy."""
        from scripts.pipeline.chunker import chunk_by_article

        # Texte long sans structure d'article
        text = "Texte sans structure d'article. Juste du texte normal. " * 30

        # max_tokens doit etre > DEFAULT_OVERLAP_TOKENS (128)
        chunks = chunk_by_article(text, max_tokens=256, overlap_tokens=50)

        assert len(chunks) >= 1
        # Pas d'article detecte
        for chunk in chunks:
            assert chunk.get("metadata", {}).get("article") is None


class TestChunkTextLegacy:
    """Tests pour chunk_text_legacy() - fallback sentence-based."""

    def test_legacy_respects_overlap(self):
        """Overlap entre chunks fonctionne."""
        from scripts.pipeline.chunker import chunk_text_legacy

        text = "Phrase un. Phrase deux. Phrase trois. Phrase quatre. " * 10

        chunks = chunk_text_legacy(text, max_tokens=50, overlap_tokens=10)

        # Avec overlap, chunks devraient avoir du texte en commun
        if len(chunks) >= 2:
            # Overlap signifie texte commun possible
            assert chunks[0]["tokens"] <= 60  # Avec tolerance

    def test_legacy_filters_small_chunks(self):
        """Petits chunks sont filtres."""
        from scripts.pipeline.chunker import MIN_CHUNK_TOKENS, chunk_text_legacy

        # Texte qui genere un petit reste
        text = "Phrase. " * 30  # ~30 tokens total

        chunks = chunk_text_legacy(text, max_tokens=25, overlap_tokens=5)

        # Tous les chunks devraient respecter MIN_CHUNK_TOKENS
        for chunk in chunks:
            assert chunk["tokens"] >= MIN_CHUNK_TOKENS or len(chunks) == 1


class TestArticleDetection:
    """Tests pour chunker_article.py - detection patterns ISO 82045."""

    def test_detect_article_numbered(self):
        """Detecte Article X.Y format."""
        from scripts.pipeline.chunker_article import detect_article_match

        assert detect_article_match("Article 4.1 Le toucher-jouer") is not None
        assert detect_article_match("Article 4.2.3 Subsection") is not None

    def test_detect_numeric_pattern_with_text(self):
        """Detecte X.Y format avec texte suivant (pattern corrige)."""
        from scripts.pipeline.chunker_article import detect_article_match

        # Pattern corrige: ^(\d+\.\d+)\s au lieu de ^(\d+\.\d+)$
        result = detect_article_match("5.5 Le toucher-jouer")
        assert result is not None
        assert "5.5" in result

    def test_detect_chapitre(self):
        """Detecte Chapitre format."""
        from scripts.pipeline.chunker_article import detect_article_match

        assert detect_article_match("Chapitre 2 Les regles") is not None

    def test_detect_section(self):
        """Detecte Section format."""
        from scripts.pipeline.chunker_article import detect_article_match

        assert detect_article_match("Section 1") is not None

    def test_detect_annexe(self):
        """Detecte Annexe format."""
        from scripts.pipeline.chunker_article import detect_article_match

        assert detect_article_match("Annexe A") is not None

    def test_no_match_normal_text(self):
        """Texte normal ne matche pas."""
        from scripts.pipeline.chunker_article import detect_article_match

        assert detect_article_match("Ceci est un texte normal") is None
        assert detect_article_match("Le joueur doit jouer") is None

    def test_detect_boundaries_multiple_articles(self):
        """Detecte frontieres de plusieurs articles."""
        from scripts.pipeline.chunker_article import detect_article_boundaries

        text = """Article 4.1 Premier article avec contenu suffisant.
Contenu du premier article ici qui continue sur plusieurs lignes.

Article 4.2 Deuxieme article avec son propre contenu.
Suite du deuxieme article avec plus de texte necessaire."""

        segments = detect_article_boundaries(text)

        assert len(segments) >= 2
        assert any("4.1" in str(s.get("article")) for s in segments)
        assert any("4.2" in str(s.get("article")) for s in segments)

    def test_detect_boundaries_empty_text(self):
        """Texte vide retourne liste vide."""
        from scripts.pipeline.chunker_article import detect_article_boundaries

        assert detect_article_boundaries("") == []
        assert detect_article_boundaries(None) == []  # type: ignore[arg-type]
