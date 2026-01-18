"""
Tests for parent_child_chunker.py - Parent-Document Retrieval Pattern

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements
"""

import pytest
from unittest.mock import Mock, patch

import tiktoken


class TestExtractArticleNumber:
    """Tests for extract_article_number function."""

    def test_article_format(self):
        """Should extract 'Article X.Y.Z' format."""
        from scripts.pipeline.parent_child_chunker import extract_article_number

        assert extract_article_number("Article 4.3.1 stipule que") == "4.3.1"
        assert extract_article_number("Article 9 - Regles") == "9"
        assert extract_article_number("voir l'Article 12.5") == "12.5"

    def test_art_abbreviated_format(self):
        """Should extract 'Art. X' format."""
        from scripts.pipeline.parent_child_chunker import extract_article_number

        assert extract_article_number("Art. 5 concernant") == "5"
        assert extract_article_number("Art.7.2 modifie") == "7.2"

    def test_paragraph_format(self):
        """Should extract paragraph symbol format."""
        from scripts.pipeline.parent_child_chunker import extract_article_number

        assert extract_article_number("Selon le §3.4") == "3.4"
        assert extract_article_number("§12 du reglement") == "12"

    def test_no_article_returns_none(self):
        """Should return None when no article found."""
        from scripts.pipeline.parent_child_chunker import extract_article_number

        assert extract_article_number("Texte sans article") is None
        assert extract_article_number("") is None


class TestExtractSectionHeader:
    """Tests for extract_section_header function."""

    def test_chapitre_format(self):
        """Should extract 'Chapitre X' format."""
        from scripts.pipeline.parent_child_chunker import extract_section_header

        result = extract_section_header("Chapitre IV - Les penalites")
        assert result is not None
        assert "Chapitre" in result

    def test_titre_format(self):
        """Should extract 'TITRE X' format."""
        from scripts.pipeline.parent_child_chunker import extract_section_header

        result = extract_section_header("TITRE III : Dispositions")
        assert result is not None
        assert "TITRE" in result

    def test_section_format(self):
        """Should extract 'Section X' format."""
        from scripts.pipeline.parent_child_chunker import extract_section_header

        result = extract_section_header("Section 2 - Cadences")
        assert result is not None
        assert "Section" in result

    def test_annexe_format(self):
        """Should extract 'ANNEXE X' format."""
        from scripts.pipeline.parent_child_chunker import extract_section_header

        result = extract_section_header("ANNEXE A - Definitions")
        assert result is not None
        assert "ANNEXE" in result

    def test_no_section_returns_none(self):
        """Should return None when no section found."""
        from scripts.pipeline.parent_child_chunker import extract_section_header

        assert extract_section_header("Simple text without section") is None
        assert extract_section_header("") is None


class TestCreateSplitter:
    """Tests for create_splitter function."""

    def test_creates_splitter_with_params(self):
        """Should create RecursiveCharacterTextSplitter with token counting."""
        from scripts.pipeline.parent_child_chunker import create_splitter
        from scripts.pipeline.token_utils import get_tokenizer

        tokenizer = get_tokenizer()
        splitter = create_splitter(
            chunk_size=800,
            chunk_overlap=100,
            tokenizer=tokenizer,
        )

        assert splitter is not None
        assert splitter._chunk_size == 800
        assert splitter._chunk_overlap == 100


class TestChunkDocumentParentChild:
    """Tests for chunk_document_parent_child function."""

    def test_empty_text_returns_empty(self):
        """Empty text should return empty lists."""
        from scripts.pipeline.parent_child_chunker import (
            chunk_document_parent_child,
            create_splitter,
        )
        from scripts.pipeline.token_utils import get_tokenizer

        tokenizer = get_tokenizer()
        parent_splitter = create_splitter(800, 100, tokenizer)
        child_splitter = create_splitter(300, 60, tokenizer)

        parents, children = chunk_document_parent_child(
            text="",
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
        )

        assert parents == []
        assert children == []

    def test_short_text_returns_empty(self):
        """Very short text should return empty lists."""
        from scripts.pipeline.parent_child_chunker import (
            chunk_document_parent_child,
            create_splitter,
        )
        from scripts.pipeline.token_utils import get_tokenizer

        tokenizer = get_tokenizer()
        parent_splitter = create_splitter(800, 100, tokenizer)
        child_splitter = create_splitter(300, 60, tokenizer)

        parents, children = chunk_document_parent_child(
            text="Hi",
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
        )

        assert parents == []
        assert children == []

    def test_parent_child_relationship(self):
        """Children should reference their parent's ID."""
        from scripts.pipeline.parent_child_chunker import (
            chunk_document_parent_child,
            create_splitter,
        )
        from scripts.pipeline.token_utils import get_tokenizer

        tokenizer = get_tokenizer()
        parent_splitter = create_splitter(800, 100, tokenizer)
        child_splitter = create_splitter(300, 60, tokenizer)

        # Use substantial text to generate chunks (>800 tokens for parent)
        text = """Article 4.3 Le joueur ou la joueuse doit jouer selon les regles etablies.
        Cette regle s'applique dans tous les cas de tournois officiels.
        Les arbitres doivent verifier que cette regle est respectee.
        En cas de non-respect, des penalites peuvent etre appliquees.
        Le reglement prevoit differentes sanctions selon la gravite.
        """ * 20  # Repeat to ensure enough tokens for chunking

        parents, children = chunk_document_parent_child(
            text=text,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            source="test.pdf",
            page=5,
            tokenizer=tokenizer,
        )

        # DETERMINISTIC: Must produce at least one parent
        assert len(parents) >= 1, "Should produce at least one parent chunk"

        # Check parent structure
        parent = parents[0]
        assert parent["chunk_type"] == "parent"
        assert parent["source"] == "test.pdf"
        assert parent["page"] == 5
        assert "id" in parent
        assert "tokens" in parent

        # Children may or may not exist depending on parent size
        if children:
            child = children[0]
            assert child["chunk_type"] == "child"
            assert "parent_id" in child
            # Child should reference a valid parent
            parent_ids = {p["id"] for p in parents}
            assert child["parent_id"] in parent_ids

    def test_metadata_extraction(self):
        """Should extract article_num and section from text."""
        from scripts.pipeline.parent_child_chunker import (
            chunk_document_parent_child,
            create_splitter,
        )
        from scripts.pipeline.token_utils import get_tokenizer

        tokenizer = get_tokenizer()
        parent_splitter = create_splitter(800, 100, tokenizer)
        child_splitter = create_splitter(300, 60, tokenizer)

        # Enough text to produce at least one parent chunk
        text = """Chapitre III - Regles de competition

        Article 4.3.1 Le joueur ou la joueuse doit respecter les cadences.
        Cette disposition s'applique a tous les tournois homologues.
        Le temps de reflexion est defini par l'organisateur selon les normes.
        L'arbitre principal est responsable du controle des pendules.
        """ * 10

        parents, children = chunk_document_parent_child(
            text=text,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            source="LA-2025.pdf",
            page=10,
            tokenizer=tokenizer,
        )

        # DETERMINISTIC: Must produce parents
        assert len(parents) >= 1, "Should produce at least one parent chunk"

        parent = parents[0]
        # Should have metadata fields (may be None if not found)
        assert "article_num" in parent
        assert "section" in parent


class TestConstants:
    """Tests for module constants."""

    def test_parent_child_size_relationship(self):
        """Parent chunks should be larger than child chunks."""
        from scripts.pipeline.parent_child_chunker import (
            PARENT_CHUNK_SIZE,
            CHILD_CHUNK_SIZE,
        )

        assert PARENT_CHUNK_SIZE > CHILD_CHUNK_SIZE
        # Industry best practice: parent 2-4x child size
        ratio = PARENT_CHUNK_SIZE / CHILD_CHUNK_SIZE
        assert 2.0 <= ratio <= 4.0

    def test_overlap_ratios(self):
        """Overlap should be 10-25% of chunk size."""
        from scripts.pipeline.parent_child_chunker import (
            PARENT_CHUNK_SIZE,
            PARENT_CHUNK_OVERLAP,
            CHILD_CHUNK_SIZE,
            CHILD_CHUNK_OVERLAP,
        )

        parent_ratio = PARENT_CHUNK_OVERLAP / PARENT_CHUNK_SIZE
        child_ratio = CHILD_CHUNK_OVERLAP / CHILD_CHUNK_SIZE

        assert 0.10 <= parent_ratio <= 0.25
        assert 0.10 <= child_ratio <= 0.25

    def test_min_chunk_tokens(self):
        """Min chunk should be reasonable (20-50 tokens)."""
        from scripts.pipeline.parent_child_chunker import MIN_CHUNK_TOKENS

        assert 20 <= MIN_CHUNK_TOKENS <= 50


class TestProcessCorpusParentChild:
    """Tests for process_corpus_parent_child function (ISO 29119)."""

    def test_processes_extraction_files(self, tmp_path):
        """Should process all JSON extraction files in directory."""
        import json
        from scripts.pipeline.parent_child_chunker import process_corpus_parent_child

        # Create input directory with extraction files
        input_dir = tmp_path / "raw"
        input_dir.mkdir()

        # Create mock extraction file
        extraction_data = {
            "source": "test_doc.pdf",
            "pages": [
                {
                    "page_num": 1,
                    "text": """Article 1.1 Les regles du jeu d'echecs.
                    Le joueur doit respecter les regles etablies par la FIDE.
                    Cette disposition s'applique a tous les tournois officiels.
                    """ * 10,
                },
            ],
        }

        with open(input_dir / "test_doc.json", "w", encoding="utf-8") as f:
            json.dump(extraction_data, f)

        output_file = tmp_path / "output" / "chunks.json"

        report = process_corpus_parent_child(
            input_dir=input_dir,
            output_file=output_file,
            corpus="fr",
        )

        # Verify report structure
        assert report["corpus"] == "fr"
        assert report["chunker"] == "parent_child"
        assert "total_parents" in report
        assert "total_children" in report
        assert "total_pages" in report
        assert report["total_pages"] == 1

        # Verify output file created
        assert output_file.exists()

        with open(output_file, encoding="utf-8") as f:
            output_data = json.load(f)

        assert output_data["corpus"] == "fr"
        assert "parents" in output_data
        assert "children" in output_data
        assert "parent_lookup" in output_data

    def test_empty_directory_returns_zero(self, tmp_path):
        """Empty directory should return empty report."""
        import json
        from scripts.pipeline.parent_child_chunker import process_corpus_parent_child

        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        output_file = tmp_path / "output.json"

        report = process_corpus_parent_child(
            input_dir=input_dir,
            output_file=output_file,
            corpus="fr",
        )

        assert report["total_parents"] == 0
        assert report["total_children"] == 0
        assert report["total_pages"] == 0

    def test_adds_corpus_metadata(self, tmp_path):
        """Should add corpus field to each chunk."""
        import json
        from scripts.pipeline.parent_child_chunker import process_corpus_parent_child

        input_dir = tmp_path / "raw"
        input_dir.mkdir()

        extraction_data = {
            "source": "fide_laws.pdf",
            "pages": [
                {
                    "page_num": 1,
                    "text": """FIDE Laws of Chess. Article 1: The nature and objectives.
                    The game of chess is played between two opponents.
                    """ * 15,
                },
            ],
        }

        with open(input_dir / "fide.json", "w", encoding="utf-8") as f:
            json.dump(extraction_data, f)

        output_file = tmp_path / "chunks_intl.json"

        process_corpus_parent_child(
            input_dir=input_dir,
            output_file=output_file,
            corpus="intl",
        )

        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        # Check corpus in parents
        if data["parents"]:
            assert data["parents"][0]["corpus"] == "intl"

        # Check corpus in children
        if data["children"]:
            assert data["children"][0]["corpus"] == "intl"
