"""
Fixtures pytest pour les tests du pipeline.

ISO Reference: ISO/IEC 29119 - Test execution
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_extracted_data() -> dict:
    """Donnees extraites de test."""
    return {
        "filename": "test_document.pdf",
        "pages": [
            {
                "page_num": 1,
                "text": "Article 4.1 - Le toucher-jouer\n\nLorsqu'un joueur...",
                "section": "Article 4.1 - Le toucher-jouer",
            },
            {
                "page_num": 2,
                "text": "Suite de l'article 4.1...",
                "section": None,
            },
        ],
        "total_pages": 2,
        "extraction_date": "2026-01-14T10:00:00",
    }


@pytest.fixture
def sample_chunk() -> dict:
    """Chunk de test conforme au schema."""
    return {
        "id": "FR-001-015-01",
        "text": "Article 4.1 - Le toucher-jouer. Lorsqu'un joueur ayant le trait touche deliberement sur l'echiquier...",
        "source": "LA-octobre2025.pdf",
        "page": 15,
        "tokens": 78,
        "metadata": {
            "section": "Article 4.1",
            "corpus": "fr",
            "extraction_date": "2026-01-14",
            "version": "1.0",
        },
    }


@pytest.fixture
def sample_long_text() -> str:
    """Texte long pour tester le chunking."""
    return """
Article 4.1 - Le toucher-jouer

Lorsqu'un joueur ayant le trait touche deliberement sur l'echiquier,
avec l'intention de jouer ou de prendre:
- une ou plusieurs de ses propres pieces, il doit jouer la premiere
  piece touchee qui peut etre jouee, ou
- une ou plusieurs pieces de son adversaire, il doit prendre la
  premiere piece touchee qui peut etre prise.

Article 4.2 - L'ajustement des pieces

Un joueur ayant le trait peut ajuster une ou plusieurs pieces sur
leurs cases, a condition qu'il exprime au prealable son intention
(par exemple en disant "j'adoube" ou "I adjust").

Article 4.3 - Consequences du toucher-jouer

Sauf dans les cas prevus a l'article 4.2:
a) Si le joueur ayant le trait touche deliberement son roi et une
   tour, il doit roquer de ce cote si c'est legal.
b) Si le joueur ayant le trait touche deliberement une tour puis
   son roi, il n'est pas autorise a roquer de ce cote.
""".strip()


@pytest.fixture
def temp_corpus_dir(tmp_path: Path) -> Path:
    """Dossier temporaire pour tests corpus."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create subdirectories
    (corpus_dir / "fr").mkdir()
    (corpus_dir / "intl").mkdir()
    (corpus_dir / "processed").mkdir()

    return corpus_dir


@pytest.fixture
def sample_chunks_file(tmp_path: Path, sample_chunk: dict) -> Path:
    """Fichier de chunks pour tests."""
    chunks_data = {
        "metadata": {
            "corpus": "fr",
            "generated": "2026-01-14T10:30:00",
            "total_chunks": 1,
            "schema_version": "1.0",
        },
        "chunks": [sample_chunk],
    }

    chunks_file = tmp_path / "chunks_test.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f)

    return chunks_file
