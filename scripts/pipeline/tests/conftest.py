"""
Fixtures pytest pour les tests du pipeline.

ISO Reference: ISO/IEC 29119 - Test execution

Note: Seules les fixtures utilisees par les tests actifs sont definies.
Les fixtures pour tests futurs seront ajoutees lors de l'implementation.
"""

import pytest

# =============================================================================
# Custom Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (requires corpus)")
    config.addinivalue_line(
        "markers", "iso_blocking: marks tests as ISO blocking criteria"
    )
    # Suppress docling internal deprecation for generate_table_images (docling >= 2.68)
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Field `generate_table_images` is deprecated:DeprecationWarning",
    )


@pytest.fixture
def sample_chunk() -> dict:
    """
    Chunk de test conforme au schema CHUNK_SCHEMA.md.

    Source: LA-octobre2025.pdf page 41, Article 4.3 (toucher-jouer)
    Texte extrait VERBATIM du PDF via PyMuPDF.
    """
    # Texte VERBATIM extrait de LA-octobre2025.pdf page 41 via PyMuPDF
    # Caractères spéciaux normalisés ASCII pour compatibilité tests
    return {
        "id": "FR-001-041-01",
        "text": "4.3. En dehors du cadre precise par l'Article 4.2 si le "
        "joueur ou la joueuse au trait touche sur l'echiquier, avec "
        "l'intention de deplacer ou de prendre : 4.3.1. Une ou plusieurs "
        "de ses propres pieces, il/elle doit deplacer la premiere piece "
        "touchee pouvant etre deplacee.",
        "source": "LA-octobre2025.pdf",
        "page": 41,
        "tokens": 70,
        "metadata": {
            "section": "4.3",
            "corpus": "fr",
            "extraction_date": "2026-01-14",
            "version": "1.0",
        },
    }


# =============================================================================
# NOTE: Fixtures supprimees
# =============================================================================
#
# Les fixtures suivantes ont ete SUPPRIMEES car elles n'etaient utilisees
# que par des tests skipped (maintenant supprimes):
# - sample_extracted_data: pour TestExtractPdf
# - sample_long_text: pour TestChunkText
# - temp_corpus_dir: pour TestExtractCorpus
# - sample_chunks_file: pour tests futurs
#
# Ces fixtures seront restaurees lors de l'implementation en Phase 1A.
# =============================================================================
