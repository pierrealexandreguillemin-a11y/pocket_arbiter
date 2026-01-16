"""
Tests unitaires pour export_sdk.py

ISO Reference: ISO/IEC 29119 - Test execution

Ce fichier teste les fonctions d'export SqliteVectorStore:
- embedding_to_blob() / blob_to_embedding() - Serialisation
- create_vector_db() - Creation base SQLite
- retrieve_similar() - Retrieval cosine similarity
- validate_export() - Validation integrite
- export_corpus() - Pipeline complet
"""

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.export_sdk import (
    SCHEMA_VERSION,
    blob_to_embedding,
    create_vector_db,
    embedding_to_blob,
    export_corpus,
    retrieve_similar,
    validate_export,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Embeddings de test normalises."""
    # 5 embeddings de dimension 8 (simplifie pour tests)
    embeddings = np.random.randn(5, 8).astype(np.float32)
    # Normaliser
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_chunks_for_db() -> list[dict]:
    """Chunks conformes au schema pour tests DB."""
    return [
        {
            "id": "FR-001-001-01",
            "text": "Article 4.3 - Le toucher-jouer en competition.",
            "source": "LA-octobre2025.pdf",
            "page": 41,
            "tokens": 12,
            "metadata": {"section": "4.3", "corpus": "fr"},
        },
        {
            "id": "FR-001-001-02",
            "text": "En cas de pat, la partie est declaree nulle.",
            "source": "LA-octobre2025.pdf",
            "page": 42,
            "tokens": 10,
            "metadata": {"section": "5.1", "corpus": "fr"},
        },
        {
            "id": "FR-001-002-01",
            "text": "L'arbitre doit verifier l'echiquier avant la partie.",
            "source": "RI-octobre2025.pdf",
            "page": 5,
            "tokens": 11,
            "metadata": {"section": "1.1", "corpus": "fr"},
        },
        {
            "id": "FR-001-002-02",
            "text": "Le temps de reflexion est controle par une pendule.",
            "source": "RI-octobre2025.pdf",
            "page": 6,
            "tokens": 10,
            "metadata": {"section": "1.2", "corpus": "fr"},
        },
        {
            "id": "FR-001-002-03",
            "text": "Les pieces blanches jouent toujours en premier.",
            "source": "RI-octobre2025.pdf",
            "page": 7,
            "tokens": 9,
            "metadata": {"section": "1.3", "corpus": "fr"},
        },
    ]


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Chemin temporaire pour base de test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


# =============================================================================
# Tests: Serialization
# =============================================================================


class TestSerialization:
    """Tests pour embedding_to_blob() et blob_to_embedding()."""

    def test_embedding_to_blob(self):
        """Convertit embedding en blob."""
        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        blob = embedding_to_blob(emb)

        assert isinstance(blob, bytes)
        assert len(blob) == 4 * 4  # 4 floats * 4 bytes

    def test_blob_to_embedding(self):
        """Convertit blob en embedding."""
        emb_orig = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        blob = embedding_to_blob(emb_orig)
        emb_recovered = blob_to_embedding(blob, dim=4)

        np.testing.assert_array_almost_equal(emb_orig, emb_recovered)

    def test_roundtrip(self):
        """Serialisation aller-retour preservee."""
        emb = np.random.randn(768).astype(np.float32)
        blob = embedding_to_blob(emb)
        emb_back = blob_to_embedding(blob, dim=768)

        np.testing.assert_array_almost_equal(emb, emb_back)

    def test_blob_wrong_size_raises(self):
        """Leve ValueError si taille blob incorrecte."""
        blob = b"\x00" * 16  # 4 floats
        with pytest.raises(ValueError, match="Blob size"):
            blob_to_embedding(blob, dim=8)  # Attend 8 floats

    def test_2d_embedding_raises(self):
        """Leve ValueError pour embedding 2D."""
        emb = np.random.randn(5, 768).astype(np.float32)
        with pytest.raises(ValueError, match="must be 1D"):
            embedding_to_blob(emb)


# =============================================================================
# Tests: create_vector_db
# =============================================================================


class TestCreateVectorDB:
    """Tests pour create_vector_db()."""

    def test_create_db(self, temp_db_path, sample_chunks_for_db, sample_embeddings):
        """Cree une base SQLite valide."""
        report = create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        assert temp_db_path.exists()
        assert report["total_chunks"] == 5
        assert report["embedding_dim"] == 8
        assert report["db_size_mb"] > 0

    def test_db_schema(self, temp_db_path, sample_chunks_for_db, sample_embeddings):
        """Verifie le schema de la base."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "chunks" in tables
        assert "metadata" in tables

        conn.close()

    def test_metadata_values(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Verifie les metadonnees."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT key, value FROM metadata ORDER BY key")
        metadata = dict(cursor.fetchall())

        assert metadata["embedding_dim"] == "8"
        assert metadata["total_chunks"] == "5"
        assert metadata["schema_version"] == SCHEMA_VERSION

        conn.close()

    def test_chunks_stored(self, temp_db_path, sample_chunks_for_db, sample_embeddings):
        """Verifie que les chunks sont stockes."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        assert count == 5

        cursor.execute("SELECT id, source, page FROM chunks ORDER BY id")
        rows = cursor.fetchall()
        assert rows[0] == ("FR-001-001-01", "LA-octobre2025.pdf", 41)

        conn.close()

    def test_embeddings_stored(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Verifie que les embeddings sont stockes."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT embedding FROM chunks WHERE id = 'FR-001-001-01'")
        blob = cursor.fetchone()[0]

        emb = blob_to_embedding(blob, dim=8)
        np.testing.assert_array_almost_equal(emb, sample_embeddings[0])

        conn.close()

    def test_empty_chunks_raises(self, temp_db_path):
        """Leve ValueError pour liste vide."""
        with pytest.raises(ValueError, match="empty"):
            create_vector_db(temp_db_path, [], np.array([]))

    def test_mismatched_counts_raises(self, temp_db_path, sample_chunks_for_db):
        """Leve ValueError si chunks != embeddings."""
        embeddings = np.random.randn(3, 8).astype(np.float32)  # 3 != 5 chunks

        with pytest.raises(ValueError, match="count"):
            create_vector_db(temp_db_path, sample_chunks_for_db, embeddings)

    def test_missing_field_raises(self, temp_db_path, sample_embeddings):
        """Leve ValueError si chunk manque un champ."""
        invalid_chunks = [
            {"id": "FR-001-001-01", "text": "Test"}
        ]  # Missing source, page, tokens

        with pytest.raises(ValueError, match="missing"):
            create_vector_db(temp_db_path, invalid_chunks, sample_embeddings[:1])

    def test_overwrites_existing(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Remplace une base existante."""
        # Create first version
        create_vector_db(temp_db_path, sample_chunks_for_db[:2], sample_embeddings[:2])

        # Create second version (should overwrite)
        report = create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        assert report["total_chunks"] == 5


# =============================================================================
# Tests: retrieve_similar
# =============================================================================


class TestRetrieveSimilar:
    """Tests pour retrieve_similar()."""

    def test_retrieve_returns_results(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Retourne des resultats de retrieval."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        # Query with first embedding (should match itself best)
        results = retrieve_similar(temp_db_path, sample_embeddings[0], top_k=3)

        assert len(results) == 3
        assert results[0]["id"] == "FR-001-001-01"
        assert results[0]["score"] > 0.99  # Cosine with itself

    def test_retrieve_top_k(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Retourne exactement top_k resultats."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        results = retrieve_similar(temp_db_path, sample_embeddings[0], top_k=2)
        assert len(results) == 2

    def test_retrieve_sorted_by_score(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Resultats tries par score decroissant."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        results = retrieve_similar(temp_db_path, sample_embeddings[0], top_k=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_returns_fields(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Resultats contiennent tous les champs."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        results = retrieve_similar(temp_db_path, sample_embeddings[0], top_k=1)
        result = results[0]

        assert "id" in result
        assert "text" in result
        assert "source" in result
        assert "page" in result
        assert "tokens" in result
        assert "metadata" in result
        assert "score" in result

    def test_retrieve_db_not_found(self):
        """Leve FileNotFoundError si base inexistante."""
        with pytest.raises(FileNotFoundError):
            retrieve_similar(Path("/nonexistent.db"), np.zeros(8))

    def test_retrieve_wrong_dim_raises(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Leve ValueError si dimension query incorrecte."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        wrong_dim_query = np.zeros(16)  # DB has dim 8
        with pytest.raises(ValueError, match="dim"):
            retrieve_similar(temp_db_path, wrong_dim_query, top_k=3)


# =============================================================================
# Tests: validate_export
# =============================================================================


class TestValidateExport:
    """Tests pour validate_export()."""

    def test_valid_export(self, temp_db_path, sample_chunks_for_db, sample_embeddings):
        """Valide une base correcte."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        errors = validate_export(temp_db_path, expected_chunks=5)
        assert errors == []

    def test_wrong_chunk_count(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Detecte nombre de chunks incorrect."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        errors = validate_export(temp_db_path, expected_chunks=10)
        assert any("mismatch" in e.lower() for e in errors)

    def test_db_not_found(self):
        """Detecte base inexistante."""
        errors = validate_export(Path("/nonexistent.db"))
        assert any("not found" in e.lower() for e in errors)


# =============================================================================
# Tests: export_corpus (Integration)
# =============================================================================


class TestExportCorpus:
    """Tests pour export_corpus()."""

    def test_export_full_pipeline(self):
        """Pipeline complet d'export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Creer fichiers sources
            chunks_file = tmpdir / "chunks.json"
            embeddings_file = tmpdir / "embeddings.npy"
            output_db = tmpdir / "corpus.db"

            # Chunks
            chunks_data = {
                "chunks": [
                    {
                        "id": "FR-001-001-01",
                        "text": "Texte un.",
                        "source": "test.pdf",
                        "page": 1,
                        "tokens": 3,
                        "metadata": {},
                    },
                    {
                        "id": "FR-001-001-02",
                        "text": "Texte deux.",
                        "source": "test.pdf",
                        "page": 2,
                        "tokens": 3,
                        "metadata": {},
                    },
                ],
                "total": 2,
            }

            from scripts.pipeline.utils import save_json

            save_json(chunks_data, chunks_file)

            # Embeddings
            embeddings = np.random.randn(2, 384).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.save(embeddings_file, embeddings)

            # Export
            report = export_corpus(chunks_file, embeddings_file, output_db)

            assert output_db.exists()
            assert report["total_chunks"] == 2
            assert report["embedding_dim"] == 384
            assert report["validation_errors"] == []

    def test_export_chunks_not_found(self):
        """Leve FileNotFoundError si chunks manquants."""
        with pytest.raises(FileNotFoundError, match="Chunks"):
            export_corpus(
                Path("/nonexistent_chunks.json"),
                Path("/embeddings.npy"),
                Path("/output.db"),
            )

    def test_export_embeddings_not_found(self):
        """Leve FileNotFoundError si embeddings manquants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_file = Path(tmpdir) / "chunks.json"

            from scripts.pipeline.utils import save_json

            save_json(
                {
                    "chunks": [
                        {"id": "1", "text": "t", "source": "s", "page": 1, "tokens": 1}
                    ]
                },
                chunks_file,
            )

            with pytest.raises(FileNotFoundError, match="Embeddings"):
                export_corpus(
                    chunks_file,
                    Path("/nonexistent_embeddings.npy"),
                    Path(tmpdir) / "output.db",
                )


# =============================================================================
# Tests: Integration end-to-end
# =============================================================================


class TestIntegration:
    """Tests d'integration complets."""

    def test_create_and_retrieve(
        self, temp_db_path, sample_chunks_for_db, sample_embeddings
    ):
        """Cree une base et effectue un retrieval."""
        create_vector_db(temp_db_path, sample_chunks_for_db, sample_embeddings)

        # Validate
        errors = validate_export(temp_db_path, expected_chunks=5)
        assert errors == []

        # Retrieve
        results = retrieve_similar(temp_db_path, sample_embeddings[2], top_k=3)

        # Le chunk 2 doit etre le meilleur match pour son propre embedding
        assert results[0]["id"] == sample_chunks_for_db[2]["id"]
        assert results[0]["score"] > 0.99

    def test_semantic_retrieval(self, temp_db_path):
        """Test de retrieval semantique basique."""
        # Creer des embeddings "semantiques" simples
        # Chunk 0 et 1 sont proches, chunk 2 est different
        chunks = [
            {
                "id": "FR-001-001-01",
                "text": "Toucher une piece oblige a la jouer.",
                "source": "LA.pdf",
                "page": 1,
                "tokens": 10,
                "metadata": {},
            },
            {
                "id": "FR-001-001-02",
                "text": "Si vous touchez une piece, vous devez la deplacer.",
                "source": "LA.pdf",
                "page": 2,
                "tokens": 12,
                "metadata": {},
            },
            {
                "id": "FR-001-002-01",
                "text": "Le chat mange la souris dans le jardin.",
                "source": "other.pdf",
                "page": 1,
                "tokens": 9,
                "metadata": {},
            },
        ]

        # Embeddings: 0 et 1 similaires, 2 different
        emb0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb1 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)  # Proche de emb0
        emb2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # Different

        embeddings = np.stack([emb0, emb1, emb2])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        create_vector_db(temp_db_path, chunks, embeddings)

        # Query similaire a emb0/emb1
        query = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
        results = retrieve_similar(temp_db_path, query, top_k=3)

        # Les deux premiers resultats doivent etre les chunks toucher-jouer
        top_2_ids = {results[0]["id"], results[1]["id"]}
        assert "FR-001-001-01" in top_2_ids
        assert "FR-001-001-02" in top_2_ids

        # Le chunk "chat" doit etre dernier
        assert results[2]["id"] == "FR-001-002-01"
