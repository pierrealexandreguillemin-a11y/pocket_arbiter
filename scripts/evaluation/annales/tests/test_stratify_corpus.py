"""
Tests for stratify_corpus module (Phase 0: BY DESIGN generation).

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO 42001 A.6.2.2 - Provenance tracking
"""

import pytest

from scripts.evaluation.annales.stratify_corpus import (
    ChunkInfo,
    Stratum,
    classify_source,
    compute_coverage,
    compute_quotas,
    stratify_chunks,
    validate_stratification,
)


class TestClassifySource:
    """Tests for source document classification."""

    def test_la_pattern(self) -> None:
        """Should classify LA- documents as LA stratum."""
        assert classify_source("LA-octobre2025.pdf") == "LA"
        assert classify_source("LA_2024.pdf") == "LA"

    def test_r01_pattern(self) -> None:
        """Should classify R01 documents as R01 stratum."""
        assert classify_source("R01_reglement_interieur.pdf") == "R01"
        assert classify_source("R01-v2.pdf") == "R01"
        assert classify_source("Reglement_Interieur_2024.pdf") == "R01"

    def test_r02_homologation_pattern(self) -> None:
        """Should classify R02/Homologation documents."""
        assert classify_source("R02_homologation.pdf") == "R02_homologation"
        assert classify_source("Homologation_tournois.pdf") == "R02_homologation"

    def test_r03_classement_pattern(self) -> None:
        """Should classify R03/Classement documents."""
        assert classify_source("R03_classement_elo.pdf") == "R03_classement"
        assert classify_source("Classement_FFE.pdf") == "R03_classement"

    def test_interclubs_pattern(self) -> None:
        """Should classify Interclubs documents."""
        assert classify_source("Interclubs_2024.pdf") == "Interclubs"
        assert classify_source("Top_12_reglement.pdf") == "Interclubs"
        assert classify_source("Nationale_1.pdf") == "Interclubs"

    def test_jeunes_pattern(self) -> None:
        """Should classify Jeunes documents."""
        assert classify_source("Reglement_Jeunes.pdf") == "Jeunes"
        assert classify_source("Junior_championship.pdf") == "Jeunes"
        assert classify_source("Cadets_rules.pdf") == "Jeunes"

    def test_fide_pattern(self) -> None:
        """Should classify FIDE documents."""
        assert classify_source("FIDE_handbook.pdf") == "FIDE"
        assert classify_source("Laws_of_Chess_2023.pdf") == "FIDE"

    def test_other_pattern(self) -> None:
        """Should classify unknown documents as other."""
        assert classify_source("random_document.pdf") == "other"
        assert classify_source("notes.txt") == "other"


class TestStratifyChunks:
    """Tests for chunk stratification."""

    def test_empty_chunks(self) -> None:
        """Should handle empty chunk list."""
        strata = stratify_chunks([])
        assert len(strata) == 0

    def test_single_chunk(self) -> None:
        """Should stratify single chunk correctly."""
        chunks = [
            {"id": "chunk1", "source": "LA-octobre2025.pdf", "page": 1, "tokens": 100}
        ]
        strata = stratify_chunks(chunks)
        assert "LA" in strata
        assert len(strata["LA"].chunks) == 1

    def test_multiple_strata(self) -> None:
        """Should create multiple strata from mixed sources."""
        chunks = [
            {"id": "c1", "source": "LA-octobre2025.pdf", "page": 1, "tokens": 100},
            {"id": "c2", "source": "R01_reglement.pdf", "page": 1, "tokens": 100},
            {"id": "c3", "source": "FIDE_rules.pdf", "page": 1, "tokens": 100},
        ]
        strata = stratify_chunks(chunks)
        assert "LA" in strata
        assert "R01" in strata
        assert "FIDE" in strata
        assert len(strata) == 3

    def test_chunk_info_extraction(self) -> None:
        """Should extract chunk info correctly."""
        chunks = [
            {
                "id": "test_chunk",
                "source": "LA-octobre2025.pdf",
                "page": 5,
                "pages": [5, 6],
                "tokens": 150,
                "text": "Lorem ipsum dolor sit amet...",
                "section": "Article 1",
            }
        ]
        strata = stratify_chunks(chunks)
        chunk_info = strata["LA"].chunks[0]
        assert chunk_info.id == "test_chunk"
        assert chunk_info.source == "LA-octobre2025.pdf"
        assert chunk_info.page == 5
        assert chunk_info.tokens == 150


class TestComputeQuotas:
    """Tests for quota computation."""

    def test_basic_quota_distribution(self) -> None:
        """Should distribute quotas based on priority and chunk count."""
        strata = {
            "LA": Stratum(
                name="LA",
                description="Laws",
                priority=1,
                chunks=[
                    ChunkInfo(f"c{i}", "LA.pdf", i, [], 100, "") for i in range(50)
                ],
            ),
            "R01": Stratum(
                name="R01",
                description="Reglement",
                priority=2,
                chunks=[
                    ChunkInfo(f"c{i}", "R01.pdf", i, [], 100, "") for i in range(30)
                ],
            ),
        }

        result = compute_quotas(strata, target_total=100)

        # Priority 1 should get more quota (40% target)
        assert result["LA"].quota > 0
        assert result["R01"].quota > 0

    def test_minimum_quota_per_stratum(self) -> None:
        """Should ensure minimum quota per stratum."""
        strata = {
            "small": Stratum(
                name="small",
                description="Small",
                priority=2,
                chunks=[
                    ChunkInfo(f"c{i}", "small.pdf", i, [], 100, "") for i in range(10)
                ],
            ),
        }

        result = compute_quotas(strata, target_total=100, min_per_stratum=5)
        assert result["small"].quota >= 5

    def test_quota_capped_by_chunks(self) -> None:
        """Should cap quota based on available chunks."""
        strata = {
            "tiny": Stratum(
                name="tiny",
                description="Tiny",
                priority=1,
                chunks=[ChunkInfo("c1", "tiny.pdf", 1, [], 100, "")],  # Only 1 chunk
            ),
        }

        result = compute_quotas(strata, target_total=1000)
        # Can't generate more questions than 0.5 * chunk count
        assert result["tiny"].quota <= 1


class TestComputeCoverage:
    """Tests for coverage computation."""

    def test_full_coverage(self) -> None:
        """Should compute 100% coverage when all docs selected."""
        chunks = [
            {"id": "c1", "source": "doc1.pdf", "text": "..."},
            {"id": "c2", "source": "doc2.pdf", "text": "..."},
        ]
        strata = {
            "test": Stratum(
                name="test",
                description="Test",
                priority=1,
                selected_chunks=["c1", "c2"],
            ),
        }

        coverage = compute_coverage(strata, chunks)
        assert coverage["coverage_ratio"] == 1.0
        assert coverage["total_documents"] == 2
        assert coverage["covered_documents"] == 2

    def test_partial_coverage(self) -> None:
        """Should compute partial coverage correctly."""
        chunks = [
            {"id": "c1", "source": "doc1.pdf", "text": "..."},
            {"id": "c2", "source": "doc2.pdf", "text": "..."},
            {"id": "c3", "source": "doc3.pdf", "text": "..."},
        ]
        strata = {
            "test": Stratum(
                name="test",
                description="Test",
                priority=1,
                selected_chunks=["c1"],  # Only 1 of 3 docs
            ),
        }

        coverage = compute_coverage(strata, chunks)
        assert coverage["coverage_ratio"] == pytest.approx(1 / 3, rel=0.01)


class TestValidateStratification:
    """Tests for stratification validation."""

    def test_passes_with_enough_strata(self) -> None:
        """Should pass G0-1 when >= 5 strata with quota."""
        strata = {
            f"s{i}": Stratum(name=f"s{i}", description="", priority=1, quota=10)
            for i in range(5)
        }
        coverage = {"coverage_ratio": 0.85}

        passed, errors, warnings = validate_stratification(strata, coverage)
        assert passed
        assert len(errors) == 0

    def test_fails_with_few_strata(self) -> None:
        """Should fail G0-1 when < 5 strata with quota."""
        strata = {
            f"s{i}": Stratum(name=f"s{i}", description="", priority=1, quota=10)
            for i in range(3)
        }
        coverage = {"coverage_ratio": 0.85}

        passed, errors, warnings = validate_stratification(strata, coverage)
        assert not passed
        assert any("G0-1" in e for e in errors)

    def test_warns_on_low_coverage(self) -> None:
        """Should warn G0-2 when coverage < 80%."""
        strata = {
            f"s{i}": Stratum(name=f"s{i}", description="", priority=1, quota=10)
            for i in range(5)
        }
        coverage = {"coverage_ratio": 0.70}  # Below 80%

        passed, errors, warnings = validate_stratification(strata, coverage)
        assert passed  # G0-2 is warning, not blocking
        assert any("G0-2" in w for w in warnings)
