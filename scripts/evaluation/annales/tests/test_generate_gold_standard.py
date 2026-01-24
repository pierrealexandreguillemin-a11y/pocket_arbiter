"""
Tests for generate_gold_standard module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

import json
import tempfile
from pathlib import Path

from scripts.evaluation.annales.generate_gold_standard import (
    _extract_keywords,
    _generate_question_id,
    _normalize_category,
    generate_gold_standard,
)


class TestGenerateQuestionId:
    """Tests for question ID generation."""

    def test_generates_unique_ids_per_uv(self) -> None:
        """Should generate unique IDs for each UV."""
        counter: dict[str, int] = {}
        id1 = _generate_question_id("dec2024", "UVR", 1, counter)
        id2 = _generate_question_id("dec2024", "UVR", 2, counter)
        id3 = _generate_question_id("dec2024", "UVC", 1, counter)

        assert id1 == "FR-ANN-UVR-001"
        assert id2 == "FR-ANN-UVR-002"
        assert id3 == "FR-ANN-UVC-001"

    def test_counter_persists_across_sessions(self) -> None:
        """Should continue counter across sessions for same UV."""
        counter: dict[str, int] = {}
        _generate_question_id("dec2024", "UVR", 1, counter)
        _generate_question_id("dec2024", "UVR", 2, counter)
        id3 = _generate_question_id("jun2025", "UVR", 1, counter)

        # Counter should continue from 2
        assert id3 == "FR-ANN-UVR-003"

    def test_formats_with_leading_zeros(self) -> None:
        """Should format ID with 3-digit leading zeros."""
        counter: dict[str, int] = {}
        id1 = _generate_question_id("dec2024", "UVR", 1, counter)
        assert id1.endswith("-001")

    def test_handles_different_uvs_independently(self) -> None:
        """Should track counters independently per UV."""
        counter: dict[str, int] = {}
        _generate_question_id("dec2024", "UVR", 1, counter)
        _generate_question_id("dec2024", "UVR", 2, counter)
        id_uvc = _generate_question_id("dec2024", "UVC", 1, counter)

        assert id_uvc == "FR-ANN-UVC-001"
        assert counter["UVR"] == 2
        assert counter["UVC"] == 1


class TestNormalizeCategory:
    """Tests for category normalization."""

    def test_uvr_to_regles_jeu(self) -> None:
        """UVR should map to regles_jeu."""
        assert _normalize_category("UVR", "Article 1.3") == "regles_jeu"

    def test_uvc_with_r01_to_regles_ffe(self) -> None:
        """UVC with R01 reference should map to regles_ffe."""
        assert _normalize_category("UVC", "R01 - 5.3") == "regles_ffe"

    def test_uvc_with_r03_to_competitions(self) -> None:
        """UVC with R03 reference should map to competitions."""
        assert _normalize_category("UVC", "R03 - 2.1") == "competitions"

    def test_uvc_with_a02_to_interclubs(self) -> None:
        """UVC with A02 reference should map to interclubs."""
        assert _normalize_category("UVC", "A02 - 4.5") == "interclubs"

    def test_uvc_default_to_competitions(self) -> None:
        """UVC without specific reference should default to competitions."""
        assert _normalize_category("UVC", "") == "competitions"

    def test_uvo_to_open(self) -> None:
        """UVO should map to open."""
        assert _normalize_category("UVO", "") == "open"

    def test_uvt_to_tournoi(self) -> None:
        """UVT should map to tournoi."""
        assert _normalize_category("UVT", "") == "tournoi"

    def test_unknown_uv_to_general(self) -> None:
        """Unknown UV should map to general."""
        assert _normalize_category("UNKNOWN", "") == "general"


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_extracts_chess_terms(self) -> None:
        """Should extract chess-specific terms."""
        keywords = _extract_keywords("Le roque est-il légal?")
        assert "roque" in keywords

    def test_extracts_arbitrage_terms(self) -> None:
        """Should extract arbitrage-related terms."""
        keywords = _extract_keywords("Quand arrêter la pendule?")
        assert "pendule" in keywords

    def test_extracts_multiple_terms(self) -> None:
        """Should extract multiple relevant terms."""
        keywords = _extract_keywords("Le joueur veut abandonner la partie")
        assert "joueur" in keywords
        assert "abandon" in keywords
        assert "partie" in keywords

    def test_limits_to_five_keywords(self) -> None:
        """Should limit to 5 keywords maximum."""
        # Text with many chess terms
        text = "Le joueur adversaire fait un coup illegal sur l'échiquier avec une pièce"
        keywords = _extract_keywords(text)
        assert len(keywords) <= 5

    def test_handles_text_without_keywords(self) -> None:
        """Should return empty list for text without keywords."""
        keywords = _extract_keywords("Bonjour le monde")
        assert keywords == []

    def test_case_insensitive_matching(self) -> None:
        """Should match keywords case-insensitively."""
        keywords = _extract_keywords("LE ROQUE EST VALIDE")
        assert "roque" in keywords


class TestGenerateGoldStandard:
    """Tests for generate_gold_standard function."""

    def test_generates_from_mapped_files(self) -> None:
        """Should generate Gold Standard from mapped JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Create a sample mapped file (must be named mapped_*.json)
            mapped_data = {
                "session": "dec2024",
                "source_file": "test.json",
                "units": [
                    {
                        "uv": "UVR",
                        "questions": [
                            {
                                "num": 1,
                                "text": "Question about roque?",
                                "choices": {"A": "Vrai", "B": "Faux"},
                                "correct_answer": "A",
                                "article_reference": "Article 3.8",
                                "document_mapping": {
                                    "document": "regles_jeu.pdf",
                                    "confidence": 0.95,
                                    "pages": [10],
                                },
                            }
                        ],
                    }
                ],
            }
            (input_dir / "mapped_dec2024.json").write_text(
                json.dumps(mapped_data), encoding="utf-8"
            )

            output_file = Path(tmpdir) / "gold_standard.json"
            stats = generate_gold_standard(input_dir, output_file)

            assert stats["included"] == 1
            # Check output file was created
            assert output_file.exists()
            gs = json.loads(output_file.read_text(encoding="utf-8"))
            assert gs["questions"][0]["id"].startswith("FR-ANN-UVR")

    def test_filters_by_confidence(self) -> None:
        """Should filter questions by minimum confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Create mapped data with low confidence
            mapped_data = {
                "session": "dec2024",
                "source_file": "test.json",
                "units": [
                    {
                        "uv": "UVR",
                        "questions": [
                            {
                                "num": 1,
                                "text": "Low confidence question",
                                "choices": {"A": "Yes"},
                                "correct_answer": "A",
                                "article_reference": "Article 1.1",
                                "document_mapping": {
                                    "document": "regles.pdf",
                                    "confidence": 0.1,  # Below threshold
                                },
                            }
                        ],
                    }
                ],
            }
            (input_dir / "mapped_dec2024.json").write_text(
                json.dumps(mapped_data), encoding="utf-8"
            )

            output_file = Path(tmpdir) / "gold_standard.json"
            stats = generate_gold_standard(input_dir, output_file, min_confidence=0.5)

            # Low confidence question should be filtered out
            assert stats["skipped_low_confidence"] == 1
            assert stats["included"] == 0

    def test_raises_for_empty_directory(self) -> None:
        """Should raise ValueError for empty input directory."""
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            output_file = Path(tmpdir) / "gold_standard.json"
            with pytest.raises(ValueError, match="No mapped files found"):
                generate_gold_standard(input_dir, output_file)

    def test_handles_multiple_uvs(self) -> None:
        """Should generate IDs for multiple UVs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            mapped_data = {
                "session": "dec2024",
                "source_file": "test.json",
                "units": [
                    {
                        "uv": "UVR",
                        "questions": [
                            {
                                "num": 1,
                                "text": "UVR Question",
                                "choices": {"A": "Yes"},
                                "correct_answer": "A",
                                "article_reference": "Article 1.1",
                                "document_mapping": {
                                    "document": "regles.pdf",
                                    "confidence": 0.9,
                                },
                            }
                        ],
                    },
                    {
                        "uv": "UVC",
                        "questions": [
                            {
                                "num": 1,
                                "text": "UVC Question",
                                "choices": {"A": "Yes"},
                                "correct_answer": "A",
                                "article_reference": "R01 - 5.3",
                                "document_mapping": {
                                    "document": "r01.pdf",
                                    "confidence": 0.85,
                                },
                            }
                        ],
                    },
                ],
            }
            (input_dir / "mapped_dec2024.json").write_text(
                json.dumps(mapped_data), encoding="utf-8"
            )

            output_file = Path(tmpdir) / "gold_standard.json"
            stats = generate_gold_standard(input_dir, output_file)

            assert stats["included"] == 2
            assert stats["by_uv"]["UVR"] == 1
            assert stats["by_uv"]["UVC"] == 1
