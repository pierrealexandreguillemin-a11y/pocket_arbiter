"""
Tests for session_utils module.

ISO Reference:
    - ISO/IEC 29119 - Software testing
"""

from scripts.evaluation.annales.session_utils import (
    MONTH_NUMBERS,
    MONTH_PATTERNS,
    detect_session_from_filename,
    normalize_session_id,
)


class TestDetectSessionFromFilename:
    """Tests for detect_session_from_filename function."""

    def test_decembre_2024(self) -> None:
        """Should detect December 2024."""
        assert detect_session_from_filename("Annales-Decembre-2024.json") == "dec2024"

    def test_juin_2025(self) -> None:
        """Should detect June 2025."""
        assert detect_session_from_filename("Annales-Juin-2025.pdf") == "jun2025"

    def test_accented_decembre(self) -> None:
        """Should handle accented décembre."""
        assert detect_session_from_filename("Annales-Décembre-2023.json") == "dec2023"

    def test_english_december(self) -> None:
        """Should handle English 'december'."""
        assert detect_session_from_filename("Annales-December-2022.json") == "dec2022"

    def test_yyyymm_format(self) -> None:
        """Should handle YYYYMM format (e.g., 201812)."""
        result = detect_session_from_filename("201812_annales.pdf")
        assert result == "dec2018"

    def test_yyyymm_june(self) -> None:
        """Should handle YYYYMM format for June."""
        result = detect_session_from_filename("201906_examen.json")
        assert result == "jun2019"

    def test_no_month_fallback(self) -> None:
        """Should fallback to session_YYYY when no month found."""
        result = detect_session_from_filename("Annales-2020.json")
        assert result == "session_2020"

    def test_no_year_fallback(self) -> None:
        """Should handle missing year."""
        result = detect_session_from_filename("Annales-Juin.json")
        assert result == "jununknown"

    def test_case_insensitive(self) -> None:
        """Should be case insensitive."""
        assert detect_session_from_filename("ANNALES-DECEMBRE-2024.JSON") == "dec2024"

    def test_janvier(self) -> None:
        """Should detect January."""
        assert detect_session_from_filename("Annales-Janvier-2024.json") == "jan2024"

    def test_mars(self) -> None:
        """Should detect March."""
        assert detect_session_from_filename("Session-Mars-2024.pdf") == "mar2024"

    def test_aout_accented(self) -> None:
        """Should detect August with accent."""
        assert detect_session_from_filename("Annales-Août-2024.json") == "aug2024"


class TestNormalizeSessionId:
    """Tests for normalize_session_id function."""

    def test_lowercase(self) -> None:
        """Should lowercase."""
        assert normalize_session_id("Dec2024") == "dec2024"

    def test_removes_spaces(self) -> None:
        """Should remove spaces."""
        result = normalize_session_id("Dec 2024")
        assert " " not in result

    def test_removes_dashes(self) -> None:
        """Should remove dashes."""
        result = normalize_session_id("dec-2024")
        assert "-" not in result

    def test_normalizes_month_names(self) -> None:
        """Should normalize month names."""
        result = normalize_session_id("decembre2024")
        assert result == "dec2024"


class TestConstants:
    """Tests for module constants."""

    def test_month_numbers_complete(self) -> None:
        """Should have all 12 months."""
        assert len(MONTH_NUMBERS) == 12
        for i in range(1, 13):
            assert i in MONTH_NUMBERS

    def test_month_patterns_has_french(self) -> None:
        """Should have French month names."""
        assert "décembre" in MONTH_PATTERNS
        assert "juin" in MONTH_PATTERNS

    def test_month_patterns_has_english(self) -> None:
        """Should have English month names."""
        assert "december" in MONTH_PATTERNS
        assert "june" in MONTH_PATTERNS
