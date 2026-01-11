#!/usr/bin/env python3
"""Tests for phase validation."""

import json
from unittest.mock import patch, MagicMock
import pytest

from ..phases import PhaseValidator


def make_phase_validator(tmp_path):
    """Helper to create phase validator."""
    errors, warnings, passed = [], [], []
    return PhaseValidator(tmp_path, errors, warnings, passed), errors, warnings, passed


class TestPhaseValidation:
    """Tests for phase-specific validations."""

    def test_validate_phase_invalid(self, full_project):
        """Test validation of invalid phase."""
        validator, errors, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(99)
        assert result is False
        assert any("non reconnue" in e for e in errors)

    def test_validate_phase0(self, full_project):
        """Test Phase 0 validation."""
        validator, _, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(0)
        assert result is True

    def test_validate_phase1_with_pdf(self, full_project):
        """Test Phase 1 with corpus."""
        (full_project / "corpus" / "fr" / "test.pdf").write_bytes(b"%PDF-1.4")
        validator, _, _, passed = make_phase_validator(full_project)
        result = validator.validate_phase(1)
        assert result is True
        assert any("PDF" in p for p in passed)

    def test_validate_phase1_no_pdf(self, full_project):
        """Test Phase 1 fails without PDFs."""
        validator, errors, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(1)
        # Phase 1 requires PDFs - must fail without them
        assert result is False
        assert any("PDF" in e for e in errors)

    def test_validate_phase1_with_tests(self, full_project):
        """Test Phase 1 runs pytest when test files exist."""
        (full_project / "corpus" / "fr" / "test.pdf").write_bytes(b"%PDF-1.4")
        (full_project / "scripts" / "test_example.py").write_text(
            "def test_pass():\n    assert True\n"
        )
        validator, _, _, _ = make_phase_validator(full_project)
        with patch.object(validator.gates, 'gate_pytest') as mock_pytest:
            mock_pytest.return_value = True
            result = validator.validate_phase(1)
            assert result is True

    def test_validate_phase2_no_android(self, full_project):
        """Test Phase 2 fails without Android."""
        validator, errors, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(2)
        assert result is False
        assert any("android" in e.lower() for e in errors)

    def test_validate_phase2_with_android(self, full_project):
        """Test Phase 2 with Android content."""
        (full_project / "android" / "app" / "src").mkdir(parents=True)
        (full_project / "android" / "app" / "src" / "Main.kt").write_text("fun main() {}")
        validator, _, _, passed = make_phase_validator(full_project)
        result = validator.validate_phase(2)
        assert result is True
        assert any("Android" in p or "Kotlin" in p for p in passed)

    def test_validate_phase2_empty_android(self, full_project):
        """Test Phase 2 fails with empty android/app."""
        (full_project / "android" / "app" / "src").mkdir(parents=True)
        validator, errors, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(2)
        assert result is False
        assert any("vide" in e.lower() or "empty" in e.lower() for e in errors)

    def test_validate_phase3_no_prompt(self, full_project):
        """Test Phase 3 fails without prompt."""
        validator, errors, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(3)
        assert result is False
        assert any("interpretation" in e.lower() or "prompt" in e.lower() for e in errors)

    def test_validate_phase3_with_prompt(self, full_project):
        """Test Phase 3 with required files."""
        (full_project / "prompts" / "interpretation_v1.txt").write_text(
            "Cite la source et article."
        )
        (full_project / "tests" / "data" / "adversarial.json").write_text('[{}]')
        validator, _, _, passed = make_phase_validator(full_project)
        result = validator.validate_phase(3)
        assert result is True
        assert any("citation" in p.lower() for p in passed)

    def test_validate_phase3_prompt_no_citation(self, full_project):
        """Test Phase 3 warns when prompt lacks citation instructions."""
        (full_project / "prompts" / "interpretation_v1.txt").write_text(
            "Just answer the question."
        )
        (full_project / "tests" / "data" / "adversarial.json").write_text('[{}]')
        validator, _, warnings, _ = make_phase_validator(full_project)
        result = validator.validate_phase(3)
        assert result is True
        assert any("citation" in w.lower() for w in warnings)

    def test_validate_phase4_success(self, full_project):
        """Test Phase 4 with all gates passing."""
        validator, _, _, passed = make_phase_validator(full_project)
        with patch.object(validator.gates, 'gate_pytest') as mock_pytest:
            with patch.object(validator.gates, 'gate_lint') as mock_lint:
                with patch.object(validator.gates, 'gate_coverage') as mock_cov:
                    mock_pytest.return_value = True
                    mock_lint.return_value = True
                    mock_cov.return_value = True
                    result = validator.validate_phase(4)
                    assert result is True
                    mock_pytest.assert_called_once_with("scripts/", required=True)
                    mock_cov.assert_called_once_with(target=0.60, required=True)

    def test_validate_phase4_pytest_fails(self, full_project):
        """Test Phase 4 fails when pytest fails."""
        validator, errors, _, _ = make_phase_validator(full_project)
        with patch.object(validator.gates, 'gate_pytest') as mock_pytest:
            with patch.object(validator.gates, 'gate_lint') as mock_lint:
                with patch.object(validator.gates, 'gate_coverage') as mock_cov:
                    mock_pytest.return_value = False
                    mock_lint.return_value = True
                    mock_cov.return_value = True
                    result = validator.validate_phase(4)
                    assert result is False

    def test_validate_phase4_coverage_fails(self, full_project):
        """Test Phase 4 fails when coverage below target."""
        validator, errors, _, _ = make_phase_validator(full_project)
        with patch.object(validator.gates, 'gate_pytest') as mock_pytest:
            with patch.object(validator.gates, 'gate_lint') as mock_lint:
                with patch.object(validator.gates, 'gate_coverage') as mock_cov:
                    mock_pytest.return_value = True
                    mock_lint.return_value = True
                    mock_cov.return_value = False
                    result = validator.validate_phase(4)
                    assert result is False

    def test_validate_phase5_no_docs(self, full_project):
        """Test Phase 5 fails without docs."""
        validator, errors, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(5)
        assert result is False
        assert any("USER_GUIDE" in e or "RELEASE" in e for e in errors)

    def test_validate_phase5_no_apk(self, full_project):
        """Test Phase 5 fails without APK."""
        (full_project / "docs" / "USER_GUIDE.md").write_text("# Guide\n")
        (full_project / "docs" / "RELEASE_NOTES.md").write_text("# Release\n")
        validator, errors, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(5)
        assert result is False
        assert any("APK" in e for e in errors)

    def test_validate_phase5_success(self, full_project):
        """Test Phase 5 with all requirements."""
        (full_project / "docs" / "USER_GUIDE.md").write_text("# Guide\n")
        (full_project / "docs" / "RELEASE_NOTES.md").write_text("# Release\n")
        (full_project / "android" / "app" / "build" / "outputs").mkdir(parents=True)
        (full_project / "android" / "app" / "build" / "outputs" / "app.apk").write_bytes(b"APK")
        validator, _, _, passed = make_phase_validator(full_project)
        result = validator.validate_phase(5)
        assert result is True
        assert any("APK" in p for p in passed)
