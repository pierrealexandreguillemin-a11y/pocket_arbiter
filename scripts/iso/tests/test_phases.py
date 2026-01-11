#!/usr/bin/env python3
"""Tests for phase validation."""

import json
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
        assert result is False or any("PDF" in e for e in errors)

    def test_validate_phase2_no_android(self, full_project):
        """Test Phase 2 fails without Android."""
        validator, _, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(2)
        assert result is False

    def test_validate_phase2_with_android(self, full_project):
        """Test Phase 2 with Android content."""
        (full_project / "android" / "app" / "src").mkdir(parents=True)
        (full_project / "android" / "app" / "src" / "Main.kt").write_text("fun main() {}")
        validator, _, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(2)
        assert result is True

    def test_validate_phase3_no_prompt(self, full_project):
        """Test Phase 3 fails without prompt."""
        validator, _, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(3)
        assert result is False

    def test_validate_phase3_with_prompt(self, full_project):
        """Test Phase 3 with required files."""
        (full_project / "prompts" / "interpretation_v1.txt").write_text(
            "Cite la source et article."
        )
        (full_project / "tests" / "data" / "adversarial.json").write_text('[{}]')
        validator, _, _, passed = make_phase_validator(full_project)
        result = validator.validate_phase(3)
        assert result is True

    def test_validate_phase5_no_docs(self, full_project):
        """Test Phase 5 fails without docs."""
        validator, _, _, _ = make_phase_validator(full_project)
        result = validator.validate_phase(5)
        assert result is False
