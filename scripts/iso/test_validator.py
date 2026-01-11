#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for ISO Validator - Pocket Arbiter
==============================================
ISO/IEC 29119 compliant test suite.

Run with: pytest scripts/iso/test_validator.py -v
"""

import json
import shutil
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

# Import validator (handle both direct run and pytest)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from validate_project import ISOValidator, Icons, Colors


class TestISOValidator:
    """Test suite for ISO Validator."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a minimal valid project structure."""
        # Create required directories
        (tmp_path / "android").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "corpus" / "fr").mkdir(parents=True)
        (tmp_path / "corpus" / "intl").mkdir(parents=True)
        (tmp_path / "docs").mkdir()
        (tmp_path / "prompts").mkdir()
        (tmp_path / "tests" / "data").mkdir(parents=True)
        (tmp_path / "tests" / "reports").mkdir(parents=True)
        (tmp_path / ".git").mkdir()
        (tmp_path / ".iso").mkdir()

        # Create required files
        (tmp_path / "README.md").write_text("# Test Project\n" * 10)
        (tmp_path / "CLAUDE_CODE_INSTRUCTIONS.md").write_text("# Instructions\n" * 10)
        (tmp_path / ".gitignore").write_text("*.pyc\n")

        # Create required docs
        (tmp_path / "docs" / "VISION.md").write_text("# Vision\n" * 10)
        (tmp_path / "docs" / "AI_POLICY.md").write_text("# AI Policy\n" * 10)
        (tmp_path / "docs" / "QUALITY_REQUIREMENTS.md").write_text("# Quality\n" * 10)
        (tmp_path / "docs" / "TEST_PLAN.md").write_text("# Test Plan\n" * 10)

        # Create prompts
        (tmp_path / "prompts" / "README.md").write_text("# Prompts\n")
        (tmp_path / "prompts" / "CHANGELOG.md").write_text("# Changelog\n")
        (tmp_path / "prompts" / "test.txt").write_text("Test prompt\n")

        # Create test data
        (tmp_path / "tests" / "data" / "test.json").write_text('{"test": true}')
        (tmp_path / "corpus" / "INVENTORY.md").write_text("# Inventory\n")

        # Create ISO config
        (tmp_path / ".iso" / "config.json").write_text('{"version": "1.0"}')

        # Create scripts
        (tmp_path / "scripts" / "requirements.txt").write_text("pytest\n")

        return tmp_path

    def test_validator_creation(self, temp_project):
        """Test validator can be instantiated."""
        validator = ISOValidator(temp_project)
        assert validator.root == temp_project
        assert len(validator.errors) == 0
        assert len(validator.warnings) == 0
        assert len(validator.passed) == 0

    def test_check_file_exists_success(self, temp_project):
        """Test file exists check passes for existing file."""
        validator = ISOValidator(temp_project)
        result = validator.check_file_exists("README.md", "README")
        assert result is True
        assert len(validator.passed) == 1
        assert len(validator.errors) == 0

    def test_check_file_exists_failure(self, temp_project):
        """Test file exists check fails for missing file."""
        validator = ISOValidator(temp_project)
        result = validator.check_file_exists("NONEXISTENT.md", "Missing file")
        assert result is False
        assert len(validator.errors) == 1
        assert len(validator.passed) == 0

    def test_check_dir_exists_success(self, temp_project):
        """Test directory exists check passes for existing dir."""
        validator = ISOValidator(temp_project)
        result = validator.check_dir_exists("docs", "Documentation")
        assert result is True
        assert len(validator.passed) == 1

    def test_check_dir_exists_failure(self, temp_project):
        """Test directory exists check fails for missing dir."""
        validator = ISOValidator(temp_project)
        result = validator.check_dir_exists("nonexistent", "Missing dir")
        assert result is False
        assert len(validator.errors) == 1

    def test_iso12207_structure_valid(self, temp_project):
        """Test ISO 12207 structure validation passes for valid project."""
        validator = ISOValidator(temp_project)
        result = validator.validate_iso12207_structure()
        assert result is True

    def test_iso12207_structure_missing_dir(self, temp_project):
        """Test ISO 12207 structure validation fails for missing dir."""
        # Remove a required directory (use shutil for non-empty dirs on Windows)
        shutil.rmtree(temp_project / "docs")

        validator = ISOValidator(temp_project)
        result = validator.validate_iso12207_structure()
        assert result is False
        assert any("docs" in e for e in validator.errors)

    def test_iso42001_policy_valid(self, temp_project):
        """Test ISO 42001 AI policy validation passes."""
        validator = ISOValidator(temp_project)
        result = validator.validate_iso42001_policy()
        assert result is True

    def test_iso42001_policy_missing(self, temp_project):
        """Test ISO 42001 validation fails without AI policy."""
        (temp_project / "docs" / "AI_POLICY.md").unlink()

        validator = ISOValidator(temp_project)
        result = validator.validate_iso42001_policy()
        assert result is False

    def test_iso25010_quality_valid(self, temp_project):
        """Test ISO 25010 quality validation passes."""
        validator = ISOValidator(temp_project)
        result = validator.validate_iso25010_quality()
        assert result is True

    def test_iso29119_testing_valid(self, temp_project):
        """Test ISO 29119 testing validation passes."""
        validator = ISOValidator(temp_project)
        result = validator.validate_iso29119_testing()
        assert result is True

    def test_iso29119_invalid_json(self, temp_project):
        """Test ISO 29119 validation fails for invalid JSON."""
        (temp_project / "tests" / "data" / "bad.json").write_text("not valid json")

        validator = ISOValidator(temp_project)
        result = validator.validate_iso29119_testing()
        assert result is False
        assert any("JSON invalide" in e for e in validator.errors)

    def test_validate_all_success(self, temp_project):
        """Test full validation passes for valid project."""
        validator = ISOValidator(temp_project)
        success, results = validator.validate_all()
        assert success is True
        assert results["errors"] == 0

    def test_validate_phase0(self, temp_project):
        """Test Phase 0 validation."""
        validator = ISOValidator(temp_project)
        result = validator.validate_phase(0)
        assert result is True

    def test_validate_phase1_with_corpus(self, temp_project):
        """Test Phase 1 validation with corpus."""
        # Add a PDF
        (temp_project / "corpus" / "fr" / "test.pdf").write_bytes(b"%PDF-1.4")

        validator = ISOValidator(temp_project)
        result = validator.validate_phase(1)
        assert result is True
        assert any("PDF" in p for p in validator.passed)

    def test_validate_phase1_without_corpus(self, temp_project):
        """Test Phase 1 validation fails without PDFs."""
        validator = ISOValidator(temp_project)
        result = validator.validate_phase(1)
        # Should fail because no PDFs
        assert result is False or any("PDF" in e for e in validator.errors)


class TestIconsAndColors:
    """Test cross-platform icons and colors."""

    def test_icons_exist(self):
        """Test all icons are defined."""
        assert hasattr(Icons, 'FOLDER')
        assert hasattr(Icons, 'CHECK')
        assert hasattr(Icons, 'CROSS')
        assert hasattr(Icons, 'WARN')

    def test_colors_exist(self):
        """Test all colors are defined."""
        assert hasattr(Colors, 'RED')
        assert hasattr(Colors, 'GREEN')
        assert hasattr(Colors, 'YELLOW')
        assert hasattr(Colors, 'NC')


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_project(self, tmp_path):
        """Test validation of empty project."""
        validator = ISOValidator(tmp_path)
        success, results = validator.validate_all()
        assert success is False
        assert results["errors"] > 0

    def test_verbose_mode(self, tmp_path):
        """Test verbose mode doesn't crash."""
        validator = ISOValidator(tmp_path, verbose=True)
        validator.log("Test message")
        # Should not raise

    def test_unicode_in_files(self, tmp_path):
        """Test handling of unicode content."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "test.md").write_text(
            "# Test avec accents: éàü\n日本語\n",
            encoding='utf-8'
        )

        validator = ISOValidator(tmp_path)
        # Should not crash
        validator.check_file_exists("docs/test.md", "Unicode file")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
