#!/usr/bin/env python3
"""Tests for main ISOValidator class."""

import shutil
import pytest
from pathlib import Path

from ..validate_project import ISOValidator
from ..utils import Icons, Colors


class TestISOValidator:
    """Test suite for main ISO Validator."""

    def test_validator_creation(self, temp_project):
        """Test validator can be instantiated."""
        validator = ISOValidator(temp_project)
        assert validator.root == temp_project
        assert len(validator.errors) == 0

    def test_validate_all_success(self, temp_project):
        """Test full validation passes for valid project."""
        validator = ISOValidator(temp_project)
        success, results = validator.validate_all()
        assert success is True
        assert results["errors"] == 0

    def test_validate_all_with_gates(self, temp_project):
        """Test validate_all with run_gates=True."""
        validator = ISOValidator(temp_project)
        success, results = validator.validate_all(run_gates=True)
        assert isinstance(success, bool)
        assert "passed" in results


class TestIconsAndColors:
    """Test cross-platform icons and colors."""

    def test_icons_exist(self):
        """Test all icons are defined."""
        assert hasattr(Icons, 'FOLDER')
        assert hasattr(Icons, 'CHECK')
        assert hasattr(Icons, 'CROSS')

    def test_colors_exist(self):
        """Test all colors are defined."""
        assert hasattr(Colors, 'RED')
        assert hasattr(Colors, 'GREEN')
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
        # Should not raise

    def test_unicode_in_files(self, tmp_path):
        """Test handling of unicode content."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "test.md").write_text(
            "# Test: éàü 日本語\n", encoding='utf-8'
        )
        validator = ISOValidator(tmp_path)
        # Should not crash
