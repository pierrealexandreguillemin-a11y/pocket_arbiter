#!/usr/bin/env python3
"""Tests for ISO check modules."""

import json
import shutil
import pytest

from ..checks import ISO12207Checks, ISO25010Checks, ISO29119Checks, ISO42001Checks


def make_checker(cls, tmp_path):
    """Helper to create checker with shared state."""
    errors, warnings, passed = [], [], []
    return cls(tmp_path, errors, warnings, passed), errors, warnings, passed


class TestISO12207Checks:
    """Tests for ISO 12207 structure and docs."""

    def test_structure_valid(self, temp_project):
        """Test structure validation passes."""
        checker, errors, _, _ = make_checker(ISO12207Checks, temp_project)
        result = checker.validate_structure()
        assert result is True

    def test_structure_missing_dir(self, temp_project):
        """Test structure fails for missing dir."""
        shutil.rmtree(temp_project / "docs")
        checker, errors, _, _ = make_checker(ISO12207Checks, temp_project)
        result = checker.validate_structure()
        assert result is False
        assert any("docs" in e for e in errors)

    def test_docs_valid(self, temp_project):
        """Test docs validation passes."""
        checker, errors, _, _ = make_checker(ISO12207Checks, temp_project)
        result = checker.validate_docs()
        assert result is True

    def test_get_current_phase_no_config(self, tmp_path):
        """Test phase detection without config."""
        checker, _, _, _ = make_checker(ISO12207Checks, tmp_path)
        assert checker.get_current_phase() == 0

    def test_get_current_phase_with_completed(self, tmp_path):
        """Test phase detection with completed phases."""
        (tmp_path / ".iso").mkdir()
        config = {
            "version": "1.0",
            "phases": [
                {"id": 0, "status": "completed"},
                {"id": 1, "status": "completed"},
                {"id": 2, "status": "in_progress"},
            ]
        }
        (tmp_path / ".iso" / "config.json").write_text(json.dumps(config))
        checker, _, _, _ = make_checker(ISO12207Checks, tmp_path)
        assert checker.get_current_phase() == 2

    def test_get_current_phase_in_progress(self, tmp_path):
        """Test phase detection with in_progress phase."""
        (tmp_path / ".iso").mkdir()
        config = {
            "version": "1.0",
            "phases": [
                {"id": 0, "status": "completed"},
                {"id": 1, "status": "in_progress"},
            ]
        }
        (tmp_path / ".iso" / "config.json").write_text(json.dumps(config))
        checker, _, _, _ = make_checker(ISO12207Checks, tmp_path)
        assert checker.get_current_phase() == 1

    def test_get_current_phase_invalid_json(self, tmp_path):
        """Test phase detection with invalid JSON."""
        (tmp_path / ".iso").mkdir()
        (tmp_path / ".iso" / "config.json").write_text("not json")
        checker, _, _, _ = make_checker(ISO12207Checks, tmp_path)
        assert checker.get_current_phase() == 0

    def test_structure_android_required_phase2(self, tmp_path):
        """Test android is required at phase 2."""
        # Create minimal structure
        for d in ["scripts", "corpus", "docs", "prompts", "tests"]:
            (tmp_path / d).mkdir()
        (tmp_path / "README.md").write_text("# Test\n")
        (tmp_path / "CLAUDE_CODE_INSTRUCTIONS.md").write_text("# Instructions\n")
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        # Set phase 2
        (tmp_path / ".iso").mkdir()
        config = {"phases": [{"id": 0, "status": "completed"}, {"id": 1, "status": "completed"}]}
        (tmp_path / ".iso" / "config.json").write_text(json.dumps(config))

        checker, errors, _, _ = make_checker(ISO12207Checks, tmp_path)
        result = checker.validate_structure()
        assert result is False
        assert any("android" in e.lower() for e in errors)


class TestISO42001Checks:
    """Tests for ISO 42001 AI governance."""

    def test_policy_valid(self, temp_project):
        """Test AI policy validation passes."""
        checker, errors, _, _ = make_checker(ISO42001Checks, temp_project)
        result = checker.validate_policy()
        assert result is True

    def test_policy_missing(self, temp_project):
        """Test fails without AI policy."""
        (temp_project / "docs" / "AI_POLICY.md").unlink()
        checker, errors, _, _ = make_checker(ISO42001Checks, temp_project)
        result = checker.validate_policy()
        assert result is False

    def test_antihallu_clean(self, tmp_path):
        """Test clean AI code passes."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "ai.py").write_text(
            "def generate_with_context(q, ctx):\n    return ctx\n"
        )
        checker, errors, _, _ = make_checker(ISO42001Checks, tmp_path)
        result = checker.validate_antihallu()
        assert result is True

    def test_antihallu_dangerous(self, tmp_path):
        """Test dangerous patterns detected."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "ai.py").write_text(
            "def generate_without_context(q):\n    return 'fake'\n"
        )
        checker, errors, _, _ = make_checker(ISO42001Checks, tmp_path)
        result = checker.validate_antihallu()
        assert result is False or len(errors) > 0


class TestISO25010Checks:
    """Tests for ISO 25010 quality."""

    def test_quality_valid(self, temp_project):
        """Test quality validation passes."""
        checker, _, _, _ = make_checker(ISO25010Checks, temp_project)
        result = checker.validate_quality()
        assert result is True


class TestISO29119Checks:
    """Tests for ISO 29119 testing."""

    def test_testing_valid(self, temp_project):
        """Test testing validation passes."""
        checker, _, _, _ = make_checker(ISO29119Checks, temp_project)
        result = checker.validate_testing()
        assert result is True

    def test_invalid_json(self, temp_project):
        """Test fails for invalid JSON."""
        (temp_project / "tests" / "data" / "bad.json").write_text("not json")
        checker, errors, _, _ = make_checker(ISO29119Checks, temp_project)
        result = checker.validate_testing()
        assert result is False
        assert any("JSON" in e for e in errors)
