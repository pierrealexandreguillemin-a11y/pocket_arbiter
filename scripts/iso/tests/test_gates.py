#!/usr/bin/env python3
"""Tests for executable gates."""

import json
import pytest

from ..gates import ExecutableGates


def make_gates(tmp_path):
    """Helper to create gates with shared state."""
    errors, warnings, passed = [], [], []
    return ExecutableGates(tmp_path, errors, warnings, passed), errors, warnings, passed


class TestExecutableGates:
    """Tests for executable gates."""

    @pytest.fixture
    def gate_project(self, tmp_path):
        """Minimal project for gate tests."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".iso").mkdir()
        (tmp_path / "tests" / "data").mkdir(parents=True)
        (tmp_path / ".iso" / "config.json").write_text('{"version": "1.0"}')
        (tmp_path / "tests" / "data" / "test.json").write_text('{"valid": true}')
        return tmp_path

    def test_run_command_success(self, gate_project):
        """Test run_command with successful command."""
        gates, _, _, _ = make_gates(gate_project)
        success, output = gates.run_command(["python", "--version"])
        assert success is True
        assert "Python" in output

    def test_run_command_failure(self, gate_project):
        """Test run_command with failing command."""
        gates, _, _, _ = make_gates(gate_project)
        success, output = gates.run_command(["nonexistent_xyz"])
        assert success is False

    def test_gate_json_valid_success(self, gate_project):
        """Test JSON gate passes for valid JSON."""
        gates, errors, _, _ = make_gates(gate_project)
        result = gates.gate_json_valid(["tests/data/*.json", ".iso/*.json"])
        assert result is True
        assert len(errors) == 0

    def test_gate_json_valid_failure(self, gate_project):
        """Test JSON gate fails for invalid JSON."""
        (gate_project / "tests" / "data" / "bad.json").write_text("not json")
        gates, errors, _, _ = make_gates(gate_project)
        result = gates.gate_json_valid(["tests/data/*.json"])
        assert result is False
        assert len(errors) > 0

    def test_gate_git_status_success(self, gate_project):
        """Test git gate with .git directory."""
        gates, _, _, _ = make_gates(gate_project)
        result = gates.gate_git_status()
        assert result is True

    def test_gate_git_status_no_git(self, tmp_path):
        """Test git gate without .git directory."""
        gates, errors, _, _ = make_gates(tmp_path)
        result = gates.gate_git_status()
        assert result is False
        assert len(errors) > 0

    def test_gate_lint_success(self, gate_project):
        """Test lint gate with clean Python."""
        (gate_project / "scripts").mkdir()
        (gate_project / "scripts" / "clean.py").write_text(
            "def hello():\n    return 'world'\n"
        )
        gates, _, _, _ = make_gates(gate_project)
        result = gates.gate_lint("scripts/")
        assert result is True
