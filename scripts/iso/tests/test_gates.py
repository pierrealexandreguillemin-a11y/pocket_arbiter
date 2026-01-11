#!/usr/bin/env python3
"""Tests for executable gates."""

import json
import subprocess
from unittest.mock import patch, MagicMock
import pytest

from ..gates import ExecutableGates


def make_gates(tmp_path, verbose=False):
    """Helper to create gates with shared state."""
    errors, warnings, passed = [], [], []
    return ExecutableGates(tmp_path, errors, warnings, passed, verbose), errors, warnings, passed


class TestExecutableGates:
    """Tests for executable gates."""

    @pytest.fixture
    def gate_project(self, tmp_path):
        """Minimal project for gate tests."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".iso").mkdir()
        (tmp_path / "tests" / "data").mkdir(parents=True)
        (tmp_path / "scripts").mkdir()
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

    def test_run_command_timeout(self, gate_project):
        """Test run_command timeout handling."""
        gates, _, _, _ = make_gates(gate_project)
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=1)
            success, output = gates.run_command(["test"])
            assert success is False
            assert "timed out" in output.lower()

    def test_run_command_exception(self, gate_project):
        """Test run_command generic exception."""
        gates, _, _, _ = make_gates(gate_project)
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Generic error")
            success, output = gates.run_command(["test"])
            assert success is False
            assert "Generic error" in output

    def test_log_verbose(self, gate_project, capsys):
        """Test log method in verbose mode."""
        gates, _, _, _ = make_gates(gate_project, verbose=True)
        gates.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_log_not_verbose(self, gate_project, capsys):
        """Test log method in non-verbose mode."""
        gates, _, _, _ = make_gates(gate_project, verbose=False)
        gates.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" not in captured.out

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

    def test_gate_json_valid_default_patterns(self, gate_project):
        """Test JSON gate with default patterns."""
        gates, errors, _, passed = make_gates(gate_project)
        result = gates.gate_json_valid()
        assert result is True

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

    def test_gate_git_status_with_remote(self, gate_project):
        """Test git gate with remote configured."""
        gates, _, _, passed = make_gates(gate_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (True, "origin\tgit@github.com:test/repo.git")
            result = gates.gate_git_status()
            assert result is True
            assert any("remote" in p.lower() for p in passed)

    def test_gate_git_status_no_remote(self, gate_project):
        """Test git gate without remote."""
        gates, _, warnings, _ = make_gates(gate_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (True, "")
            result = gates.gate_git_status()
            assert result is True
            assert any("remote" in w.lower() for w in warnings)

    def test_gate_lint_success(self, gate_project):
        """Test lint gate with clean Python."""
        (gate_project / "scripts" / "clean.py").write_text(
            "def hello():\n    return 'world'\n"
        )
        gates, _, _, _ = make_gates(gate_project)
        result = gates.gate_lint("scripts/")
        assert result is True

    def test_gate_lint_failure(self, gate_project):
        """Test lint gate with errors."""
        gates, errors, _, _ = make_gates(gate_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (False, "E999 syntax error\nE999 another error")
            result = gates.gate_lint("scripts/")
            assert result is False
            assert len(errors) > 0


class TestGatePytest:
    """Tests for gate_pytest method."""

    @pytest.fixture
    def pytest_project(self, tmp_path):
        """Project with test files."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "test_example.py").write_text(
            "def test_pass():\n    assert True\n"
        )
        return tmp_path

    def test_gate_pytest_not_installed_required(self, pytest_project):
        """Test pytest gate when pytest not installed and required."""
        gates, errors, _, _ = make_gates(pytest_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (False, "not found")
            result = gates.gate_pytest("scripts/", required=True)
            assert result is False
            assert any("pytest" in e.lower() for e in errors)

    def test_gate_pytest_not_installed_optional(self, pytest_project):
        """Test pytest gate when pytest not installed and optional."""
        gates, errors, _, _ = make_gates(pytest_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (False, "not found")
            result = gates.gate_pytest("scripts/", required=False)
            assert result is True
            assert len(errors) == 0

    def test_gate_pytest_path_not_found(self, pytest_project):
        """Test pytest gate when path doesn't exist."""
        gates, errors, _, _ = make_gates(pytest_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (True, "pytest 7.0.0")
            result = gates.gate_pytest("nonexistent/", required=True)
            assert result is True  # Skipped, not failed

    def test_gate_pytest_success(self, pytest_project):
        """Test pytest gate with passing tests."""
        gates, errors, _, passed = make_gates(pytest_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.side_effect = [
                (True, "pytest 7.0.0"),  # version check
                (True, "1 passed"),       # test run
            ]
            result = gates.gate_pytest("scripts/", required=True)
            assert result is True
            assert any("pytest" in p.lower() for p in passed)

    def test_gate_pytest_failure(self, pytest_project):
        """Test pytest gate with failing tests."""
        gates, errors, _, _ = make_gates(pytest_project, verbose=True)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.side_effect = [
                (True, "pytest 7.0.0"),   # version check
                (False, "1 failed"),       # test run
            ]
            result = gates.gate_pytest("scripts/", required=True)
            assert result is False
            assert any("pytest" in e.lower() for e in errors)


class TestGateCoverage:
    """Tests for gate_coverage method."""

    @pytest.fixture
    def coverage_project(self, tmp_path):
        """Project for coverage tests."""
        (tmp_path / "scripts").mkdir()
        return tmp_path

    def test_gate_coverage_no_file_required(self, coverage_project):
        """Test coverage gate when coverage.json missing and required."""
        gates, errors, _, _ = make_gates(coverage_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (True, "")
            result = gates.gate_coverage(target=0.60, required=True)
            assert result is False
            assert any("coverage" in e.lower() for e in errors)

    def test_gate_coverage_no_file_optional(self, coverage_project):
        """Test coverage gate when coverage.json missing and optional."""
        gates, errors, _, _ = make_gates(coverage_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (True, "")
            result = gates.gate_coverage(target=0.60, required=False)
            assert result is True

    def test_gate_coverage_meets_target(self, coverage_project):
        """Test coverage gate when coverage meets target."""
        cov_data = {"totals": {"percent_covered": 75.0}}
        (coverage_project / "coverage.json").write_text(json.dumps(cov_data))
        gates, errors, _, passed = make_gates(coverage_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (True, "")
            result = gates.gate_coverage(target=0.60, required=True)
            assert result is True
            assert any("75" in p for p in passed)

    def test_gate_coverage_below_target(self, coverage_project):
        """Test coverage gate when coverage below target."""
        cov_data = {"totals": {"percent_covered": 50.0}}
        (coverage_project / "coverage.json").write_text(json.dumps(cov_data))
        gates, errors, _, _ = make_gates(coverage_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (True, "")
            result = gates.gate_coverage(target=0.60, required=True)
            assert result is False
            assert any("50" in e for e in errors)

    def test_gate_coverage_invalid_json(self, coverage_project):
        """Test coverage gate with invalid JSON file."""
        (coverage_project / "coverage.json").write_text("not json")
        gates, errors, _, _ = make_gates(coverage_project)
        with patch.object(gates, 'run_command') as mock_run:
            mock_run.return_value = (True, "")
            result = gates.gate_coverage(target=0.60, required=False)
            assert result is True  # Skipped on error when not required
