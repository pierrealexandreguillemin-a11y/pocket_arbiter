#!/usr/bin/env python3
"""Tests for DVC setup script."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from ..setup_dvc import run_command, check_dvc_installed


class TestRunCommand:
    """Tests for run_command function."""

    def test_run_command_success(self):
        """Test successful command execution."""
        success, output = run_command(["python", "--version"])
        assert success is True
        assert "Python" in output

    def test_run_command_failure(self):
        """Test failed command execution."""
        success, output = run_command(["nonexistent_command_xyz"])
        assert success is False

    def test_run_command_with_cwd(self, tmp_path):
        """Test command with custom working directory."""
        success, output = run_command(["python", "--version"], cwd=tmp_path)
        assert success is True

    def test_run_command_timeout(self):
        """Test command timeout handling."""
        with patch('subprocess.run') as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=1)
            success, output = run_command(["test"])
            assert success is False
            assert "timed out" in output.lower() or "timeout" in output.lower()

    def test_run_command_exception(self):
        """Test generic exception handling."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Test error")
            success, output = run_command(["test"])
            assert success is False
            assert "Test error" in output


class TestCheckDvcInstalled:
    """Tests for check_dvc_installed function."""

    def test_dvc_not_installed(self):
        """Test when DVC is not installed."""
        import scripts.iso.setup_dvc as setup_module
        with patch.object(setup_module, 'run_command') as mock_run:
            # Both direct call and python -m call fail
            mock_run.return_value = (False, "not found")
            result = setup_module.check_dvc_installed()
            assert result is False
            assert mock_run.call_count == 2  # Tries both methods

    def test_dvc_installed_direct(self):
        """Test when DVC is installed directly."""
        import scripts.iso.setup_dvc as setup_module
        with patch.object(setup_module, 'run_command') as mock_run:
            mock_run.return_value = (True, "3.0.0")
            result = setup_module.check_dvc_installed()
            assert result is True

    def test_dvc_installed_via_python(self):
        """Test when DVC is installed via Python module."""
        import scripts.iso.setup_dvc as setup_module
        with patch.object(setup_module, 'run_command') as mock_run:
            # First call fails (dvc not in PATH), second succeeds (python -m dvc)
            mock_run.side_effect = [(False, "not found"), (True, "3.0.0")]
            result = setup_module.check_dvc_installed()
            assert result is True


class TestMainFunction:
    """Tests for main function."""

    def test_main_dvc_not_installed(self, tmp_path, capsys):
        """Test main when DVC is not installed."""
        import scripts.iso.setup_dvc as setup_module
        with patch.object(setup_module, 'check_dvc_installed') as mock_check:
            mock_check.return_value = False
            with patch('sys.argv', ['setup_dvc.py']):
                result = setup_module.main()
                assert result == 1

    def test_main_init_only(self, tmp_path):
        """Test main with --init-only flag."""
        import scripts.iso.setup_dvc as setup_module
        with patch.object(setup_module, 'check_dvc_installed') as mock_check:
            mock_check.return_value = True
            with patch.object(setup_module, 'run_command') as mock_run:
                mock_run.return_value = (True, "ok")
                with patch('sys.argv', ['setup_dvc.py', '--init-only']):
                    # Verify no crash and returns 0
                    result = setup_module.main()
                    assert result == 0

    def test_main_with_remote(self, tmp_path):
        """Test main with --remote flag."""
        import scripts.iso.setup_dvc as setup_module
        with patch.object(setup_module, 'check_dvc_installed') as mock_check:
            mock_check.return_value = True
            with patch.object(setup_module, 'run_command') as mock_run:
                mock_run.return_value = (True, "ok")
                with patch('sys.argv', ['setup_dvc.py', '--remote', 's3://bucket']):
                    result = setup_module.main()
                    assert result == 0
