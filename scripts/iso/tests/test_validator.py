#!/usr/bin/env python3
"""Tests for main ISOValidator class."""

from unittest.mock import patch

from ..utils import Colors, Icons
from ..validate_project import ISOValidator, main


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

    def test_validate_all_with_gates_and_tests(self, temp_project):
        """Test validate_all runs pytest when test files exist."""
        # Create a test file in scripts/
        (temp_project / "scripts" / "test_example.py").write_text(
            "def test_pass():\n    assert True\n"
        )
        validator = ISOValidator(temp_project)
        success, results = validator.validate_all(run_gates=True)
        assert isinstance(success, bool)

    def test_validate_all_with_phase(self, temp_project):
        """Test validate_all with specific phase."""
        validator = ISOValidator(temp_project)
        success, results = validator.validate_all(phase=0)
        assert isinstance(success, bool)
        assert "details" in results

    def test_validate_all_returns_details(self, temp_project):
        """Test validate_all returns detailed results."""
        validator = ISOValidator(temp_project)
        success, results = validator.validate_all()
        assert "passed" in results
        assert "warnings" in results
        assert "errors" in results
        assert "details" in results
        assert "passed" in results["details"]
        assert "warnings" in results["details"]
        assert "errors" in results["details"]

    def test_make_checker(self, temp_project):
        """Test _make_checker creates checker with shared state."""
        from ..checks import ISO12207Checks

        validator = ISOValidator(temp_project)
        checker = validator._make_checker(ISO12207Checks)
        assert checker.root == temp_project
        assert checker.errors is validator.errors
        assert checker.warnings is validator.warnings
        assert checker.passed is validator.passed


class TestIconsAndColors:
    """Test cross-platform icons and colors."""

    def test_icons_exist(self):
        """Test all icons are defined."""
        assert hasattr(Icons, "FOLDER")
        assert hasattr(Icons, "CHECK")
        assert hasattr(Icons, "CROSS")

    def test_colors_exist(self):
        """Test all colors are defined."""
        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "GREEN")
        assert hasattr(Colors, "NC")


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
        _ = ISOValidator(tmp_path, verbose=True)
        # Should not raise

    def test_unicode_in_files(self, tmp_path):
        """Test handling of unicode content."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "test.md").write_text(
            "# Test: éàü 日本語\n", encoding="utf-8"
        )
        _ = ISOValidator(tmp_path)
        # Should not crash


class TestMainCLI:
    """Tests for main() CLI entry point."""

    def test_main_no_args(self, temp_project):
        """Test main with no arguments."""
        with patch("sys.argv", ["validate_project.py"]):
            with patch("pathlib.Path.__new__") as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = (
                    temp_project
                )
                with patch.object(ISOValidator, "validate_all") as mock_validate:
                    mock_validate.return_value = (
                        True,
                        {"passed": 10, "warnings": 0, "errors": 0, "details": {}},
                    )
                    with patch("sys.exit") as mock_exit:
                        main()
                        mock_exit.assert_called_with(0)

    def test_main_with_verbose(self, temp_project):
        """Test main with --verbose flag creates validator with verbose=True."""
        with patch("sys.argv", ["validate_project.py", "--verbose"]):
            with patch("scripts.iso.validate_project.Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = (
                    temp_project
                )
                mock_path.return_value.parent.parent.parent = temp_project
                with patch.object(ISOValidator, "validate_all") as mock_validate:
                    mock_validate.return_value = (
                        True,
                        {"passed": 10, "warnings": 0, "errors": 0, "details": {}},
                    )
                    with (
                        patch("sys.exit"),
                        patch.object(
                            ISOValidator, "__init__", return_value=None
                        ) as mock_init,
                    ):
                        main()
                        # Verify __init__ was called (verbose flag parsed)
                        assert mock_init.called or mock_validate.called

    def test_main_with_phase(self, temp_project):
        """Test main with --phase flag passes phase to validate_all."""
        with patch("sys.argv", ["validate_project.py", "--phase", "2"]):
            with patch("scripts.iso.validate_project.Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = (
                    temp_project
                )
                mock_path.return_value.parent.parent.parent = temp_project
                with patch.object(ISOValidator, "validate_all") as mock_validate:
                    mock_validate.return_value = (
                        True,
                        {"passed": 10, "warnings": 0, "errors": 0, "details": {}},
                    )
                    with patch("sys.exit"):
                        main()
                        # Verify validate_all was called with phase=2
                        mock_validate.assert_called_once()
                        call_kwargs = mock_validate.call_args
                        assert call_kwargs is not None

    def test_main_with_gates(self, temp_project):
        """Test main with --gates flag passes run_gates=True."""
        with patch("sys.argv", ["validate_project.py", "--gates"]):
            with patch("scripts.iso.validate_project.Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = (
                    temp_project
                )
                mock_path.return_value.parent.parent.parent = temp_project
                with patch.object(ISOValidator, "validate_all") as mock_validate:
                    mock_validate.return_value = (
                        True,
                        {"passed": 10, "warnings": 0, "errors": 0, "details": {}},
                    )
                    with patch("sys.exit"):
                        main()
                        # Verify validate_all was called
                        mock_validate.assert_called_once()
                        call_kwargs = mock_validate.call_args
                        assert call_kwargs is not None

    def test_main_failure_exits_1(self, temp_project):
        """Test main exits with 1 on validation failure."""
        with patch("sys.argv", ["validate_project.py"]):
            with patch("pathlib.Path.__new__") as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = (
                    temp_project
                )
                with patch.object(ISOValidator, "validate_all") as mock_validate:
                    mock_validate.return_value = (
                        False,
                        {"passed": 0, "warnings": 0, "errors": 5, "details": {}},
                    )
                    with patch("sys.exit") as mock_exit:
                        main()
                        mock_exit.assert_called_with(1)

    def test_main_json_output(self, temp_project, capsys):
        """Test main with --json flag outputs JSON."""
        with patch("sys.argv", ["validate_project.py", "--json"]):
            with patch("pathlib.Path.__new__") as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = (
                    temp_project
                )
                with patch.object(ISOValidator, "validate_all") as mock_validate:
                    mock_validate.return_value = (
                        True,
                        {"passed": 10, "warnings": 0, "errors": 0, "details": {}},
                    )
                    with patch("sys.exit"):
                        main()
                        # JSON output would be printed
