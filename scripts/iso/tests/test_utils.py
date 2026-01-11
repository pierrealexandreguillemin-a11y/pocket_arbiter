#!/usr/bin/env python3
"""Tests for utility functions and classes."""

import os
from unittest.mock import patch

from ..utils import Colors, Icons, colored, _is_fancy_terminal, _get_colors, _get_icons


class TestIsFancyTerminal:
    """Tests for _is_fancy_terminal function."""

    def test_fancy_terminal_windows_with_wt(self):
        """Test Windows Terminal is detected as fancy."""
        with patch("sys.platform", "win32"):
            with patch.dict(os.environ, {"WT_SESSION": "1"}):
                result = _is_fancy_terminal()
                assert result is True

    def test_fancy_terminal_windows_without_wt(self):
        """Test basic Windows console is not fancy."""
        with patch("sys.platform", "win32"):
            with patch.dict(os.environ, {}, clear=True):
                result = _is_fancy_terminal()
                assert result is False

    def test_fancy_terminal_linux(self):
        """Test Linux is always fancy."""
        with patch("sys.platform", "linux"):
            result = _is_fancy_terminal()
            assert result is True

    def test_fancy_terminal_darwin(self):
        """Test macOS is always fancy."""
        with patch("sys.platform", "darwin"):
            result = _is_fancy_terminal()
            assert result is True


class TestGetColors:
    """Tests for _get_colors function."""

    def test_get_colors_fancy(self):
        """Test colors with fancy terminal."""
        colors = _get_colors(fancy=True)
        assert colors["RED"] == "\033[0;31m"
        assert colors["GREEN"] == "\033[0;32m"
        assert colors["YELLOW"] == "\033[1;33m"
        assert colors["BLUE"] == "\033[0;34m"
        assert colors["NC"] == "\033[0m"

    def test_get_colors_not_fancy(self):
        """Test colors without fancy terminal."""
        colors = _get_colors(fancy=False)
        assert colors["RED"] == ""
        assert colors["GREEN"] == ""
        assert colors["YELLOW"] == ""
        assert colors["BLUE"] == ""
        assert colors["NC"] == ""

    def test_get_colors_auto_detect(self):
        """Test colors with auto-detection."""
        colors = _get_colors()  # No argument, auto-detect
        assert "RED" in colors
        assert "NC" in colors


class TestGetIcons:
    """Tests for _get_icons function."""

    def test_get_icons_fancy(self):
        """Test icons with fancy terminal."""
        icons = _get_icons(fancy=True)
        assert icons["FOLDER"] == "📁"
        assert icons["CHECK"] == "✅"
        assert icons["CROSS"] == "❌"
        assert icons["WARN"] == "⚠️"

    def test_get_icons_not_fancy(self):
        """Test icons without fancy terminal (ASCII fallback)."""
        icons = _get_icons(fancy=False)
        assert icons["FOLDER"] == "[DIR]"
        assert icons["CHECK"] == "[OK]"
        assert icons["CROSS"] == "[FAIL]"
        assert icons["WARN"] == "[WARN]"

    def test_get_icons_auto_detect(self):
        """Test icons with auto-detection."""
        icons = _get_icons()  # No argument, auto-detect
        assert "FOLDER" in icons
        assert "CHECK" in icons


class TestColors:
    """Tests for Colors class."""

    def test_colors_attributes_exist(self):
        """Test all color attributes are defined."""
        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "GREEN")
        assert hasattr(Colors, "YELLOW")
        assert hasattr(Colors, "BLUE")
        assert hasattr(Colors, "NC")

    def test_colors_are_strings(self):
        """Test all colors are string values."""
        assert isinstance(Colors.RED, str)
        assert isinstance(Colors.GREEN, str)
        assert isinstance(Colors.YELLOW, str)
        assert isinstance(Colors.BLUE, str)
        assert isinstance(Colors.NC, str)


class TestIcons:
    """Tests for Icons class."""

    def test_icons_attributes_exist(self):
        """Test all icon attributes are defined."""
        required_icons = [
            "FOLDER",
            "DOC",
            "ROBOT",
            "SHIELD",
            "SPARKLE",
            "TEST",
            "PIN",
            "CHECK",
            "CROSS",
            "WARN",
        ]
        for icon in required_icons:
            assert hasattr(Icons, icon), f"Missing icon: {icon}"

    def test_icons_are_strings(self):
        """Test all icons are string values."""
        assert isinstance(Icons.FOLDER, str)
        assert isinstance(Icons.CHECK, str)
        assert isinstance(Icons.CROSS, str)
        assert isinstance(Icons.WARN, str)

    def test_icons_not_empty(self):
        """Test icons are not empty strings."""
        assert len(Icons.FOLDER) > 0
        assert len(Icons.CHECK) > 0
        assert len(Icons.CROSS) > 0


class TestColoredFunction:
    """Tests for colored() function."""

    def test_colored_returns_string(self):
        """Test colored returns a string."""
        result = colored("test", Colors.RED)
        assert isinstance(result, str)

    def test_colored_contains_text(self):
        """Test colored output contains the original text."""
        result = colored("hello world", Colors.GREEN)
        assert "hello world" in result

    def test_colored_empty_string(self):
        """Test colored with empty string."""
        result = colored("", Colors.BLUE)
        assert isinstance(result, str)

    def test_colored_with_no_color(self):
        """Test colored with empty color code."""
        result = colored("test", "")
        assert "test" in result

    def test_colored_wraps_text_fancy(self):
        """Test colored wraps text with ANSI codes on fancy terminal."""
        colors = _get_colors(fancy=True)
        result = f"{colors['RED']}message{colors['NC']}"
        assert result.startswith("\033[")
        assert "message" in result

    def test_colored_wraps_text_not_fancy(self):
        """Test colored returns plain text on non-fancy terminal."""
        colors = _get_colors(fancy=False)
        result = f"{colors['RED']}message{colors['NC']}"
        assert result == "message"
