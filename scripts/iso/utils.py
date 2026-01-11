#!/usr/bin/env python3
"""Utilities for ISO validation."""

import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def _is_fancy_terminal() -> bool:
    """Check if terminal supports colors and unicode."""
    if sys.platform == 'win32':
        return bool(os.environ.get('WT_SESSION'))
    return True


def _get_colors(fancy: bool = None):
    """Get color codes based on terminal support."""
    if fancy is None:
        fancy = _is_fancy_terminal()
    if fancy:
        return {
            'RED': '\033[0;31m',
            'GREEN': '\033[0;32m',
            'YELLOW': '\033[1;33m',
            'BLUE': '\033[0;34m',
            'NC': '\033[0m',
        }
    return {'RED': '', 'GREEN': '', 'YELLOW': '', 'BLUE': '', 'NC': ''}


def _get_icons(fancy: bool = None):
    """Get icons based on terminal support."""
    if fancy is None:
        fancy = _is_fancy_terminal()
    if fancy:
        return {
            'FOLDER': '📁', 'DOC': '📝', 'ROBOT': '🤖', 'SHIELD': '🛡️',
            'SPARKLE': '✨', 'TEST': '🧪', 'PIN': '📌', 'CHECK': '✅',
            'CROSS': '❌', 'WARN': '⚠️',
        }
    return {
        'FOLDER': '[DIR]', 'DOC': '[DOC]', 'ROBOT': '[AI]', 'SHIELD': '[SEC]',
        'SPARKLE': '[QA]', 'TEST': '[TEST]', 'PIN': '[PHASE]', 'CHECK': '[OK]',
        'CROSS': '[FAIL]', 'WARN': '[WARN]',
    }


class Colors:
    """Terminal colors (disabled on basic Windows console)."""
    _colors = _get_colors()
    RED = _colors['RED']
    GREEN = _colors['GREEN']
    YELLOW = _colors['YELLOW']
    BLUE = _colors['BLUE']
    NC = _colors['NC']


class Icons:
    """Icons (ASCII fallback for Windows)."""
    _icons = _get_icons()
    FOLDER = _icons['FOLDER']
    DOC = _icons['DOC']
    ROBOT = _icons['ROBOT']
    SHIELD = _icons['SHIELD']
    SPARKLE = _icons['SPARKLE']
    TEST = _icons['TEST']
    PIN = _icons['PIN']
    CHECK = _icons['CHECK']
    CROSS = _icons['CROSS']
    WARN = _icons['WARN']


def colored(text: str, color: str) -> str:
    """Return colored text for terminal."""
    return f"{color}{text}{Colors.NC}"
