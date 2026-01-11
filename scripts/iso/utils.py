#!/usr/bin/env python3
"""Utilities for ISO validation."""

import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


class Colors:
    """Terminal colors (disabled on basic Windows console)."""
    if sys.platform == 'win32' and not os.environ.get('WT_SESSION'):
        RED = GREEN = YELLOW = BLUE = NC = ''
    else:
        RED = '\033[0;31m'
        GREEN = '\033[0;32m'
        YELLOW = '\033[1;33m'
        BLUE = '\033[0;34m'
        NC = '\033[0m'


class Icons:
    """Icons (ASCII fallback for Windows)."""
    if sys.platform == 'win32' and not os.environ.get('WT_SESSION'):
        FOLDER = '[DIR]'
        DOC = '[DOC]'
        ROBOT = '[AI]'
        SHIELD = '[SEC]'
        SPARKLE = '[QA]'
        TEST = '[TEST]'
        PIN = '[PHASE]'
        CHECK = '[OK]'
        CROSS = '[FAIL]'
        WARN = '[WARN]'
    else:
        FOLDER = '📁'
        DOC = '📝'
        ROBOT = '🤖'
        SHIELD = '🛡️'
        SPARKLE = '✨'
        TEST = '🧪'
        PIN = '📌'
        CHECK = '✅'
        CROSS = '❌'
        WARN = '⚠️'


def colored(text: str, color: str) -> str:
    """Return colored text for terminal."""
    return f"{color}{text}{Colors.NC}"
