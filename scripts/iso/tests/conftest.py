#!/usr/bin/env python3
"""Shared fixtures for ISO validator tests."""

import json
import pytest
from pathlib import Path


@pytest.fixture
def temp_project(tmp_path):
    """Create a minimal valid project structure."""
    for d in ["android", "scripts", "corpus/fr", "corpus/intl",
              "docs", "prompts", "tests/data", "tests/reports", ".git", ".iso"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    (tmp_path / "README.md").write_text("# Test Project\n" * 10)
    (tmp_path / "CLAUDE_CODE_INSTRUCTIONS.md").write_text("# Instructions\n" * 10)
    (tmp_path / ".gitignore").write_text("*.pyc\n")

    (tmp_path / "docs" / "VISION.md").write_text("# Vision\n" * 10)
    (tmp_path / "docs" / "ARCHITECTURE.md").write_text("# Architecture\n" * 10)
    (tmp_path / "docs" / "AI_POLICY.md").write_text("# AI Policy\n" * 10)
    (tmp_path / "docs" / "QUALITY_REQUIREMENTS.md").write_text("# Quality\n" * 10)
    (tmp_path / "docs" / "TEST_PLAN.md").write_text("# Test Plan\n" * 10)

    (tmp_path / "prompts" / "README.md").write_text("# Prompts\n")
    (tmp_path / "prompts" / "CHANGELOG.md").write_text("# Changelog\n")
    (tmp_path / "prompts" / "test.txt").write_text("Test prompt\n")

    (tmp_path / "tests" / "data" / "test.json").write_text('{"test": true}')
    (tmp_path / "corpus" / "INVENTORY.md").write_text("# Inventory\n")
    (tmp_path / ".iso" / "config.json").write_text('{"version": "1.0"}')
    (tmp_path / "scripts" / "requirements.txt").write_text("pytest\n")

    return tmp_path


@pytest.fixture
def full_project(tmp_path):
    """Create a complete project structure for phase testing."""
    for d in ["android/app", "scripts", "corpus/fr", "corpus/intl",
              "docs", "prompts", "tests/data", "tests/reports", ".git", ".iso"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    (tmp_path / "README.md").write_text("# Project\n" * 10)
    (tmp_path / "CLAUDE_CODE_INSTRUCTIONS.md").write_text("# Instructions\n" * 10)
    (tmp_path / ".gitignore").write_text("*.pyc\n")
    (tmp_path / "docs" / "VISION.md").write_text("# Vision\n" * 10)
    (tmp_path / "docs" / "ARCHITECTURE.md").write_text("# Architecture\n" * 10)
    (tmp_path / "docs" / "AI_POLICY.md").write_text("# AI Policy\n" * 10)
    (tmp_path / "docs" / "QUALITY_REQUIREMENTS.md").write_text("# Quality\n" * 10)
    (tmp_path / "docs" / "TEST_PLAN.md").write_text("# Test Plan\n" * 10)
    (tmp_path / "prompts" / "README.md").write_text("# Prompts\n")
    (tmp_path / "prompts" / "CHANGELOG.md").write_text("# Changelog\n")
    (tmp_path / "prompts" / "test.txt").write_text("Test prompt\n")
    (tmp_path / "tests" / "data" / "questions_fr.json").write_text('[{"q": "test"}]')
    (tmp_path / "corpus" / "INVENTORY.md").write_text("# Inventory\n")
    (tmp_path / ".iso" / "config.json").write_text('{"version": "1.0"}')
    (tmp_path / "scripts" / "requirements.txt").write_text("pytest\n")

    return tmp_path
