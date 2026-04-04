"""Shared fixtures for pipeline tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_markdown_hierarchical() -> str:
    """Markdown with real heading levels (post-hierarchical-pdf)."""
    return (
        "# REGLES GENERALES\n\n"
        "## 1. Licences\n\n"
        "Les joueurs doivent etre licencies.\n\n"
        "### 1.1. Licence A\n\n"
        "Pour cadence >= 60 min.\n\n"
        "### 1.2. Licence B\n\n"
        "Pour cadence < 60 min.\n\n"
        "## 2. Statut\n\n"
        "### 2.1. Nationalite\n\n"
        "En cas de reserve sur la nationalite, le club justifie dans 15 jours.\n\n"
        "## 3. Forfaits\n\n"
    )


@pytest.fixture
def sample_markdown_flat() -> str:
    """Markdown with all ## headings (current docling output, fallback)."""
    return (
        "## REGLES GENERALES\n\n"
        "## 1. Licences\n\n"
        "Les joueurs doivent etre licencies.\n\n"
        "## 1.1. Licence A\n\n"
        "Pour cadence >= 60 min.\n\n"
        "## 1.2. Licence B\n\n"
        "Pour cadence < 60 min.\n\n"
        "## 2. Statut\n\n"
        "## 2.1. Nationalite\n\n"
        "En cas de reserve sur la nationalite, le club justifie dans 15 jours.\n\n"
        "## 3. Forfaits\n\n"
    )
