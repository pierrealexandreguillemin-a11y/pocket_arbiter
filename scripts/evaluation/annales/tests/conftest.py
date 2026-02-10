"""
Shared fixtures for GS BY DESIGN pipeline tests.

ISO Reference:
    - ISO/IEC 29119 - Test infrastructure
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent


@pytest.fixture()
def sample_chunk() -> dict:
    """Chunk minimal with rich text (rules, definitions, articles)."""
    return {
        "id": "test-source.pdf-p001-parent001-child00",
        "text": (
            "## Article 5.1 - Obligations de l'arbitre\n"
            "L'arbitre doit veiller au bon deroulement de la competition. "
            "Il est interdit de quitter la salle de jeu pendant une partie. "
            "En cas de litige, l'arbitre doit prendre une decision dans les 5 minutes. "
            "Le joueur peut demander une pause de 10 minutes maximum. "
            "La pendule doit etre placee du cote de l'arbitre pour une meilleure visibilite. "
            "Un joueur ne peut pas utiliser de telephone portable dans la salle de jeu."
        ),
        "source": "test-source.pdf",
        "page": 1,
        "pages": [1],
        "section": "Obligations",
        "tokens": 120,
        "corpus": "fr",
        "chunk_type": "child",
    }


@pytest.fixture()
def sample_chunk_short() -> dict:
    """Chunk with < 50 chars (edge case for generation)."""
    return {
        "id": "short.pdf-p001-parent001-child00",
        "text": "Texte trop court pour generer.",
        "source": "short.pdf",
        "page": 1,
    }


@pytest.fixture()
def sample_gs_question_answerable() -> dict:
    """Complete Schema v2 question (is_impossible=False)."""
    return {
        "id": "gs:scratch:answerable:0001:abc12345",
        "legacy_id": "",
        "content": {
            "question": "Que doit faire l'arbitre en cas de litige?",
            "expected_answer": "L'arbitre doit prendre une decision dans les 5 minutes.",
            "is_impossible": False,
        },
        "mcq": {
            "original_question": "Que doit faire l'arbitre en cas de litige?",
            "choices": {"A": "Option A", "B": "Option B"},
            "mcq_answer": "A",
            "correct_answer": "L'arbitre doit prendre une decision dans les 5 minutes.",
            "original_answer": "L'arbitre doit prendre une decision dans les 5 minutes.",
        },
        "provenance": {
            "chunk_id": "test-source.pdf-p001-parent001-child00",
            "docs": ["test-source.pdf"],
            "pages": [1],
            "article_reference": "Art. 5.1",
            "answer_explanation": 'Source: test-source, Art. 5.1. Extrait: "l\'arbitre doit..."',
            "annales_source": {"exam_year": 2024, "session": "octobre"},
        },
        "classification": {
            "category": "arbitrage",
            "keywords": ["arbitre", "litige", "decision", "minutes"],
            "difficulty": 0.5,
            "question_type": "procedural",
            "cognitive_level": "Understand",
            "reasoning_type": "single-hop",
            "reasoning_class": "reasoning",
            "answer_type": "extractive",
            "hard_type": "ANSWERABLE",
        },
        "validation": {
            "status": "VALIDATED",
            "method": "by_design_generation",
            "reviewer": "claude_code",
            "answer_current": True,
            "verified_date": "2026-01-01",
            "pages_verified": True,
            "batch": "test_batch",
        },
        "processing": {
            "chunk_match_score": 100,
            "chunk_match_method": "by_design_input",
            "reasoning_class_method": "generation_prompt",
            "triplet_ready": True,
            "extraction_flags": ["by_design"],
            "answer_source": "chunk_extraction",
            "quality_score": 0.8,
            "priority_boost": 0.1,
        },
        "audit": {
            "history": "[BY DESIGN] Generated on 2026-01-01",
            "qat_revalidation": "passed",
            "requires_inference": False,
        },
    }


@pytest.fixture()
def sample_gs_question_unanswerable() -> dict:
    """Schema v2 question (is_impossible=True, hard_type=INSUFFICIENT_INFO)."""
    return {
        "id": "gs:scratch:unanswerable:0001:def67890",
        "legacy_id": "",
        "content": {
            "question": "Quel est le salaire d'un arbitre de niveau A1?",
            "expected_answer": "",
            "is_impossible": True,
        },
        "mcq": {
            "original_question": "Quel est le salaire d'un arbitre de niveau A1?",
            "choices": {},
            "mcq_answer": "",
            "correct_answer": "",
            "original_answer": "",
        },
        "provenance": {
            "chunk_id": "test-source.pdf-p001-parent001-child00",
            "docs": ["test-source.pdf"],
            "pages": [1],
            "article_reference": "",
            "answer_explanation": "",
            "annales_source": None,
        },
        "classification": {
            "category": "arbitrage",
            "keywords": ["arbitre", "salaire"],
            "difficulty": 0.8,
            "question_type": "adversarial",
            "cognitive_level": "Analyze",
            "reasoning_type": "single-hop",
            "reasoning_class": "adversarial",
            "answer_type": "unanswerable",
            "hard_type": "INSUFFICIENT_INFO",
        },
        "validation": {
            "status": "VALIDATED",
            "method": "by_design_generation",
            "reviewer": "claude_code",
            "answer_current": True,
            "verified_date": "2026-01-01",
            "pages_verified": True,
            "batch": "test_batch",
        },
        "processing": {
            "chunk_match_score": 100,
            "chunk_match_method": "by_design_input",
            "reasoning_class_method": "generation_prompt",
            "triplet_ready": False,
            "extraction_flags": ["by_design"],
            "answer_source": "unanswerable",
            "quality_score": 0.8,
            "priority_boost": 0.0,
        },
        "audit": {
            "history": "[BY DESIGN] Generated on 2026-01-01",
            "qat_revalidation": None,
            "requires_inference": False,
        },
    }


@pytest.fixture()
def chunks_by_id_small(sample_chunk: dict) -> dict[str, dict]:
    """3 chunks indexed by ID."""
    chunk1 = sample_chunk
    chunk2 = {
        "id": "rules.pdf-p002-parent002-child00",
        "text": (
            "Le roi peut se deplacer d'une case dans toutes les directions. "
            "Le roque est un mouvement special impliquant le roi et une tour. "
            "La promotion permet a un pion atteignant la derniere rangee de devenir "
            "une autre piece."
        ),
        "source": "rules.pdf",
        "page": 2,
    }
    chunk3 = {
        "id": "cadences.pdf-p003-parent003-child00",
        "text": (
            "En cadence rapide, chaque joueur dispose de 15 minutes pour toute la partie. "
            "En blitz, le temps est reduit a 5 minutes par joueur. "
            "L'increment Fischer ajoute des secondes apres chaque coup."
        ),
        "source": "cadences.pdf",
        "page": 3,
    }
    return {c["id"]: c for c in [chunk1, chunk2, chunk3]}


@pytest.fixture(scope="session")
def gs_scratch_data() -> dict:
    """Load tests/data/gs_scratch_v1.json (session-scoped, loaded once)."""
    path = _PROJECT_ROOT / "tests" / "data" / "gs_scratch_v1.json"
    if not path.exists():
        pytest.skip(f"GS scratch file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def chunks_data() -> dict:
    """Load corpus/processed/chunks_mode_b_fr.json (session-scoped)."""
    path = _PROJECT_ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json"
    if not path.exists():
        pytest.skip(f"Chunks file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def chunk_index(chunks_data: dict) -> dict[str, str]:
    """chunk_id -> text mapping from chunks_data."""
    chunks = chunks_data.get("chunks", [])
    return {c["id"]: c["text"] for c in chunks}
