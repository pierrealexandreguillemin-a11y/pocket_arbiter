"""
Retrieval Logger - Pocket Arbiter

Logging structuré pour les opérations de retrieval.
Format dual: console (INFO) + fichier JSON (DEBUG) pour analytics.

ISO Reference:
    - ISO/IEC 42001 - AI traceability
    - ISO/IEC 25010 - Maintainability
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "retrieval.jsonl"  # JSON Lines pour analytics

# =============================================================================
# Data Classes (structured logging)
# =============================================================================


def _generate_query_id() -> str:
    """Genere un ID court unique pour correlation."""
    return uuid.uuid4().hex[:8]


@dataclass
class RetrievalLogEntry:
    """Structure d'une entrée de log retrieval."""

    timestamp: str
    query_id: str  # Pour correlation session
    query: str
    is_definition: bool
    matched_pattern: str | None
    boost_applied: bool
    boost_factor: float
    source_filter: str | None
    results_count: int
    glossary_hits: int
    fallback_used: bool
    top_scores: list[float]
    top_sources: list[str]


# =============================================================================
# Logger Setup
# =============================================================================


def _setup_logger() -> logging.Logger:
    """Configure le logger avec dual output: console + fichier JSONL."""
    logger = logging.getLogger("retrieval")

    # Éviter les doublons si déjà configuré
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler (INFO only)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

    # File handler (DEBUG - JSON Lines with rotation)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(message)s"))  # Raw JSON
    logger.addHandler(file_handler)

    return logger


_logger = _setup_logger()


# =============================================================================
# Public API
# =============================================================================


def log_retrieval(
    query: str,
    is_definition: bool,
    matched_pattern: str | None,
    boost_applied: bool,
    boost_factor: float,
    source_filter: str | None,
    results_count: int,
    glossary_hits: int,
    fallback_used: bool,
    top_scores: list[float],
    top_sources: list[str],
) -> None:
    """
    Log une opération de retrieval (fichier JSONL + console si warning).

    Args:
        query: Texte de la requête.
        is_definition: True si question de définition détectée.
        matched_pattern: Pattern de définition matché (ou None).
        boost_applied: True si boost glossaire appliqué.
        boost_factor: Facteur de boost (ex: 3.5).
        source_filter: Filtre source appliqué (ou None).
        results_count: Nombre de résultats retournés.
        glossary_hits: Nombre de chunks glossaire dans les résultats.
        fallback_used: True si fallback sans boost utilisé.
        top_scores: Top-3 scores des résultats.
        top_sources: Top-3 sources des résultats.
    """
    entry = RetrievalLogEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        query_id=_generate_query_id(),
        query=query[:100],  # Truncate for log
        is_definition=is_definition,
        matched_pattern=matched_pattern,
        boost_applied=boost_applied,
        boost_factor=boost_factor,
        source_filter=source_filter,
        results_count=results_count,
        glossary_hits=glossary_hits,
        fallback_used=fallback_used,
        top_scores=top_scores,
        top_sources=top_sources,
    )

    # JSON Lines format (1 entry per line)
    _logger.debug(json.dumps(asdict(entry), ensure_ascii=False))

    # Console warning si fallback
    if fallback_used:
        _logger.warning(f"FALLBACK: '{query[:50]}...' -> retry sans boost")


def log_fallback_warning(query: str) -> None:
    """Log un warning quand le fallback est déclenché."""
    _logger.warning(f"FALLBACK: No glossary chunk for '{query[:50]}...' -> retry sans boost")
