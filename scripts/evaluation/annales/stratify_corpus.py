#!/usr/bin/env python3
"""
Phase 0: Stratification du corpus de chunks pour generation GS BY DESIGN.

Repartit les 1,857 chunks en strates par source/page/type pour garantir
la diversite dans la generation du Gold Standard.

ISO Reference:
- ISO 42001 A.6.2.2: Provenance tracking
- ISO 29119-3: Test data coverage

Usage:
    python stratify_corpus.py [--chunks PATH] [--output PATH] [--target N]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import TypedDict  # noqa: E402

from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402


class StratumDef(TypedDict):
    """Type definition for stratum configuration."""

    pattern: str
    description: str
    priority: int


# Document categories for stratification
STRATA_DEFINITIONS: dict[str, StratumDef] = {
    "LA": {
        "pattern": r"^LA[-_]",
        "description": "Lois de l'arbitrage",
        "priority": 1,  # High priority - core content
    },
    "R01": {
        "pattern": r"^R01[-_]|Reglement.*Interieur",
        "description": "Reglement Interieur General",
        "priority": 2,
    },
    "R02_homologation": {
        "pattern": r"^R02[-_]|Homologation",
        "description": "Homologation des tournois",
        "priority": 2,
    },
    "R03_classement": {
        "pattern": r"^R03[-_]|Classement",
        "description": "Classement Elo",
        "priority": 2,
    },
    "Interclubs": {
        "pattern": r"Interclub|Top.*12|Nationale",
        "description": "Competitions par equipes",
        "priority": 2,
    },
    "Jeunes": {
        "pattern": r"Jeunes|Junior|Cadets|Minimes|Benjamins",
        "description": "Reglements jeunes",
        "priority": 2,
    },
    "Disciplinaire": {
        "pattern": r"Disciplin|Ethique|Deontologie",
        "description": "Reglement disciplinaire",
        "priority": 3,
    },
    "Titres": {
        "pattern": r"Titres|Arbitre.*AF|A[123F]",
        "description": "Titres d'arbitre",
        "priority": 2,
    },
    "Rapide_Blitz": {
        "pattern": r"Rapide|Blitz|Cadence",
        "description": "Parties rapides et blitz",
        "priority": 2,
    },
    "FIDE": {
        "pattern": r"FIDE|Laws.*Chess|Handbook",
        "description": "Regles FIDE internationales",
        "priority": 1,
    },
    "other": {
        "pattern": r".*",  # Catch-all
        "description": "Autres documents",
        "priority": 4,
    },
}


@dataclass
class ChunkInfo:
    """Information about a chunk for stratification."""

    id: str
    source: str
    page: int
    pages: list[int]
    tokens: int
    text_preview: str
    section: str = ""


@dataclass
class Stratum:
    """A stratum of chunks with quota."""

    name: str
    description: str
    priority: int
    chunks: list[ChunkInfo] = field(default_factory=list)
    quota: int = 0
    selected_chunks: list[str] = field(default_factory=list)


def classify_source(source: str) -> str:
    """
    Classify a source document into a stratum.

    Args:
        source: Document filename

    Returns:
        Stratum name
    """
    for stratum_name, definition in STRATA_DEFINITIONS.items():
        if stratum_name == "other":
            continue
        if re.search(definition["pattern"], source, re.IGNORECASE):
            return stratum_name
    return "other"


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunks from JSON file."""
    data = load_json(chunks_path)
    chunks = data.get("chunks", data)
    if isinstance(chunks, dict):
        chunks = list(chunks.values())
    return chunks


def extract_covered_chunk_ids(gs_data: dict) -> set[str]:
    """Extract chunk IDs covered by answerable questions in GS.

    Args:
        gs_data: Full GS JSON data (Schema v2 nested format).

    Returns:
        Set of chunk_ids with at least one answerable question.
    """
    covered: set[str] = set()
    for q in gs_data.get("questions", []):
        if q.get("content", {}).get("is_impossible", True):
            continue
        chunk_id = q.get("provenance", {}).get("chunk_id", "")
        if chunk_id:
            covered.add(chunk_id)
    return covered


def filter_uncovered_chunks(
    chunks: list[dict],
    gs_path: Path,
) -> list[dict]:
    """Filter chunks to only those not covered by existing GS questions.

    Args:
        chunks: All chunks.
        gs_path: Path to GS JSON file.

    Returns:
        List of uncovered chunks.
    """
    gs_data = load_json(gs_path)
    covered_ids = extract_covered_chunk_ids(gs_data)
    return [c for c in chunks if c["id"] not in covered_ids]


def stratify_chunks(chunks: list[dict]) -> dict[str, Stratum]:
    """
    Stratify chunks by source document category.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Dictionary mapping stratum name to Stratum object
    """
    # Initialize strata
    strata: dict[str, Stratum] = {}
    for name, definition in STRATA_DEFINITIONS.items():
        strata[name] = Stratum(
            name=name,
            description=definition["description"],
            priority=definition["priority"],
        )

    # Classify each chunk
    for chunk in chunks:
        source = chunk.get("source", "")
        stratum_name = classify_source(source)

        chunk_info = ChunkInfo(
            id=chunk["id"],
            source=source,
            page=chunk.get("page", 0),
            pages=chunk.get("pages", []),
            tokens=chunk.get("tokens", 0),
            text_preview=chunk.get("text", "")[:100],
            section=chunk.get("section", ""),
        )

        strata[stratum_name].chunks.append(chunk_info)

    # Remove empty strata
    return {k: v for k, v in strata.items() if v.chunks}


def compute_quotas(
    strata: dict[str, Stratum],
    target_total: int,
    min_per_stratum: int = 5,
) -> dict[str, Stratum]:
    """
    Compute quotas for each stratum.

    Distribution strategy:
    - Priority 1 strata get 40% of target
    - Priority 2 strata get 45% of target
    - Priority 3+ strata get 15% of target
    - Each stratum gets at least min_per_stratum (if chunks available)

    Args:
        strata: Dictionary of strata
        target_total: Target total questions to generate
        min_per_stratum: Minimum quota per stratum

    Returns:
        Updated strata with quotas set
    """
    # Group by priority
    by_priority: dict[int, list[str]] = defaultdict(list)
    for name, stratum in strata.items():
        by_priority[stratum.priority].append(name)

    # Target distribution by priority
    priority_targets = {
        1: int(target_total * 0.40),
        2: int(target_total * 0.45),
        3: int(target_total * 0.10),
        4: int(target_total * 0.05),
    }

    # Distribute quotas
    for priority, target in priority_targets.items():
        stratum_names = by_priority.get(priority, [])
        if not stratum_names:
            continue

        # Distribute proportionally to chunk count within priority
        total_chunks = sum(len(strata[n].chunks) for n in stratum_names)
        if total_chunks == 0:
            continue

        for name in stratum_names:
            stratum = strata[name]
            # Proportional quota, but at least min_per_stratum
            raw_quota = int(target * len(stratum.chunks) / total_chunks)
            stratum.quota = max(min_per_stratum, raw_quota)
            # Cap at available chunks (approx 0.5 questions per chunk)
            max_questions = int(len(stratum.chunks) * 0.5)
            stratum.quota = min(stratum.quota, max_questions)

    return strata


def select_chunks_for_generation(
    strata: dict[str, Stratum],
    chunks: list[dict],
) -> dict[str, Stratum]:
    """
    Select specific chunks to use for question generation.

    Selection strategy:
    - Diverse page coverage within each stratum
    - Prefer chunks with substantive content (>50 tokens)
    - Avoid table-of-contents and header chunks

    Args:
        strata: Dictionary of strata with quotas
        chunks: Original chunk list

    Returns:
        Updated strata with selected_chunks populated
    """
    # Build chunk index
    chunk_index = {c["id"]: c for c in chunks}

    for stratum in strata.values():
        # Skip if no quota
        if stratum.quota == 0:
            continue

        # Filter chunks: prefer substantive content
        good_chunks = []
        for chunk_info in stratum.chunks:
            chunk = chunk_index.get(chunk_info.id)
            if not chunk:
                continue

            text = chunk.get("text", "")
            tokens = chunk.get("tokens", 0)

            # Skip very short chunks (likely headers/TOC)
            if tokens < 50:
                continue

            # Skip obvious non-content
            if text.strip().startswith("##") and len(text.strip()) < 100:
                continue

            # Skip table of contents patterns
            if re.search(r"\.{3,}\s*\d+$", text):
                continue

            good_chunks.append(chunk_info)

        # Ensure page diversity: group by page
        by_page: dict[int, list[ChunkInfo]] = defaultdict(list)
        for chunk_info in good_chunks:
            by_page[chunk_info.page].append(chunk_info)

        # Round-robin selection across pages
        selected: list[ChunkInfo] = []
        pages = sorted(by_page.keys())
        page_idx = 0
        while len(selected) < stratum.quota * 2 and pages:  # 2x for margin
            page = pages[page_idx % len(pages)]
            if by_page[page]:
                selected.append(by_page[page].pop(0))
            else:
                # Page exhausted, remove
                pages = [p for p in pages if by_page[p]]
            page_idx += 1

        stratum.selected_chunks = [c.id for c in selected]

    return strata


def compute_coverage(strata: dict[str, Stratum], chunks: list[dict]) -> dict:
    """
    Compute corpus coverage statistics.

    Args:
        strata: Dictionary of strata
        chunks: Original chunk list

    Returns:
        Coverage statistics
    """
    # Unique documents in corpus
    all_docs = set(c.get("source", "") for c in chunks)

    # Documents covered by selected chunks
    selected_ids = set()
    for stratum in strata.values():
        selected_ids.update(stratum.selected_chunks)

    covered_docs = set()
    for chunk in chunks:
        if chunk["id"] in selected_ids:
            covered_docs.add(chunk.get("source", ""))

    return {
        "total_documents": len(all_docs),
        "covered_documents": len(covered_docs),
        "coverage_ratio": len(covered_docs) / len(all_docs) if all_docs else 0,
        "total_chunks": len(chunks),
        "selected_chunks": len(selected_ids),
    }


def validate_stratification(
    strata: dict[str, Stratum],
    coverage: dict,
    min_strata: int = 5,
    min_coverage: float = 0.80,
) -> tuple[bool, list[str], list[str]]:
    """
    Validate stratification against quality gates.

    Gates:
    - G0-1: len(strata) >= min_strata (BLOCKING)
    - G0-2: coverage_documents >= min_coverage (WARNING)

    Args:
        strata: Dictionary of strata
        coverage: Coverage statistics
        min_strata: Minimum number of strata (G0-1)
        min_coverage: Minimum document coverage (G0-2)

    Returns:
        Tuple of (blocking_passed, blocking_errors, warnings)
    """
    blocking_errors = []
    warnings = []

    # G0-1: Minimum strata count (BLOCKING)
    active_strata = sum(1 for s in strata.values() if s.quota > 0)
    if active_strata < min_strata:
        blocking_errors.append(
            f"G0-1 FAIL: {active_strata} strata < {min_strata} minimum"
        )

    # G0-2: Minimum coverage (WARNING - not blocking)
    coverage_ratio = coverage["coverage_ratio"]
    if coverage_ratio < min_coverage:
        warnings.append(
            f"G0-2 WARN: {coverage_ratio:.1%} coverage < {min_coverage:.0%} target"
        )

    return len(blocking_errors) == 0, blocking_errors, warnings


def format_report(
    strata: dict[str, Stratum],
    coverage: dict,
    target_total: int,
) -> str:
    """Format stratification report for display."""
    lines = [
        "=" * 70,
        "PHASE 0: CORPUS STRATIFICATION REPORT",
        "=" * 70,
        "",
        f"Target questions: {target_total}",
        f"Total chunks: {coverage['total_chunks']}",
        f"Selected chunks: {coverage['selected_chunks']}",
        "",
        "DOCUMENT COVERAGE:",
        f"  Documents in corpus: {coverage['total_documents']}",
        f"  Documents covered: {coverage['covered_documents']}",
        f"  Coverage ratio: {coverage['coverage_ratio']:.1%}",
        "",
        "STRATA DISTRIBUTION:",
        "",
    ]

    # Sort by priority then by chunk count
    sorted_strata = sorted(
        strata.values(),
        key=lambda s: (s.priority, -len(s.chunks)),
    )

    for stratum in sorted_strata:
        lines.append(
            f"  [{stratum.priority}] {stratum.name:20s} "
            f"chunks={len(stratum.chunks):4d}  "
            f"quota={stratum.quota:3d}  "
            f"selected={len(stratum.selected_chunks):3d}"
        )
        lines.append(f"      {stratum.description}")

    lines.append("")
    total_quota = sum(s.quota for s in strata.values())
    lines.append(f"TOTAL QUOTA: {total_quota}")

    return "\n".join(lines)


def run_stratification(
    chunks_path: Path,
    output_path: Path,
    target_total: int = 700,
    exclude_covered_gs: Path | None = None,
) -> dict:
    """
    Run complete stratification pipeline.

    Args:
        chunks_path: Path to chunks JSON
        output_path: Path to output strata JSON
        target_total: Target total questions
        exclude_covered_gs: If set, filter out chunks covered by this GS file

    Returns:
        Stratification result dictionary
    """
    print(f"Loading chunks from {chunks_path}...")
    chunks = load_chunks(chunks_path)
    print(f"  Loaded {len(chunks)} chunks")

    if exclude_covered_gs is not None:
        original_count = len(chunks)
        chunks = filter_uncovered_chunks(chunks, exclude_covered_gs)
        print(
            f"  Filtered to {len(chunks)} uncovered chunks "
            f"(excluded {original_count - len(chunks)} covered)"
        )

    print("\nStratifying by source document...")
    strata = stratify_chunks(chunks)
    print(f"  Created {len(strata)} strata")

    print("\nComputing quotas...")
    strata = compute_quotas(strata, target_total)

    print("\nSelecting chunks for generation...")
    strata = select_chunks_for_generation(strata, chunks)

    print("\nComputing coverage...")
    coverage = compute_coverage(strata, chunks)

    # Validate
    passed, blocking_errors, warnings = validate_stratification(strata, coverage)

    # Format report
    report = format_report(strata, coverage, target_total)
    print("\n" + report)

    if blocking_errors:
        print("\nBLOCKING ERRORS:")
        for error in blocking_errors:
            print(f"  {error}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  {warning}")

    # Prepare output
    result = {
        "version": "1.0",
        "date": get_date(),
        "target_total": target_total,
        "coverage": coverage,
        "validation": {
            "passed": passed,
            "blocking_errors": blocking_errors,
            "warnings": warnings,
        },
        "strata": {
            name: {
                "description": s.description,
                "priority": s.priority,
                "chunk_count": len(s.chunks),
                "quota": s.quota,
                "selected_chunks": s.selected_chunks,
            }
            for name, s in strata.items()
        },
    }

    # Save
    save_json(result, output_path)
    print(f"\nSaved stratification to {output_path}")

    return result


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stratify corpus chunks for GS generation"
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        default=Path("corpus/processed/chunks_mode_b_fr.json"),
        help="Input chunks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/gs_generation/chunk_strata.json"),
        help="Output strata JSON file",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=int,
        default=700,
        help="Target total questions (default: 700)",
    )
    parser.add_argument(
        "--exclude-covered",
        type=Path,
        default=None,
        help="GS JSON file — exclude chunks already covered by answerable Qs",
    )

    args = parser.parse_args()

    result = run_stratification(
        args.chunks,
        args.output,
        args.target,
        exclude_covered_gs=args.exclude_covered,
    )

    if not result["validation"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
