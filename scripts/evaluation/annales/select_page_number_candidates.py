"""Select page-number questions as candidates for Phase B-P3a replacement.

Identifies answerable questions matching the "Quelle regle a la page X?"
pattern, which are unsuitable for RAG retrieval (no semantic anchor).

ISO Reference:
- ISO 42001 A.6.2.2: Provenance tracking
- ISO 29119-3: Test data quality

Usage:
    python -m scripts.evaluation.annales.select_page_number_candidates \
        --gs tests/data/gs_scratch_v1_step1.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output data/gs_generation/p3_page_candidates.json
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402

# Pattern matching page-number references in question text
PAGE_NUMBER_PATTERN = re.compile(
    r"(?i)\bpage\s*\d+",
)


@dataclass
class PageCandidate:
    """A page-number question targeted for replacement."""

    old_id: str
    chunk_id: str
    chunk_text: str
    source: str
    question_preview: str


def is_page_number_question(question_text: str) -> bool:
    """Check if a question references a specific page number.

    Args:
        question_text: The question text to check.

    Returns:
        True if the question contains a page-number reference.
    """
    return bool(PAGE_NUMBER_PATTERN.search(question_text))


def select_page_number_questions(
    gs_data: dict,
    chunk_index: dict[str, str],
    source_index: dict[str, str] | None = None,
) -> list[PageCandidate]:
    """Select answerable questions that reference page numbers.

    Args:
        gs_data: Full GS JSON data.
        chunk_index: Mapping chunk_id -> chunk_text.
        source_index: Optional mapping chunk_id -> source filename.

    Returns:
        List of PageCandidate for replacement.
    """
    questions = gs_data.get("questions", [])
    candidates: list[PageCandidate] = []

    for q in questions:
        if q.get("content", {}).get("is_impossible", False):
            continue

        question_text = q.get("content", {}).get("question", "")
        if not is_page_number_question(question_text):
            continue

        chunk_id = q.get("provenance", {}).get("chunk_id", "")
        chunk_text = chunk_index.get(chunk_id, "")
        source = ""
        if source_index:
            source = source_index.get(chunk_id, "")

        candidates.append(
            PageCandidate(
                old_id=q["id"],
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                source=source,
                question_preview=question_text[:120],
            )
        )

    return candidates


def build_chunk_indexes(
    chunks_data: dict,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build chunk_id -> text and chunk_id -> source mappings.

    Args:
        chunks_data: Raw chunks JSON data.

    Returns:
        Tuple of (chunk_text_index, chunk_source_index).
    """
    chunks = chunks_data.get("chunks", chunks_data)
    if isinstance(chunks, dict):
        chunks = list(chunks.values())

    text_index: dict[str, str] = {}
    source_index: dict[str, str] = {}
    for c in chunks:
        cid = c["id"]
        text_index[cid] = c.get("text", "")
        source_index[cid] = c.get("source", "")
    return text_index, source_index


def format_report(candidates: list[PageCandidate]) -> str:
    """Format a human-readable report of selected candidates.

    Args:
        candidates: List of PageCandidate.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        "P3a: PAGE-NUMBER CANDIDATES FOR REPLACEMENT",
        "=" * 60,
        f"Total candidates: {len(candidates)}",
        "",
    ]

    # Group by source
    source_counter: Counter[str] = Counter()
    for c in candidates:
        source_counter[c.source or "unknown"] += 1

    lines.append("By document source:")
    for src, count in source_counter.most_common():
        lines.append(f"  {src[:55]:55s} {count:3d}")

    lines.append("")
    lines.append("First 10 candidates:")
    for c in candidates[:10]:
        lines.append(f"  {c.old_id[:40]:40s} {c.question_preview[:60]}")

    return "\n".join(lines)


def save_candidates(
    candidates: list[PageCandidate],
    output_path: Path,
) -> dict:
    """Save candidates to JSON for downstream processing.

    Args:
        candidates: List of PageCandidate.
        output_path: Path to output JSON file.

    Returns:
        Output data dict.
    """
    output = {
        "version": "1.0",
        "date": get_date(),
        "phase": "B-P3a",
        "description": "Page-number questions for replacement",
        "total": len(candidates),
        "candidates": [asdict(c) for c in candidates],
    }
    save_json(output, output_path)
    return output


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Select page-number questions for Phase B-P3a replacement",
    )
    parser.add_argument(
        "--gs",
        type=Path,
        default=Path("tests/data/gs_scratch_v1_step1.json"),
        help="GS JSON file (default: tests/data/gs_scratch_v1_step1.json)",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("corpus/processed/chunks_mode_b_fr.json"),
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/gs_generation/p3_page_candidates.json"),
        help="Output candidates JSON",
    )
    args = parser.parse_args()

    gs_data = load_json(args.gs)
    chunks_data = load_json(args.chunks)
    text_index, source_index = build_chunk_indexes(chunks_data)

    candidates = select_page_number_questions(gs_data, text_index, source_index)

    report = format_report(candidates)
    print(report)

    save_candidates(candidates, args.output)
    print(f"\nSaved {len(candidates)} candidates to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
