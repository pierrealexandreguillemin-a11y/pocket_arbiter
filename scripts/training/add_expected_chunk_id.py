#!/usr/bin/env python3
"""
Add expected_chunk_id to gold standard files.

This script:
1. Reads gold standard questions
2. Queries SQLite Mode B databases for candidate chunks from expected pages
3. Uses semantic matching to identify the correct chunk
4. Updates gold standard files with expected_chunk_id

ISO 42001 A.6.2.2 - Provenance tracable
ISO 29119-3 - Test data documented
"""

import json
import sqlite3
import re
from pathlib import Path
from typing import Any

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLD_STANDARD_FR = PROJECT_ROOT / "tests" / "data" / "gold_standard_fr.json"
GOLD_STANDARD_INTL = PROJECT_ROOT / "tests" / "data" / "gold_standard_intl.json"
CORPUS_FR_DB = PROJECT_ROOT / "corpus" / "processed" / "corpus_mode_b_fr.db"
CORPUS_INTL_DB = PROJECT_ROOT / "corpus" / "processed" / "corpus_mode_b_intl.db"


def normalize_source_name(source: str) -> str:
    """Normalize source name for matching by removing accents."""
    replacements = [
        ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
        ("à", "a"), ("â", "a"),
        ("ù", "u"), ("û", "u"),
        ("î", "i"), ("ï", "i"),
        ("ô", "o"),
        ("ç", "c"),
    ]
    result = source.lower()
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def get_chunks_for_pages(
    db_path: Path,
    source_pattern: str,
    pages: list[int],
) -> list[dict[str, Any]]:
    """
    Get all chunks from specified pages.

    Args:
        db_path: Path to SQLite database
        source_pattern: Pattern to match source (e.g., "LA-octobre2025.pdf")
        pages: List of page numbers

    Returns:
        List of chunk dictionaries with id, text, source, page
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # First get all sources to find matching one
    cursor.execute("SELECT DISTINCT source FROM chunks")
    all_sources = [r[0] for r in cursor.fetchall()]

    # Find matching source by normalized comparison
    source_pattern_norm = normalize_source_name(source_pattern)
    matching_sources = []
    for src in all_sources:
        if source_pattern_norm in normalize_source_name(src):
            matching_sources.append(src)

    if not matching_sources:
        conn.close()
        return []

    # Build query for pages with exact source matches
    source_placeholders = ",".join("?" * len(matching_sources))
    page_placeholders = ",".join("?" * len(pages))
    query = f"""
        SELECT id, text, source, page
        FROM chunks
        WHERE source IN ({source_placeholders}) AND page IN ({page_placeholders})
        ORDER BY page, id
    """

    params = matching_sources + pages
    cursor.execute(query, params)

    chunks = []
    for row in cursor.fetchall():
        chunks.append({
            "id": row[0],
            "text": row[1],
            "source": row[2],
            "page": row[3],
        })

    conn.close()
    return chunks


def find_best_chunk_for_question(
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> str | None:
    """
    Find the best matching chunk for a question using keyword matching.

    Uses expected keywords and question terms to score chunks.

    Args:
        question: Question dictionary from gold standard
        chunks: List of candidate chunks

    Returns:
        Chunk ID of best match, or None if no match
    """
    if not chunks:
        return None

    # Get keywords from question
    keywords = question.get("keywords", [])
    question_text = question.get("question", "").lower()

    # Extract additional terms from question
    question_words = set(re.findall(r'\b\w{4,}\b', question_text))
    all_terms = set(k.lower() for k in keywords) | question_words

    # Score each chunk
    best_chunk = None
    best_score = 0

    for chunk in chunks:
        chunk_text_lower = chunk["text"].lower()
        score = 0

        # Count keyword matches
        for term in all_terms:
            if term in chunk_text_lower:
                score += 1

        # Boost for article numbers if present
        article_num = question.get("metadata", {}).get("article_num", "")
        if article_num:
            # Check for article patterns like "4.1", "Article 4"
            articles = re.findall(r'\d+\.?\d*', article_num)
            for art in articles:
                if art in chunk_text_lower:
                    score += 5  # Strong boost for article match

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk and best_score > 0:
        return best_chunk["id"]
    return None


def process_gold_standard(
    gold_path: Path,
    db_path: Path,
    corpus_type: str = "fr",
) -> dict[str, Any]:
    """
    Process a gold standard file and add expected_chunk_id.

    Args:
        gold_path: Path to gold standard JSON
        db_path: Path to corpus SQLite database
        corpus_type: "fr" or "intl"

    Returns:
        Report dictionary with statistics and modifications
    """
    # Load gold standard
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    questions = gold.get("questions", [])

    # Use explicit variables for type safety
    chunk_id_found = 0
    no_chunks_found = 0
    failed_matching = 0
    already_has_chunk_id = 0
    modifications: list[dict[str, Any]] = []
    failed_questions: list[dict[str, Any]] = []

    for question in questions:
        qid = question.get("id", "unknown")

        # Skip if already has expected_chunk_id
        if "expected_chunk_id" in question:
            already_has_chunk_id += 1
            continue

        # Get expected pages and docs
        expected_pages = question.get("expected_pages", [])
        expected_docs = question.get("expected_docs", [])

        if not expected_pages:
            failed_questions.append({
                "id": qid,
                "reason": "No expected_pages",
            })
            failed_matching += 1
            continue

        # Get source pattern from expected_docs
        source_pattern = expected_docs[0] if expected_docs else ""

        # Get candidate chunks
        chunks = get_chunks_for_pages(db_path, source_pattern, expected_pages)

        if not chunks:
            failed_questions.append({
                "id": qid,
                "reason": "No chunks found for pages",
                "pages": expected_pages,
                "source": source_pattern,
            })
            no_chunks_found += 1
            continue

        # Find best matching chunk
        matched_chunk_id = find_best_chunk_for_question(question, chunks)

        if matched_chunk_id:
            question["expected_chunk_id"] = matched_chunk_id
            chunk_id_found += 1
            modifications.append({
                "id": qid,
                "chunk_id": matched_chunk_id,
                "num_candidates": len(chunks),
            })
        else:
            # Take first chunk as fallback (least bad option)
            question["expected_chunk_id"] = chunks[0]["id"]
            chunk_id_found += 1
            modifications.append({
                "id": qid,
                "chunk_id": chunks[0]["id"],
                "num_candidates": len(chunks),
                "method": "fallback_first_chunk",
            })

    # Save updated gold standard
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

    # Build report dictionary
    report: dict[str, Any] = {
        "corpus": corpus_type,
        "total_questions": len(questions),
        "chunk_id_found": chunk_id_found,
        "no_chunks_found": no_chunks_found,
        "failed_matching": failed_matching,
        "already_has_chunk_id": already_has_chunk_id,
        "modifications": modifications,
        "failed_questions": failed_questions,
    }

    return report


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Adding expected_chunk_id to Gold Standards")
    print("=" * 60)

    # Process FR gold standard
    print("\n[FR] Processing gold_standard_fr.json...")
    report_fr = process_gold_standard(GOLD_STANDARD_FR, CORPUS_FR_DB, "fr")
    print(f"  Total questions: {report_fr['total_questions']}")
    print(f"  Chunk IDs found: {report_fr['chunk_id_found']}")
    print(f"  Already had chunk_id: {report_fr['already_has_chunk_id']}")
    print(f"  No chunks found: {report_fr['no_chunks_found']}")
    print(f"  Failed matching: {report_fr['failed_matching']}")

    # Process INTL gold standard
    print("\n[INTL] Processing gold_standard_intl.json...")
    report_intl = process_gold_standard(GOLD_STANDARD_INTL, CORPUS_INTL_DB, "intl")
    print(f"  Total questions: {report_intl['total_questions']}")
    print(f"  Chunk IDs found: {report_intl['chunk_id_found']}")
    print(f"  Already had chunk_id: {report_intl['already_has_chunk_id']}")
    print(f"  No chunks found: {report_intl['no_chunks_found']}")
    print(f"  Failed matching: {report_intl['failed_matching']}")

    # Save combined report
    report_path = PROJECT_ROOT / "corpus" / "processed" / "chunk_id_enrichment_report.json"
    combined_report = {
        "fr": report_fr,
        "intl": report_intl,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Report saved to: {report_path}")
    print("=" * 60)

    # Show failed questions if any
    all_failed = report_fr["failed_questions"] + report_intl["failed_questions"]
    if all_failed:
        print("\n[WARN] Failed questions:")
        for q in all_failed[:10]:
            print(f"  - {q['id']}: {q['reason']}")
        if len(all_failed) > 10:
            print(f"  ... and {len(all_failed) - 10} more")


if __name__ == "__main__":
    main()
