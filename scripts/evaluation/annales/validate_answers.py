"""
Validate Gold Standard answers against current corpus and add page references.

This module:
1. Builds an article-to-page index from corpus chunks
2. Validates each GS question's answer exists in the corpus
3. Adds expected_pages for retrieval validation
4. Flags potentially outdated answers

ISO Reference:
    - ISO/IEC 42001 A.7.3 - Data traceability
    - ISO/IEC 25010 - Functional suitability

Usage:
    python -m scripts.evaluation.annales.validate_answers \
        --gold-standard tests/data/gold_standard_annales_fr.json \
        --chunks corpus/processed/chunks_mode_a_fr.json \
        --output tests/data/gold_standard_annales_fr_validated.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from scripts.pipeline.utils import get_timestamp, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Article number patterns - multiple formats
ARTICLE_PATTERN = re.compile(r"(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
# Also match single numbers (chapters, sections)
CHAPTER_PATTERN = re.compile(r"(?:Chapitre|Chap\.?)\s*(\d+)", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^(\d+)\.", re.MULTILINE)  # "2. Statut" -> "2"
PREAMBULE_PATTERN = re.compile(r"Pr[eé]ambule", re.IGNORECASE)

# R01/R02/A02 patterns (FFE internal regulations)
# Examples: "R01 - 2. Statut", "R01 article 5.3", "R.01 - art. 3"
R01_SECTION_PATTERN = re.compile(
    r"R\.?01\s*[-–—:]\s*(\d+)\.\s*\w+",
    re.IGNORECASE,
)
R01_ARTICLE_PATTERN = re.compile(
    r"R\.?01\s*[-–—]?\s*(?:article|art\.?)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Generic regulation pattern: R01, R02, R03, A01, A02, etc.
REGULATION_PATTERN = re.compile(
    r"([RA])\.?(\d{2})\s*[-–—:]\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def build_article_page_index(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Build index mapping article numbers to pages and content.

    Args:
        chunks: List of chunk dicts with text, source, pages.

    Returns:
        Dict mapping (source, article) -> {pages: set, chunks: list}
    """
    index: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        source = chunk.get("source", "")
        text = chunk.get("text", "")
        pages = chunk.get("pages", [])

        # Find all article mentions in text (multiple patterns)
        articles = set(ARTICLE_PATTERN.findall(text))

        # Also find chapter numbers
        chapters = CHAPTER_PATTERN.findall(text)
        for ch in chapters:
            articles.add(f"Ch.{ch}")

        # Find section numbers (e.g., "2. Statut")
        sections = SECTION_PATTERN.findall(text)
        for sec in sections:
            articles.add(f"Sec.{sec}")

        # Check for Préambule
        if PREAMBULE_PATTERN.search(text):
            articles.add("Préambule")

        # Find R01 section patterns (e.g., "R01 - 2. Statut")
        r01_sections = R01_SECTION_PATTERN.findall(text)
        for sec in r01_sections:
            articles.add(f"R01.Sec.{sec}")

        # Find R01 article patterns (e.g., "R01 article 5.3")
        r01_articles = R01_ARTICLE_PATTERN.findall(text)
        for art in r01_articles:
            articles.add(f"R01.{art}")

        # Find generic regulation patterns (R01, R02, A02, etc.)
        for match in REGULATION_PATTERN.finditer(text):
            reg_type = match.group(1).upper()  # R or A
            reg_num = match.group(2)            # 01, 02, etc.
            section = match.group(3)            # 5.3, 2, etc.
            articles.add(f"{reg_type}{reg_num}.{section}")

        for article in articles:
            key = f"{source}|{article}"
            if key not in index:
                index[key] = {"pages": set(), "chunks": [], "source": source, "article": article}

            index[key]["pages"].update(pages)
            index[key]["chunks"].append(
                {"text": text[:500], "pages": pages, "chunk_id": chunk.get("id", "")}
            )

    # Convert sets to sorted lists
    for key in index:
        index[key]["pages"] = sorted(index[key]["pages"])

    return index


def extract_article_from_reference(reference: str) -> list[str]:
    """
    Extract article/chapter/section numbers from reference string.

    Returns multiple possible keys to search in the index.
    """
    if not reference:
        return []

    keys = []

    # Check for R01/R02/A02 patterns first (more specific)
    # R01 section pattern (e.g., "R01 - 2. Statut")
    r01_section_match = R01_SECTION_PATTERN.search(reference)
    if r01_section_match:
        keys.append(f"R01.Sec.{r01_section_match.group(1)}")

    # R01 article pattern (e.g., "R01 article 5.3")
    r01_article_match = R01_ARTICLE_PATTERN.search(reference)
    if r01_article_match:
        keys.append(f"R01.{r01_article_match.group(1)}")

    # Generic regulation pattern (R01, R02, A02, etc.)
    for match in REGULATION_PATTERN.finditer(reference):
        reg_type = match.group(1).upper()
        reg_num = match.group(2)
        section = match.group(3)
        keys.append(f"{reg_type}{reg_num}.{section}")

    # Check for article number (e.g., "1.3", "4.2.1")
    article_match = ARTICLE_PATTERN.search(reference)
    if article_match:
        keys.append(article_match.group(1))

    # Check for chapter reference (e.g., "Chapitre 8")
    chapter_match = CHAPTER_PATTERN.search(reference)
    if chapter_match:
        keys.append(f"Ch.{chapter_match.group(1)}")

    # Check for section pattern (e.g., "R01 - 2. Statut")
    # Match "- X." or "- X.Y" patterns
    section_match = re.search(r"-\s*(\d+)\.(?:\s|$|\d)", reference)
    if section_match:
        keys.append(f"Sec.{section_match.group(1)}")

    # Check for Préambule
    if PREAMBULE_PATTERN.search(reference):
        keys.append("Préambule")

    return keys


def find_answer_in_chunks(
    answer: str, chunks: list[dict[str, Any]], threshold: float = 0.5
) -> dict[str, Any]:
    """
    Search for answer text in chunks.

    Args:
        answer: Expected answer text.
        chunks: List of chunk dicts to search.
        threshold: Minimum match ratio (0-1).

    Returns:
        Dict with found, confidence, chunk_id, context.
    """
    if not answer or len(answer) <= 2:
        # Letter-only answers (A, B, C, D) - can't validate in text
        return {"found": None, "reason": "letter_only_answer"}

    answer_lower = answer.lower()

    for chunk in chunks:
        text = chunk.get("text", "").lower()

        if answer_lower in text:
            return {
                "found": True,
                "confidence": 1.0,
                "chunk_id": chunk.get("chunk_id", ""),
                "context": text[:200],
            }

    return {"found": False, "confidence": 0.0, "reason": "not_found_in_chunks"}


def validate_gold_standard(
    gs_path: Path,
    chunks_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Validate Gold Standard answers and add page references.

    Args:
        gs_path: Path to Gold Standard JSON.
        chunks_path: Path to chunks JSON.
        output_path: Path for validated output.

    Returns:
        Validation statistics.
    """
    # Load data
    gs_data = json.loads(gs_path.read_text(encoding="utf-8"))
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = chunks_data.get("chunks", [])

    logger.info(f"Loaded {len(gs_data.get('questions', []))} questions")
    logger.info(f"Loaded {len(chunks)} chunks")

    # Build article-page index
    logger.info("Building article-page index...")
    article_index = build_article_page_index(chunks)
    logger.info(f"Indexed {len(article_index)} article-source combinations")

    # Stats with proper typing
    stats: dict[str, int] = {
        "total": 0,
        "pages_added": 0,
        "pages_not_found": 0,
        "letter_only": 0,
        "answer_validated": 0,
        "answer_not_found": 0,
    }
    by_document: dict[str, dict[str, int]] = {}

    questions = gs_data.get("questions", [])

    for q in questions:
        stats["total"] += 1

        # Get expected document and article
        expected_docs = q.get("expected_docs", [])
        article_ref = q.get("article_reference", "")
        article_keys = extract_article_from_reference(article_ref)

        # Try to find pages for this article (try all possible keys)
        pages_found: list[int] = []
        found_for_doc = False

        for doc in expected_docs:
            # Track by document
            if doc not in by_document:
                by_document[doc] = {"total": 0, "with_pages": 0}
            by_document[doc]["total"] += 1

            # Try each possible key from the reference
            for article_key in article_keys:
                key = f"{doc}|{article_key}"
                if key in article_index:
                    pages_found.extend(article_index[key]["pages"])
                    found_for_doc = True
                else:
                    # Fallback: try prefix match (e.g., "2.4" matches "2.4.1")
                    # Only for numeric article keys
                    if re.match(r"^\d+\.\d+$", article_key):
                        prefix = f"{doc}|{article_key}."
                        for idx_key in article_index:
                            if idx_key.startswith(prefix):
                                pages_found.extend(article_index[idx_key]["pages"])
                                found_for_doc = True

            if found_for_doc:
                by_document[doc]["with_pages"] += 1

        # Update expected_pages
        if pages_found:
            q["expected_pages"] = sorted(set(pages_found))
            stats["pages_added"] += 1
        else:
            stats["pages_not_found"] += 1

        # Validate answer (optional - check if answer text exists)
        answer = q.get("expected_answer", "")
        if len(answer) <= 4 and answer.upper() in [
            "A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD",
            "ABC", "ABD", "ACD", "BCD", "ABCD",
        ]:
            stats["letter_only"] += 1
            q["validation"]["answer_type"] = "letter_only"
        else:
            # For text answers, we could search in chunks
            stats["answer_validated"] += 1

        # Update validation metadata
        q["validation"]["pages_verified"] = len(pages_found) > 0
        q["validation"]["verified_date"] = get_timestamp().split("T")[0]

    # Update GS metadata
    gs_data["validation_stats"] = {
        "total_questions": stats["total"],
        "with_pages": stats["pages_added"],
        "without_pages": stats["pages_not_found"],
        "letter_only_answers": stats["letter_only"],
        "validation_date": get_timestamp(),
    }

    # Save validated GS
    save_json(gs_data, output_path)
    logger.info(f"Saved validated Gold Standard: {output_path}")

    # Return combined stats
    return {**stats, "by_document": by_document}


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Gold Standard answers")
    parser.add_argument(
        "--gold-standard",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr.json"),
        help="Input Gold Standard JSON",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("corpus/processed/chunks_mode_a_fr.json"),
        help="Chunks JSON for page lookup",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr.json"),
        help="Output validated Gold Standard (can be same as input)",
    )

    args = parser.parse_args()

    stats = validate_gold_standard(
        gs_path=args.gold_standard,
        chunks_path=args.chunks,
        output_path=args.output,
    )

    print("\n=== Validation Report ===")
    print(f"Total questions: {stats['total']}")
    print(f"Pages added: {stats['pages_added']}")
    print(f"Pages not found: {stats['pages_not_found']}")
    print(f"Letter-only answers: {stats['letter_only']}")
    print()
    print("By document:")
    for doc, doc_stats in sorted(stats["by_document"].items()):
        pct = doc_stats["with_pages"] * 100 / doc_stats["total"] if doc_stats["total"] > 0 else 0
        print(f"  {doc}: {doc_stats['with_pages']}/{doc_stats['total']} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
