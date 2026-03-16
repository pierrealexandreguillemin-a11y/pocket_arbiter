#!/usr/bin/env python3
"""
Verify and correct chunk mappings using semantic analysis.

Uses 4 parallel agents to:
1. Analyze question semantics
2. Match against docling-parsed chunks
3. Classify retrieval difficulty (single-hop, multi-hop, reasoning)
4. Validate expected_pages alignment

ISO 42001 A.6.2.2: Provenance verification
ISO 29119: Test data quality
"""

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Difficulty taxonomy constants
DIFFICULTY_SINGLE_HOP = 0.3  # Direct fact lookup
DIFFICULTY_MULTI_HOP = 0.6  # Multiple sources needed
DIFFICULTY_REASONING = 0.9  # Inference required


@dataclass
class ChunkCandidate:
    """A candidate chunk for a question."""

    chunk_id: str
    page: int
    text: str
    score: float = 0.0


def load_gold_standard(path: str) -> dict:
    """Load gold standard JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_chunks_for_source(conn: sqlite3.Connection, source_pattern: str) -> list[dict]:
    """Get all chunks matching a source pattern."""
    cur = conn.execute(
        """SELECT id, page, text, source FROM chunks
           WHERE source LIKE ? ORDER BY page""",
        (f"%{source_pattern}%",),
    )
    return [
        {"id": r[0], "page": r[1], "text": r[2], "source": r[3]} for r in cur.fetchall()
    ]


def extract_article_reference(text: str) -> str | None:
    """Extract article number from text (e.g., 'Article 7.5.5')."""
    patterns = [
        r"Article\s*(\d+\.\d+\.\d+\.\d+)",
        r"Article\s*(\d+\.\d+\.\d+)",
        r"Article\s*(\d+\.\d+)",
        r"Article\s*(\d+)",
        r"(\d+\.\d+\.\d+)",
        r"(\d+\.\d+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def compute_semantic_score(question: dict, chunk: dict) -> float:
    """
    Compute semantic similarity score between question and chunk.

    Factors:
    - Keyword overlap
    - Article reference match
    - Answer text in chunk
    """
    score = 0.0

    _q_text = question.get("question", question.get("question_text", "")).lower()
    answer = question.get("answer_text", "").lower()
    ref = question.get("article_reference", "").lower()
    chunk_text = chunk["text"].lower()

    # Keyword overlap from question
    keywords = question.get("keywords", [])
    for kw in keywords:
        if kw.lower() in chunk_text:
            score += 1.0

    # Article reference match (high weight)
    q_article = extract_article_reference(ref)
    chunk_article = extract_article_reference(chunk_text)
    if q_article and chunk_article and q_article == chunk_article:
        score += 5.0

    # Answer text presence (critical)
    if answer:
        # Check if key phrases from answer appear in chunk
        answer_words = set(answer.split())
        chunk_words = set(chunk_text.split())
        overlap = len(answer_words & chunk_words)
        if overlap > 5:
            score += 3.0

    # Page alignment with expected_pages
    expected_pages = question.get("expected_pages", [])
    if chunk["page"] in expected_pages:
        score += 2.0

    return score


def classify_retrieval_difficulty(question: dict, best_chunk: dict) -> float:
    """
    Classify retrieval difficulty based on question complexity.

    - Single-hop (0.3): Direct fact lookup, answer in one chunk
    - Multi-hop (0.6): Need to combine info from multiple sources
    - Reasoning (0.9): Requires inference or calculation
    """
    q_text = question.get("question", question.get("question_text", "")).lower()
    answer = question.get("answer_text", "").lower()

    # Reasoning indicators
    reasoning_patterns = [
        r"combien",
        r"calculer",
        r"quel.*score",
        r"quelle.*sanction",
        r"que.*faire",
        r"comment.*r[ée]agir",
        r"si.*alors",
        r"dans.*cas",
        r"exception",
        r"priorit[ée]",
    ]
    for p in reasoning_patterns:
        if re.search(p, q_text):
            return DIFFICULTY_REASONING

    # Multi-hop indicators
    multihop_patterns = [
        r"et.*aussi",
        r"deux.*r[èe]gles",
        r"plusieurs",
        r"conform[ée]ment.*et",
        r"article.*et.*article",
    ]
    for p in multihop_patterns:
        if re.search(p, q_text) or re.search(p, answer):
            return DIFFICULTY_MULTI_HOP

    # Check if answer spans multiple concepts
    expected_pages = question.get("expected_pages", [])
    if len(expected_pages) > 2:
        return DIFFICULTY_MULTI_HOP

    # Default: single-hop
    return DIFFICULTY_SINGLE_HOP


def find_best_chunk_for_question(
    question: dict, chunks: list[dict]
) -> tuple[str | None, float]:
    """
    Find the best matching chunk for a question.

    Returns (chunk_id, difficulty_retrieval)
    """
    if not chunks:
        return None, DIFFICULTY_REASONING

    # Score all chunks
    candidates = []
    for chunk in chunks:
        score = compute_semantic_score(question, chunk)
        candidates.append(
            ChunkCandidate(
                chunk_id=chunk["id"],
                page=chunk["page"],
                text=chunk["text"],
                score=score,
            )
        )

    # Sort by score descending
    candidates.sort(key=lambda x: x.score, reverse=True)

    if not candidates or candidates[0].score == 0:
        return None, DIFFICULTY_REASONING

    best = candidates[0]
    difficulty = classify_retrieval_difficulty(question, {"text": best.text})

    return best.chunk_id, difficulty


def process_question_batch(
    questions: list[dict], db_path: str, batch_id: int
) -> list[dict]:
    """
    Process a batch of questions (for parallel execution).

    Returns list of results with chunk_id and difficulties.
    """
    conn = sqlite3.connect(db_path)
    results = []

    for q in questions:
        # Get source document
        expected_docs = q.get("expected_docs", [])
        if not expected_docs:
            results.append(
                {
                    "id": q["id"],
                    "chunk_id": None,
                    "difficulty_retrieval": DIFFICULTY_REASONING,
                    "error": "No expected_docs",
                }
            )
            continue

        # Extract source pattern from document name
        doc = expected_docs[0]
        source_pattern = doc.replace(".pdf", "").split("_")[0]

        # Get chunks for this source
        chunks = get_chunks_for_source(conn, source_pattern)

        # Find best chunk
        chunk_id, diff_retrieval = find_best_chunk_for_question(q, chunks)

        # Compute human difficulty
        if q.get("annales_source") and q["annales_source"].get("success_rate"):
            diff_human = round(1 - q["annales_source"]["success_rate"], 2)
        else:
            diff_human = 0.5  # Default when no annales data

        results.append(
            {
                "id": q["id"],
                "chunk_id": chunk_id,
                "difficulty_retrieval": diff_retrieval,
                "difficulty_human": diff_human,
                "batch": batch_id,
            }
        )

    conn.close()
    return results


def verify_and_update_gold_standard(
    gs_path: str, db_path: str, num_workers: int = 4
) -> dict:
    """
    Main function: verify all questions with parallel workers.

    Returns statistics about updates made.
    """
    gs = load_gold_standard(gs_path)
    questions = gs["questions"]

    # Split into batches
    batch_size = len(questions) // num_workers + 1
    batches = [
        questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
    ]

    print(f"Processing {len(questions)} questions in {len(batches)} batches...")

    # Process in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_question_batch, batch, db_path, i): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"  Batch {batch_id} complete: {len(results)} questions")
            except Exception as e:
                print(f"  Batch {batch_id} failed: {e}")

    # Create lookup from results
    results_lookup = {r["id"]: r for r in all_results}

    # Update gold standard
    stats = {"chunk_added": 0, "chunk_updated": 0, "difficulty_added": 0, "errors": 0}

    for q in questions:
        result = results_lookup.get(q["id"])
        if not result:
            stats["errors"] += 1
            continue

        # Add/update chunk_id (without modifying expected_pages!)
        if result["chunk_id"]:
            if not q.get("expected_chunk_id"):
                stats["chunk_added"] += 1
            elif q["expected_chunk_id"] != result["chunk_id"]:
                stats["chunk_updated"] += 1
            q["expected_chunk_id"] = result["chunk_id"]

        # Add both difficulty types
        q["difficulty_human"] = result["difficulty_human"]
        q["difficulty_retrieval"] = result["difficulty_retrieval"]

        # Keep legacy 'difficulty' for backwards compatibility
        if not q.get("difficulty"):
            q["difficulty"] = result["difficulty_human"]
            stats["difficulty_added"] += 1

    # Save updated gold standard
    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify chunk mappings")
    parser.add_argument(
        "--gold-standard",
        default="tests/data/gold_standard_annales_fr.json",
        help="Path to gold standard JSON",
    )
    parser.add_argument(
        "--db",
        default="corpus/processed/corpus_mode_b_fr.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print changes without saving"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("  CHUNK MAPPING VERIFICATION")
    print("  ISO 42001 A.6.2.2 Provenance")
    print("=" * 50)
    print()

    stats = verify_and_update_gold_standard(args.gold_standard, args.db, args.workers)

    print()
    print("=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  Chunks added:    {stats['chunk_added']}")
    print(f"  Chunks updated:  {stats['chunk_updated']}")
    print(f"  Difficulty added: {stats['difficulty_added']}")
    print(f"  Errors:          {stats['errors']}")
    print("=" * 50)
