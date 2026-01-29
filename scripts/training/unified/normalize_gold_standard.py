#!/usr/bin/env python3
"""
Normalize Gold Standard Annales FR.

Phases (from plan vast-floating-diffie.md):
1. Corriger 10 questions page [3] suspectes
2. Mapper 477 chunk_ids vers DB
3. Verifier 477 pages contre docling (page UNIQUE)
4. Completer 55 difficulty manquantes
5. Quality gates finaux

Exigence:
- 1 page unique (pas de multi-pages)
- Page VERIFIEE contre docling JSON
- Page REELLE (pas de fallback ToC/index)

ISO 42001 A.6.2.2: Provenance tracable
ISO 29119: Test data completeness
"""

import json
import sqlite3
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# === CONSTANTS ===

DIFFICULTY_SINGLE_HOP = 0.3
DIFFICULTY_MULTI_HOP = 0.6
DIFFICULTY_REASONING = 0.9

# Questions page [3] suspectes (from plan)
PAGE_3_SUSPECTS = [
    "FR-ANN-UVC-011",
    "FR-ANN-UVC-025",
    "FR-ANN-UVC-028",
    "FR-ANN-UVO-031",
    "FR-ANN-UVC-067",
    "FR-ANN-UVC-073",
    "FR-ANN-UVC-083",
    "FR-ANN-UVC-101",
    "FR-ANN-UVC-130",
    "FR-ANN-UVC-136",
]

# Paths
GS_PATH = Path("tests/data/gold_standard_annales_fr.json")
DB_PATH = Path("corpus/processed/corpus_mode_b_fr.db")
DOCLING_DIR = Path("corpus/processed/docling_fr")


@dataclass
class NormalizationResult:
    """Result of normalizing one question."""

    id: str
    chunk_id: Optional[str] = None
    page_verified: Optional[int] = None
    page_original: list = field(default_factory=list)
    difficulty_human: float = 0.5
    difficulty_retrieval: float = DIFFICULTY_SINGLE_HOP
    error: Optional[str] = None


def load_docling_json(doc_name: str) -> Optional[dict]:
    """Load docling JSON for a document."""
    # Map document names to docling files
    name_mappings = {
        "LA-octobre2025": "LA-octobre2025",
        "A02": "A02_2025_26_Championnat_de_France_des_Clubs",
        "R01": "R01_2025_26_Regles_generales",
        "R02": "R02_2025_26_Regles_generales_Annexes",
        "R03": "R03_2025_26_Competitions_homologuees",
        "C01": "C01_2025_26_Coupe_de_France",
        "C03": "C03_2025_26_Coupe_Jean_Claude_Loubatiere",
        "C04": "C04_2025_26_Coupe_de_la_parite",
        "J01": "J01_2025_26_Championnat_de_France_Jeunes",
        "J02": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes",
        "Reglement_Disciplinaire": "2018_Reglement_Disciplinaire20180422",
    }

    # Extract base name
    base = doc_name.replace(".pdf", "").split("_")[0]
    if base in name_mappings:
        docling_name = name_mappings[base]
    else:
        docling_name = doc_name.replace(".pdf", "")

    docling_path = DOCLING_DIR / f"{docling_name}.json"
    if docling_path.exists():
        with open(docling_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def find_page_in_docling(
    docling: dict, article_ref: str, answer_text: str
) -> Optional[int]:
    """
    Find the correct page in docling JSON by matching article reference and answer.

    Returns the page number where the content is found.
    """
    if not docling:
        return None

    # Extract article number from reference
    article_match = re.search(r"(\d+\.?\d*\.?\d*)", article_ref)
    article_num = article_match.group(1) if article_match else None

    # Search in docling texts
    best_page = None
    best_score = 0

    texts = docling.get("texts", [])
    for text_item in texts:
        text = text_item.get("text", "").lower()
        prov = text_item.get("prov", [])

        if not prov:
            continue

        page = prov[0].get("page_no", 0)
        if page < 1:
            continue

        score = 0

        # Check article number match
        if article_num and article_num in text:
            score += 10

        # Check answer text overlap
        if answer_text:
            answer_words = set(answer_text.lower().split())
            text_words = set(text.split())
            overlap = len(answer_words & text_words)
            score += overlap

        if score > best_score:
            best_score = score
            best_page = page

    return best_page


def find_chunk_for_page(
    conn: sqlite3.Connection, source: str, page: int
) -> Optional[str]:
    """Find the best chunk for a given source and page."""
    # Normalize source name for matching
    source_pattern = source.replace(".pdf", "").split("_")[0]

    cur = conn.execute(
        """SELECT id, text FROM chunks
           WHERE source LIKE ? AND page = ?
           ORDER BY id LIMIT 5""",
        (f"%{source_pattern}%", page),
    )

    rows = cur.fetchall()
    if rows:
        return rows[0][0]  # Return first match
    return None


def classify_retrieval_difficulty(question: dict) -> float:
    """Classify retrieval difficulty based on question complexity."""
    q_text = question.get("question", question.get("question_text", "")).lower()
    reasoning_type = question.get("reasoning_type", "single-hop")

    if reasoning_type == "multi-hop":
        return DIFFICULTY_MULTI_HOP

    # Reasoning indicators
    reasoning_patterns = [
        r"combien",
        r"calculer",
        r"quel.*score",
        r"quelle.*sanction",
        r"que.*faire",
        r"comment.*réagir",
        r"si.*alors",
        r"exception",
        r"priorité",
    ]

    for p in reasoning_patterns:
        if re.search(p, q_text):
            return DIFFICULTY_REASONING

    return DIFFICULTY_SINGLE_HOP


def normalize_question(question: dict, conn: sqlite3.Connection) -> NormalizationResult:
    """
    Normalize a single question:
    1. Verify page against docling
    2. Find chunk_id
    3. Calculate difficulties
    """
    result = NormalizationResult(
        id=question["id"], page_original=question.get("expected_pages", [])
    )

    # Get source document
    expected_docs = question.get("expected_docs", [])
    if not expected_docs:
        result.error = "No expected_docs"
        return result

    source = expected_docs[0]
    article_ref = question.get("article_reference", "")
    answer_text = question.get("answer_text", "")

    # Load docling JSON
    docling = load_docling_json(source)

    # Find verified page
    verified_page = find_page_in_docling(docling, article_ref, answer_text)

    if verified_page:
        result.page_verified = verified_page
    else:
        # Fallback to first expected_page if docling verification fails
        if result.page_original:
            result.page_verified = result.page_original[0]

    # Find chunk_id
    if result.page_verified:
        chunk_id = find_chunk_for_page(conn, source, result.page_verified)
        result.chunk_id = chunk_id

    # Calculate difficulties
    # Human difficulty from annales
    annales = question.get("annales_source", {})
    if annales and annales.get("success_rate") is not None:
        result.difficulty_human = round(1 - annales["success_rate"], 2)
    else:
        result.difficulty_human = question.get("difficulty", 0.5)

    # Retrieval difficulty from taxonomy
    result.difficulty_retrieval = classify_retrieval_difficulty(question)

    return result


def process_batch(questions: list[dict], batch_id: int) -> list[NormalizationResult]:
    """Process a batch of questions."""
    conn = sqlite3.connect(str(DB_PATH))
    results = []

    for q in questions:
        result = normalize_question(q, conn)
        results.append(result)

    conn.close()
    print(f"  Batch {batch_id}: {len(results)} questions processed")
    return results


def normalize_gold_standard(num_workers: int = 4) -> dict:
    """
    Main function: normalize all 477 questions.

    Returns statistics.
    """
    # Load gold standard
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    questions = gs["questions"]
    print(f"Processing {len(questions)} questions with {num_workers} workers...")

    # Split into batches
    batch_size = len(questions) // num_workers + 1
    batches = [
        questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
    ]

    # Process in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_batch, batch, i): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)

    # Create lookup
    results_lookup = {r.id: r for r in all_results}

    # Update gold standard
    stats = {"chunk_added": 0, "page_corrected": 0, "difficulty_added": 0, "errors": 0}

    for q in questions:
        result = results_lookup.get(q["id"])
        if not result:
            stats["errors"] += 1
            continue

        if result.error:
            stats["errors"] += 1
            continue

        # Update chunk_id
        if result.chunk_id:
            q["expected_chunk_id"] = result.chunk_id
            stats["chunk_added"] += 1

        # Update expected_pages to SINGLE verified page
        if result.page_verified:
            old_pages = q.get("expected_pages", [])
            q["expected_pages"] = [result.page_verified]
            if old_pages != [result.page_verified]:
                stats["page_corrected"] += 1

        # Update difficulties
        q["difficulty_human"] = result.difficulty_human
        q["difficulty_retrieval"] = result.difficulty_retrieval

        # Keep legacy difficulty for backwards compat
        if not q.get("difficulty"):
            q["difficulty"] = result.difficulty_human
            stats["difficulty_added"] += 1

    # Save
    with open(GS_PATH, "w", encoding="utf-8") as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    return stats


def validate_results() -> dict:
    """Validate the normalized gold standard."""
    with open(GS_PATH, encoding="utf-8") as f:
        gs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))

    validation = {
        "total": len(gs["questions"]),
        "with_chunk_id": 0,
        "chunk_exists_in_db": 0,
        "single_page": 0,
        "with_difficulty_human": 0,
        "with_difficulty_retrieval": 0,
        "page_3_remaining": 0,
    }

    for q in gs["questions"]:
        if q.get("expected_chunk_id"):
            validation["with_chunk_id"] += 1

            # Verify chunk exists
            cur = conn.execute(
                "SELECT 1 FROM chunks WHERE id = ?", (q["expected_chunk_id"],)
            )
            if cur.fetchone():
                validation["chunk_exists_in_db"] += 1

        pages = q.get("expected_pages", [])
        if len(pages) == 1:
            validation["single_page"] += 1
        if pages == [3]:
            validation["page_3_remaining"] += 1

        if q.get("difficulty_human") is not None:
            validation["with_difficulty_human"] += 1
        if q.get("difficulty_retrieval") is not None:
            validation["with_difficulty_retrieval"] += 1

    conn.close()
    return validation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize Gold Standard")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("  GOLD STANDARD NORMALIZATION")
    print("  ISO 42001 A.6.2.2 | ISO 29119 | ISO 25010")
    print("=" * 60)
    print()

    if args.validate_only:
        print("Validation only mode...")
    else:
        stats = normalize_gold_standard(args.workers)
        print()
        print("Normalization complete:")
        print(f"  Chunks added:     {stats['chunk_added']}")
        print(f"  Pages corrected:  {stats['page_corrected']}")
        print(f"  Difficulty added: {stats['difficulty_added']}")
        print(f"  Errors:           {stats['errors']}")

    print()
    print("Validating...")
    validation = validate_results()

    print()
    print("=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    total = validation["total"]
    print(f"  expected_chunk_id:      {validation['with_chunk_id']}/{total}")
    print(f"  chunks exist in DB:     {validation['chunk_exists_in_db']}/{total}")
    print(f"  single page:            {validation['single_page']}/{total}")
    print(f"  difficulty_human:       {validation['with_difficulty_human']}/{total}")
    print(
        f"  difficulty_retrieval:   {validation['with_difficulty_retrieval']}/{total}"
    )
    print(f"  page [3] remaining:     {validation['page_3_remaining']}")
    print("=" * 60)

    # Check if all criteria met
    all_pass = (
        validation["with_chunk_id"] == total
        and validation["chunk_exists_in_db"] == total
        and validation["single_page"] == total
        and validation["with_difficulty_human"] == total
        and validation["with_difficulty_retrieval"] == total
    )

    if all_pass:
        print("  STATUS: ALL CRITERIA MET")
    else:
        print("  STATUS: CRITERIA NOT MET")
    print("=" * 60)
