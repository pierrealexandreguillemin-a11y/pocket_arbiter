#!/usr/bin/env python3
"""
Audit complet des chunk_ids du Gold Standard Annales.

Vérifie que chaque question a un chunk_id valide et que la réponse
est effectivement trouvable dans le chunk associé.

ISO 42001: Vérification anti-hallucination
ISO 25010: Qualité des données
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
MIN_ANSWER_MATCH_RATIO = 0.3  # Au moins 30% des mots-clés de la réponse


def load_json(path: str) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """Save JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    # Remove accents for matching
    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ç": "c",
        "œ": "oe",
        "æ": "ae",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_keywords(text: str, min_length: int = 4) -> list[str]:
    """Extract meaningful keywords from text."""
    text = normalize_text(text)
    # Remove common words
    stopwords = {
        "pour",
        "dans",
        "avec",
        "cette",
        "celui",
        "celle",
        "sont",
        "etre",
        "avoir",
        "fait",
        "faire",
        "peut",
        "doit",
        "tous",
        "tout",
        "plus",
        "moins",
        "entre",
        "autres",
        "autre",
        "comme",
        "ainsi",
        "donc",
        "lors",
        "apres",
        "avant",
        "depuis",
        "pendant",
        "selon",
        "sans",
        "sous",
        "vers",
        "chez",
        "contre",
        "entre",
        "parmi",
        "sauf",
    }
    words = re.findall(r"\b[a-z]+\b", text)
    return [w for w in words if len(w) >= min_length and w not in stopwords]


def check_answer_in_chunk(answer: str, chunk_text: str) -> dict:
    """
    Check if answer content is findable in chunk.
    Returns detailed analysis.
    """
    answer_norm = normalize_text(answer)
    chunk_norm = normalize_text(chunk_text)

    # Method 1: Direct substring match (first 50 chars)
    direct_match = answer_norm[:50] in chunk_norm

    # Method 2: Keyword matching
    answer_keywords = extract_keywords(answer)
    if not answer_keywords:
        return {
            "direct_match": direct_match,
            "keyword_ratio": 0.0,
            "keywords_found": [],
            "keywords_missing": [],
            "answerable": direct_match,
        }

    found = [kw for kw in answer_keywords if kw in chunk_norm]
    missing = [kw for kw in answer_keywords if kw not in chunk_norm]
    ratio = len(found) / len(answer_keywords) if answer_keywords else 0

    # Consider answerable if direct match OR high keyword ratio
    answerable = direct_match or ratio >= MIN_ANSWER_MATCH_RATIO

    return {
        "direct_match": direct_match,
        "keyword_ratio": ratio,
        "keywords_found": found[:10],
        "keywords_missing": missing[:10],
        "answerable": answerable,
    }


def find_best_chunk(
    answer: str, chunks: list[dict], source_doc: str = None
) -> list[dict]:
    """
    Find chunks that best match the answer.
    Returns top candidates with scores.
    """
    candidates = []
    answer_keywords = set(extract_keywords(answer))

    if not answer_keywords:
        return []

    for chunk in chunks:
        # Filter by source document if specified
        if source_doc and source_doc not in chunk["id"]:
            continue

        chunk_norm = normalize_text(chunk["text"])
        found = sum(1 for kw in answer_keywords if kw in chunk_norm)
        ratio = found / len(answer_keywords)

        if ratio >= 0.2:  # At least 20% match
            candidates.append(
                {
                    "chunk_id": chunk["id"],
                    "score": ratio,
                    "found_keywords": found,
                    "total_keywords": len(answer_keywords),
                }
            )

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:5]


def audit_gold_standard(gs_path: str, chunks_path: str) -> dict:
    """
    Full audit of Gold Standard chunk_ids.
    """
    gs = load_json(gs_path)
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c["text"] for c in chunks}

    results = {
        "audit_date": datetime.now().isoformat(),
        "gs_version": gs.get("version", "unknown"),
        "total_questions": len(gs["questions"]),
        "summary": {
            "chunk_id_valid": 0,
            "chunk_id_missing": 0,
            "answerable": 0,
            "not_answerable": 0,
        },
        "issues": [],
        "questions_detail": [],
    }

    for q in gs["questions"]:
        qid = q["id"]
        chunk_id = q.get("expected_chunk_id", "")
        answer = q.get("expected_answer", "")
        expected_docs = q.get("expected_docs", [])

        detail = {
            "id": qid,
            "chunk_id": chunk_id,
            "answer_preview": answer[:60],
        }

        # Check 1: Chunk ID exists
        if not chunk_id:
            results["summary"]["chunk_id_missing"] += 1
            detail["status"] = "MISSING_CHUNK_ID"
            detail["issue"] = "No expected_chunk_id"
            results["issues"].append(detail.copy())
            results["questions_detail"].append(detail)
            continue

        if chunk_id not in chunk_index:
            results["summary"]["chunk_id_missing"] += 1
            detail["status"] = "INVALID_CHUNK_ID"
            detail["issue"] = "Chunk ID not found in index"

            # Try to find best matching chunk
            source_doc = expected_docs[0] if expected_docs else None
            candidates = find_best_chunk(answer, chunks, source_doc)
            if candidates:
                detail["suggested_chunks"] = candidates[:3]

            results["issues"].append(detail.copy())
            results["questions_detail"].append(detail)
            continue

        results["summary"]["chunk_id_valid"] += 1

        # Check 2: Answer in chunk
        chunk_text = chunk_index[chunk_id]
        check = check_answer_in_chunk(answer, chunk_text)
        detail["answer_check"] = check

        if check["answerable"]:
            results["summary"]["answerable"] += 1
            detail["status"] = "OK"
        else:
            results["summary"]["not_answerable"] += 1
            detail["status"] = "ANSWER_NOT_IN_CHUNK"
            detail["issue"] = (
                f"Answer keywords not found (ratio: {check['keyword_ratio']:.2f})"
            )

            # Try to find better chunk
            source_doc = expected_docs[0] if expected_docs else None
            candidates = find_best_chunk(answer, chunks, source_doc)
            if candidates and candidates[0]["score"] > check["keyword_ratio"]:
                detail["suggested_chunks"] = candidates[:3]

            results["issues"].append(detail.copy())

        results["questions_detail"].append(detail)

    return results


def main():
    """Main audit pipeline."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base_path = Path(__file__).parent.parent.parent.parent
    gs_path = base_path / "tests" / "data" / "gold_standard_annales_fr_v7.json"
    chunks_path = base_path / "corpus" / "processed" / "chunks_mode_b_fr.json"
    output_path = base_path / "tests" / "data" / "chunk_id_audit_report.json"

    print("=" * 70)
    print("AUDIT CHUNK_IDS - Gold Standard Annales")
    print("=" * 70)

    print("\nLoading data...")
    print(f"  GS: {gs_path}")
    print(f"  Chunks: {chunks_path}")

    results = audit_gold_standard(str(gs_path), str(chunks_path))

    print(f"\n{'=' * 70}")
    print("RÉSULTATS DE L'AUDIT")
    print("=" * 70)
    print(f"\nVersion GS: {results['gs_version']}")
    print(f"Total questions: {results['total_questions']}")
    print()
    print("Chunk IDs:")
    print(f"  - Valides: {results['summary']['chunk_id_valid']}")
    print(f"  - Manquants/Invalides: {results['summary']['chunk_id_missing']}")
    print()
    print("Answerability:")
    print(f"  - Réponse dans chunk: {results['summary']['answerable']}")
    print(f"  - Réponse ABSENTE: {results['summary']['not_answerable']}")
    print()

    pct_ok = 100 * results["summary"]["answerable"] / results["total_questions"]
    print(f"TAUX ANSWERABILITY: {pct_ok:.1f}%")
    print()

    if results["issues"]:
        print(f"{'=' * 70}")
        print(f"PROBLÈMES DÉTECTÉS: {len(results['issues'])}")
        print("=" * 70)

        # Group by issue type
        by_type = defaultdict(list)
        for issue in results["issues"]:
            by_type[issue["status"]].append(issue)

        for status, issues in by_type.items():
            print(f"\n[{status}] - {len(issues)} questions")
            for issue in issues[:5]:
                print(f"  - {issue['id']}")
                if "suggested_chunks" in issue:
                    best = issue["suggested_chunks"][0]
                    print(
                        f"    Suggestion: {best['chunk_id']} (score: {best['score']:.2f})"
                    )
            if len(issues) > 5:
                print(f"  ... et {len(issues) - 5} autres")

    # Save detailed report
    save_json(results, str(output_path))
    print(f"\nRapport détaillé: {output_path}")

    return 0 if pct_ok >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
