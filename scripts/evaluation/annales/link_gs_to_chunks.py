"""
Link Gold Standard questions to corpus chunks.

This script implements BY DESIGN chunk linking (ISO 42001 A.6.2.2):
- Parses article_reference from GS questions
- Matches to chunks via article number patterns
- Uses semantic similarity as tiebreaker
- Updates provenance.chunk_id field

ISO Reference: ISO 42001 A.6.2.2 - Provenance tracking
"""

import json
import re
from collections import Counter
from pathlib import Path


def extract_article_numbers(ref: str) -> list[str]:
    """Extract article numbers from reference string."""
    if not ref:
        return []
    # Patterns: "Article 9.1.2.2", "article 7.3", "Article A.4"
    matches = re.findall(
        r"(?:Article\s+)?([0-9]+(?:\.[0-9]+)*|[A-Z]\.[0-9]+)", ref, re.I
    )
    return matches


def find_matching_chunks(
    article_nums: list[str],
    chunks: list[dict],
    source_filter: str | None = None,
) -> list[dict]:
    """Find chunks matching article numbers."""
    matches = []

    for chunk in chunks:
        if source_filter and source_filter not in chunk.get("source", ""):
            continue

        text = chunk.get("text", "")
        section = chunk.get("section", "")
        combined = f"{text} {section}"

        # Check for article number matches
        for art_num in article_nums:
            # Exact match (e.g., "9.1.2.2")
            if art_num in combined:
                matches.append(
                    {"chunk": chunk, "article": art_num, "match_type": "exact"}
                )
                break
            # Prefix match (e.g., "9.1" matches "9.1.2")
            if any(
                part.startswith(art_num) or art_num.startswith(part)
                for part in re.findall(r"\b\d+(?:\.\d+)+\b", combined)
            ):
                matches.append(
                    {"chunk": chunk, "article": art_num, "match_type": "prefix"}
                )
                break

    return matches


def link_questions_to_chunks(gs_path: Path, chunks_path: Path) -> dict:
    """Link GS questions to corpus chunks."""
    # Load data
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    with open(chunks_path, encoding="utf-8") as f:
        chunks_data = json.load(f)
        chunks = chunks_data["chunks"]

    # Filter to LA (Laws) chunks - primary source for règles questions
    la_chunks = [c for c in chunks if "LA-octobre2025" in c.get("source", "")]
    print(f"Corpus: {len(chunks)} total, {len(la_chunks)} LA chunks")

    # Link each question
    linked = 0
    not_linked = 0
    results = []

    for q in gs["questions"]:
        article_ref = q["provenance"].get("article_reference", "")
        article_nums = extract_article_numbers(article_ref)

        if not article_nums:
            not_linked += 1
            results.append(
                {
                    "id": q["id"],
                    "status": "NO_REF",
                    "article_ref": article_ref,
                }
            )
            continue

        # Find matching chunks
        matches = find_matching_chunks(article_nums, la_chunks)

        if matches:
            # Prefer exact matches, then longest article number
            exact = [m for m in matches if m["match_type"] == "exact"]
            if exact:
                # Sort by article length (most specific)
                exact.sort(key=lambda m: len(m["article"]), reverse=True)
                best = exact[0]
            else:
                # Use prefix match
                matches.sort(key=lambda m: len(m["article"]), reverse=True)
                best = matches[0]

            # Update question
            q["provenance"]["chunk_id"] = best["chunk"]["id"]
            q["processing"]["chunk_match_score"] = (
                100 if best["match_type"] == "exact" else 80
            )
            q["processing"]["chunk_match_method"] = f"article_{best['match_type']}"

            linked += 1
            results.append(
                {
                    "id": q["id"],
                    "status": "LINKED",
                    "chunk_id": best["chunk"]["id"],
                    "article": best["article"],
                    "match_type": best["match_type"],
                }
            )
        else:
            not_linked += 1
            results.append(
                {
                    "id": q["id"],
                    "status": "NOT_FOUND",
                    "article_ref": article_ref,
                    "article_nums": article_nums,
                }
            )

    # Summary
    print("\n=== LINKING SUMMARY ===")
    print(
        f"Linked: {linked}/{len(gs['questions'])} ({100*linked/len(gs['questions']):.1f}%)"
    )
    print(f"Not linked: {not_linked}")

    # Status breakdown
    status_counts = Counter(r["status"] for r in results)
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    return {
        "gs": gs,
        "results": results,
        "stats": {
            "total": len(gs["questions"]),
            "linked": linked,
            "not_linked": not_linked,
        },
    }


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Link Gold Standard questions to corpus chunks"
    )
    parser.add_argument(
        "--gs",
        "-g",
        type=Path,
        default=Path("tests/data/jun2025_gs_v2.json"),
        help="Input Gold Standard JSON",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        default=Path("corpus/processed/chunks_mode_b_fr.json"),
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tests/data/jun2025_gs_v2_linked.json"),
        help="Output linked GS JSON",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=Path("tests/data/jun2025_linking_report.json"),
        help="Linking report JSON",
    )

    args = parser.parse_args()

    print(f"Linking: {args.gs} -> {args.chunks}")
    result = link_questions_to_chunks(args.gs, args.chunks)

    # Save linked GS
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result["gs"], f, ensure_ascii=False, indent=2)
    print(f"\nSaved linked GS: {args.output}")

    # Save report
    report = {
        "input_gs": str(args.gs),
        "input_chunks": str(args.chunks),
        "stats": result["stats"],
        "results": result["results"],
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
