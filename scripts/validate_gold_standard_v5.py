"""
Validate Gold Standard v5 against extracted corpus.

Assigns expected_pages by finding keywords in corpus text.

ISO Reference:
    - ISO/IEC 42001 A.7.3 - Validation independence
    - ISO/IEC 25010 - Recall measurement

Document ID: SCRIPT-VAL-001
Version: 1.0
Date: 2026-01-16
Author: Claude Opus 4.5
"""

import json
from pathlib import Path
from typing import TypedDict


class PageMatch(TypedDict):
    """Type for page match result."""

    page: int
    source: str
    keywords_found: list[str]
    score: int


def load_corpus(raw_dir: Path) -> dict[str, dict[int, str]]:
    """Load extracted corpus indexed by source and page."""
    corpus: dict[str, dict[int, str]] = {}
    for json_file in raw_dir.glob("*.json"):
        if json_file.name == "extraction_report.json":
            continue
        with open(json_file, encoding="utf-8") as f:
            doc = json.load(f)
        source = doc.get("filename", json_file.stem + ".pdf")
        corpus[source] = {}
        for page in doc.get("pages", []):
            page_num = page.get("page_num", page.get("page", 0))
            text = page.get("text", "").lower()
            corpus[source][page_num] = text
    return corpus


def find_pages_with_keywords(
    corpus: dict[str, dict[int, str]],
    expected_docs: list[str],
    keywords: list[str],
    max_pages: int = 3,
) -> list[PageMatch]:
    """Find pages containing keywords."""
    matches: list[PageMatch] = []
    kw_lower = [kw.lower() for kw in keywords]

    for doc in expected_docs:
        # Try to find the doc (may have slight name variations)
        doc_key = None
        for key in corpus.keys():
            if doc.replace(".pdf", "") in key or key.replace(".pdf", "") in doc:
                doc_key = key
                break

        if not doc_key:
            continue

        for page_num, text in corpus[doc_key].items():
            found_keywords = [kw for kw in kw_lower if kw in text]
            if len(found_keywords) >= 2:  # At least 2 keywords
                matches.append(PageMatch(
                    page=page_num,
                    source=doc_key,
                    keywords_found=found_keywords,
                    score=len(found_keywords),
                ))

    # Sort by score (most keywords) and return top pages
    matches.sort(key=lambda x: (-x["score"], x["page"]))
    return matches[:max_pages]


def validate_gold_standard(
    gold_path: Path, corpus_dir: Path, output_path: Path
) -> dict:
    """Validate gold standard against corpus."""
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    corpus = load_corpus(corpus_dir)
    print(f"Corpus: {len(corpus)} documents")

    validated_count = 0
    for q in gold["questions"]:
        matches = find_pages_with_keywords(
            corpus, q["expected_docs"], q["keywords"]
        )

        if matches:
            q["expected_pages"] = [m["page"] for m in matches]
            q["validation"] = {
                "status": "VALIDATED",
                "method": "keyword_in_corpus",
                "details": matches,
            }
            validated_count += 1
        else:
            q["expected_pages"] = []
            q["validation"] = {
                "status": "NOT_FOUND",
                "method": "keyword_in_corpus",
                "details": [],
            }

        status = q["validation"]["status"]
        pages = q["expected_pages"]
        print(f"{q['id']}: {pages} [{status}]")

    # Update statistics
    gold["statistics"]["validated_questions"] = validated_count
    gold["statistics"]["validation_rate"] = f"{validated_count/len(gold['questions']):.1%}"

    # Save validated version
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

    return {
        "total": len(gold["questions"]),
        "validated": validated_count,
        "rate": validated_count / len(gold["questions"]),
    }


def main() -> None:
    """Validate gold standard v5."""
    print("=== VALIDATION GOLD STANDARD V5 ===")
    print()

    result = validate_gold_standard(
        gold_path=Path("tests/data/questions_fr_v5_arbiter.json"),
        corpus_dir=Path("corpus/processed/raw_fr"),
        output_path=Path("tests/data/questions_fr.json"),
    )

    print()
    print("=== RÉSULTAT ===")
    print(f"Questions validées: {result['validated']}/{result['total']} ({result['rate']:.1%})")
    print("Sauvegardé: tests/data/questions_fr.json")


if __name__ == "__main__":
    main()
